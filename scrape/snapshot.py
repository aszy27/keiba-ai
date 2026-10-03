# scrape/snapshot.py
# 開催日のレースについて、発走前のオッズを何度も保存する（購入時点のオッズでの検証と、オッズの動きの記録のため）。
# 既定は発走の 60/30/20/15/10/7/5/3/2/1 分前。判定に使うのは3分前（docs/rebuild_plan.md「前向き検証」）だが、
# 取りこぼし対策と、締切直前のオッズの動きを残すために多めに取る。
# 開催日の朝に起動しておくと、最終レースまで待機しながら取得を続ける（PCがスリープしないようにしておく）。
# 実際にはタスクスケジューラ（keiba-odds-snapshot → live/run_snapshot_hidden.vbs → live/run_snapshot.ps1）が毎日起動する。
# 前向き検証の判定に使うデータなので、取得のタイミング・保存形式は判定まで変えないこと。
#
# 使い方: python -m scrape snapshot                          # 今日
#         python -m scrape snapshot --minutes 10,5,3
#         python -m scrape snapshot --date 20260913 --now    # 指定日の全レースを今すぐ1回取得（動作確認用）
#         （動作確認で本番のデータに混ぜたくないときは --out-dir で保存先を変える）
import json
import random
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from scrape.common import ODDS_API, http_get
from paths import V2_DIR

OUT_DIR = V2_DIR / "odds_snapshots"
LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36",
           "Referer": "https://race.netkeiba.com/", "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_odds_min", "place_odds_max",
        "minutes_before", "seconds_to_post", "post_time", "fetched_at", "official_datetime", "api_status"]
LADDER = [10, 3]   # 発走の何分前に取るか。判定に使うのは 3分前（無ければ10分前）。API は10〜15分に5回ほどで制限をかけるので絞った（2026-10-03。以前は 60/30/20/15/10/7/5/3/2/1）
TICK_SEC = 20        # 予定を確認する間隔
GRACE_SEC = 60       # 発走後この秒数までは取得を試みる（締切直後の値も残す）
API_SPACING_SEC = (15, 20)   # オッズの API への間隔（秒）。短い間に続けて取ると status=limit（制限）になる（2026-10-03 に判明）
LIMIT_STREAK = 3     # API の status=limit がこの回数続いたら控える
LIMIT_PAUSE_MIN = 6   # 次の予定（3分前など）に間に合うよう短めに


def race_schedule(date):
    """その日の (race_id, 発走時刻) の一覧"""
    r, status = http_get(LIST_URL.format(date), referer="https://race.netkeiba.com/", honor_cooldown=False)
    if r is None:
        raise RuntimeError(f"レース一覧を取得できない（{status}）")
    html = r.content.decode("utf-8", errors="replace")
    races = {}
    for item in html.split('<li class="RaceList_DataItem')[1:]:
        m_id = re.search(r"race_id=(\d{12})", item)
        m_time = re.search(r'RaceList_Itemtime">\s*(\d{1,2}:\d{2})', item)
        if m_id and m_time:
            races[m_id.group(1)] = datetime.strptime(f"{date} {m_time.group(1)}", "%Y%m%d %H:%M")
    return sorted(races.items(), key=lambda x: x[1])


def fetch_odds(race_id):
    r, status = http_get(ODDS_API.format(race_id=race_id, bet_type=1, action="update"),
                         referer="https://race.netkeiba.com/", honor_cooldown=False)
    if r is None:
        return [], f"error: {status}"
    try:
        js = r.json()
    except ValueError as e:
        return [], f"error: {e}"
    data = js.get("data") if isinstance(js.get("data"), dict) else {}
    odds = data.get("odds") or {}
    win, place = odds.get("1") or {}, odds.get("2") or {}
    rows = [{"race_id": race_id, "horse_number": int(num), "win_odds": v[0], "popularity": v[2],
             "place_odds_min": (place.get(num) or [None, None])[0], "place_odds_max": (place.get(num) or [None, None])[1],
             "official_datetime": data.get("official_datetime", "")} for num, v in win.items()]
    return rows, js.get("status", "")


def load_done(path):
    """途中から起動しても、取得済みの (race_id, 何分前) は飛ばす。API が制限中（status=limit）で取れなかった行は数えない"""
    if not path.exists():
        return set()
    d = pd.read_csv(path, usecols=["race_id", "minutes_before", "api_status"], dtype={"race_id": str}, on_bad_lines="skip")
    d = d[d["api_status"].astype(str).str.strip() != "limit"]
    return set(zip(d["race_id"], pd.to_numeric(d["minutes_before"], errors="coerce")))


class LimitBackoff:
    """API が status=limit（制限中）を返し続けたら控える（2026-10-03 の制限への対策）。
    LIMIT_STREAK 回続いたら LIMIT_PAUSE_MIN 分は取りに行かず、そのあと1回だけ試す。それも limit ならまた控える。
    控えている間も予定（発走後 GRACE_SEC 秒まで）は捨てないので、途中で解ければその時点のオッズを取れる。
    状態はファイルに残すので、タスクスケジューラが30分ごとに起動し直しても控え続ける。制限が無いときの動きは変えない"""

    def __init__(self, path):
        self.path = path
        self.streak = 0

    def _until(self):
        try:
            return datetime.fromisoformat(json.loads(self.path.read_text(encoding="utf-8"))["until"])
        except (OSError, ValueError, KeyError):
            return None

    def allow(self, now):
        until = self._until()
        return until is None or now >= until

    def record(self, status, now):
        if str(status).strip() == "limit":
            self.streak += 1
            if self.streak >= LIMIT_STREAK or self._until() is not None:
                until = now + timedelta(minutes=LIMIT_PAUSE_MIN)
                self.path.write_text(json.dumps({"until": until.isoformat(timespec="seconds")}), encoding="utf-8")
                print(f"  API が制限中（status=limit）なので {until:%H:%M} まで控える", flush=True)
        else:
            self.streak = 0
            if self.path.exists():
                self.path.unlink()
                print("  制限が解けた", flush=True)


def save(path, rows, race_id, minutes, post, status):
    now = datetime.now()
    df = pd.DataFrame(rows or [{"race_id": race_id}]).assign(
        minutes_before=minutes, seconds_to_post=round((post - now).total_seconds()), post_time=f"{post:%H:%M}",
        fetched_at=now.isoformat(timespec="seconds"), api_status=status)
    df.reindex(columns=COLS).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8")
    print(f"  {now:%H:%M:%S} {race_id} 発走{minutes}分前（実際 {round((post - now).total_seconds())}秒前） "
          f"{len(rows)}頭 status={status}", flush=True)


def run(args):
    """args: date / minutes / now / out_dir"""
    out_dir = args.out_dir or OUT_DIR
    schedule = race_schedule(args.date)
    if not schedule:
        print(f"{args.date} のレースが見つかりません（開催日でないか、ページの形式が変わった）")
        return
    print(f"{args.date}: {len(schedule)}R（{schedule[0][1]:%H:%M}〜{schedule[-1][1]:%H:%M}）", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.date}.csv"

    if args.now:
        for rid, post in schedule:
            rows, status = fetch_odds(rid)
            save(out, rows, rid, -1, post, status)
            time.sleep(random.uniform(*API_SPACING_SEC))
        print("完了:", out)
        return

    minutes = sorted((int(m) for m in args.minutes.split(",")) if args.minutes else LADDER, reverse=True)
    done = load_done(out)
    backoff = LimitBackoff(out_dir / "limit_state.json")
    print(f"取得予定: 1レースあたり {minutes} 分前 / 取得済み {len(done)} 件", flush=True)
    while True:
        now = datetime.now()
        # 予定時刻を過ぎていて、まだ取っていないものを「発走が近い順」に処理する
        due = [(post, rid, m) for rid, post in schedule for m in minutes
               if (rid, m) not in done and post - timedelta(minutes=m) <= now <= post + timedelta(seconds=GRACE_SEC)]
        if not due:
            if all(now > post + timedelta(seconds=GRACE_SEC) for _, post in schedule):
                break
            time.sleep(TICK_SEC)
            continue
        # 起動し直した直後などで同じレースの予定がいくつもたまっていたら、1回だけ取って一番近い予定として保存し、
        # たまっていた古い予定はまとめて片付ける（同じオッズを続けて何度も取らない）。普段は1レースに1つしか同時にたまらない
        latest = {}
        for post, rid, m in due:
            latest.setdefault(rid, []).append(m)
        for post, rid, m in sorted({(p_, r_, min(latest[r_])) for p_, r_, _ in due}):
            if not backoff.allow(datetime.now()):
                break                     # 控えている間は取りに行かない（予定は捨てない）
            rows, status = fetch_odds(rid)
            save(out, rows, rid, m, post, status)
            if str(status).strip() != "limit":
                done.update((rid, mm) for mm in latest[rid])   # 制限中で取れなかったものは、解けたら取り直す
            backoff.record(status, datetime.now())
            time.sleep(random.uniform(*API_SPACING_SEC))
        if not backoff.allow(datetime.now()):
            time.sleep(TICK_SEC)
    print("完了:", out)
