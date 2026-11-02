# scrape/snapshot.py
# 開催日のレースについて、発走前のオッズを保存する（購入時点のオッズでの検証のため）。
# 発走の 30/10/3 分前に取る。判定に使うのは 3分前 → 10分前 → 30分前（docs/rebuild_plan.md「前向き検証」）。
# オッズの動き（時系列）は使わないので、判定に使う時点だけを取る（2026-10-06）。
# 開催日の朝に起動しておくと、最終レースまで待機しながら取得を続ける（PCがスリープしないようにしておく）。
# 実際にはタスクスケジューラ（keiba-odds-snapshot → live/run_snapshot_hidden.vbs → live/run_snapshot.ps1）が毎日起動する。
# 前向き検証の判定に使うデータなので、取得のタイミング・保存形式は判定まで変えないこと。
# 2026-10-26 から、同じ時点の3連複・3連単（全通り）も JRA 公式から取って <保存先>/exotic/YYYYMMDD_{trio,trifecta}.csv に残す
# （組み合わせ券の前向き検証用。docs/rebuild_plan.md「組み合わせ券の前向き検証と掛け金の決め方」）。単勝・複勝のファイルの形式は変えず、
# 取る順番も単勝・複勝が先（その時点に予定のあるレースの単勝・複勝を取り終えてから3連複・3連単を取る）。
# 2026-11-02 から馬連・馬単（全通り）も同じ時点で取り、<保存先>/exotic/YYYYMMDD_{quinella,exacta}.csv に残す（3連複・3連単の後に取る）。
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
LADDER = [30, 10, 3]   # 発走の何分前に取るか（判定に使うのは 3分前 → 10分前 → 30分前）
TICK_SEC = 20        # 予定を確認する間隔
GRACE_SEC = 60       # 発走後この秒数までは取得を試みる（締切直後の値も残す）
API_SPACING_SEC = (5, 8)   # オッズの取得の間隔（秒）。JRA 公式から取る。netkeiba の API は閲覧回数の制限がある（2026-10-03）
LIMIT_STREAK = 3     # API の status=limit がこの回数続いたら控える
LIMIT_PAUSE_MIN = 6   # 次の予定（3分前など）に間に合うよう短めに
EXOTIC_BETS = ["trio", "trifecta", "quinella", "exacta"]   # 取る順番（前向き検証中の3連複・3連単が先）
EXOTIC_COLS = ["race_id", "combo", "odds", "minutes_before", "seconds_to_post", "post_time", "fetched_at",
               "official_datetime", "api_status"]


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


def fetch_odds(race_id, date=None):
    """発売中の単勝・複勝 → (行, 状態)。JRA 公式から取り、取れなければ netkeiba の API（ログインしていない人は1日5回まで）"""
    if date:
        from scrape import jra_odds
        rows, status = jra_odds.fetch_odds(race_id, date)
        if rows or status == "jra_presale":
            for r in rows:
                r.setdefault("official_datetime", "")
            return rows, status
    return fetch_odds_netkeiba(race_id)


def fetch_odds_netkeiba(race_id):
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


def exotic_path(out_dir, date, bet):
    return out_dir / "exotic" / f"{date}_{bet}.csv"


def load_done_exotic(out_dir, date):
    """取り終えた (race_id, 何分前, 券種)。券種ごとに数えるので、券種を足した日に起動し直しても取り済みの券種は取り直さない"""
    done = set()
    for bet in EXOTIC_BETS:
        p = exotic_path(out_dir, date, bet)
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=["race_id", "minutes_before"], dtype={"race_id": str}, on_bad_lines="skip")
        done.update((r, m, bet) for r, m in zip(d["race_id"], pd.to_numeric(d["minutes_before"], errors="coerce")))
    return done


def save_exotic(out_dir, date, rows, race_id, minutes, post, status, bet):
    now = datetime.now()
    path = exotic_path(out_dir, date, bet)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows or [{"race_id": race_id}]).assign(
        minutes_before=minutes, seconds_to_post=round((post - now).total_seconds()), post_time=f"{post:%H:%M}",
        fetched_at=now.isoformat(timespec="seconds"), api_status=status)
    df.reindex(columns=EXOTIC_COLS).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8")
    print(f"  {now:%H:%M:%S} {race_id} 発走{minutes}分前 {bet} {len(rows)}通り status={status}", flush=True)


def fetch_save_exotic(out_dir, date, race_id, minutes, post, bets=EXOTIC_BETS):
    from scrape import jra_odds
    for bet in bets:
        rows, status = jra_odds.fetch_exotic(race_id, date, bet)
        save_exotic(out_dir, date, rows, race_id, minutes, post, status, bet)
        time.sleep(random.uniform(*API_SPACING_SEC))


def exotic_due(schedule, minutes, done, done_x, now):
    """組み合わせ券: 単勝・複勝を取り終えた時点のうち、まだ取っていない券種があるもの（予定の窓の中だけ）→ {race_id: (発走, [何分前])}。
    単勝・複勝の予定が無くても見るので、単勝・複勝と組み合わせ券の間で止まって起動し直した時点も取る"""
    due_x = {}
    for rid, post in schedule:
        ms = [m for m in minutes if (rid, m) in done
              and any((rid, m, b) not in done_x for b in EXOTIC_BETS)
              and post - timedelta(minutes=m) <= now <= post + timedelta(seconds=GRACE_SEC)]
        if ms:
            due_x[rid] = (post, ms)
    return due_x


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
            rows, status = fetch_odds(rid, args.date)
            save(out, rows, rid, -1, post, status)
            time.sleep(random.uniform(*API_SPACING_SEC))
            if not args.no_exotic:
                fetch_save_exotic(out_dir, args.date, rid, -1, post)
        print("完了:", out)
        return

    minutes = sorted((int(m) for m in args.minutes.split(",")) if args.minutes else LADDER, reverse=True)
    done = load_done(out)
    done_x = set() if args.no_exotic else load_done_exotic(out_dir, args.date)
    backoff = LimitBackoff(out_dir / "limit_state.json")
    print(f"取得予定: 1レースあたり {minutes} 分前 / 取得済み {len(done)} 件", flush=True)
    while True:
        now = datetime.now()
        # 予定時刻を過ぎていて、まだ取っていないものを「発走が近い順」に処理する
        due = [(post, rid, m) for rid, post in schedule for m in minutes
               if (rid, m) not in done and post - timedelta(minutes=m) <= now <= post + timedelta(seconds=GRACE_SEC)]
        if not due and (args.no_exotic or not exotic_due(schedule, minutes, done, done_x, now)):
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
            rows, status = fetch_odds(rid, args.date)
            save(out, rows, rid, m, post, status)
            if str(status).strip() != "limit":
                done.update((rid, mm) for mm in latest[rid])   # 制限中で取れなかったものは、解けたら取り直す
            backoff.record(status, datetime.now())
            time.sleep(random.uniform(*API_SPACING_SEC))
        if not args.no_exotic:
            due_x = exotic_due(schedule, minutes, done, done_x, datetime.now())
            for rid, (post, ms) in sorted(due_x.items(), key=lambda x: x[1][0]):
                bets = [b for b in EXOTIC_BETS if any((rid, m, b) not in done_x for m in ms)]
                fetch_save_exotic(out_dir, args.date, rid, min(ms), post, bets)
                done_x.update((rid, m, b) for m in ms for b in bets)
        if not backoff.allow(datetime.now()):
            time.sleep(TICK_SEC)
    print("完了:", out)
