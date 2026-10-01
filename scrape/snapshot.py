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
import random
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from scrape.common import ODDS_API
from paths import V2_DIR

OUT_DIR = V2_DIR / "odds_snapshots"
LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36",
           "Referer": "https://race.netkeiba.com/", "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_odds_min", "place_odds_max",
        "minutes_before", "seconds_to_post", "post_time", "fetched_at", "official_datetime", "api_status"]
LADDER = [60, 30, 20, 15, 10, 7, 5, 3, 2, 1]   # 発走の何分前に取るか
TICK_SEC = 20        # 予定を確認する間隔
GRACE_SEC = 60       # 発走後この秒数までは取得を試みる（締切直後の値も残す）


def race_schedule(date):
    """その日の (race_id, 発走時刻) の一覧"""
    r = requests.get(LIST_URL.format(date), headers=HEADERS, timeout=20)
    r.raise_for_status()
    html = r.content.decode("utf-8", errors="replace")
    races = {}
    for item in html.split('<li class="RaceList_DataItem')[1:]:
        m_id = re.search(r"race_id=(\d{12})", item)
        m_time = re.search(r'RaceList_Itemtime">\s*(\d{1,2}:\d{2})', item)
        if m_id and m_time:
            races[m_id.group(1)] = datetime.strptime(f"{date} {m_time.group(1)}", "%Y%m%d %H:%M")
    return sorted(races.items(), key=lambda x: x[1])


def fetch_odds(race_id):
    try:
        js = requests.get(ODDS_API.format(race_id=race_id, bet_type=1, action="update"), headers=HEADERS,
                          timeout=20).json()
    except (requests.RequestException, ValueError) as e:
        return [], f"error: {e}"
    data = js.get("data") if isinstance(js.get("data"), dict) else {}
    odds = data.get("odds") or {}
    win, place = odds.get("1") or {}, odds.get("2") or {}
    rows = [{"race_id": race_id, "horse_number": int(num), "win_odds": v[0], "popularity": v[2],
             "place_odds_min": (place.get(num) or [None, None])[0], "place_odds_max": (place.get(num) or [None, None])[1],
             "official_datetime": data.get("official_datetime", "")} for num, v in win.items()]
    return rows, js.get("status", "")


def load_done(path):
    """途中から起動しても、取得済みの (race_id, 何分前) は飛ばす"""
    if not path.exists():
        return set()
    d = pd.read_csv(path, usecols=["race_id", "minutes_before"], dtype={"race_id": str}, on_bad_lines="skip")
    return set(zip(d["race_id"], pd.to_numeric(d["minutes_before"], errors="coerce")))


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
            time.sleep(random.uniform(1.0, 2.0))
        print("完了:", out)
        return

    minutes = sorted((int(m) for m in args.minutes.split(",")) if args.minutes else LADDER, reverse=True)
    done = load_done(out)
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
        for post, rid, m in sorted(due):
            rows, status = fetch_odds(rid)
            save(out, rows, rid, m, post, status)
            done.add((rid, m))
            time.sleep(random.uniform(1.0, 2.0))
    print("完了:", out)
