# v2/scrape_odds_snapshot.py
# 開催日のレースについて、発走の数分前の単勝・複勝オッズを保存する。
# 目的: 実際に買える時点のオッズと確定オッズの差を測り、確定オッズで計算した回収率をどれだけ割り引くべきかを決める。
# 開催日の朝に起動しておくと、最終レースまで待機しながら取得を続ける（PCがスリープしないようにしておく）。
#
# 使い方: python -m v2.scrape_odds_snapshot                          # 今日。発走30分前・10分前・3分前
#         python -m v2.scrape_odds_snapshot --minutes 15,5
#         python -m v2.scrape_odds_snapshot --date 20260913 --now    # 指定日の全レースを今すぐ1回ずつ取得（動作確認用）
import argparse
import random
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from v2.paths import V2_DIR

OUT_DIR = V2_DIR / "odds_snapshots"
LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={}"
API = ("https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&input=UTF-8&output=json"
       "&race_id={}&type=1&action=update&sort=odds&compress=0")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36",
           "Referer": "https://race.netkeiba.com/", "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_odds_min", "place_odds_max",
        "minutes_before", "post_time", "fetched_at", "official_datetime", "api_status"]


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
        js = requests.get(API.format(race_id), headers=HEADERS, timeout=20).json()
    except (requests.RequestException, ValueError) as e:
        return [], f"error: {e}"
    data = js.get("data") if isinstance(js.get("data"), dict) else {}
    odds = data.get("odds") or {}
    win, place = odds.get("1") or {}, odds.get("2") or {}
    rows = [{"race_id": race_id, "horse_number": int(num), "win_odds": v[0], "popularity": v[2],
             "place_odds_min": (place.get(num) or [None, None])[0], "place_odds_max": (place.get(num) or [None, None])[1],
             "official_datetime": data.get("official_datetime", "")} for num, v in win.items()]
    return rows, js.get("status", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--minutes", default="30,10,3", help="発走の何分前に取るか（カンマ区切り）")
    ap.add_argument("--now", action="store_true", help="待たずに全レースを1回ずつ取得する")
    args = ap.parse_args()

    schedule = race_schedule(args.date)
    if not schedule:
        print(f"{args.date} のレースが見つかりません（開催日でないか、ページの形式が変わった）")
        return
    print(f"{args.date}: {len(schedule)}R（{schedule[0][1]:%H:%M}〜{schedule[-1][1]:%H:%M}）")
    if args.now:
        jobs = [(datetime.now(), rid, post, None) for rid, post in schedule]
    else:
        minutes = [int(m) for m in args.minutes.split(",")]
        jobs = sorted((post - timedelta(minutes=m), rid, post, m) for rid, post in schedule for m in minutes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.date}.csv"
    for due, rid, post, m in jobs:
        wait = (due - datetime.now()).total_seconds()
        if wait < -120:
            continue  # 2分以上過ぎた取得予定は飛ばす（途中から起動した場合）
        if wait > 0:
            print(f"  次: {rid} 発走{m}分前（{due:%H:%M}）まで待機")
            time.sleep(wait)
        rows, status = fetch_odds(rid)
        now = datetime.now().isoformat(timespec="seconds")
        df = pd.DataFrame(rows or [{"race_id": rid}]).assign(minutes_before=m, post_time=f"{post:%H:%M}",
                                                              fetched_at=now, api_status=status)
        df.reindex(columns=COLS).to_csv(out, mode="a", index=False, header=not out.exists(), encoding="utf-8")
        print(f"  {now} {rid} {len(rows)}頭 status={status}")
        time.sleep(random.uniform(0.5, 1.0))
    print("完了:", out)


if __name__ == "__main__":
    main()
