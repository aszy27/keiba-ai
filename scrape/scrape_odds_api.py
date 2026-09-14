# scrape/scrape_odds_api.py
# netkeiba のオッズAPIから各馬の確定単勝オッズ・人気・複勝オッズ（下限/上限）を1レース1リクエストで取得する。
# 例: python scrape/scrape_odds_api.py --years 2024,2025,2026
import argparse
import glob
import os
import random
import sys
import time

import pandas as pd
from tqdm import tqdm

from scrape_other_data import DATA_DIR, DATA_SEARCH_DIRS, create_session, get_headers

FILE_OUT = str(DATA_DIR / "odds_api_progress.csv")
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_min", "place_max", "official_datetime"]
API = ("https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&input=UTF-8&output=json"
       "&race_id={}&type=1&action=init&sort=odds&compress=0")


def fetch(session, rid):
    try:
        r = session.get(API.format(rid), headers=get_headers(), timeout=20)
    except Exception:
        return "NETWORK_ERROR"
    if r.status_code != 200:
        return "BLOCK"
    try:
        js = r.json()
    except ValueError:
        return "BLOCK"
    data = js.get("data") if isinstance(js.get("data"), dict) else {}
    odds = data.get("odds") or {}
    if js.get("status") != "result" or "1" not in odds:
        return None
    place = odds.get("2", {})
    rows = []
    for num, v in odds["1"].items():
        p = place.get(num, [None, None, None])
        rows.append({"race_id": rid, "horse_number": int(num), "win_odds": v[0], "popularity": v[2],
                     "place_min": p[0], "place_max": p[1], "official_datetime": data.get("official_datetime", "")})
    return rows


def load_done():
    if not os.path.exists(FILE_OUT):
        return set()
    return set(pd.read_csv(FILE_OUT, usecols=["race_id"], dtype=str, on_bad_lines="skip")["race_id"])


def save(buf):
    if not buf:
        return
    pd.DataFrame(buf, columns=COLS).to_csv(FILE_OUT, mode="a", index=False, header=not os.path.exists(FILE_OUT),
                                           encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2026", help="カンマ区切りの年 (例: 2024,2025,2026)")
    years = [y.strip() for y in ap.parse_args().years.split(",")]

    ids = set()
    for d in DATA_SEARCH_DIRS:
        for y in years:
            for f in glob.glob(os.path.join(d, f"race_data_{y}.csv")):
                ids |= set(pd.read_csv(f, usecols=["race_id"], dtype=str)["race_id"])
    done = load_done()
    todo = sorted(ids - done)
    print(f"対象 {len(ids)}R / 取得済み {len(ids & done)}R / 残り {len(todo)}R")

    session = create_session()
    buf = []
    for i, rid in enumerate(tqdm(todo, desc="OddsAPI"), 1):
        rows = fetch(session, rid)
        if rows in ("BLOCK", "NETWORK_ERROR"):
            save(buf)
            print(f"\n{rows} を検知したため停止します（{rid}）。時間をおいて再実行すれば続きから取得します。")
            sys.exit(1)
        buf.extend(rows or [{"race_id": rid, "horse_number": -1}])
        time.sleep(random.uniform(1.0, 2.0))
        if i % 50 == 0:
            save(buf)
            buf = []
            time.sleep(10)
    save(buf)
    print("完了:", FILE_OUT)


if __name__ == "__main__":
    main()
