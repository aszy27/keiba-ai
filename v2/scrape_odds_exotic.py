# v2/scrape_odds_exotic.py
# 組み合わせ券の確定オッズ（全通り）を netkeiba のオッズAPIから取る。
# 払戻は「当たった組み合わせのオッズ×100」で計算できるので、払戻データが無いレースも検証に使える。
# type: 4=馬連 / 5=ワイド（下限・上限）/ 7=3連複 / 8=3連単
# 1レース1リクエスト。3連単は最大4,896通りあるので、年・券種ごとに parquet に分けて保存する。
# 途中で止めても再実行すれば続きから取る。
#
# 使い方: python -m v2.scrape_odds_exotic --years 2020 --types 7,8
#         python -m v2.scrape_odds_exotic --years 2019,2020 --types 4,5 --limit 500
import argparse
import random
import time

import numpy as np
import pandas as pd
import requests

from v2.paths import V2_DIR, table_path

OUT_DIR = V2_DIR / "odds_exotic"
API = ("https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&input=UTF-8&output=json"
       "&race_id={}&type={}&action=update&sort=odds&compress=0")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36",
           "Referer": "https://race.netkeiba.com/", "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
NAMES = {4: "馬連", 5: "ワイド", 7: "3連複", 8: "3連単"}
SAVE_EVERY = 100


def out_path(year, bet_type):
    return OUT_DIR / f"{year}_type{bet_type}.parquet"


def fetch(race_id, bet_type, session):
    """戻り値: (行のリスト, 状態)。行は (組み合わせ, オッズ下限, オッズ上限)"""
    try:
        js = session.get(API.format(race_id, bet_type), headers=HEADERS, timeout=20).json()
    except (requests.RequestException, ValueError) as e:
        return [], f"error: {type(e).__name__}"
    data = js.get("data") if isinstance(js.get("data"), dict) else {}
    odds = (data.get("odds") or {}).get(str(bet_type)) or {}
    rows = []
    for combo, v in odds.items():
        lo = str(v[0]).replace(",", "")
        hi = str(v[1]).replace(",", "") if len(v) > 1 else "0"
        try:
            lo = float(lo)
        except ValueError:
            continue                      # 取消などで "---" が入る
        try:
            hi = float(hi)
        except ValueError:
            hi = 0.0
        rows.append((combo, np.float32(lo), np.float32(hi if hi > 0 else lo)))
    return rows, js.get("status", "")


def save(buf, year, bet_type):
    if not buf:
        return
    df = pd.DataFrame(buf, columns=["race_id", "combo", "odds_min", "odds_max"])
    path = out_path(year, bet_type)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True).drop_duplicates(["race_id", "combo"], keep="last")
    df.to_parquet(path, index=False)


def done_races(year, bet_type):
    path = out_path(year, bet_type)
    return set(pd.read_parquet(path, columns=["race_id"])["race_id"].unique()) if path.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True)
    ap.add_argument("--types", default="7,8")
    ap.add_argument("--limit", type=int, help="1回の実行で取るレース数の上限")
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
    races["year"] = races["race_date"].dt.year
    session = requests.Session()

    for year in [int(y) for y in args.years.split(",")]:
        ids = races.loc[races["year"] == year, "race_id"].sort_values().tolist()
        for bet_type in [int(t) for t in args.types.split(",")]:
            have = done_races(year, bet_type)
            todo = [r for r in ids if r not in have][:args.limit]
            print(f"{year}年 {NAMES[bet_type]}: 取得済み {len(have)}R / 残り {len(todo)}R", flush=True)
            buf, t0 = [], time.time()
            for i, rid in enumerate(todo, 1):
                rows, status = fetch(rid, bet_type, session)
                if not rows:
                    print(f"  {rid} 取得できず（{status}）", flush=True)
                buf += [(rid, c, lo, hi) for c, lo, hi in rows]
                time.sleep(args.sleep * random.uniform(0.8, 1.3))
                if i % SAVE_EVERY == 0:
                    save(buf, year, bet_type)
                    buf = []
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}R 保存（{rate * 60:.0f}R/分、残り {(len(todo) - i) / rate / 60:.0f}分）", flush=True)
            save(buf, year, bet_type)
            print(f"{year}年 {NAMES[bet_type]}: 完了 {len(todo)}R", flush=True)


if __name__ == "__main__":
    main()
