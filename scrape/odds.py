# scrape/odds.py
# netkeiba のオッズAPIから確定オッズを取る。1レース1リクエスト。途中で止めても再実行で続きから取れる。
#  - 単勝・複勝: 各馬の確定単勝オッズ・人気・複勝オッズ（下限/上限）→ data/odds_api_progress.csv
#  - 組み合わせ券（全通り）: 4=馬連 / 5=ワイド（下限・上限）/ 7=3連複 / 8=3連単 → data/v2/odds_exotic/<年>_type<券種>.parquet
#    払戻は「当たった組み合わせのオッズ×100」で計算できるので、払戻データが無いレースも検証に使える。
#    3連単は最大4,896通りあるので、年・券種ごとに分けて保存する。
# 発走前のオッズ（前向き検証のスナップショット）は scrape/snapshot.py。
#
# 使い方: python -m scrape odds --years 2024,2025,2026
#         python -m scrape odds-exotic --years 2020 --types 7,8 [--limit 500]
# 注意: 2026-09-17 に短時間で大量に取って API から一時ブロックされた。間隔を詰めないこと。
import glob
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from scrape.common import DATA_DIR, DATA_SEARCH_DIRS, ODDS_API, create_session, get_headers

FILE_OUT = str(DATA_DIR / "odds_api_progress.csv")
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_min", "place_max", "official_datetime"]
EXOTIC_NAMES = {4: "馬連", 5: "ワイド", 7: "3連複", 8: "3連単"}
EXOTIC_SAVE_EVERY = 100


# ---------- 単勝・複勝 ----------

def fetch(session, rid):
    try:
        r = session.get(ODDS_API.format(race_id=rid, bet_type=1, action="init"), headers=get_headers(), timeout=20)
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


def run(years):
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


# ---------- 組み合わせ券 ----------

def exotic_path(year, bet_type):
    from paths import V2_DIR
    return V2_DIR / "odds_exotic" / f"{year}_type{bet_type}.parquet"


def fetch_exotic(race_id, bet_type, session):
    """戻り値: (行のリスト, 状態)。行は (組み合わせ, オッズ下限, オッズ上限)"""
    try:
        js = session.get(ODDS_API.format(race_id=race_id, bet_type=bet_type, action="update"),
                         headers=get_headers(), timeout=20).json()
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


def save_exotic(buf, year, bet_type):
    if not buf:
        return
    df = pd.DataFrame(buf, columns=["race_id", "combo", "odds_min", "odds_max"])
    path = exotic_path(year, bet_type)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True).drop_duplicates(["race_id", "combo"], keep="last")
    df.to_parquet(path, index=False)


def done_exotic(year, bet_type):
    path = exotic_path(year, bet_type)
    return set(pd.read_parquet(path, columns=["race_id"])["race_id"].unique()) if path.exists() else set()


def run_exotic(years, types, limit=None, sleep=0.7):
    from paths import table_path
    exotic_path(0, 0).parent.mkdir(parents=True, exist_ok=True)
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
    races["year"] = races["race_date"].dt.year
    session = requests.Session()

    for year in years:
        ids = races.loc[races["year"] == year, "race_id"].sort_values().tolist()
        for bet_type in types:
            have = done_exotic(year, bet_type)
            todo = [r for r in ids if r not in have][:limit]
            print(f"{year}年 {EXOTIC_NAMES[bet_type]}: 取得済み {len(have)}R / 残り {len(todo)}R", flush=True)
            buf, t0 = [], time.time()
            for i, rid in enumerate(todo, 1):
                rows, status = fetch_exotic(rid, bet_type, session)
                if not rows:
                    print(f"  {rid} 取得できず（{status}）", flush=True)
                buf += [(rid, c, lo, hi) for c, lo, hi in rows]
                time.sleep(sleep * random.uniform(0.8, 1.3))
                if i % EXOTIC_SAVE_EVERY == 0:
                    save_exotic(buf, year, bet_type)
                    buf = []
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}R 保存（{rate * 60:.0f}R/分、残り {(len(todo) - i) / rate / 60:.0f}分）",
                          flush=True)
            save_exotic(buf, year, bet_type)
            print(f"{year}年 {EXOTIC_NAMES[bet_type]}: 完了 {len(todo)}R", flush=True)
