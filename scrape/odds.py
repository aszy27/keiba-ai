# scrape/odds.py
# netkeiba のオッズAPIから確定オッズを取る。1レース1リクエスト。途中で止めても再実行で続きから取れる。
#  - 単勝・複勝: 各馬の確定単勝オッズ・人気・複勝オッズ（下限/上限）→ data/odds_api_progress.csv
#  - 組み合わせ券（全通り）: 4=馬連 / 5=ワイド（下限・上限）/ 6=馬単 / 7=3連複 / 8=3連単 → data/v2/odds_exotic/<年>_type<券種>.parquet
#    払戻は「当たった組み合わせのオッズ×100」で計算できるので、払戻データが無いレースも検証に使える。
#    3連単は最大4,896通りあるので、年・券種ごとに分けて保存する。
# 発走前のオッズ（前向き検証のスナップショット）は scrape/snapshot.py。
#
# 使い方: python -m scrape odds --years 2024,2025,2026
#         python -m scrape odds-exotic --years 2020 --types 7,8 [--limit 500]
#         python -m scrape odds-exotic --job 2021-2025:7,8 --job 2020-2025:4,6 --max-requests 3000   # 前から順に、上限を共有して取る
# 注意: 2026-09-17 に短時間で大量に取って API から一時ブロックされた。間隔を詰めないこと。
import datetime
import glob
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from scrape.common import DATA_DIR, DATA_SEARCH_DIRS, ODDS_API, create_session, http_get

FILE_OUT = str(DATA_DIR / "odds_api_progress.csv")
COLS = ["race_id", "horse_number", "win_odds", "popularity", "place_min", "place_max", "official_datetime"]
EXOTIC_NAMES = {4: "馬連", 5: "ワイド", 6: "馬単", 7: "3連複", 8: "3連単"}
EXOTIC_SAVE_EVERY = 100


# ---------- 単勝・複勝 ----------

def fetch(session, rid):
    r, status = http_get(ODDS_API.format(race_id=rid, bet_type=1, action="init"), referer="https://race.netkeiba.com/")
    if r is None:
        return "NETWORK_ERROR" if status == "network_error" else "BLOCK"
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
    r, status = http_get(ODDS_API.format(race_id=race_id, bet_type=bet_type, action="update"),
                         referer="https://race.netkeiba.com/")
    if r is None:
        return [], f"error: {status}"
    try:
        js = r.json()
    except ValueError as e:
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


def empty_path():
    from paths import V2_DIR
    return V2_DIR / "odds_exotic" / "empty.csv"


def load_empty():
    """オッズが返ってこなかった（取消・中止などで、ブロックではない）(レース, type)。次からは取りに行かない"""
    p = empty_path()
    if not p.exists():
        return set()
    d = pd.read_csv(p, dtype={"race_id": str})
    return set(zip(d["race_id"], d["bet_type"]))


def run_exotic(years, types, limit=None, sleep=0.7, max_requests=None, stop_after_fail=20, on_request=None):
    """年 → 券種の順に、取得していないレースを取る。max_requests は今回の実行全体の上限、
    stop_after_fail 回続けて取れなければブロックの兆候とみて止める（翌日に続きから取れる）。
    on_request はリクエストのたびに呼ぶ（1日の使用回数の記録用）。
    戻り値: (今回のリクエスト数, 止まった理由)。理由は None（最後まで取り終えた）/ "上限" / "制限" / "ブロックの兆候"""
    from paths import table_path
    exotic_path(0, 0).parent.mkdir(parents=True, exist_ok=True)
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
    races["year"] = races["race_date"].dt.year
    session = create_session()
    empty = load_empty()
    n_req, n_fail = 0, 0

    for year in years:
        ids = races.loc[races["year"] == year, "race_id"].sort_values().tolist()
        for bet_type in types:
            have = done_exotic(year, bet_type)
            todo = [r for r in ids if r not in have and (r, bet_type) not in empty][:limit]
            print(f"{year}年 {EXOTIC_NAMES[bet_type]}: 取得済み {len(have)}R / 残り {len(todo)}R", flush=True)
            buf, t0 = [], time.time()
            for i, rid in enumerate(todo, 1):
                if max_requests is not None and n_req >= max_requests:
                    save_exotic(buf, year, bet_type)
                    print(f"今回の上限 {max_requests}回に達したので終了（続きは次回）", flush=True)
                    return n_req, "上限"
                rows, status = fetch_exotic(rid, bet_type, session)
                n_req += 1
                if on_request:
                    on_request()
                if not rows and str(status).strip() == "limit":
                    # 短い間に続けて取ると API が制限をかける（2026-10-03 に判明）。オッズが無いのではないので記録せず、15分控える
                    n_fail += 1
                    print(f"  {rid} API が制限中（status=limit）。15分控える（{n_fail}回目）", flush=True)
                    if n_fail >= 3:
                        save_exotic(buf, year, bet_type)
                        print("制限が続くので今日は終了（続きは次回）", flush=True)
                        return n_req, "制限"
                    time.sleep(15 * 60)
                    continue
                if not rows:
                    print(f"  {rid} 取得できず（{status}）", flush=True)
                    if str(status).startswith("error"):
                        n_fail += 1
                        if n_fail >= stop_after_fail:
                            save_exotic(buf, year, bet_type)
                            print(f"{stop_after_fail}回続けて取得できない（ブロックの兆候）ので終了", flush=True)
                            return n_req, "ブロックの兆候"
                    else:   # API は答えたがオッズが無い（取消・中止など）→ 記録して次から飛ばす
                        pd.DataFrame([{"race_id": rid, "bet_type": bet_type, "status": status}]).to_csv(
                            empty_path(), mode="a", index=False, header=not empty_path().exists())
                else:
                    n_fail = 0
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
    return n_req, None


def parse_job(text):
    """"2021-2025:7,8" → ([2021, ..., 2025], [7, 8])。年は "2020,2022" のようにカンマ区切りでもよい"""
    years, types = text.split(":")
    ys = []
    for part in years.split(","):
        a, _, b = part.partition("-")
        ys += list(range(int(a), int(b or a) + 1))
    return ys, [int(t) for t in types.split(",")]


def daily_path():
    from paths import V2_DIR
    return V2_DIR / "odds_exotic" / "daily_state.json"


def load_daily(today):
    """今日（0時区切り）の {"date", "used": 使ったリクエスト数, "stopped": 制限・ブロックの兆候で止めた理由}"""
    p = daily_path()
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
        if st.get("date") == today:
            return st
    return {"date": today, "used": 0, "stopped": None}


def save_daily(st):
    p = daily_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def run_exotic_jobs(jobs, limit=None, sleep=0.7, max_requests=None, daily_max=None):
    """(年のリスト, 券種のリスト) を前から順に取る。リクエスト数の上限は全体で共有し、
    前のジョブを取り終えてから次へ進む（上限・制限・ブロックの兆候で止まったらそこで終わり）。
    daily_max は1日（0時区切り）の上限。使った回数を daily_path() に1回ごとに残すので、途中でPCを切って
    起動し直しても同じ日は残りの回数だけ取る。制限・ブロックの兆候で止まった日は、起動し直しても取らない"""
    st, on_request = None, None
    if daily_max is not None:
        st = load_daily(datetime.date.today().strftime("%Y%m%d"))
        if st["stopped"]:
            print(f"今日は「{st['stopped']}」で止めたので取らない（続きは明日）", flush=True)
            return
        rest = daily_max - st["used"]
        print(f"今日の使用 {st['used']}/{daily_max}回", flush=True)
        if rest <= 0:
            print(f"今日の上限 {daily_max}回に達しているので取らない（続きは明日）", flush=True)
            return
        max_requests = rest if max_requests is None else min(max_requests, rest)

        def on_request():
            st["used"] += 1
            save_daily(st)
    used = 0
    for years, types in jobs:
        left = None if max_requests is None else max_requests - used
        if left is not None and left <= 0:
            print(f"今回の上限 {max_requests}回に達したので終了（続きは次回）", flush=True)
            return
        n, reason = run_exotic(years, types, limit, sleep, left, on_request=on_request)
        used += n
        if reason:
            if st is not None and reason in ("制限", "ブロックの兆候"):
                st["stopped"] = reason
                save_daily(st)
            return
