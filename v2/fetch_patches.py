# v2/fetch_patches.py
# 検査で見つかったデータの穴を netkeiba から取り直し、data/v2/patches/ に保存する（旧 CSV は書き換えない）。
#  - レース情報（芝ダ・距離・馬場・天候・レース名）が欠けているレース → patches/race_info.csv
#  - 1開催日のレース数が11未満の日に抜けているレース                 → patches/race_rows/<race_id>.csv
# 保存後に python -m v2.ingest を実行すると反映され、検査もやり直される。
#
# 使い方: python -m v2.fetch_patches            # 取得して保存
#         python -m v2.fetch_patches --dry-run  # 対象の一覧だけ表示
import argparse
import random
import time

import pandas as pd

from scrape.repair import parse_race_info
from scrape.rescrape import fetch_race
from v2.paths import PATCH_DIR, table_path

INFO_COLS = ["race_id", "race_name", "race_time", "type", "length", "handed", "weather", "condition"]


def missing_info_ids(races):
    ng = races["surface"].isna() | races["distance"].isna() | races["going"].isna() | races["weather"].isna()
    return races.loc[ng, "race_id"].tolist()


def missing_race_ids(races):
    days = races.groupby(["race_date", "place"])["race_id"].agg(list)
    out = []
    for ids in days[days.map(len) < 11]:
        base = ids[0][:10]
        out += [f"{base}{n:02d}" for n in range(1, 13) if f"{base}{n:02d}" not in set(ids)]
    return out


def save_race_info(rows):
    path = PATCH_DIR / "race_info.csv"
    new = pd.DataFrame(rows).reindex(columns=INFO_COLS)
    if path.exists():
        new = pd.concat([pd.read_csv(path, dtype=str, encoding="utf-8-sig"), new], ignore_index=True)
    new.drop_duplicates("race_id", keep="last").to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    races = pd.read_parquet(table_path("races"))
    info_ids, race_ids = missing_info_ids(races), missing_race_ids(races)
    print(f"レース情報の欠損: {len(info_ids)}R / 抜けているレース: {len(race_ids)}R")
    print("  抜けているレース:", ", ".join(race_ids))
    if args.dry_run:
        return
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for i, rid in enumerate(info_ids, 1):
        info = parse_race_info(rid)
        if info.get("type") and info.get("length"):
            rows.append({k: info.get(k) for k in INFO_COLS} | {"race_id": rid})
        else:
            failed.append(rid)
        print(f"\r  レース情報 {i}/{len(info_ids)}（失敗 {len(failed)}）", end="")
        if len(rows) % 20 == 0 and rows:
            save_race_info(rows)
        time.sleep(random.uniform(1.5, 3.0))
    if rows:
        save_race_info(rows)
    print(f"\n  保存: {len(rows)}R{'、取得できず: ' + ', '.join(failed) if failed else ''}")

    if race_ids:
        runners = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_id"])
        existing = runners.merge(races[["race_id", "race_date"]], on="race_id")
        existing["race_date"] = existing["race_date"].dt.strftime("%Y-%m-%d")
        (PATCH_DIR / "race_rows").mkdir(exist_ok=True)
        for rid in race_ids:
            print(f"\n[{rid}]")
            df, source, attempts = fetch_race(rid, existing.astype(str))
            if df.empty:
                print("   取得できず:", " / ".join(f"{a}: {b}" for a, b in attempts))
            else:
                df.to_csv(PATCH_DIR / "race_rows" / f"{rid}.csv", index=False, encoding="utf-8-sig")
                top = df.sort_values("rank").head(3)["horse_name"].tolist()
                print(f"   保存 [{source}] {df['race_date'].iloc[0]} {df['race_name'].iloc[0]} {len(df)}頭 1〜3着: {top}")
            time.sleep(random.uniform(2.0, 4.0))
    print("\n次に python -m v2.ingest を実行して反映・検査する")


if __name__ == "__main__":
    main()
