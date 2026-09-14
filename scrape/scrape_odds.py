# scrape/scrape_odds.py
# netkeiba の結果ページから各馬の確定単勝オッズ・人気を取得する。 例: python scrape/scrape_odds.py --years 2026
import argparse
import glob
import os
import random
import sys
import time

import pandas as pd
from tqdm import tqdm

from scrape_other_data import DATA_DIR, DATA_SEARCH_DIRS, create_session, get_soup

FILE_ODDS = str(DATA_DIR / "odds_data_progress.csv")
ODDS_COLS = ["race_id", "horse_number", "horse_id", "popularity", "win_odds"]


def parse_odds(soup, race_id):
    table = soup.find("table", id="All_Result_Table")
    if table is None:
        return None
    rows = []
    for tr in table.find_all("tr")[1:]:
        num = tr.select_one("td.Num.Txt_C")
        odds_cells = tr.find_all("td", class_="Odds")
        if num is None or len(odds_cells) < 2:
            continue
        link = tr.select_one("td.Horse_Info a[href*='/horse/']")
        rows.append({
            "race_id": race_id,
            "horse_number": num.get_text(strip=True),
            "horse_id": link["href"].rstrip("/").split("/")[-1] if link else "",
            "popularity": odds_cells[0].get_text(strip=True),
            "win_odds": odds_cells[1].get_text(strip=True),
        })
    return rows or None


def load_done():
    if not os.path.exists(FILE_ODDS):
        return set()
    return set(pd.read_csv(FILE_ODDS, usecols=["race_id"], dtype=str)["race_id"])


def save(buf):
    if not buf:
        return
    # 列順を ODDS_COLS に固定して追記する（払戻データで起きた列ずれを防ぐ）
    pd.DataFrame(buf, columns=ODDS_COLS).to_csv(
        FILE_ODDS, mode="a", index=False, header=not os.path.exists(FILE_ODDS), encoding="utf-8")


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
    for i, rid in enumerate(tqdm(todo, desc="Odds"), 1):
        soup = get_soup(session, f"https://race.netkeiba.com/race/result.html?race_id={rid}")
        if isinstance(soup, str) and soup in ("BLOCK", "NETWORK_ERROR"):
            save(buf)
            print(f"\n{soup} を検知したため停止します（{rid}）。時間をおいて再実行すれば続きから取得します。")
            sys.exit(1)
        rows = None if isinstance(soup, str) else parse_odds(soup, rid)
        buf.extend(rows or [{"race_id": rid, "horse_number": "NO_DATA"}])
        time.sleep(random.uniform(1.0, 2.0))
        if i % 50 == 0:
            save(buf)
            buf = []
            time.sleep(10)
    save(buf)
    print("完了:", FILE_ODDS)


if __name__ == "__main__":
    main()
