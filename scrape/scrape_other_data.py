# scrape/scrape_other_data.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re
from tqdm import tqdm
import glob
import random
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# ==========================================
# ★パスの自動動的解決 (サブディレクトリ移動対策)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_SEARCH_DIRS = [
    str(DATA_DIR / "train"),
    str(DATA_DIR / "val"),
    str(DATA_DIR / "test")
]

MASTER_FILE = str(DATA_DIR / "master_horse_data.csv")
FILE_BREEDER = str(DATA_DIR / "breeder_data_progress.csv")
FILE_LAP = str(DATA_DIR / "lap_data_progress.csv")
FILE_RETURN = str(DATA_DIR / "return_data_progress.csv")
FILE_TRAIN = str(DATA_DIR / "training_data_progress.csv")

LAP_COLS = ["race_id", "lap_headers", "lap_times", "first_3f", "last_3f_race", "pace_type"]
RETURN_COLS = ["race_id", "tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]
TRAIN_COLS = ["race_id", "horse_id", "oikiri_rank", "oikiri_last1f"]
BREEDER_COLS = ["horse_id", "breeder", "owner"]

for f in [FILE_BREEDER, FILE_LAP, FILE_RETURN, FILE_TRAIN]:
    os.makedirs(os.path.dirname(f), exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]


def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://race.netkeiba.com/",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    }


def get_soup(session, url):
    try:
        res = session.get(url, headers=get_headers(), timeout=20)
        if res.status_code == 429:
            print("\n🚨 アクセス過多 (429)。5分待機...")
            time.sleep(300)
            return "NETWORK_ERROR"
        if res.status_code == 404:
            return "NO_DATA"

        try:
            html = res.content.decode('euc-jp')
        except UnicodeDecodeError:
            try:
                html = res.content.decode('shift_jis')
            except UnicodeDecodeError:
                html = res.content.decode('utf-8', errors='replace')

        return BeautifulSoup(html, "html.parser")
    except:
        return "NETWORK_ERROR"


def parse_lap(soup, race_id):
    lap_block = {"race_id": race_id}
    try:
        pace_type = "Unknown"
        pace_elem = soup.find("span", class_=re.compile("Pace_Icon"))
        if pace_elem: pace_type = pace_elem.get_text(strip=True)

        table = soup.find("table", summary="ラップタイム")
        if not table: table = soup.find("table", class_=re.compile("RaceLap"))

        if table:
            rows = table.find_all("tr")
            headers = []
            times = []
            for row in rows:
                th_list = row.find_all("th")
                td_list = row.find_all("td")
                if th_list and not headers: headers = [th.get_text(strip=True) for th in th_list]
                if td_list:
                    row_data = [td.get_text(strip=True) for td in td_list]
                    if any(re.match(r"\d+\.\d", x) for x in row_data): times = row_data

            if times:
                lap_block["lap_headers"] = "-".join(headers)
                lap_block["lap_times"] = "-".join(times)
                try:
                    vals = [float(t) for t in times if t.replace('.', '').isdigit()]
                    if len(vals) >= 3:
                        f3 = sum(vals[:3])
                        l3 = sum(vals[-3:])
                        lap_block["first_3f"] = f3
                        lap_block["last_3f_race"] = l3
                        if pace_type == "Unknown":
                            if f3 < l3 - 0.5:
                                pace_type = "H"
                            elif f3 > l3 + 0.5:
                                pace_type = "S"
                            else:
                                pace_type = "M"
                except:
                    pass
        lap_block["pace_type"] = pace_type
        if "lap_times" not in lap_block: return "NO_DATA"
        return lap_block
    except:
        return "NO_DATA"


def parse_return(soup, race_id):
    try:
        tables = soup.find_all("table", class_="Payout_Detail_Table")
        if not tables: return "NO_DATA"
        return_data = {"race_id": race_id}
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                th = row.find("th")
                tds = row.find_all("td")
                if not th or len(tds) < 2: continue
                type_name = th.get_text(strip=True)
                clean_payouts = [p.replace(",", "").replace("円", "").strip() for p in list(tds[1].stripped_strings) if
                                 p.replace(",", "").replace("円", "").strip().isdigit()]
                key_map = {
                    "単勝": "tansho", "複勝": "fukusho", "枠連": "wakuren",
                    "馬連": "umaren", "ワイド": "wide", "馬単": "umatan",
                    "三連複": "sanrenpuku", "3連複": "sanrenpuku",
                    "三連単": "sanrentan", "3連単": "sanrentan"
                }
                if type_name in key_map and clean_payouts:
                    return_data[key_map[type_name]] = "|".join(clean_payouts)
        if len(return_data) <= 1: return "NO_DATA"
        return return_data
    except:
        return "NO_DATA"


def parse_training(session, race_id):
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    soup = get_soup(session, url)
    if isinstance(soup, str): return soup
    try:
        table = soup.find("table", class_=re.compile(r"OikiriTable|race_table_01"))
        if not table: return "NO_DATA"
        rows = table.find_all("tr")
        data_list = []
        for i, row in enumerate(rows):
            try:
                horse_a = row.find("a", href=re.compile(r"/horse/\d+"))
                if not horse_a: continue
                horse_id = re.search(r'/horse/(\d+)', horse_a['href']).group(1)
                oikiri_rank = "C"
                rank_elem = row.find("td", class_=re.compile(r"Oikiri_Rank|Rank"))
                if rank_elem: oikiri_rank = rank_elem.get_text(strip=True) or "C"

                last_1f = 99.9
                search_rows = [row]

                if i + 1 < len(rows):
                    next_row = rows[i + 1]
                    if not next_row.find("a", href=re.compile(r"/horse/\d+")):
                        search_rows.append(next_row)

                found_time = False
                for r in search_rows:
                    time_tds = r.find_all("td", class_=re.compile("Oikiri_Time|Time"))
                    if time_tds:
                        try:
                            val = time_tds[-1].get_text(strip=True)
                            if val and val.replace('.', '', 1).isdigit():
                                last_1f = float(val)
                                found_time = True
                        except:
                            pass
                    else:
                        for td in reversed(r.find_all("td")):
                            if re.match(r"^\d{1,2}\.\d$", td.get_text(strip=True)):
                                last_1f = float(td.get_text(strip=True))
                                found_time = True
                                break
                    if found_time: break
                data_list.append({
                    "race_id": race_id, "horse_id": horse_id,
                    "oikiri_rank": oikiri_rank, "oikiri_last1f": last_1f
                })
            except:
                continue
        return pd.DataFrame(data_list) if data_list else "NO_DATA"
    except:
        return "NO_DATA"


def parse_breeder(session, horse_id):
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    soup = get_soup(session, url)
    if isinstance(soup, str): return soup
    try:
        summary = soup.find("table", class_=re.compile(r"db_prof_table|summary_01"))
        if not summary: return "NO_DATA"
        breeder = "unknown"
        owner = "unknown"
        for row in summary.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td:
                ht = th.get_text(strip=True)
                if "生産者" in ht:
                    breeder = td.get_text(strip=True)
                elif "馬主" in ht:
                    owner = td.get_text(strip=True)
        return {"horse_id": horse_id, "breeder": breeder, "owner": owner}
    except:
        return "NO_DATA"


def save_buffer_to_csv(buf, cols, filepath, is_concat=False):
    if not buf: return
    if is_concat:
        df = pd.concat(buf, ignore_index=True)
    else:
        df = pd.DataFrame(buf)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    header = not os.path.exists(filepath)
    if os.path.exists(filepath):
        try:
            existing_cols = pd.read_csv(filepath, nrows=0, encoding='utf-8-sig').columns.tolist()
            for c in existing_cols:
                if c not in df.columns: df[c] = np.nan
            df = df[existing_cols]
            df.to_csv(filepath, index=False, encoding='utf-8-sig', mode='a', header=False)
            return
        except:
            pass

    df.to_csv(filepath, index=False, encoding='utf-8-sig', mode='a', header=header)


def load_done(path, key):
    if os.path.exists(path):
        try:
            return set(pd.read_csv(path, dtype=str, encoding='utf-8-sig')[key].dropna().unique())
        except Exception as e:
            print(f"⚠️ 既取得リストの読み込みに失敗 ({path}): {e}")
            return set()
    return set()


# ★ 年数指定の個別スキャンから、存在する全ての race_data_*.csv を一括全スキャンする方式に変更
def get_all_race_ids():
    found_ids = []
    for d in DATA_SEARCH_DIRS:
        files = glob.glob(os.path.join(d, "race_data_*.csv"))
        for fpath in files:
            try:
                df = pd.read_csv(fpath, usecols=['race_id'], dtype=str, encoding='utf-8-sig')
                ids = df['race_id'].unique().tolist()
                found_ids.extend(ids)
                print(f"   📂 {os.path.basename(fpath)} から {len(ids)} レース検出")
            except Exception as e:
                print(f"   ⚠️ 読み込み失敗: {fpath} ({e})")
    return sorted(list(set(found_ids)))


def main():
    print("🛡️ ステルスデータ収集 (Smart Target & Bug-Free Mode)")
    session = create_session()

    print("\n=== [Phase 1] レース拡張データ収集 (自動ファイルスキャン版) ===")

    done_lap = load_done(FILE_LAP, 'race_id')
    done_return = load_done(FILE_RETURN, 'race_id')
    done_train = load_done(FILE_TRAIN, 'race_id')

    print("📅 既存のフォルダ内レースデータをスキャン中...")
    target_ids = get_all_race_ids()

    if not target_ids:
        print("❌ 対象となるベースのレースデータ（race_data_*.csv）が見つかりません。")
    else:
        needed_ids = []
        for rid in target_ids:
            try:
                # race_id（12桁）の先頭4桁から自動で年を割り出し
                year = int(str(rid)[:4])
            except:
                continue

            is_return_target = (year >= 2024)
            missing_lap = (rid not in done_lap)
            missing_train = (rid not in done_train)
            missing_return = (is_return_target and rid not in done_return)

            if missing_lap or missing_train or missing_return:
                needed_ids.append(rid)

        if not needed_ids:
            print("   ✅ すべてのレースの拡張データは収集済みです。")
        else:
            random.shuffle(needed_ids)
            print(f"   🎯 差分収集の対象レース数: {len(needed_ids)} 件")

            buf_lap = []
            buf_return = []
            buf_train = []
            request_count = 0

            for rid in tqdm(needed_ids, desc="Scraping Progress"):
                try:
                    year = int(str(rid)[:4])
                except:
                    continue

                is_return_target = (year >= 2024)
                missing_lap = (rid not in done_lap)
                missing_train = (rid not in done_train)
                missing_return = (is_return_target and rid not in done_return)

                request_count += 1

                if missing_lap or missing_return:
                    url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
                    soup = get_soup(session, url)
                    if isinstance(soup, BeautifulSoup):
                        if missing_lap:
                            d = parse_lap(soup, rid)
                            if d != "NO_DATA":
                                buf_lap.append(d)
                            else:
                                buf_lap.append({"race_id": rid, "pace_type": "Unknown"})
                            done_lap.add(rid)

                        if missing_return:
                            d = parse_return(soup, rid)
                            if d != "NO_DATA":
                                buf_return.append(d)
                            else:
                                buf_return.append({"race_id": rid})
                            done_return.add(rid)

                    time.sleep(random.uniform(1.0, 2.0))

                if missing_train:
                    res_t = parse_training(session, rid)
                    if isinstance(res_t, pd.DataFrame):
                        buf_train.append(res_t)
                        done_train.add(rid)
                    elif res_t == "NO_DATA":
                        buf_train.append(pd.DataFrame([{"race_id": rid, "horse_id": "unknown"}]))
                        done_train.add(rid)

                if request_count % 10 == 0:
                    time.sleep(random.uniform(2.0, 4.0))

                if request_count % 50 == 0:
                    save_buffer_to_csv(buf_lap, LAP_COLS, FILE_LAP)
                    buf_lap = []

                    save_buffer_to_csv(buf_return, RETURN_COLS, FILE_RETURN)
                    buf_return = []

                    save_buffer_to_csv(buf_train, TRAIN_COLS, FILE_TRAIN, is_concat=True)
                    buf_train = []

                    tqdm.write(f"☕ 休憩... ({request_count} requests)")
                    time.sleep(10)

            # ループ終了後に残ったバッファをすべて書き出し
            save_buffer_to_csv(buf_lap, LAP_COLS, FILE_LAP)
            save_buffer_to_csv(buf_return, RETURN_COLS, FILE_RETURN)
            save_buffer_to_csv(buf_train, TRAIN_COLS, FILE_TRAIN, is_concat=True)

    print("\n=== [Phase 2] 生産者データ収集 ===")
    if not os.path.exists(MASTER_FILE):
        print("❌ マスタファイルなし。生産者収集をスキップします。")
    else:
        try:
            df_master = pd.read_csv(MASTER_FILE, dtype=str, encoding='utf-8-sig')
            all_horses = df_master['horse_id'].dropna().unique()
            done_breeder = load_done(FILE_BREEDER, 'horse_id')

            targets_breeder = [h for h in all_horses if h not in done_breeder]
            random.shuffle(targets_breeder)
            print(f"   🎯 新規対象: {len(targets_breeder)} 頭")

            buf_breed = []
            request_count = 0

            for hid in tqdm(targets_breeder, desc="Breeder Loop"):
                d = parse_breeder(session, hid)
                if d == "NO_DATA":
                    buf_breed.append({"horse_id": hid, "breeder": "unknown", "owner": "unknown"})
                elif isinstance(d, dict):
                    buf_breed.append(d)

                request_count += 1

                if request_count % 100 == 0:
                    time.sleep(10)
                    save_buffer_to_csv(buf_breed, BREEDER_COLS, FILE_BREEDER)
                    buf_breed = []

                time.sleep(random.uniform(1.5, 3.0))

            save_buffer_to_csv(buf_breed, BREEDER_COLS, FILE_BREEDER)

        except Exception as e:
            print(f"❌ 生産者収集エラー: {e}")

    print("\n🎉 全データ収集完了！")


if __name__ == "__main__":
    main()