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
import sys

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

# ==========================================
# ★ 固定カラム定義 (タイム列は完全排除を維持)
# ==========================================
LAP_COLS = ["race_id", "lap_headers", "lap_times", "first_3f", "last_3f_race", "pace_type"]
RETURN_COLS = ["race_id", "tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]
TRAIN_COLS = ["race_id", "horse_id", "oikiri_rank"]
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

        if res.status_code == 404:
            return "NO_DATA"

        if res.status_code != 200:
            return "BLOCK"

        try:
            html = res.content.decode('euc-jp')
        except UnicodeDecodeError:
            try:
                html = res.content.decode('shift_jis')
            except UnicodeDecodeError:
                html = res.content.decode('utf-8', errors='replace')

        soup = BeautifulSoup(html, "html.parser")

        if soup.title:
            title_text = soup.title.get_text()
            if any(w in title_text for w in
                   ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限", "大変混み合って"]):
                return "BLOCK"

        if not soup.find(id=re.compile("header|container|main")) and not soup.find(
                class_=re.compile("Race|Header|Layout")):
            return "BLOCK"

        return soup
    except:
        return "NETWORK_ERROR"


def parse_lap(soup, race_id):
    if isinstance(soup, str) and soup in ["BLOCK", "NETWORK_ERROR"]: return "BLOCK"
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
    if isinstance(soup, str) and soup in ["BLOCK", "NETWORK_ERROR"]: return "BLOCK"
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
    if isinstance(soup, str): return soup  # 文字列ならそのままエラーコードを返す
    try:
        table = soup.find("table", class_=re.compile(r"OikiriTable|race_table_01"))
        if not table: return "NO_DATA"
        rows = table.find_all("tr")
        data_list = []
        for row in rows:
            try:
                horse_a = row.find("a", href=re.compile(r"/horse/\d+"))
                if not horse_a: continue
                horse_id = re.search(r'/horse/(\d+)', horse_a['href']).group(1)

                oikiri_rank = "C"
                rank_elem = row.find("td", class_=re.compile(r"Oikiri_Rank|Rank"))
                if rank_elem: oikiri_rank = rank_elem.get_text(strip=True) or "C"

                data_list.append({
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "oikiri_rank": oikiri_rank
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


def load_done_with_filter(path, key, category=None):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
        if df.empty: return set()

        if category == 'train':
            invalid_mask = df['horse_id'].isnull() | (df['horse_id'] == 'unknown') | df['oikiri_rank'].isnull()
            invalid_ids = set(df.loc[invalid_mask, 'race_id'].dropna().unique())
            return set(df['race_id'].dropna().unique()) - invalid_ids

        elif category == 'lap':
            invalid_mask = df['lap_times'].isnull() | (df['lap_times'] == '')
            invalid_ids = set(df.loc[invalid_mask, 'race_id'].dropna().unique())
            return set(df['race_id'].dropna().unique()) - invalid_ids

        elif category == 'return':
            check_cols = [c for c in ['tansho', 'fukusho', 'sanrenpuku'] if c in df.columns]
            if check_cols:
                invalid_mask = df[check_cols].isnull().all(axis=1)
                invalid_ids = set(df.loc[invalid_mask, 'race_id'].dropna().unique())
                return set(df['race_id'].dropna().unique()) - invalid_ids

        return set(df[key].dropna().unique())
    except:
        return set()


def sanitize_existing_files():
    print("🧹 既存の進捗CSVから不完全なデータ（unknown馬など）を自動検知・クレンジング中...")
    if os.path.exists(FILE_TRAIN):
        try:
            df = pd.read_csv(FILE_TRAIN, dtype=str, encoding='utf-8-sig')
            before = len(df)
            df = df[df['race_id'].notnull() & df['horse_id'].notnull() & (df['horse_id'] != 'unknown')]
            if len(df) < before:
                df.to_csv(FILE_TRAIN, index=False, encoding='utf-8-sig')
                print(f"   📊 調教データ: 過去の不完全な {before - len(df)} 行を抹消し、適正化しました。")
        except Exception as e:
            print(f"   ⚠️ 調教データの自動クレンジング失敗: {e}")

    if os.path.exists(FILE_LAP):
        try:
            df = pd.read_csv(FILE_LAP, dtype=str, encoding='utf-8-sig')
            before = len(df)
            df = df[df['race_id'].notnull() & df['lap_times'].notnull() & (df['lap_times'] != '')]
            if len(df) < before:
                df.to_csv(FILE_LAP, index=False, encoding='utf-8-sig')
                print(f"   📊 ラップデータ: 過去の不完全な {before - len(df)} 行を抹消し、適正化しました。")
        except Exception as e:
            print(f"   ⚠️ ラップデータの自動クレンジング失敗: {e}")


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
            except:
                pass
    return sorted(list(set(found_ids)))


def flush_buffers(buf_lap, buf_return, buf_train):
    if buf_lap: save_buffer_to_csv(buf_lap, LAP_COLS, FILE_LAP)
    if buf_return: save_buffer_to_csv(buf_return, RETURN_COLS, FILE_RETURN)
    if buf_train: save_buffer_to_csv(buf_train, TRAIN_COLS, FILE_TRAIN, is_concat=True)
    print("💾 取得済みのバッファデータをすべて安全にセーブしました。")


def main():
    print("🛡️ ステルスデータ収集 (Smart Target & BLOCK緊急停止 & 自動穴埋め版)")
    sanitize_existing_files()
    session = create_session()

    print("\n=== [Phase 1] レース拡張データ収集 (自動ファイルスキャン版) ===")

    done_lap = load_done_with_filter(FILE_LAP, 'race_id', 'lap')
    done_return = load_done_with_filter(FILE_RETURN, 'race_id', 'return')
    done_train = load_done_with_filter(FILE_TRAIN, 'race_id', 'train')

    print("📅 既存のフォルダ内レースデータをスキャン中...")
    target_ids = get_all_race_ids()

    if not target_ids:
        print("❌ 対象となるベースのレースデータ（race_data_*.csv）が見つかりません。")
        return

    needed_ids = []
    for rid in target_ids:
        try:
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
        print("   ✅ すべてのレースの拡張データは収集・補完済みです。")
    else:
        random.shuffle(needed_ids)
        print(f"   🎯 差分・穴埋め収集の対象レース数: {len(needed_ids)} 件")

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

            # --- 1. 結果ページ (Lap / Return) ---
            if missing_lap or missing_return:
                url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
                soup = get_soup(session, url)

                # ★ 修正: isinstance を追加して DataFrame による比較クラッシュを完全防御
                if isinstance(soup, str) and soup in ["BLOCK", "NETWORK_ERROR"]:
                    print(f"\n🚨 アクセス制限または通信異常(BLOCK)を検知しました。処理を即座に安全停止します。")
                    flush_buffers(buf_lap, buf_return, buf_train)
                    sys.exit(1)

                if isinstance(soup, BeautifulSoup):
                    if missing_lap:
                        d = parse_lap(soup, rid)
                        if isinstance(d, str) and d == "BLOCK":
                            print(f"\n🚨 アクセス制限（BLOCK）を検知しました。")
                            flush_buffers(buf_lap, buf_return, buf_train)
                            sys.exit(1)
                        elif d != "NO_DATA":
                            buf_lap.append(d)
                        else:
                            buf_lap.append({"race_id": rid, "pace_type": "Unknown"})
                        done_lap.add(rid)

                    if missing_return:
                        d = parse_return(soup, rid)
                        if isinstance(d, str) and d == "BLOCK":
                            print(f"\n🚨 アクセス制限（BLOCK）を検知しました。")
                            flush_buffers(buf_lap, buf_return, buf_train)
                            sys.exit(1)
                        elif d != "NO_DATA":
                            buf_return.append(d)
                        else:
                            buf_return.append({"race_id": rid})
                        done_return.add(rid)

                time.sleep(random.uniform(1.0, 2.0))

            # --- 2. 追い切りページ (調教評価) ---
            if missing_train:
                res_t = parse_training(session, rid)

                # ★ 修正: isinstance を追加して DataFrame 判定による ValueError を完全克服！
                if isinstance(res_t, str) and res_t in ["BLOCK", "NETWORK_ERROR"]:
                    print(f"\n🚨 アクセス制限または通信異常(BLOCK)を検知しました。処理を即座に安全停止します。")
                    flush_buffers(buf_lap, buf_return, buf_train)
                    sys.exit(1)

                if isinstance(res_t, pd.DataFrame):
                    buf_train.append(res_t)
                    done_train.add(rid)
                elif isinstance(res_t, str) and res_t == "NO_DATA":
                    buf_train.append(pd.DataFrame([{"race_id": rid, "horse_id": "unknown", "oikiri_rank": "C"}]))
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

        flush_buffers(buf_lap, buf_return, buf_train)

    # --- Phase 2: 生産者データ収集 ---
    print("\n=== [Phase 2] 生産者データ収集 ===")
    if not os.path.exists(MASTER_FILE):
        print("❌ マスタファイルなし。生産者収集をスキップします。")
    else:
        try:
            df_master = pd.read_csv(MASTER_FILE, dtype=str, encoding='utf-8-sig')
            all_horses = df_master['horse_id'].dropna().unique()
            done_breeder = load_done_with_filter(FILE_BREEDER, 'horse_id')

            targets_breeder = [h for h in all_horses if h not in done_breeder]
            random.shuffle(targets_breeder)
            print(f"   🎯 新規対象: {len(targets_breeder)} 頭")

            buf_breed = []
            request_count = 0

            for hid in tqdm(targets_breeder, desc="Breeder Loop"):
                d = parse_breeder(session, hid)

                if isinstance(d, str) and d in ["BLOCK", "NETWORK_ERROR"]:
                    print(f"\n🚨 血統DB側からアクセス制限（BLOCK）を検知しました。処理を即座に安全停止します。")
                    if buf_breed: save_buffer_to_csv(buf_breed, BREEDER_COLS, FILE_BREEDER)
                    sys.exit(1)

                if isinstance(d, str) and d == "NO_DATA":
                    buf_breed.append({"horse_id": hid, "breeder": "unknown", "owner": "unknown"})
                elif isinstance(d, dict):
                    buf_breed.append(d)

                request_count += 1

                if request_count % 100 == 0:
                    time.sleep(10)
                    save_buffer_to_csv(buf_breed, BREEDER_COLS, FILE_BREEDER)
                    buf_breed = []

                time.sleep(random.uniform(1.5, 3.0))

            if buf_breed:
                save_buffer_to_csv(buf_breed, BREEDER_COLS, FILE_BREEDER)

        except Exception as e:
            print(f"❌ 生産者収集エラー: {e}")

    print("\n🎉 全データ収集・補完完了！")


if __name__ == "__main__":
    main()