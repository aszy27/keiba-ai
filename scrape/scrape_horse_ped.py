# scrape/scrape_horse_ped.py
import pandas as pd
import time
import os
import sys
import glob
import random
import requests
from bs4 import BeautifulSoup
import re
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

warnings.simplefilter('ignore')

# ==========================================
# ★パスの自動動的解決 (サブディレクトリ移動対策)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ==========================================
# ★設定エリア
# ==========================================
MIN_SLEEP = 1.0  # 血統データベースは規制が厳しいため少し慎重に
MAX_SLEEP = 3.0
SAVE_INTERVAL = 100

# 保存先
FILENAME = str(DATA_DIR / "master_horse_data.csv")

# 読み込むデータの場所
DATA_DIRS = [
    str(DATA_DIR / "train"),
    str(DATA_DIR / "val"),
    str(DATA_DIR / "test"),
    str(DATA_DIR)
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]


# ==========================================

def create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://db.netkeiba.com/"
    })
    return session


def extract_id_from_url(url):
    if not url: return None
    match = re.search(r'/horse/([a-zA-Z0-9]+)', url)
    return match.group(1) if match else None


def scrape_pedigree_page(horse_id, session):
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"

    try:
        response = session.get(url, timeout=10)
        if response.status_code in [429, 403]:
            return "BLOCK"

        response.encoding = 'euc-jp'
        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title and ("アクセスが集中" in soup.title.get_text() or "お手数ですが" in soup.title.get_text()):
            return "BLOCK"

        data = {
            'horse_id': horse_id,
            'horse_name': None,
            'sire_id': None, 'sire_name': None,
            'dam_id': None, 'dam_name': None
        }

        if soup.title:
            title_text = soup.title.get_text(strip=True)
            if "|" in title_text:
                data['horse_name'] = title_text.split("|")[0].strip()
            else:
                data['horse_name'] = title_text

        blood_table = soup.find("table", class_="blood_table")
        if blood_table:
            tds = blood_table.find_all("td", attrs={"rowspan": True})
            candidates = []
            for td in tds:
                try:
                    candidates.append((td, int(td.get("rowspan"))))
                except:
                    pass

            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) >= 1:
                links = candidates[0][0].find_all("a")
                if links:
                    data['sire_id'] = extract_id_from_url(links[0].get('href'))
                    data['sire_name'] = links[0].get_text(strip=True)

            if len(candidates) >= 2 and candidates[1][1] == candidates[0][1]:
                links = candidates[1][0].find_all("a")
                if links:
                    data['dam_id'] = extract_id_from_url(links[0].get('href'))
                    data['dam_name'] = links[0].get_text(strip=True)

        return data

    except Exception:
        return None


def get_all_horse_ids():
    print("🔍 各フォルダのレースデータから馬IDを収集中...")
    files = []

    for d in DATA_DIRS:
        if not os.path.exists(d): continue
        path_pattern = os.path.join(d, "race_data_*.csv")
        found = glob.glob(path_pattern)
        if found:
            print(f"   -> {d} フォルダ: {len(found)} ファイル発見")
            files.extend(found)

    if not files:
        print("❌ レースデータが見つかりません。")
        return set()

    horse_ids = set()
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['horse_id'], dtype={'horse_id': str}, encoding='utf-8-sig')
            horse_ids.update(df['horse_id'].dropna().astype(str))
        except:
            pass

    return horse_ids


def save_to_csv_safe(new_data, COLUMNS):
    if not new_data: return
    df_new = pd.DataFrame(new_data)

    for col in COLUMNS:
        if col not in df_new.columns: df_new[col] = None

    if os.path.exists(FILENAME):
        try:
            existing_cols = pd.read_csv(FILENAME, nrows=0, encoding='utf-8-sig').columns.tolist()
            for col in existing_cols:
                if col not in df_new.columns: df_new[col] = None
            df_new = df_new[existing_cols]
            df_new.to_csv(FILENAME, index=False, encoding='utf-8-sig', mode='a', header=False)
            return
        except:
            pass

    df_new = df_new[COLUMNS]
    df_new.to_csv(FILENAME, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(FILENAME))


def main():
    print(f"\n🚀 血統データの収集 (フォルダ整理版) を開始します")
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)

    existing_ids = set()
    if os.path.exists(FILENAME):
        try:
            df_exist = pd.read_csv(FILENAME, dtype={'horse_id': str}, encoding='utf-8-sig')
            if 'horse_id' in df_exist.columns:
                existing_ids = set(df_exist['horse_id'].astype(str))
                print(f"📚 既存ID数: {len(existing_ids)}")

                if 'sire_id' in df_exist.columns:
                    failed = df_exist[df_exist['sire_id'].isnull() | (df_exist['sire_id'] == '')]['horse_id'].astype(str)
                    if len(failed) > 0:
                        print(f"♻️ データ欠損がある {len(failed)} 頭をリトライ対象に戻します")
                        existing_ids = existing_ids - set(failed)
        except Exception as e:
            print(f"⚠️ ファイル読み込みエラー: {e}")

    all_targets = get_all_horse_ids()
    clean_targets = {hid.replace('.0', '') for hid in all_targets}
    existing_ids = {hid.replace('.0', '') for hid in existing_ids}

    target_ids = list(clean_targets - existing_ids)
    print(f"🚀 今回収集するターゲット: {len(target_ids)} 頭")

    if not target_ids:
        print("✅ すべて収集済みです。")
        return

    random.shuffle(target_ids)

    session = create_session()
    new_data = []
    COLUMNS = ['horse_id', 'horse_name', 'sire_id', 'sire_name', 'dam_id', 'dam_name']

    try:
        for i, horse_id in enumerate(target_ids):
            sys.stdout.write(f"\r🐎 Processing: {i + 1}/{len(target_ids)} [ID: {horse_id}] ... ")
            sys.stdout.flush()

            data = scrape_pedigree_page(horse_id, session)

            if data == "BLOCK":
                print("\n🚨 ネット競馬側からアクセス制限（BLOCK）を検知しました。")
                print("   安全のため、ここまでのデータを保存してスクリプトを一時停止します。")
                break

            if data:
                new_data.append(data)
                sire = data.get('sire_name', 'X')
                dam = data.get('dam_name', 'X')
                if sire and sire != 'X':
                    sys.stdout.write(f"✅ 父:{sire} / 母:{dam}")
                else:
                    sys.stdout.write("⚠️ Missing (Empty)")
            else:
                sys.stdout.write("⚠️ Error")

            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

            if len(new_data) >= SAVE_INTERVAL:
                save_to_csv_safe(new_data, COLUMNS)
                new_data = []
                print(f" 💾 Saved")

    except KeyboardInterrupt:
        print("\n🛑 中断！データを保存します")

    if new_data:
        save_to_csv_safe(new_data, COLUMNS)

    print("\n🎉 処理を終了しました。")


if __name__ == "__main__":
    main()