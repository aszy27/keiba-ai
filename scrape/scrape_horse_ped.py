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
MIN_SLEEP = 1.0
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

        if response.status_code == 404:
            return None
        if response.status_code != 200:
            return "BLOCK"

        response.encoding = 'euc-jp'
        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title:
            title_text = soup.title.get_text()
            if any(w in title_text for w in
                   ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限", "大変混み合って"]):
                return "BLOCK"

        if not soup.find(id=re.compile("header|container|main")) and not soup.find(
                class_=re.compile("Race|Header|Layout|db")):
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


def sanitize_existing_ped_file():
    print("🧹 既存の血統マスタCSVから不完全なデータ（ブロック警告の巻き込み行など）を自動クレンジング中...")
    if os.path.exists(FILENAME):
        try:
            df = pd.read_csv(FILENAME, dtype=str, encoding='utf-8-sig')
            before = len(df)

            invalid_keywords = ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限", "大変混み合って"]
            mask = df['horse_name'].isnull()
            for kw in invalid_keywords:
                mask = mask | df['horse_name'].str.contains(kw, na=False)

            df_clean = df[~mask & df['horse_id'].notnull()]
            if len(df_clean) < before:
                df_clean.to_csv(FILENAME, index=False, encoding='utf-8-sig')
                print(f"   📊 血統マスタ: 過去の不完全な {before - len(df_clean)} 行を抹消し、適正化しました。")
        except Exception as e:
            print(f"   ⚠️ 血統マスタの自動クレンジング失敗: {e}")


def main():
    print(f"\n🚀 血統データの収集 (フォルダ整理版) を開始します")
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)

    sanitize_existing_ped_file()

    existing_ids = set()
    if os.path.exists(FILENAME):
        try:
            df_exist = pd.read_csv(FILENAME, dtype={'horse_id': str}, encoding='utf-8-sig')
            if 'horse_id' in df_exist.columns:
                invalid_keywords = ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限",
                                    "大変混み合って"]
                mask = df_exist['horse_name'].isnull()
                for kw in invalid_keywords:
                    mask = mask | df_exist['horse_name'].str.contains(kw, na=False)

                df_valid = df_exist[~mask]
                existing_ids = set(df_valid['horse_id'].dropna().astype(str))
                print(f"📚 健全な既存マスタ登録数: {len(existing_ids)} 頭")

                if 'sire_id' in df_valid.columns:
                    failed = df_valid[df_valid['sire_id'].isnull() | (df_valid['sire_id'] == '')]['horse_id'].astype(
                        str)
                    if len(failed) > 0:
                        print(f"♻️ 親IDに欠損がある {len(failed)} 頭を自動で再取得（穴埋め）対象に戻します")
                        existing_ids = existing_ids - set(failed)
        except Exception as e:
            print(f"⚠️ ファイル読み込みエラー: {e}")

    all_targets = get_all_horse_ids()
    clean_targets = {hid.replace('.0', '') for hid in all_targets}
    existing_ids = {hid.replace('.0', '') for hid in existing_ids}

    target_ids = list(clean_targets - existing_ids)
    print(f"🚀 今回収集するターゲット（自動穴埋め対象含む）: {len(target_ids)} 頭")

    if not target_ids:
        print("✅ すべて収集・補完済みです。")
        return

    random.shuffle(target_ids)

    session = create_session()
    new_data = []
    COLUMNS = ['horse_id', 'horse_name', 'sire_id', 'sire_name', 'dam_id', 'dam_name']

    # =========================================================================
    # 🔥 【追加】 前衛ヘルスチェック：実在が確定している名馬（ディープインパクト）へテストアクセス
    # =========================================================================
    print("🩺 通信環境のヘルスチェックを実行中 (実在する有名馬へのテストアクセス)...")
    health_check = scrape_pedigree_page("2002100816", session)

    if health_check == "BLOCK":
        print(f"\n🚨 【通信遮断を検知】現在、IPアドレスがサイト側から完全にブロックされています。")
        print(
            f"🛡️  マスタの汚染を防ぐため、'NO_PED'の書き込みを行わずにスクリプトを安全停止します。IP環境を変えてから再実行してください。")
        sys.exit(1)

    print("✅ 通信環境は100%正常です（IP BANはされていません）。残りのゴーストIDの同期を開始します。")
    # =========================================================================

    consecutive_block_count = 0  # 連続してBLOCKされた頭数（別々の馬にまたがるIP BAN検知用）

    try:
        for i, horse_id in enumerate(target_ids):
            success = False
            horse_retry_count = 0  # その馬単体でのリトライ回数

            while not success:
                sys.stdout.write(f"\r🐎 Processing: {i + 1}/{len(target_ids)} [ID: {horse_id}] ... ")
                sys.stdout.flush()

                data = scrape_pedigree_page(horse_id, session)

                if data == "BLOCK":
                    # 🔴 FIX: 起動時のヘルスチェックが通っただけで以後の全BLOCKを「ゴーストID」と
                    #          断定すると、途中からIP制限がかかった場合に実在する馬まで誤って
                    #          'NO_PED' で汚染し、二度と再取得されなくなる。
                    #          ここでは NO_PED を書き込まず、必ずリトライ＆冷却を経由する。
                    horse_retry_count += 1
                    print(f"\n🚨 制限検知(BLOCK) [その馬のリトライ: {horse_retry_count}/2回目]")

                    if new_data:
                        save_to_csv_safe(new_data, COLUMNS)
                        new_data = []

                    # 1回目のBLOCKなら5分待ってセッションを変えて同じ馬を再開
                    if horse_retry_count < 2:
                        print("⏳ 冷却とIP規制解除を待つため、5分間（300秒）完全停止します...")
                        time.sleep(300)
                        print("♻️ 5分経過しました。セッションとUser-Agentをリフレッシュして同じ馬から再挑戦します。")
                        session = create_session()
                        continue
                    else:
                        # 2回連続で同じ馬がBLOCKされた場合、その馬の取得を諦めてスマートスキップ
                        # （NO_PEDは刻印しない＝次回実行時に再挑戦できるようにする）
                        print(
                            f"⚠️ 5分待っても同じ馬 {horse_id} で即ブロックされるため、この馬は一旦スキップして次へ突き進みます。")
                        consecutive_block_count += 1
                        break  # whileループを強制脱出して、次の馬（forの次のループ）へ進む

                if data:
                    new_data.append(data)
                    consecutive_block_count = 0  # 正常に1頭でも取れたら連続ブロックカウントをリセット
                    sire = data.get('sire_name', 'X')
                    dam = data.get('dam_name', 'X')
                    if sire and sire != 'X':
                        sys.stdout.write(f"✅ 父:{sire} / 母:{dam}")
                    else:
                        sys.stdout.write("⚠️ Missing (Empty)")
                else:
                    sys.stdout.write("⚠️ Error (No Data)")

                success = True
                consecutive_block_count = 0  # 正常終了時はリセット
                time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

            # --- 【鉄壁のプロテクトガード】 ---
            # スキップして次の馬、その次の馬に進んでも、5分待機➔即ブロックが「連続3頭」続く場合、
            # それは馬が原因ではなくグローバルIP自体が完全に焼き切れている。
            # 永久BANを防ぐため、データを全セーブして安全に終了する。
            if consecutive_block_count >= 3:
                print(
                    f"\n🚨 深刻なエラー: 連続して3頭の別々の馬で待機後も即ブロックされました。現在のIPアドレスが完全にロックされています。")
                print(
                    f"🛡️ 安全のため、ここまでのデータをすべて保存してスクリプトを終了します。IP環境を変えてから再実行してください。")
                if new_data:
                    save_to_csv_safe(new_data, COLUMNS)
                sys.exit(1)

            if len(new_data) >= SAVE_INTERVAL:
                save_to_csv_safe(new_data, COLUMNS)
                new_data = []
                print(f" 💾 Saved")

    except KeyboardInterrupt:
        print("\n🛑 中断！データを保存します")

    if new_data:
        save_to_csv_safe(new_data, COLUMNS)

    print("\n🎉 全ての有効な血統データの同期が完了しました！")


if __name__ == "__main__":
    main()