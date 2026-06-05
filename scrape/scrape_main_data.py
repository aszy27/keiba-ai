# scrape/scrape_main_data.py
import pandas as pd
import time
import os
import sys
import random
import warnings
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

# ★ keibascraperライブラリを使用
try:
    import keibascraper
except ImportError:
    print("❌ エラー: 'keibascraper' ライブラリが見つかりません。")
    print("pip install keibascraper を実行してください。")
    sys.exit()

warnings.simplefilter('ignore')

# ==========================================
# ★パスの自動動的解決 (サブディレクトリ移動対策)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ==========================================
# ★設定エリア: ここに集めたい年を指定するだけ！
# ==========================================
TARGET_YEAR = 2026  # 👈 取りたい年をここに変更するだけ（例: 2018 や 2024 など）
SAVE_INTERVAL = 20  # 保存間隔

# ★ 年数に応じて保存先フォルダ（train / val / test）を自動判定
if TARGET_YEAR <= 2023:
    FOLDER_TYPE = "train"
elif TARGET_YEAR == 2024:
    FOLDER_TYPE = "val"
else:
    FOLDER_TYPE = "test"

# 自動決定されたフォルダをセット
OUTPUT_DIR = str(DATA_DIR / FOLDER_TYPE)

# ★ BAN対策用設定
MIN_SLEEP = 2.0  # 通常待機 (秒)
MAX_SLEEP = 5.0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]


def check_network_status(race_id):
    """ ★ 追加: 400エラーや正常コード200を装う偽装ブロックを1発目で完全に検知する前衛防衛ロジック """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://race.netkeiba.com/"
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)

        # 1. ページが物理的に存在しない404だけは「データなし(正常)」としてスルー
        if res.status_code == 404:
            return "NO_DATA"

        # 2. 【最重要】400(Bad Request), 403, 429など、200以外のエラーコードはすべて一撃でBLOCKとする
        if res.status_code != 200:
            return "BLOCK"

        try:
            html = res.content.decode('euc-jp')
        except:
            try:
                html = res.content.decode('shift_jis')
            except:
                html = res.content.decode('utf-8', errors='replace')

        soup = BeautifulSoup(html, "html.parser")

        # 3. タイトルによるアクセスブロック画面の検知
        if soup.title:
            title_text = soup.title.get_text()
            if any(w in title_text for w in
                   ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限", "大変混み合って"]):
                return "BLOCK"

        # 4. 中身が空っぽ、あるいは正常な競馬ページ構造（共通タグ）が皆無な場合もブロックとみなす
        if not soup.find(id=re.compile("header|container|main")) and not soup.find(
                class_=re.compile("Race|Header|Layout")):
            return "BLOCK"

        return "OK"
    except:
        return "NETWORK_ERROR"


def safe_load(category, race_id):
    """ ★ 修正: ネットワークチェックを噛ませてサイレント突き進みを完全ブロック """
    status = check_network_status(race_id)
    if status == "BLOCK" or status == "NETWORK_ERROR":
        return "BLOCK"
    if status == "NO_DATA":
        return None

    max_retries = 2
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            data = keibascraper.load(category, race_id)
            return data
        except Exception as e:
            error_msg = str(e)
            # 単なるパースエラー、欠損エラーならデータなしとして諦める
            if "strptime" in error_msg or "NoneType" in error_msg or "argument 1 must be str" in error_msg:
                return None

            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return "BLOCK"
    return None


def scrape_race_data_safe(year):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"race_data_{year}.csv")

    print(f"\n🚀 {year}年のデータ収集を開始します (擬態BAN完全防御版)")
    print(f"   💾 保存先: {filename}")
    print("--------------------------------------------------")

    existing_ids = set()

    if os.path.exists(filename):
        try:
            df_exist = pd.read_csv(filename, dtype={'race_id': str, 'id': str}, encoding='utf-8-sig')
            if 'race_id' in df_exist.columns:
                existing_ids = set(df_exist['race_id'].dropna().astype(str))
            elif 'id' in df_exist.columns:
                existing_ids = set(df_exist['id'].dropna().astype(str))
            print(f"📂 既存ファイルを発見: {len(existing_ids)} レース済み -> 続きから再開します")
        except Exception as e:
            print(f"⚠️ 既存ファイルの読み込みに失敗 ({e})。新規作成として扱います。")

    place_codes = range(1, 11)
    kais = range(1, 7)
    days = range(1, 13)
    rounds = range(1, 13)

    session_data = []

    try:
        for place in place_codes:
            for kai in kais:
                for day in days:

                    race_id_base = f"{year}{str(place).zfill(2)}{str(kai).zfill(2)}{str(day).zfill(2)}"

                    # --- 1Rチェック (ここでもBANを検知できるようにガード) ---
                    check_id = f"{race_id_base}01"
                    if check_id in existing_ids:
                        pass
                    else:
                        check_data = safe_load("result", check_id)
                        if check_data == "BLOCK":
                            print(f"\n🚨 ネット競馬側からアクセス制限（BLOCK）を検知しました。処理を即座に安全停止します。")
                            if session_data: save_to_csv(session_data, filename)
                            sys.exit(1)
                        if check_data is None:
                            continue

                    print(f"\n📅 開催確認: {check_id[:-2]}")

                    # --- 全レース取得 ---
                    for r in rounds:
                        race_id = f"{race_id_base}{str(r).zfill(2)}"

                        if race_id in existing_ids:
                            sys.stdout.write(f"\r    Skipping: {race_id} (Done)   ")
                            sys.stdout.flush()
                            continue

                        sys.stdout.write(f"\r    Running: {race_id} ... ")
                        sys.stdout.flush()

                        try:
                            data = safe_load("result", race_id)

                            # BLOCKを検知したらダミーを入れずにバッファを保存して即終了
                            if data == "BLOCK":
                                print(f"\n🚨 アクセス制限（BLOCK）を検知しました。処理を即座に安全停止します。")
                                if session_data: save_to_csv(session_data, filename)
                                sys.exit(1)

                            if data and len(data) >= 2 and len(data[0]) > 0:
                                race_info = data[0][0]
                                horse_results = data[1]

                                df_temp = pd.DataFrame(horse_results)
                                for key, val in race_info.items():
                                    df_temp[key] = val

                                df_temp['race_id'] = race_id

                                session_data.append(df_temp)
                                existing_ids.add(str(race_id))

                                sys.stdout.write(f"✅ OK ({len(df_temp)}頭)\n")
                            else:
                                sys.stdout.write("Skip (No Data)\n")

                            if len(session_data) >= SAVE_INTERVAL:
                                save_to_csv(session_data, filename)
                                session_data = []
                                print(f"    💾 保存完了 (計 {len(existing_ids)} レース)")

                        except Exception as e:
                            print(f"\n❌ エラー ({race_id}): {e}")
                            continue

    except KeyboardInterrupt:
        print("\n\n🛑 中断！現在保持しているデータを保存します。")

    if len(session_data) > 0:
        save_to_csv(session_data, filename)

    print(f"\n🎉 終了しました。現在の総レース数: {len(existing_ids)}")


def save_to_csv(data_list, filename):
    if not data_list: return
    df_new = pd.concat(data_list, ignore_index=True)

    cols = ['race_id', 'rank', 'bracket', 'horse_number', 'horse_id', 'horse_name',
            'gender', 'age', 'burden', 'jockey_id', 'jockey_name', 'rap_time',
            'diff_time', 'passage_rank', 'last_3f', 'weight', 'weight_diff',
            'trainer_id', 'trainer_name', 'prize', 'id', 'race_number', 'race_name',
            'race_date', 'race_time', 'type', 'length', 'length_class', 'handed',
            'weather', 'condition', 'place', 'course', 'round', 'days',
            'head_count', 'max_prize']

    if os.path.exists(filename):
        try:
            existing_cols = pd.read_csv(filename, nrows=0, encoding='utf-8-sig').columns.tolist()
            for c in existing_cols:
                if c not in df_new.columns: df_new[c] = None
            df_new = df_new[existing_cols]
            df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='a', header=False)
            return
        except:
            pass

    for c in cols:
        if c not in df_new.columns: df_new[c] = None
    df_new = df_new[cols]
    df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(filename))


if __name__ == "__main__":
    scrape_race_data_safe(TARGET_YEAR)