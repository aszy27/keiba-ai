# scrape/scrape_main_data.py
import pandas as pd
import time
import os
import sys
import random
import warnings
from pathlib import Path

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
BLOCK_WAIT = 120  # 通信エラー時の待機 (秒)

# ==========================================

def safe_load(category, race_id):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            data = keibascraper.load(category, race_id)
            if data is None:
                raise ValueError("Data is None")
            return data
        except Exception as e:
            error_msg = str(e)
            if "strptime" in error_msg or "NoneType" in error_msg or "argument 1 must be str" in error_msg:
                return None
            if attempt < max_retries - 1:
                print(f"\n🚨 通信エラー感知。{BLOCK_WAIT}秒間待機してリトライします... ({e})")
                time.sleep(BLOCK_WAIT)
            else:
                return None
    return None


def scrape_race_data_safe(year):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"race_data_{year}.csv")

    print(f"\n🚀 {year}年のデータ収集を開始します (自動フォルダ配置版)")
    print(f"   📂 判定された配置先: data/{FOLDER_TYPE}/")
    print(f"   💾 保存ファイルパス: {filename}")
    print("--------------------------------------------------")

    existing_ids = set()

    if os.path.exists(filename):
        try:
            df_exist = pd.read_csv(filename, dtype={'race_id': str, 'id': str})
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
                    check_id = f"{race_id_base}01"
                    if check_id in existing_ids:
                        pass
                    else:
                        try:
                            check_data = safe_load("result", check_id)
                            if check_data is None:
                                continue
                        except Exception:
                            continue

                    print(f"\n📅 開催確認: {check_id[:-2]}")

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
            existing_cols = pd.read_csv(filename, nrows=0).columns.tolist()
            for c in existing_cols:
                if c not in df_new.columns:
                    df_new[c] = None
            df_new = df_new[existing_cols]
            df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='a', header=False)
            return
        except Exception as e:
            print(f"⚠️ 既存の列順取得に失敗したため、デフォルト列順で追記します: {e}")

    for c in cols:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new[cols]
    df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='w' if not os.path.exists(filename) else 'a',
                  header=not os.path.exists(filename))


if __name__ == "__main__":
    scrape_race_data_safe(TARGET_YEAR)