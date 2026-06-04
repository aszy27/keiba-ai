# core/data_loader.py
import pandas as pd
import os
import glob
import re
from core.config import BASE_DIR
import unicodedata

DATA_DIR = os.path.join(BASE_DIR, "data")

FILE_BREEDER = os.path.join(DATA_DIR, "breeder_data_progress.csv")
FILE_LAP = os.path.join(DATA_DIR, "lap_data_progress.csv")
FILE_TRAIN = os.path.join(DATA_DIR, "training_data_progress.csv")

# グレード正規化用マップ
GRADE_MAP = {
    'G1': 'G1', 'GI': 'G1', 'JGI': 'G1', 'JG1': 'G1', 'JPNI': 'G1', 'JPN1': 'G1',
    'G2': 'G2', 'GII': 'G2', 'JGII': 'G2', 'JG2': 'G2', 'JPNII': 'G2', 'JPN2': 'G2',
    'G3': 'G3', 'GIII': 'G3', 'JGIII': 'G3', 'JG3': 'G3', 'JPNIII': 'G3', 'JPN3': 'G3',
    'L': 'L', 'OP': 'OP'
}


def extract_grade_from_name(name) -> str:
    """
    race_name からグレードを全方位・高精度で抽出する
    """
    if pd.isna(name): return 'OP'

    # 🔥 FIX 1: 末尾アンカーを撤廃し、文字列内のすべてのカッコ中身をリストで抽出
    blocks = re.findall(r'[\(（]([^\)）]+)[\)）]', str(name))
    if not blocks: return 'OP'

    # カッコを順番に精査し、最初にヒットした有効なグレードを返す
    for block in blocks:
        # 🔥 FIX 2: 空白除去に加え、障害重賞特有のドット「.」や中黒「・」も一括で除去
        tag = block.strip().replace(' ', '').replace('.', '').replace('・', '').upper()
        tag = unicodedata.normalize('NFKC', tag)

        if tag in GRADE_MAP and GRADE_MAP[tag] != 'OP':
            return GRADE_MAP[tag]

    return 'OP'


def load_and_merge_all_data(data_dirs):
    print("   -> 🗂️ ベースデータと拡張データを動的マージ中 (V7 Bulletproof Loader)...")

    frames = []
    id_dtypes = {'horse_id': str, 'jockey_id': str, 'trainer_id': str, 'race_id': str}

    for dname in data_dirs:
        if not os.path.exists(dname): continue
        for f in sorted(glob.glob(os.path.join(dname, "race_data_*.csv"))):
            try:
                frames.append(pd.read_csv(f, dtype=id_dtypes, low_memory=False))
            except Exception as e:
                print(f"   ⚠️ 読込スキップ: {f} ({e})")

    if not frames:
        raise FileNotFoundError(f"ベースデータが見つかりません: {data_dirs}")

    df_base = pd.concat(frames, ignore_index=True)
    df_base = df_base.drop_duplicates(subset=['race_id', 'horse_id'], keep='last')

    # 1. Breeder データのマージ
    if os.path.exists(FILE_BREEDER):
        try:
            df_b = pd.read_csv(FILE_BREEDER, usecols=['horse_id', 'breeder'], dtype={'horse_id': str})
            df_b = df_b.drop_duplicates(subset=['horse_id'], keep='last')
            if 'breeder' in df_base.columns: df_base = df_base.drop(columns=['breeder'])
            df_base = pd.merge(df_base, df_b, on='horse_id', how='left')
        except Exception as e:
            print(f"   ⚠️ Breederデータ読込失敗: {e}")

    # 2. Train(追い切り) データのマージ
    if os.path.exists(FILE_TRAIN):
        try:
            df_t = pd.read_csv(FILE_TRAIN, usecols=['race_id', 'horse_id', 'oikiri_rank', 'oikiri_last1f'],
                               dtype={'race_id': str, 'horse_id': str})
            df_t = df_t.drop_duplicates(subset=['race_id', 'horse_id'], keep='last')
            for c in ['oikiri_rank', 'oikiri_last1f']:
                if c in df_base.columns: df_base = df_base.drop(columns=[c])
            df_base = pd.merge(df_base, df_t, on=['race_id', 'horse_id'], how='left')
        except Exception as e:
            print(f"   ⚠️ 追い切りデータ読込失敗: {e}")

    # 3. Lap データのマージ
    if os.path.exists(FILE_LAP):
        try:
            df_l = pd.read_csv(FILE_LAP, usecols=['race_id', 'pace_type', 'first_3f', 'last_3f_race'],
                               dtype={'race_id': str})
            df_l = df_l.drop_duplicates(subset=['race_id'], keep='last')
            for c in ['pace_type', 'first_3f', 'last_3f_race']:
                if c in df_base.columns: df_base = df_base.drop(columns=[c])
            df_base = pd.merge(df_base, df_l, on='race_id', how='left')
        except Exception as e:
            print(f"   ⚠️ ラップデータ読込失敗: {e}")

    # 🔥 FIX 3: 特徴量エンジニアリング側で NaN センチネル（不当な格下げ）を発生させない防衛
    if 'grade' in df_base.columns:
        df_base['grade'] = df_base['grade'].fillna('OP').replace('', 'OP').astype(str)
    else:
        df_base['grade'] = 'OP'

    # グレード補完 (全方位ベクトルパース)
    if 'race_name' in df_base.columns:
        extracted = df_base['race_name'].apply(extract_grade_from_name)
        mask = extracted != 'OP'
        df_base.loc[mask, 'grade'] = extracted[mask]

    return df_base