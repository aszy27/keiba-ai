# core/features.py
import pandas as pd
import numpy as np
import os
import warnings
from core.config import BASE_DIR, PEDIGREE_FILE, HORSE_VECTOR_FILE, VEC_DIM

MASTER_COURSE_FILE = os.path.join(BASE_DIR, "data", "master_course_data.csv")


def safe_transform(mapping, labels):
    unknown_idx = mapping.get('unknown', 0)
    return [mapping.get(str(x), unknown_idx) for x in labels]


GRADE_MAP = {'G1': 10, 'G2': 9, 'G3': 8, 'L': 7, 'OP': 6}
OIKIRI_MAP = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
PACE_MAP = {'H': 3, 'M': 2, 'S': 1}


def feature_engineering(df, weight_mean: float = None, burden_mean: float = None):
    """
    特徴量エンジニアリング (V8 学習/バックテスト/本番 完全整合版)

    設計原則: あるレースの特徴量は「そのレース日より前に確定していた情報」のみから
    計算される。これにより、学習(train_main)・バックテスト(evaluate_main)・
    本番(predict_main)の3経路が全く同じ条件で特徴量を生成する。
    """
    if weight_mean is None or burden_mean is None:
        raise ValueError(
            "⚠️ weight_mean / burden_mean は必ず train データの統計値を渡してください。"
            " val/test 内の平均で埋めるとデータリークになります。"
        )

    print("   -> 🛠️ 特徴量エンジニアリング (V8 Train/Backtest/Production Aligned Edition)...")
    df = df.copy()

    # =========================================================================
    # 🔴 FIX: 全経路（学習・バックテスト・本番）で rank と表記を統一する
    # =========================================================================
    # rank: train_main.py は欠損rankを99で埋めるが、evaluate/predict は埋めずに
    #       to_numeric→NaN→0 になっており、prev_rank_* / avg_rank_5 が
    #       学習時(99)と推論時(0)で食い違っていた。ここで一元的に正規化する。
    #       ・history行（結果確定済み）: 非数値(中止・取消等)は 99 に統一
    #       ・target行（本番の未確定レース）: NaN のまま維持（is_win等は自動的に0になる）
    _rank_num = pd.to_numeric(df['rank'], errors='coerce') if 'rank' in df.columns else pd.Series(np.nan, index=df.index)
    if 'split_tag' in df.columns:
        _hist_mask = (df['split_tag'] == 'history')
    else:
        _hist_mask = pd.Series(True, index=df.index)
    df['rank'] = _rank_num.where(~(_hist_mask & _rank_num.isna()), 99)

    # condition: 学習データ(keibascraper)は「稍」表記だが、本番スクレイプは「稍重」に
    #            なり得るため、カテゴリ特徴が unknown 化していた。「稍」に統一する。
    if 'condition' in df.columns:
        df['condition'] = df['condition'].astype(str).str.strip().replace({'稍重': '稍'})

    # grade: keibascraperの一部年度は非重賞を「一般」、grade列が無い年度は「OP」となり、
    #        同じクラスのレースが年度によって別カテゴリ＆grade_score(1 vs 6)に割れていた。
    #        本番predictは「OP」を出すため、「一般」→「OP」に統一する。
    if 'grade' in df.columns:
        df['grade'] = df['grade'].fillna('OP').replace({'一般': 'OP'})
    else:
        df['grade'] = 'OP'

    # グレードスコア
    df['grade_score'] = df['grade'].astype(str).map(GRADE_MAP).fillna(1).astype(int)
    df['race_date'] = pd.to_datetime(df['race_date'])

    # ID列のクレンジング
    for id_col in ['horse_id', 'jockey_id', 'trainer_id']:
        if id_col in df.columns:
            df[id_col] = df[id_col].fillna('unknown').astype(str).str.replace(r'\.0$', '', regex=True)

    df['head_count'] = df.groupby('race_id')['horse_id'].transform('count') if 'race_id' in df.columns else 1

    if 'breeder' not in df.columns:
        df['breeder'] = 'unknown'
    df['breeder'] = df['breeder'].fillna('unknown').astype(str).str.strip().str.replace(r'\s+', '', regex=True)

    # ★修正：oikiri_last1f に依存していたロジックを完全抹消
    # 🔴 FIX: 旧実装は「文字列がそのままS/A/B/C/Dであること」を前提にしていたため、
    #          本番スクレイプで欠損馬が ''(空文字) になると isna()=False となり
    #          oikiri_missing=0 と誤判定されていた（学習時はNaN→1）。
    #          S/A/B/C/D の1文字を頑健に抽出し、抽出できない場合を missing=1 に統一する。
    if 'oikiri_rank' in df.columns:
        # NaN を先にマスクする（astype(str) で 'NAN' となり 'A' が誤抽出されるのを防ぐ）
        _oik = df['oikiri_rank'].astype(str).str.upper().str.extract(r'([SABCD])', expand=False)
        _oik = _oik.mask(df['oikiri_rank'].isna())
        df['oikiri_score'] = _oik.map(OIKIRI_MAP).fillna(0).astype(int)
        df['oikiri_missing'] = _oik.isna().astype(int)  # 追い切り評価自体がない馬を1とする
    else:
        df['oikiri_score'], df['oikiri_missing'] = 0, 1

    if 'pace_type' in df.columns:
        df['pace_score'] = df['pace_type'].astype(str).str.upper().map(PACE_MAP).fillna(0).astype(int)
        df['first_3f'] = pd.to_numeric(df['first_3f'], errors='coerce').fillna(0.0)
        df['last_3f_race'] = pd.to_numeric(df['last_3f_race'], errors='coerce').fillna(0.0)
    else:
        df['pace_score'], df['first_3f'], df['last_3f_race'] = 0, 0.0, 0.0

    df['is_win'] = (pd.to_numeric(df['rank'], errors='coerce') == 1).astype(int)
    df['is_top3'] = (pd.to_numeric(df['rank'], errors='coerce') <= 3).astype(int)

    # コース情報の結合
    place_str = df['place'].fillna('unknown').astype(str)
    type_str = df['type'].fillna('unknown').astype(str)
    df['course_id'] = place_str + type_str
    df['course'] = place_str + type_str + df['length'].fillna(0).astype(int).astype(str)

    if os.path.exists(MASTER_COURSE_FILE):
        try:
            course_master = pd.read_csv(MASTER_COURSE_FILE, encoding='utf-8-sig')
            course_master = course_master.rename(columns={'straight_len': 'straight', 'slope_height': 'elevation'})
            df = pd.merge(df, course_master[['course_id', 'straight', 'elevation', 'has_slope']], on='course_id',
                          how='left')
            df['straight'] = pd.to_numeric(df['straight'], errors='coerce').fillna(0.0)
            df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce').fillna(0.0)
            df['has_slope'] = pd.to_numeric(df['has_slope'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"   ⚠️ コースマスタ読み込み失敗: {e}")
            df['straight'], df['elevation'], df['has_slope'] = 0.0, 0.0, 0
    else:
        df['straight'], df['elevation'], df['has_slope'] = 0.0, 0.0, 0
    df = df.drop(columns=['course_id'])

    # ========================================
    # 馬単位の過去成績
    # ========================================
    df = df.sort_values(['horse_id', 'race_date', 'race_id']).reset_index(drop=True)
    grouped = df.groupby('horse_id', sort=False)

    df['run_count'] = grouped.cumcount()

    for i in range(1, 6):
        df[f'prev_rank_{i}'] = pd.to_numeric(grouped['rank'].shift(i), errors='coerce').fillna(0)
        df[f'prev_diff_{i}'] = grouped['diff_time'].shift(i).fillna(1.0)
        df[f'prev_3f_{i}'] = grouped['last_3f'].shift(i).fillna(0.0)

    df['prev_pace_1'] = grouped['pace_score'].shift(1).fillna(0)
    df['prev_first_3f_1'] = grouped['first_3f'].shift(1).fillna(0.0)
    df['prev_last_3f_race_1'] = grouped['last_3f_race'].shift(1).fillna(0.0)

    if 'passage_rank' in df.columns:
        last_pos = grouped['passage_rank'].shift(1).astype(str).str.extract(r'(\d+)[^\d]*$')[0]
        df['prev_pos_1'] = pd.to_numeric(last_pos, errors='coerce').fillna(10).astype(int)
        df['is_nige'] = (df['prev_pos_1'] == 1).astype(int)
    else:
        df['prev_pos_1'], df['is_nige'] = 10, 0

    df['is_pre_senkou'] = (df['prev_pos_1'] <= 4).astype(int)

    if 'race_id' in df.columns:
        df['race_senkou_count'] = df.groupby('race_id')['is_pre_senkou'].transform('sum')
        df['race_senkou_ratio'] = (df['race_senkou_count'] / df['head_count']).fillna(0)

    df['prev_date'] = grouped['race_date'].shift(1)
    df['interval'] = (df['race_date'] - df['prev_date']).dt.days.fillna(0)
    df['prev_is_win'] = (df['prev_rank_1'] == 1).astype(int)

    if 'jockey_id' in df.columns:
        df['jockey_change'] = (df['jockey_id'] != grouped['jockey_id'].shift(1).fillna('unknown')).astype(int)
    else:
        df['jockey_change'] = 0

    rank_cols = [f'prev_rank_{i}' for i in range(1, 6)]
    df['avg_rank_5'] = df[rank_cols].replace(0, np.nan).mean(axis=1).fillna(0)

    # 血統ベクトル結合
    if os.path.exists(HORSE_VECTOR_FILE):
        try:
            bv = pd.read_csv(HORSE_VECTOR_FILE, dtype={'horse_id': str})
            bv['horse_id'] = bv['horse_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            df = df.drop(columns=[c for c in df.columns if c.startswith('p_vec_')], errors='ignore')
            df = pd.merge(df, bv, on='horse_id', how='left')
        except Exception as e:
            print(f"   ⚠️ 血統ベクトル読み込み失敗: {e}")

    vec_cols = [f'p_vec_{i}' for i in range(VEC_DIM)]
    for col in vec_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[vec_cols] = df[vec_cols].fillna(0.0)

    # ========================================
    # 累積成績 (V8: 日付単位の排他集計に統一)
    # ========================================
    # 🔴 FIX: 旧実装は (race_date, race_id) 順の累積だったため、学習時は「同日の
    #          先行レースの結果」が特徴量に混入していた。本番の推論時点では同日の
    #          結果は知り得ないため、学習と本番で分布がズレる（train/serving skew）。
    #          さらに evaluate では target行の勝利(is_win)は加算されるのに騎乗数
    #          (_is_history_ride=0)は加算されず、勝率が水増しされるバグがあった。
    #          → 「その日より前（date-exclusive）」の累積に統一し、
    #            出走カウントは _ride_flag（結果が確定した行のみ1）で数える。
    #          これにより 学習・バックテスト・本番 の3経路が完全に同一条件になる。
    if 'split_tag' in df.columns:
        # history行(結果確定済み) または rankが数値の行(=バックテストのtarget行)は出走1件。
        # 本番predictのtarget行(rank未確定=NaN)のみ0となる。
        df['_ride_flag'] = ((df['split_tag'] == 'history') | df['rank'].notna()).astype(int)
    else:
        df['_ride_flag'] = 1

    for col, prefix in [('jockey_id', 'jockey'), ('trainer_id', 'trainer'), ('breeder', 'breeder')]:
        if col not in df.columns: continue

        grp = df.groupby([col, 'race_date']).agg(
            d_wins=('is_win', 'sum'),
            d_top3=('is_top3', 'sum'),
            d_rides=('_ride_flag', 'sum')
        ).reset_index()

        grp = grp.sort_values('race_date')

        # 当日分を差し引き、「その日より前」までの累積にする
        grp['cum_win'] = grp.groupby(col)['d_wins'].cumsum() - grp['d_wins']
        grp['cum_top3'] = grp.groupby(col)['d_top3'].cumsum() - grp['d_top3']
        grp['cum_rides'] = grp.groupby(col)['d_rides'].cumsum() - grp['d_rides']

        grp[f'{prefix}_win_rate'] = (grp['cum_win'] / grp['cum_rides']).fillna(0.0)
        grp[f'{prefix}_top3_rate'] = (grp['cum_top3'] / grp['cum_rides']).fillna(0.0)

        df = pd.merge(df, grp[[col, 'race_date', f'{prefix}_win_rate', f'{prefix}_top3_rate']],
                      on=[col, 'race_date'], how='left')
        df[f'{prefix}_win_rate'] = df[f'{prefix}_win_rate'].fillna(0.0)
        df[f'{prefix}_top3_rate'] = df[f'{prefix}_top3_rate'].fillna(0.0)

    # 血統マスタ（sire）の結合
    if os.path.exists(PEDIGREE_FILE):
        ped = pd.read_csv(PEDIGREE_FILE, dtype=str)[['horse_id', 'sire_name']].rename(columns={'sire_name': 'sire'})
        ped['horse_id'] = ped['horse_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df = df.drop(columns=['sire'], errors='ignore')
        df = pd.merge(df, ped, on='horse_id', how='left')
        df['sire'] = df['sire'].fillna('unknown')
    else:
        df['sire'] = 'unknown'

    # ========================================
    # コース別累積成績 (V8: 日付単位の排他集計に統一)
    # ========================================
    for col, feat in [('jockey_id', 'jockey_course_win'), ('trainer_id', 'trainer_course_win'),
                      ('sire', 'sire_course_win')]:
        if col not in df.columns:
            df[feat] = 0.0
            continue

        grp2 = df.groupby([col, 'course', 'race_date']).agg(
            d_wins=('is_win', 'sum'),
            d_rides=('_ride_flag', 'sum')
        ).reset_index()

        grp2 = grp2.sort_values('race_date')

        grp2['cum_win'] = grp2.groupby([col, 'course'])['d_wins'].cumsum() - grp2['d_wins']
        grp2['cum_rides'] = grp2.groupby([col, 'course'])['d_rides'].cumsum() - grp2['d_rides']

        grp2[feat] = (grp2['cum_win'] / grp2['cum_rides']).fillna(0.0)

        df = df.drop(columns=[feat], errors='ignore')
        df = pd.merge(df, grp2[[col, 'course', 'race_date', feat]], on=[col, 'course', 'race_date'], how='left')
        df[feat] = df[feat].fillna(0.0)

    if '_ride_flag' in df.columns:
        df = df.drop(columns=['_ride_flag'])

    # 数値列の欠損補完
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(
            weight_mean if weight_mean else df['weight'].mean())
    if 'burden' in df.columns:
        df['burden'] = pd.to_numeric(df['burden'], errors='coerce').fillna(
            burden_mean if burden_mean else df['burden'].mean())
    if 'weight_diff' in df.columns:
        df['weight_diff'] = pd.to_numeric(df['weight_diff'].astype(str).str.replace('±', '', regex=False),
                                          errors='coerce').fillna(0.0)
    else:
        df['weight_diff'] = 0.0

    return df
