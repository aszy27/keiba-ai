# build_edge_dataset.py
# オッズへの上積みを検証するための馬単位データ（既存特徴量＋当日バイアス特徴量）を2024年以降について作る。
import numpy as np
import pandas as pd

from core.config import DATA_DIRS_ALL, DATA_DIR, FEATURE_COLS
from core.data_loader import load_and_merge_all_data
from core.features import feature_engineering

OUT_FILE = str(DATA_DIR / "edge_dataset.pkl")
START = "2021-01-01"
BASE_FEATURES = [c for c in FEATURE_COLS if not c.startswith('dae_')]
BIAS_FEATURES = ['day_n', 'day_num_dev', 'day_pos_dev', 'horse_rel_num', 'horse_style', 'inner_x', 'front_x']
CHANGE_FEATURES = ['dist_change', 'surface_change', 'place_change', 'burden_change',
                   'jockey_form30', 'trainer_form30', 'prev_mkt_resid']
FILE_ODDS = str(DATA_DIR / "odds_api_progress.csv")


def day_bias(df):
    d = df[['race_id', 'horse_id', 'race_date', 'place', 'type', 'race_number', 'horse_number',
            'passage_rank', 'rank_raw']].copy()
    d['rank_num'] = pd.to_numeric(d['rank_raw'], errors='coerce')
    d['finished'] = d['rank_num'].notna()
    n_fin = d.groupby('race_id')['finished'].transform('sum')
    denom = (n_fin - 1).clip(lower=1)
    d['horse_rel_num'] = (pd.to_numeric(d['horse_number'], errors='coerce') - 1) / denom
    first_pos = pd.to_numeric(d['passage_rank'].astype(str).str.split('-').str[0], errors='coerce')
    d['rel_pos1'] = (first_pos - 1) / denom

    # その馬の過去5走の1コーナー相対位置（当日の同レースは含まない）
    d = d.sort_values(['horse_id', 'race_date', 'race_id'])
    d['horse_style'] = (d.groupby('horse_id')['rel_pos1']
                        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()))

    t3 = d[d['rank_num'].between(1, 3)].groupby('race_id').agg(t3_num=('horse_rel_num', 'mean'),
                                                                  t3_pos=('rel_pos1', 'mean'))
    races = d.drop_duplicates('race_id')[['race_id', 'race_date', 'place', 'type', 'race_number']].merge(
        t3, on='race_id', how='left')
    races['race_number'] = pd.to_numeric(races['race_number'], errors='coerce')

    # 同じ日・同じ競馬場・同じ芝ダで、それより前に行われたレースの上位3頭の傾向
    races = races.sort_values(['race_date', 'place', 'type', 'race_number']).reset_index(drop=True)
    g = races.groupby(['race_date', 'place', 'type'], sort=False)
    for c in ['t3_num', 't3_pos']:
        races[f'{c}_sum'] = g[c].transform(lambda s: s.fillna(0).cumsum().shift(1)).fillna(0)
    races['day_n'] = g['t3_num'].transform(lambda s: s.notna().cumsum().shift(1)).fillna(0)

    # 競馬場×芝ダごとの長期平均（その日より前の日のみ）
    daily = races.groupby(['place', 'type', 'race_date']).agg(s_num=('t3_num', 'sum'), s_pos=('t3_pos', 'sum'),
                                                              n=('t3_num', 'count')).reset_index()
    daily = daily.sort_values(['place', 'type', 'race_date'])
    gd = daily.groupby(['place', 'type'])
    for c in ['s_num', 's_pos', 'n']:
        daily[f'cum_{c}'] = gd[c].cumsum() - daily[c]
    daily['base_num'] = daily['cum_s_num'] / daily['cum_n'].replace(0, np.nan)
    daily['base_pos'] = daily['cum_s_pos'] / daily['cum_n'].replace(0, np.nan)
    races = races.merge(daily[['place', 'type', 'race_date', 'base_num', 'base_pos']],
                        on=['place', 'type', 'race_date'], how='left')

    has_day = races['day_n'] > 0
    races['day_num_dev'] = np.where(has_day, races['t3_num_sum'] / races['day_n'].clip(lower=1) - races['base_num'], 0)
    races['day_pos_dev'] = np.where(has_day, races['t3_pos_sum'] / races['day_n'].clip(lower=1) - races['base_pos'], 0)

    d = d.merge(races[['race_id', 'day_n', 'day_num_dev', 'day_pos_dev']], on='race_id', how='left')
    for c in ['day_num_dev', 'day_pos_dev', 'horse_style']:
        d[c] = d[c].fillna(0)
    d['horse_style'] = d['horse_style'].where(d['horse_style'] != 0, 0.5)
    d['inner_x'] = (d['horse_rel_num'] - 0.5) * d['day_num_dev']
    d['front_x'] = (d['horse_style'] - 0.5) * d['day_pos_dev']
    return d[['race_id', 'horse_id', 'finished'] + BIAS_FEATURES]


def change_features(df):
    d = df[['race_id', 'horse_id', 'race_date', 'place', 'type', 'length', 'burden', 'jockey_id', 'trainer_id',
            'horse_number', 'rank_raw']].copy()
    d['rank_num'] = pd.to_numeric(d['rank_raw'], errors='coerce')
    for c in ['length', 'burden', 'horse_number']:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    d = d.sort_values(['horse_id', 'race_date', 'race_id'])
    prev = d.groupby('horse_id')[['length', 'type', 'place', 'burden']].shift(1)
    d['dist_change'] = d['length'] - prev['length']
    d['surface_change'] = np.where(prev['type'].isna(), np.nan, (d['type'] != prev['type']).astype(float))
    d['place_change'] = np.where(prev['place'].isna(), np.nan, (d['place'] != prev['place']).astype(float))
    d['burden_change'] = d['burden'] - prev['burden']

    # 騎手・調教師の直近30日（当日を含まない）の勝率。騎乗5回未満は欠損のまま
    for key, name in [('jockey_id', 'jockey_form30'), ('trainer_id', 'trainer_form30')]:
        k = d[key].astype(str).str.zfill(5)
        daily = (d.assign(_k=k, w=(d['rank_num'] == 1).astype(float), r=d['rank_num'].notna().astype(float))
                 .groupby(['_k', 'race_date'])[['w', 'r']].sum().reset_index().sort_values(['_k', 'race_date']))
        roll = (daily.set_index('race_date').groupby('_k')[['w', 'r']]
                .rolling('30D', closed='left').sum().reset_index())
        roll[name] = (roll['w'] / roll['r']).where(roll['r'] >= 5)
        d = d.assign(_k=k).merge(roll[['_k', 'race_date', name]], on=['_k', 'race_date'], how='left').drop(columns='_k')

    # 前走で人気より何着良かったか（オッズ取得済みのレースのみ）
    d['prev_mkt_resid'] = np.nan
    try:
        # スクレイパーが追記中でも読めるよう、途中までしか書かれていない行は飛ばす
        o = pd.read_csv(FILE_ODDS, dtype={'race_id': str}, on_bad_lines='skip')
        o['horse_number'] = pd.to_numeric(o['horse_number'], errors='coerce')
        o = o[o['horse_number'] > 0]
        # APIは取消・除外馬を オッズ-3.0 / 人気9999 で返す
        o['popularity'] = pd.to_numeric(o['popularity'], errors='coerce').where(lambda s: s < 999)
        d = d.merge(o[['race_id', 'horse_number', 'popularity']].drop_duplicates(['race_id', 'horse_number']),
                    on=['race_id', 'horse_number'], how='left')
        d = d.sort_values(['horse_id', 'race_date', 'race_id'])
        d['prev_mkt_resid'] = d.groupby('horse_id')['popularity'].shift(1) - d.groupby('horse_id')['rank_num'].shift(1)
    except FileNotFoundError:
        pass
    return d[['race_id', 'horse_id'] + CHANGE_FEATURES]


def main():
    print("1. データをロード中...")
    df = load_and_merge_all_data(DATA_DIRS_ALL)
    before = len(df)
    df = df.drop_duplicates(['race_id', 'horse_id'], keep='first')
    print(f"   重複行を {before - len(df)} 行除外")

    df['rank_raw'] = df['rank']
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce').fillna(99).astype(int)
    df['race_date'] = pd.to_datetime(df['race_date'])
    for c in ['prize', 'popularity', 'last_3f', 'diff_time']:
        if c not in df.columns:
            df[c] = 0
    df = df.sort_values(['race_date', 'race_id']).reset_index(drop=True)

    print("2. 当日バイアス特徴量...")
    bias = day_bias(df)
    print("   条件変化・直近成績の特徴量...")
    change = change_features(df)

    print("3. 既存の特徴量エンジニアリング（train_main.py と同じ手順）...")
    weight_mean = float(df['weight'][pd.to_numeric(df['weight'], errors='coerce') > 0].mean())
    burden_mean = float(df['burden'][pd.to_numeric(df['burden'], errors='coerce') > 0].mean())
    df['split_tag'] = 'history'
    df = feature_engineering(df, weight_mean=weight_mean, burden_mean=burden_mean)
    df = df[df['race_date'] >= START]
    for c in BASE_FEATURES:
        if c not in df.columns:
            df[c] = 0
    ids = ['race_id', 'race_date', 'horse_id', 'horse_number', 'rank', 'place', 'type']
    keep = ids + [c for c in BASE_FEATURES if c not in ids]
    out = df[keep].merge(bias, on=['race_id', 'horse_id'], how='left').merge(change, on=['race_id', 'horse_id'],
                                                                              how='left')
    out = out[out['finished'] == True].drop(columns=['finished'])
    out[BASE_FEATURES] = out[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    out.to_pickle(OUT_FILE)
    print(f"保存: {OUT_FILE} ({len(out)}行 / {out['race_id'].nunique()}R)")
    print(out[BIAS_FEATURES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
