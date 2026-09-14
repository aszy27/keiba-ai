# evaluate_tickets.py
"""
券種ごとの回収率を、監督AIの自信度の閾値別に測定する。

背景: 「3連複1点」という券種は、払戻データの列ずれで過大評価されたバックテスト（回収率140%）を根拠に
      選ばれていた。列ずれを修正すると3連複の回収率は控除率並みだったため、券種の選び直しが必要になった。
      このスクリプトは evaluate_main.py と同じ推論を行い、AIの上位馬から組み立てた各券種の
      的中・払戻を集計する。

買い目（すべて1レースあたりの購入点数で均等・100円）:
    単勝1点    : 上位1頭
    複勝1点    : 上位1頭
    馬連1点    : 上位2頭
    馬単1点    : 上位2頭（1着→2着の順）
    枠連1点    : 上位2頭の枠
    ワイド1点  : 上位1・2頭
    ワイド3点  : 上位3頭の総当たり（3点）
    3連複1点   : 上位3頭
    3連単1点   : 上位3頭（着順どおり）

出力: ターミナルの表 と result/ticket_roi.csv

ワイドの払戻3つの並び順は、実データで検証済み:
      「1着-2着 / 1着-3着 / 2着-3着」の着順の組で並んでいる。
      （9,088レースで「1着-2着のワイド < 馬連」が100.0%成立。馬番の昇順とみなすと81.1%しか成立しない。
        3つの払戻の中央値も 610 / 770 / 970 円と、下位の着との組ほど高くなる並びになっている）
      対応付けに依存しない下限・上限も併せて出力する。
"""
import os
import re
import random
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.config import (
    MODEL_PATHS, FEATURE_COLS, DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST, BASE_DIR, DEVICE_STR
)
from core.features import feature_engineering
from core.inference import load_all_models, generate_dl_features

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

FILE_RETURN = os.path.join(BASE_DIR, "data", "return_data_progress.csv")
OUT_CSV = os.path.join(BASE_DIR, "result", "ticket_roi.csv")
THRESHOLDS = [0.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

# 券種ごとの購入点数
TICKETS = {
    '単勝': 1, '複勝': 1, '馬連': 1, '馬単': 1, '枠連': 1,
    'ワイド1点': 1, 'ワイド3点': 3, '3連複': 1, '3連単': 1,
}

try:
    DEVICE = torch.device(DEVICE_STR)
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
except Exception:
    DEVICE = torch.device("cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _nums(text):
    """'1,230円' → [1230] / '150|240|100' → [150, 240, 100]"""
    if pd.isna(text):
        return []
    out = []
    for part in str(text).split('|'):
        digits = re.sub(r'[^0-9]', '', part)
        if digits:
            out.append(int(digits))
    return out


def prep_return_data(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, dtype={'race_id': str})
    df = df.drop_duplicates(subset=['race_id'], keep='last')
    for col in ['tansho', 'fukusho', 'umaren', 'umatan', 'wide', 'wakuren', 'sanrenpuku', 'sanrentan']:
        if col in df.columns:
            df[col] = df[col].apply(_nums)
        else:
            df[col] = [[] for _ in range(len(df))]
    return df.set_index('race_id')


def build_predictions():
    """evaluate_main.py と同じ手順で、レースごとの自信度と上位馬・着順を作る"""
    seed_everything(42)
    print("1️⃣ モデルをロード中...")
    models = load_all_models()
    gm = models['gm_params']

    print("2️⃣ データをロード中...")
    from core.data_loader import load_and_merge_all_data
    try:
        df_history = load_and_merge_all_data([DATA_DIR_TRAIN, DATA_DIR_VAL])
    except FileNotFoundError:
        df_history = pd.DataFrame()
    df_test = load_and_merge_all_data([DATA_DIR_TEST])

    df_test['race_date'] = pd.to_datetime(df_test['race_date'])
    for c in ['prize', 'popularity', 'grade', 'rank', 'last_3f', 'diff_time']:
        df_test[c] = df_test.get(c, 0 if c != 'grade' else 'OP')
    test_ids = set(df_test['race_id'].astype(str))

    if not df_history.empty:
        df_history['race_date'] = pd.to_datetime(df_history['race_date'])
        for c in ['prize', 'popularity', 'grade', 'rank', 'last_3f', 'diff_time']:
            df_history[c] = df_history.get(c, 0 if c != 'grade' else 'OP')
        df_all = pd.concat([df_history, df_test], ignore_index=True)
    else:
        df_all = df_test.copy()

    print("3️⃣ 特徴量エンジニアリング...")
    _rank = pd.to_numeric(df_all['rank'], errors='coerce')
    _is_test = df_all['race_id'].astype(str).isin(test_ids)
    df_all['split_tag'] = np.where(_is_test & _rank.notna(), 'target', 'history')
    df_all = feature_engineering(df_all, weight_mean=gm.get('weight_mean', 470.0),
                                 burden_mean=gm.get('burden_mean', 55.0))
    df_all = df_all.sort_values(['race_date', 'race_id']).reset_index(drop=True)
    df_all['rank'] = pd.to_numeric(df_all['rank'], errors='coerce').fillna(99)
    df_all = df_all.fillna(0).replace([np.inf, -np.inf], 0)

    print("4️⃣ 未来情報の隠蔽 ＆ DL推論...")
    df_hist_dl = df_all.copy()
    df_tgt = df_all[df_all['split_tag'] == 'target'].copy()
    true_labels = df_tgt[['race_id', 'horse_id', 'rank']].copy()
    for c in ['prize', 'last_3f', 'diff_time', 'popularity', 'pace_score', 'first_3f', 'last_3f_race']:
        if c in df_tgt.columns:
            df_tgt[c] = 0.0
    df_tgt['rank'] = np.nan
    df_tgt = generate_dl_features(df_tgt, models, df_history=df_hist_dl, show_progress=True)
    df_tgt = df_tgt.drop(columns=['rank']).merge(true_labels, on=['race_id', 'horse_id'], how='left')

    print("5️⃣ LightGBM 推論...")
    for feat in FEATURE_COLS:
        if feat not in df_tgt.columns:
            df_tgt[feat] = 0
    X = df_tgt[FEATURE_COLS].values
    df_tgt['lgb_rank_score'] = models['ranker'].predict(X)
    df_tgt['lgb_prob_score'] = models['clf'].predict(X)
    df_tgt['norm_rank_score'] = df_tgt.groupby('race_id')['lgb_rank_score'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5)

    w_rank, w_prob = gm.get('w_rank', 0.6), gm.get('w_prob', 0.4)
    w_trans = max(0.0, 1.0 - w_rank - w_prob)
    df_tgt['transformer_prob'] = df_tgt['transformer_prob'].fillna(0.5)
    df_tgt['final_score'] = (w_rank * df_tgt['norm_rank_score'].fillna(0.5)
                             + w_prob * df_tgt['lgb_prob_score'].fillna(0.5)
                             + w_trans * df_tgt['transformer_prob'])

    print("6️⃣ 監督AI (Meta-Model) 予測...")
    meta_cols = ['score_1st', 'score_2nd', 'score_3rd', 'gap_1_2', 'gap_3_4', 'score_std', 'score_mean',
                 'head_count', 'transformer_prob_1st']
    df_tgt['rank'] = pd.to_numeric(df_tgt['rank'], errors='coerce').fillna(99).astype(int)

    rows, rids = [], []
    for rid, grp in tqdm(df_tgt.groupby('race_id'), desc="Meta Prep"):
        grp = grp.sort_values('final_score', ascending=False)
        if len(grp) < 3:
            continue
        t1, t2, t3 = grp.iloc[0]['final_score'], grp.iloc[1]['final_score'], grp.iloc[2]['final_score']
        t4 = grp.iloc[3]['final_score'] if len(grp) > 3 else 0.0
        rows.append({'score_1st': t1, 'score_2nd': t2, 'score_3rd': t3, 'gap_1_2': t1 - t2, 'gap_3_4': t3 - t4,
                     'score_std': grp['final_score'].std() if len(grp) > 1 else 0.0,
                     'score_mean': grp['final_score'].mean(), 'head_count': len(grp),
                     'transformer_prob_1st': grp.iloc[0]['transformer_prob']})
        rids.append(rid)
    df_meta = pd.DataFrame(rows, columns=meta_cols)
    conf = models['meta_model'].predict(df_meta) * 100 if (models.get('meta_model') and not df_meta.empty) \
        else np.full(len(df_meta), -1.0)
    return df_tgt, dict(zip(rids, conf))


def evaluate_race(grp, conf, ret_row):
    """1レース分の券種ごとの投資・払戻を返す"""
    grp = grp.sort_values('final_score', ascending=False)
    ai = grp.iloc[:3]
    ai_nums = [int(x) for x in ai['horse_number'].values]
    ai_waku = [int(x) for x in ai['bracket'].values]

    placed = grp[(grp['rank'] >= 1) & (grp['rank'] <= 3)].sort_values('rank')
    if len(placed) < 3:
        return None
    act_nums = [int(x) for x in placed['horse_number'].values[:3]]
    act_waku = [int(x) for x in placed['bracket'].values[:3]]

    res = {'confidence': conf, 'race_date': grp['race_date'].iloc[0]}
    pay = lambda key, i=0: (ret_row[key][i] if (isinstance(ret_row.get(key), list) and len(ret_row[key]) > i) else 0)

    # 単勝・複勝（AI本命）
    res['単勝'] = pay('tansho') if ai_nums[0] == act_nums[0] else 0
    if ai_nums[0] in act_nums:
        res['複勝'] = pay('fukusho', act_nums.index(ai_nums[0]))
    else:
        res['複勝'] = 0

    # 馬連・馬単・枠連（AI上位2頭）
    res['馬連'] = pay('umaren') if set(ai_nums[:2]) == set(act_nums[:2]) else 0
    res['馬単'] = pay('umatan') if ai_nums[:2] == act_nums[:2] else 0
    res['枠連'] = pay('wakuren') if sorted(ai_waku[:2]) == sorted(act_waku[:2]) else 0

    # ワイド: 払戻3つは「1着-2着 / 1着-3着 / 2着-3着」の順（検証済み）
    wide_pairs = [tuple(sorted((act_nums[0], act_nums[1]))),
                  tuple(sorted((act_nums[0], act_nums[2]))),
                  tuple(sorted((act_nums[1], act_nums[2])))]
    wide_list = ret_row.get('wide') if isinstance(ret_row.get('wide'), list) else []

    def wide_pay(pair):
        key = tuple(sorted(pair))
        if key not in wide_pairs:
            return 0, 0, 0  # 想定・下限・上限
        if len(wide_list) < 3:
            return 0, 0, 0
        exact = wide_list[wide_pairs.index(key)]
        return exact, min(wide_list), max(wide_list)

    w1, w1_lo, w1_hi = wide_pay(ai_nums[:2])
    res['ワイド1点'], res['ワイド1点_lo'], res['ワイド1点_hi'] = w1, w1_lo, w1_hi

    w3 = [wide_pay(p) for p in [(ai_nums[0], ai_nums[1]), (ai_nums[0], ai_nums[2]), (ai_nums[1], ai_nums[2])]]
    res['ワイド3点'] = sum(x[0] for x in w3)
    res['ワイド3点_lo'] = sum(x[1] for x in w3)
    res['ワイド3点_hi'] = sum(x[2] for x in w3)

    # 3連複・3連単
    res['3連複'] = pay('sanrenpuku') if set(ai_nums) == set(act_nums) else 0
    res['3連単'] = pay('sanrentan') if ai_nums == act_nums else 0
    return res


def main():
    df_tgt, conf_dict = build_predictions()

    print("\n7️⃣ 券種ごとの集計...")
    df_ret = prep_return_data(FILE_RETURN)
    if df_ret is None:
        print("❌ 払戻データが見つかりません。")
        return

    records = []
    for rid, grp in tqdm(df_tgt.groupby('race_id'), desc="Result Check"):
        if rid not in conf_dict or rid not in df_ret.index:
            continue
        row = evaluate_race(grp, conf_dict[rid], df_ret.loc[rid].to_dict())
        if row:
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        print("❌ 集計できるレースがありませんでした。")
        return

    print(f"\n対象: {len(df)}R（{df['race_date'].min().date()} 〜 {df['race_date'].max().date()}）")
    print("=" * 104)
    print(f"{'自信度':<8} | {'R数':>5} | " + " | ".join(f"{k:>9}" for k in TICKETS))
    print("-" * 104)

    out = []
    for th in THRESHOLDS:
        tgt = df[df['confidence'] >= th]
        if tgt.empty:
            continue
        cells, rec = [], {'threshold': th, 'races': len(tgt)}
        for name, points in TICKETS.items():
            roi = tgt[name].sum() / (len(tgt) * 100 * points) * 100
            hit = (tgt[name] > 0).mean() * 100
            cells.append(f"{roi:>8.1f}%")
            rec[f'{name}_回収率'] = round(roi, 1)
            rec[f'{name}_的中率'] = round(hit, 1)
        out.append(rec)
        print(f"{th:>5.0f}%以上 | {len(tgt):>5} | " + " | ".join(cells))

    print("=" * 104)
    print("※ 各券種とも1レースあたり均等購入（ワイド3点は3点＝300円を分母とした回収率）")

    # ワイドの対応付けが結論を変えないかの確認（想定＝着順の組での対応付け）
    print("\n🩺 ワイドの払戻の対応付けによる幅（自信度60%以上）")
    tgt = df[df['confidence'] >= 60.0]
    if not tgt.empty:
        for name in ['ワイド1点', 'ワイド3点']:
            pts = TICKETS[name]
            base = tgt[name].sum() / (len(tgt) * 100 * pts) * 100
            lo = tgt[f'{name}_lo'].sum() / (len(tgt) * 100 * pts) * 100
            hi = tgt[f'{name}_hi'].sum() / (len(tgt) * 100 * pts) * 100
            print(f"   {name}: 想定 {base:.1f}%  （下限 {lo:.1f}% 〜 上限 {hi:.1f}%）")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(out).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 保存: {OUT_CSV}")
    print("\n💡 見方: 控除率は単勝・複勝20%、馬連・馬単・ワイド・枠連 22.5%、3連複・3連単 25%。")
    print("   回収率がその水準（80% / 77.5% / 75%）前後なら、市場と同じ評価をしているだけで優位性はない。")
    print("   どの券種も100%に届かない場合、着順予測の精度ではなくオッズを使った期待値の判断が必要になる。")


if __name__ == "__main__":
    main()
