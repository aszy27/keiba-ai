# optimize_main.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
import os
import warnings
import json
import gc

from core.config import (
    MODEL_BASE, PARAM_FILE_RANKER, DATA_DIRS, DEVICE_STR,
    FEATURE_COLS, CAT_COLS, NUM_COLS, HIST_COLS, DAE_COLS, DAE_LEAKY_COLS,
    HIST_LEN, DAE_INPUT_DIM, N_FOLDS, TS_N_SPLITS, TS_GAP, META_GAP,
    OOF_CACHE_FILE, DAE_EPOCHS, TRANSFORMER_EPOCHS
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

DEVICE = torch.device(DEVICE_STR) if DEVICE_STR else torch.device("cpu")

N_TRIALS_PHASE1 = 50
N_TRIALS_PHASE2 = 50
N_TRIALS_PHASE3 = 50
N_TRIALS_PHASE4 = 50

OPT_VAL_RATIO_PHASE1 = 0.20
OPT_VAL_RATIO_PHASE3 = 0.15
OPTUNA_DB = f"sqlite:///{MODEL_BASE}/optuna_study.db"


def make_target(rank):
    r = int(rank) if pd.notnull(rank) else 99
    if r == 1: return 4
    if r == 2: return 3
    if r == 3: return 2
    if r == 4: return 1
    return 0


# ==========================================
# 🚀 爆速・3頭カバー率算出エンジン (NumPyネイティブ)
# ==========================================
def fast_3top_cover_rate(preds, race_ids, ranks):
    """👑 【完全先祖返り】上位3頭がすべて3着以内に入る確率（3連複1点的中率）を算出"""
    unique_races, inverse_indices = np.unique(race_ids, return_inverse=True)
    n_races = len(unique_races)

    hits, total = 0, 0
    for r_idx in range(n_races):
        mask = (inverse_indices == r_idx)
        if mask.sum() < 3: continue

        total += 1
        r_preds = preds[mask]
        r_ranks = ranks[mask]

        top3_idx = np.argsort(-r_preds)[:3]
        if np.all(r_ranks[top3_idx] <= 3):
            hits += 1

    return hits / total if total > 0 else 0


def main():
    print("🚀 【全自動モジュール版】 フル特徴量 1点勝負最適化 (V7 Fast Sniper)")
    os.makedirs(MODEL_BASE, exist_ok=True)

    if not os.path.exists(OOF_CACHE_FILE):
        raise FileNotFoundError(
            f"❌ クリーンなOOFキャッシュが見つかりません: {OOF_CACHE_FILE}\n"
            f"先に 'python train_main.py' を実行して正常な特徴量ベースを生成してください。"
        )

    print(f"\n1️⃣ ♻️ 同期されたOOF特徴量キャッシュをロード中... ({OOF_CACHE_FILE})")
    df_all = pd.read_pickle(OOF_CACHE_FILE)

    # 🔴 FIX: 特徴量定義が変わった旧キャッシュ（V7以前）で最適化すると、本番と異なる
    #          特徴量分布に対してパラメータが最適化されてしまう。バージョンを検証する。
    EXPECTED_CACHE_VERSION = "v8_date_exclusive_aligned"
    cache_ver = str(df_all['__cache_version__'].iloc[0]) if '__cache_version__' in df_all.columns else "unknown"
    if cache_ver != EXPECTED_CACHE_VERSION:
        raise ValueError(
            f"❌ OOFキャッシュのバージョンが古いです (found: {cache_ver} / expected: {EXPECTED_CACHE_VERSION})。\n"
            f"   先に 'python train_main.py' を再実行してキャッシュを再生成してください。"
        )

    df_opt = df_all[df_all['has_oof']].copy().sort_values(['race_date', 'race_id']).reset_index(drop=True)
    print(f"\n2️⃣ 最適化データセットの構築 (レコード数: {len(df_opt)})")

    opt_unique_races = df_opt['race_id'].unique()
    n_races = len(opt_unique_races)

    p3_start = int(n_races * (1 - OPT_VAL_RATIO_PHASE1 - OPT_VAL_RATIO_PHASE3))
    p1_start = p3_start + int(n_races * OPT_VAL_RATIO_PHASE1)

    phase_train_races = opt_unique_races[:p3_start]
    phase1_val_races = opt_unique_races[p3_start:p1_start]
    phase3_val_races = opt_unique_races[p1_start:]

    def get_lgb_data(train_races, val_races):
        df_tr = df_opt[df_opt['race_id'].isin(train_races)].sort_values('race_id')
        df_vl = df_opt[df_opt['race_id'].isin(val_races)].sort_values('race_id')

        X_tr, y_tr = df_tr[FEATURE_COLS].values, df_tr['rank'].apply(make_target).values
        X_vl, y_vl = df_vl[FEATURE_COLS].values, df_vl['rank'].apply(make_target).values

        q_tr = df_tr.groupby('race_id', sort=False).size().values
        q_vl = df_vl.groupby('race_id', sort=False).size().values

        return X_tr, X_vl, q_tr, q_vl, y_tr, y_vl, df_vl['race_id'].values, df_vl['rank'].fillna(99).values, df_vl[
            'race_date'].values

    X_p1_tr, X_p1_vl, q_p1_tr, q_p1_vl, y_p1_tr, y_p1_vl, rids_p1_vl, ranks_p1_vl, _ = get_lgb_data(phase_train_races,
                                                                                                    phase1_val_races)

    phase3_train_races = opt_unique_races[:p1_start]
    X_p3_tr, X_p3_vl, q_p3_tr, q_p3_vl, y_p3_tr, y_p3_vl, rids_p3_vl, ranks_p3_vl, dates_p3_vl = get_lgb_data(
        phase3_train_races, phase3_val_races)

    # =========================================================================
    # Phase 1: 報酬（Label Gain）の最適化
    # =========================================================================
    print(f"\n🔹 Phase 1: 1点勝負用・報酬配点の最適探索... (Trials: {N_TRIALS_PHASE1})")

    def objective_reward(trial):
        s4 = trial.suggest_int('score_4th', 1, 30)
        s3 = trial.suggest_int('score_3rd', 30, 80)
        s2 = trial.suggest_int('score_2nd', 80, 150)
        s1 = trial.suggest_int('score_1st', 150, 300)

        params = {
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [3], 'boosting_type': 'gbdt',
            'learning_rate': 0.1, 'num_leaves': 31, 'label_gain': [0, s4, s3, s2, s1], 'verbose': -1, 'random_state': 42
        }
        model = lgb.train(params, lgb.Dataset(X_p1_tr, y_p1_tr, group=q_p1_tr),
                          valid_sets=[lgb.Dataset(X_p1_vl, y_p1_vl, group=q_p1_vl)], num_boost_round=300,
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        return fast_3top_cover_rate(model.predict(X_p1_vl), rids_p1_vl, ranks_p1_vl)

    # 過去の3要素のDBと衝突しないよう study_name を _v2 に変更
    study_r = optuna.create_study(direction='maximize', study_name='phase1_reward_v7_3top_v2', storage=OPTUNA_DB,
                                  load_if_exists=True)
    study_r.optimize(objective_reward, n_trials=N_TRIALS_PHASE1)
    best_gain = [0, study_r.best_params['score_4th'], study_r.best_params['score_3rd'], study_r.best_params['score_2nd'], study_r.best_params['score_1st']]
    print(f"   🏆 確定した最適ゲイン: {best_gain} (Best Cover Rate: {study_r.best_value * 100:.2f}%)")

    # =========================================================================
    # Phase 2: ハイパーパラメータの最適化
    # =========================================================================
    print(f"\n🔹 Phase 2: ランカー（LightGBM）パラメータの探索... (Trials: {N_TRIALS_PHASE2})")

    def objective_param(trial):
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [3], 'boosting_type': 'gbdt',
            'label_gain': best_gain, 'verbose': -1, 'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        model = lgb.train(params, lgb.Dataset(X_p1_tr, y_p1_tr, group=q_p1_tr),
                          valid_sets=[lgb.Dataset(X_p1_vl, y_p1_vl, group=q_p1_vl)], num_boost_round=1000,
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        return fast_3top_cover_rate(model.predict(X_p1_vl), rids_p1_vl, ranks_p1_vl)

    study_p = optuna.create_study(direction='maximize', study_name='phase2_param_v7_3top', storage=OPTUNA_DB,
                                  load_if_exists=True)
    study_p.optimize(objective_param, n_trials=N_TRIALS_PHASE2)
    final_params = dict(study_p.best_params)
    final_params.update({
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [3], 'boosting_type': 'gbdt',
        'verbose': -1, 'random_state': 42, 'label_gain': best_gain
    })

    # =========================================================================
    # Phase 3: w_rank / w_prob / w_trans の最適化
    # =========================================================================
    print(f"\n🔹 Phase 3: 3段ブレンド（合成比率）の探索... (Trials: {N_TRIALS_PHASE3})")
    model_p3 = lgb.train(final_params, lgb.Dataset(X_p3_tr, y_p3_tr, group=q_p3_tr),
                         valid_sets=[lgb.Dataset(X_p3_vl, y_p3_vl, group=q_p3_vl)], num_boost_round=1000,
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    # 🔴 FIX: make_target が5値(0-4)になったため、3着以内(top3)の陽性ラベルは
    #          旧: == 2 (3値時代は2=top3) → 新: >= 2 (5値では2=3着,3=2着,4=1着)
    model_clf_p3 = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'is_unbalance': True,
         'verbose': -1},
        lgb.Dataset(X_p3_tr, (y_p3_tr >= 2).astype(int)),
        valid_sets=[lgb.Dataset(X_p3_vl, (y_p3_vl >= 2).astype(int))], num_boost_round=500,
        callbacks=[lgb.early_stopping(30, verbose=False)])

    _rank_scores = model_p3.predict(X_p3_vl)
    _prob_scores = model_clf_p3.predict(X_p3_vl)

    df_norm = pd.DataFrame({'race_id': rids_p3_vl, 'score': _rank_scores})
    _norm_rank_scores = df_norm.groupby('race_id', sort=False)['score'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5).values

    _trans_scores = df_opt[df_opt['race_id'].isin(phase3_val_races)].sort_values('race_id')['transformer_prob'].values

    def objective_weight(trial):
        w_rank = trial.suggest_float('w_rank', 0.1, 0.8)
        w_prob = trial.suggest_float('w_prob', 0.1, 1.0 - w_rank)
        w_trans = max(0.0, 1.0 - w_rank - w_prob)

        final_preds = w_rank * _norm_rank_scores + w_prob * _prob_scores + w_trans * _trans_scores
        return fast_3top_cover_rate(final_preds, rids_p3_vl, ranks_p3_vl)

    study_w = optuna.create_study(direction='maximize', study_name='phase3_weight_v7_3top', storage=OPTUNA_DB,
                                  load_if_exists=True)
    study_w.optimize(objective_weight, n_trials=N_TRIALS_PHASE3)
    best_w_rank = study_w.best_params['w_rank']
    best_w_prob = study_w.best_params['w_prob']
    best_w_trans = max(0.0, 1.0 - best_w_rank - best_w_prob)
    print(f"   🏆 確定した比率: w_rank={best_w_rank:.3f}, w_prob={best_w_prob:.3f}, w_trans={best_w_trans:.3f}")

    # =========================================================================
    # Phase 4: メタモデル（監督AI）のハイパーパラメータ最適化
    # =========================================================================
    print(f"\n🔹 Phase 4: 監督AI（メタモデル）ハイパーパラメータの探索... (Trials: {N_TRIALS_PHASE4})")

    df_meta_p4 = pd.DataFrame({
        'race_id': rids_p3_vl, 'race_date': dates_p3_vl, 'rank': ranks_p3_vl,
        'transformer_prob': _trans_scores,
        'final_score': best_w_rank * _norm_rank_scores + best_w_prob * _prob_scores + best_w_trans * _trans_scores
    })

    meta_rows = []
    for rid, grp in df_meta_p4.groupby('race_id', sort=False):
        if len(grp) < 3: continue
        grp = grp.sort_values('final_score', ascending=False)
        top1, top2, top3 = grp.iloc[0]['final_score'], grp.iloc[1]['final_score'], grp.iloc[2]['final_score']
        top4 = grp.iloc[3]['final_score'] if len(grp) > 3 else 0.0

        # 👑 【完全先祖返り】 ターゲットの条件を「上位3頭の完全的中」に戻す
        target_val = 1 if np.all(grp.iloc[:3]['rank'].values <= 3) else 0

        meta_rows.append({
            'race_date': grp.iloc[0]['race_date'],
            'score_1st': top1,
            'score_2nd': top2,
            'score_3rd': top3,
            'gap_1_2': top1 - top2,
            'gap_3_4': top3 - top4,
            'score_std': grp['final_score'].std() if len(grp) > 1 else 0.0,
            'score_mean': grp['final_score'].mean(),
            'head_count': len(grp),
            'transformer_prob_1st': grp.iloc[0]['transformer_prob'],
            'target': target_val
        })

    df_meta_p4_agg = pd.DataFrame(meta_rows).sort_values('race_date').reset_index(drop=True)

    tscv_meta = TimeSeriesSplit(n_splits=TS_N_SPLITS, gap=META_GAP)
    meta_cols = ['score_1st', 'score_2nd', 'score_3rd', 'gap_1_2', 'gap_3_4', 'score_std', 'score_mean', 'head_count',
                 'transformer_prob_1st']

    def objective_meta(trial):
        meta_p = {
            'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'random_state': 42, 'is_unbalance': True,
            'learning_rate': trial.suggest_float('meta_lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('meta_leaves', 4, 31),
            'max_depth': trial.suggest_int('meta_depth', 2, 6),
            'min_data_in_leaf': trial.suggest_int('meta_min', 10, 100),
        }
        aucs = []
        for tr_idx, va_idx in tscv_meta.split(df_meta_p4_agg):
            tr_m, va_m = df_meta_p4_agg.iloc[tr_idx], df_meta_p4_agg.iloc[va_idx]
            m = lgb.train(meta_p, lgb.Dataset(tr_m[meta_cols], tr_m['target']),
                          valid_sets=[lgb.Dataset(va_m[meta_cols], va_m['target'])],
                          num_boost_round=500, callbacks=[lgb.early_stopping(30, verbose=False)])
            aucs.append(m.best_score['valid_0']['auc'])
        return np.mean(aucs)

    study_m = optuna.create_study(direction='maximize', study_name='phase4_meta_v7_3top', storage=OPTUNA_DB,
                                  load_if_exists=True)
    study_m.optimize(objective_meta, n_trials=N_TRIALS_PHASE4)
    best_meta_params = {
        'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'random_state': 42, 'is_unbalance': True,
        'learning_rate': study_m.best_params['meta_lr'], 'num_leaves': study_m.best_params['meta_leaves'],
        'max_depth': study_m.best_params['meta_depth'], 'min_data_in_leaf': study_m.best_params['meta_min']
    }
    print(f"   🏆 最適メタパラメータ: {best_meta_params} (Best Mean AUC: {study_m.best_value:.4f})")

    # =========================================================================
    # 結果保存
    # =========================================================================
    print(f"\n💾 最強設定を保存中: {PARAM_FILE_RANKER}")
    with open(PARAM_FILE_RANKER, 'w') as f:
        json.dump(final_params, f, indent=4)
    with open(os.path.join(MODEL_BASE, "best_gm_params.json"), 'w') as f:
        json.dump({'w_rank': best_w_rank, 'w_prob': best_w_prob, 'meta_params': best_meta_params}, f, indent=4)
    print("✅ 全自動最適化完了！ 1点勝負に特化した最強パラメータが保存されました。")


if __name__ == "__main__":
    main()