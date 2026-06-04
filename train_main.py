# train_main.py
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings
import gc
import random
from tqdm import tqdm

from core.config import (
    DATA_DIRS, MODEL_BASE, MODEL_PATHS, DEVICE_STR, N_FOLDS,
    RANKER_PARAMS, CLASS_PARAMS, FEATURE_COLS,
    CAT_COLS, NUM_COLS, HIST_COLS, DAE_COLS, DAE_LEAKY_COLS, HIST_LEN, DAE_INPUT_DIM,
    TS_N_SPLITS, TS_GAP, META_GAP, OOF_CACHE_FILE, TRANSFORMER_EPOCHS, DAE_EPOCHS
)
from core.models_nn import StrongDAE, RacingTransformer
from core.features import feature_engineering, safe_transform
from core.data_loader import load_and_merge_all_data

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

DEVICE = torch.device(DEVICE_STR)
PIN_MEM = (DEVICE.type == 'cuda')

assert 'rank' not in NUM_COLS + CAT_COLS, "データリーク: rank が特徴量に含まれています"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_hist_array(df_hist_scaled, hist_len=HIST_LEN):
    hist_feats = np.zeros((len(df_hist_scaled), hist_len, len(HIST_COLS)), dtype=np.float32)
    grouped_indices = df_hist_scaled.groupby('horse_id').groups
    scaled_values = df_hist_scaled[HIST_COLS].values

    for horse_id, indices in grouped_indices.items():
        idx_list = list(indices)
        for i, idx in enumerate(idx_list):
            start_i = max(0, i - hist_len)
            past_indices = idx_list[start_i:i]
            if len(past_indices) > 0:
                seq_length = len(past_indices)
                start_idx = hist_len - seq_length
                hist_feats[idx, start_idx:, :] = scaled_values[past_indices]
    return hist_feats


def make_target(x):
    r = int(x)
    return 2 if r <= 3 else (1 if r == 4 else 0)


def _load_gm_weights(model_base):
    try:
        with open(os.path.join(model_base, "best_gm_params.json"), 'r') as f:
            gm = json.load(f)
        return float(gm.get('w_rank', 0.6)), float(gm.get('w_prob', 0.4)), gm.get('meta_params')
    except Exception:
        return 0.6, 0.4, None


def get_lgb_dataset(df, target_col):
    df_sorted = df.sort_values('race_id')
    groups = df_sorted.groupby('race_id', sort=False).size().values
    return lgb.Dataset(df_sorted[FEATURE_COLS].values, df_sorted[target_col], group=groups), df_sorted.index


def main():
    seed_everything(42)
    os.makedirs(MODEL_BASE, exist_ok=True)
    print("🚀 [STRONG FULL V7] 3連複1点勝負アライメント OOFパイプライン 学習開始")

    w_rank_meta, w_prob_meta, loaded_meta_params = _load_gm_weights(MODEL_BASE)

    print("\n1️⃣ データをロード中...")
    df_all = load_and_merge_all_data(DATA_DIRS)
    df_all['rank'] = pd.to_numeric(df_all['rank'], errors='coerce').fillna(99).astype(int)
    df_all['is_win'] = (df_all['rank'] == 1).astype(int)
    df_all['is_place'] = (df_all['rank'] <= 3).astype(int)
    df_all['target_rank'] = df_all['rank'].apply(make_target)
    df_all['race_date'] = pd.to_datetime(df_all['race_date'])

    for c in ['prize', 'popularity', 'last_3f', 'diff_time']:
        if c not in df_all.columns: df_all[c] = 0

    df_all = df_all.sort_values(['race_date', 'race_id']).reset_index(drop=True)
    race_ids = df_all['race_id'].unique()

    print("\n2️⃣ 特徴量エンジニアリング実行中...")
    weight_mean = float(df_all['weight'][pd.to_numeric(df_all['weight'],
                                                       errors='coerce') > 0].mean()) if 'weight' in df_all.columns else 470.0
    burden_mean = float(df_all['burden'][pd.to_numeric(df_all['burden'],
                                                       errors='coerce') > 0].mean()) if 'burden' in df_all.columns else 55.0

    df_all['split_tag'] = 'history'
    df_all = feature_engineering(df_all, weight_mean=weight_mean, burden_mean=burden_mean)
    df_all = df_all.fillna(0).replace([np.inf, -np.inf], 0)

    lgb_rank_oof = np.zeros(len(df_all))
    lgb_prob_oof = np.zeros(len(df_all))
    oof_dl_feats = np.zeros((len(df_all), 33), dtype=np.float32)
    oof_received = np.zeros(len(df_all), dtype=bool)
    rank_iters, clf_iters = [], []

    tscv = TimeSeriesSplit(n_splits=TS_N_SPLITS, gap=TS_GAP)

    print("\n🔄 [同期型シングルループ開始] DL特徴量生成 ＆ LightGBM同時OOF学習")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(race_ids)):
        print(f"\n⚡ --- Fold {fold + 1}/{TS_N_SPLITS} ---")
        df_train_raw = df_all[df_all['race_id'].isin(race_ids[train_idx])].copy()
        df_val_raw = df_all[df_all['race_id'].isin(race_ids[val_idx])].copy()

        encoders, cat_dims = {}, []
        df_train = df_train_raw.copy()
        df_val = df_val_raw.copy()

        for c in CAT_COLS:
            le = LabelEncoder()
            df_train[c] = le.fit_transform(df_train_raw[c].astype(str))
            df_val[c] = safe_transform(dict(zip(le.classes_, le.transform(le.classes_))), df_val_raw[c].astype(str))
            encoders[c] = dict(zip(le.classes_, le.transform(le.classes_)))
            encoders[c]['unknown'] = len(le.classes_)
            cat_dims.append(len(le.classes_) + 1)

        n_scaler = StandardScaler()
        df_train[NUM_COLS] = n_scaler.fit_transform(df_train[NUM_COLS].values)
        df_val[NUM_COLS] = n_scaler.transform(df_val[NUM_COLS].values)

        h_scaler = StandardScaler()
        hist_tr = h_scaler.fit_transform(df_train[HIST_COLS].values)
        hist_va = h_scaler.transform(df_val[HIST_COLS].values)

        df_hist_comb = pd.DataFrame(np.vstack([hist_tr, hist_va]), columns=HIST_COLS)
        df_hist_comb['horse_id'] = np.concatenate([df_train['horse_id'].values, df_val['horse_id'].values])

        X_hist_comb = build_hist_array(df_hist_comb, HIST_LEN)
        X_hist_tr, X_hist_va = X_hist_comb[:len(df_train)], X_hist_comb[len(df_train):]

        trans = RacingTransformer(cat_dims, len(NUM_COLS), len(HIST_COLS), seq_len=HIST_LEN).to(DEVICE)
        opt = torch.optim.AdamW(trans.parameters(), lr=1e-3, weight_decay=1e-4)

        ds_tr = torch.utils.data.TensorDataset(
            torch.tensor(df_train[CAT_COLS].values, dtype=torch.long),
            torch.tensor(df_train[NUM_COLS].values, dtype=torch.float32),
            torch.tensor(X_hist_tr, dtype=torch.float32),
            torch.tensor(df_train['is_place'].values, dtype=torch.float32)
        )

        current_batch_size = min(512, max(32, len(df_train) // 2))
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=current_batch_size, shuffle=True, drop_last=True,
                                            pin_memory=PIN_MEM)

        trans.train()
        for _ in range(TRANSFORMER_EPOCHS):
            for bc, bn, bh, by in dl_tr:
                opt.zero_grad()
                loss = nn.BCEWithLogitsLoss()(trans(bc.to(DEVICE), bn.to(DEVICE), bh.to(DEVICE)).squeeze(-1),
                                              by.to(DEVICE))
                loss.backward()
                opt.step()

        trans.eval()
        for df_target, x_hist_target, is_val in [(df_train, X_hist_tr, False), (df_val, X_hist_va, True)]:
            x_c = torch.tensor(df_target[CAT_COLS].values, dtype=torch.long)
            x_n = torch.tensor(df_target[NUM_COLS].values, dtype=torch.float32)
            x_h = torch.tensor(x_hist_target, dtype=torch.float32)
            probs = []
            with torch.no_grad():
                for i in range(0, len(x_c), 1024):
                    out = trans(x_c[i:i + 1024].to(DEVICE), x_n[i:i + 1024].to(DEVICE),
                                x_h[i:i + 1024].to(DEVICE)).squeeze(-1)
                    probs.append(torch.sigmoid(out).cpu().numpy().reshape(-1))

            p_concat = np.concatenate(probs)
            df_target['transformer_prob'] = np.nan_to_num(p_concat, nan=0.5)
            if is_val:
                oof_dl_feats[df_target.index, 0] = df_target['transformer_prob'].values

        # DAE
        dae_tr_tgt = df_train_raw[DAE_COLS].values.astype(np.float32)
        dae_tr_in = df_train_raw[DAE_COLS].copy()
        dae_tr_in[DAE_LEAKY_COLS] = 0.0
        dae_va_in = df_val_raw[DAE_COLS].copy()
        dae_va_in[DAE_LEAKY_COLS] = 0.0

        i_sc, t_sc = StandardScaler(), StandardScaler()
        tr_in_sc = i_sc.fit_transform(dae_tr_in.values.astype(np.float32))
        tr_tgt_sc = t_sc.fit_transform(dae_tr_tgt)
        va_in_sc = i_sc.transform(dae_va_in.values.astype(np.float32))

        dae = StrongDAE(DAE_INPUT_DIM).to(DEVICE)
        d_opt = torch.optim.AdamW(dae.parameters(), lr=1e-3)
        dae_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(tr_in_sc), torch.tensor(tr_tgt_sc)),
            batch_size=current_batch_size, shuffle=True, pin_memory=PIN_MEM)

        dae.train()
        for _ in range(DAE_EPOCHS):
            for bx, by in dae_dl:
                d_opt.zero_grad()
                loss = nn.MSELoss()(dae(bx.to(DEVICE))[1], by.to(DEVICE))
                loss.backward()
                d_opt.step()

        dae.eval()
        for df_target, sc_matrix, is_val in [(df_train, tr_in_sc, False), (df_val, va_in_sc, True)]:
            dae_feats = []
            with torch.no_grad():
                for i in range(0, len(sc_matrix), 1024):
                    enc, _ = dae(torch.tensor(sc_matrix[i:i + 1024]).to(DEVICE))
                    dae_feats.append(enc.cpu().numpy())
            d_concat = np.concatenate(dae_feats, axis=0)
            for d_idx in range(32):
                df_target[f'dae_{d_idx}'] = d_concat[:, d_idx]
            if is_val:
                oof_dl_feats[df_target.index, 1:] = d_concat

        for c in FEATURE_COLS:
            if c not in df_train.columns: df_train[c] = 0
            if c not in df_val.columns: df_val[c] = 0

        lgb_tr_r, _ = get_lgb_dataset(df_train, 'target_rank')
        lgb_va_r, va_idx_r = get_lgb_dataset(df_val, 'target_rank')

        ranker = lgb.train(RANKER_PARAMS, lgb_tr_r, valid_sets=[lgb_va_r], num_boost_round=1000,
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_rank_oof[va_idx_r] = ranker.predict(df_val.sort_values('race_id')[FEATURE_COLS].values)
        rank_iters.append(ranker.best_iteration)

        lgb_tr_c, _ = get_lgb_dataset(df_train, 'is_place')
        lgb_va_c, va_idx_c = get_lgb_dataset(df_val, 'is_place')

        clf = lgb.train(CLASS_PARAMS, lgb_tr_c, valid_sets=[lgb_va_c], num_boost_round=1000,
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_prob_oof[va_idx_c] = clf.predict(df_val.sort_values('race_id')[FEATURE_COLS].values)
        clf_iters.append(clf.best_iteration)

        oof_received[df_val.index] = True

        del trans, dae, ds_tr, dl_tr, dae_dl
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df_all['transformer_prob'] = oof_dl_feats[:, 0]
    for i in range(32): df_all[f'dae_{i}'] = oof_dl_feats[:, i + 1]
    df_all['lgb_rank_score'] = lgb_rank_oof
    df_all['lgb_prob_score'] = lgb_prob_oof
    df_all['has_oof'] = oof_received
    df_all['__cache_version__'] = "v7_fully_aligned_ sniper_production"
    df_all.to_pickle(OOF_CACHE_FILE)
    print(f"   -> 💾 堅牢なOOFキャッシュを保存しました。")

    df_meta_src = df_all[df_all['has_oof']].copy()

    def glob_minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0.0 + 0.5

    df_meta_src['norm_rank'] = df_meta_src.groupby('race_id')['lgb_rank_score'].transform(glob_minmax)
    w_trans_meta = max(0.0, 1.0 - w_rank_meta - w_prob_meta)

    df_meta_src['transformer_prob'] = df_meta_src['transformer_prob'].fillna(0.5)
    df_meta_src['final_score'] = (
            w_rank_meta * df_meta_src['norm_rank'].fillna(0.5)
            + w_prob_meta * df_meta_src['lgb_prob_score'].fillna(0.5)
            + w_trans_meta * df_meta_src['transformer_prob']
    )

    meta_params = loaded_meta_params or {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03,
        'num_leaves': 7, 'max_depth': 3, 'min_data_in_leaf': 50, 'is_unbalance': True, 'verbose': -1
    }
    meta_params.setdefault('is_unbalance', True)

    meta_cols = ['score_1st', 'score_2nd', 'gap', 'score_std', 'score_mean', 'head_count', 'transformer_prob_1st']
    race_meta_data = []

    for rid, grp in df_meta_src.groupby('race_id'):
        grp = grp.sort_values('final_score', ascending=False)
        if len(grp) < 3: continue

        top1, top2 = grp.iloc[0], grp.iloc[1]

        # 👑 【完全先祖返り】 ターゲットの定義を「上位3頭の完全的中」に戻す
        ai_top3_set = set(grp.iloc[:3]['horse_id'].astype(str).values)
        act_top3_set = set(grp[grp['rank'] <= 3]['horse_id'].astype(str).values)
        is_perfect_cover = 1 if ai_top3_set.issubset(act_top3_set) else 0

        race_meta_data.append({
            'race_date': grp['race_date'].iloc[0],
            'score_1st': top1['final_score'],
            'score_2nd': top2['final_score'],
            'gap': top1['final_score'] - top2['final_score'],
            'score_std': grp['final_score'].std() if len(grp) > 1 else 0.0,
            'score_mean': grp['final_score'].mean(),
            'head_count': len(grp),
            'transformer_prob_1st': top1['transformer_prob'],
            'target': is_perfect_cover
        })

    if race_meta_data:
        df_meta = pd.DataFrame(race_meta_data).sort_values('race_date').reset_index(drop=True)
        meta_iters = []
        for tr_idx, va_idx in TimeSeriesSplit(n_splits=TS_N_SPLITS, gap=META_GAP).split(df_meta):
            _m = lgb.train(meta_params, lgb.Dataset(df_meta.iloc[tr_idx][meta_cols], df_meta.iloc[tr_idx]['target']),
                           valid_sets=[lgb.Dataset(df_meta.iloc[va_idx][meta_cols], df_meta.iloc[va_idx]['target'])],
                           num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
            meta_iters.append(_m.best_iteration)
        final_meta_rounds = int(np.mean(meta_iters) * 1.1)
    else:
        df_meta, final_meta_rounds = pd.DataFrame(), 500

    print("\n🏆 本番用 最終モデル学習 (全データフラットフィッティング)")

    # DAE 最終学習
    dae_all_in = df_all[DAE_COLS].copy()
    dae_all_in[DAE_LEAKY_COLS] = 0.0
    dae_i_sc, dae_t_sc = StandardScaler(), StandardScaler()

    dae_final = StrongDAE(DAE_INPUT_DIM).to(DEVICE)
    dfin_opt = torch.optim.AdamW(dae_final.parameters(), lr=1e-3)

    final_batch_size = min(512, len(df_all) // 2)
    dl_dae_all = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(dae_i_sc.fit_transform(dae_all_in.values.astype(np.float32))),
        torch.tensor(dae_t_sc.fit_transform(df_all[DAE_COLS].values.astype(np.float32)))
    ), batch_size=final_batch_size, shuffle=True, drop_last=True, pin_memory=PIN_MEM)

    dae_final.train()
    for _ in range(DAE_EPOCHS):
        for bx, by in dl_dae_all:
            dfin_opt.zero_grad()
            loss = nn.MSELoss()(dae_final(bx.to(DEVICE))[1], by.to(DEVICE))
            loss.backward()
            dfin_opt.step()

    torch.save(dae_final.state_dict(), MODEL_PATHS['dae_model'])
    joblib.dump(dae_i_sc, MODEL_PATHS['dae_input_scaler'])
    joblib.dump(dae_t_sc, MODEL_PATHS['dae_target_scaler'])

    # Transformer 最終学習
    df_trans_final = df_all.copy()

    enc_f, cat_dims_f = {}, []
    for c in CAT_COLS:
        le = LabelEncoder()
        df_trans_final[c] = le.fit_transform(df_trans_final[c].astype(str))
        enc_f[c] = dict(zip(le.classes_, le.transform(le.classes_)))
        enc_f[c]['unknown'] = len(le.classes_)
        cat_dims_f.append(len(le.classes_) + 1)

    n_sc_f, h_sc_f = StandardScaler(), StandardScaler()
    df_trans_final[NUM_COLS] = n_sc_f.fit_transform(df_trans_final[NUM_COLS].values)

    df_hist_f = pd.DataFrame(h_sc_f.fit_transform(df_trans_final[HIST_COLS].values), columns=HIST_COLS)
    df_hist_f['horse_id'] = df_trans_final['horse_id'].values

    trans_final = RacingTransformer(cat_dims_f, len(NUM_COLS), len(HIST_COLS), seq_len=HIST_LEN).to(DEVICE)
    tfin_opt = torch.optim.AdamW(trans_final.parameters(), lr=1e-3, weight_decay=1e-4)

    dl_tall = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(df_trans_final[CAT_COLS].values, dtype=torch.long),
        torch.tensor(df_trans_final[NUM_COLS].values, dtype=torch.float32),
        torch.tensor(build_hist_array(df_hist_f, HIST_LEN), dtype=torch.float32),
        torch.tensor(df_trans_final['is_place'].values, dtype=torch.float32)
    ), batch_size=final_batch_size, shuffle=True, drop_last=True, pin_memory=PIN_MEM)

    trans_final.train()
    for _ in range(TRANSFORMER_EPOCHS):
        for bc, bn, bh, by in dl_tall:
            tfin_opt.zero_grad()
            loss = nn.BCEWithLogitsLoss()(trans_final(bc.to(DEVICE), bn.to(DEVICE), bh.to(DEVICE)).squeeze(-1),
                                          by.to(DEVICE))
            loss.backward()
            tfin_opt.step()

    torch.save(trans_final.state_dict(), MODEL_PATHS['transformer'])
    with open(MODEL_PATHS['transformer_proc'], 'wb') as f:
        pickle.dump({'encoders': enc_f, 'cat_dims': cat_dims_f, 'num_scaler': n_sc_f, 'hist_scaler': h_sc_f,
                     'num_dim': len(NUM_COLS), 'hist_dim': len(HIST_COLS), 'hist_len': HIST_LEN,
                     'weight_mean': weight_mean, 'burden_mean': burden_mean}, f)

    # LGBM 最終学習
    for c in FEATURE_COLS:
        if c not in df_all.columns: df_all[c] = 0

    lgb_all, _ = get_lgb_dataset(df_all, 'target_rank')
    lgb.train(RANKER_PARAMS, lgb_all,
              num_boost_round=int(np.mean(rank_iters) * 1.1) if rank_iters else 1000).save_model(
        MODEL_PATHS['lgb_ranker'])

    lgb_all_c, _ = get_lgb_dataset(df_all, 'is_place')
    lgb.train(CLASS_PARAMS, lgb_all_c, num_boost_round=int(np.mean(clf_iters) * 1.1) if clf_iters else 1000).save_model(
        MODEL_PATHS['lgb_prob'])

    # 監督AI 最終学習
    if not df_meta.empty:
        lgb.train(meta_params, lgb.Dataset(df_meta[meta_cols], df_meta['target']),
                  num_boost_round=final_meta_rounds).save_model(MODEL_PATHS['meta_model'])

    with open(MODEL_PATHS['grandmaster'], 'wb') as f:
        pickle.dump(
            {'w_rank': w_rank_meta, 'w_prob': w_prob_meta, 'weight_mean': weight_mean, 'burden_mean': burden_mean}, f)

    print("\n🎉 [1点勝負仕様] 全ての学習と本番用最終モデルの保存が完了しました！")


if __name__ == "__main__":
    main()