# core/inference.py
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
import lightgbm as lgb
import os
import gc
from collections import defaultdict
from tqdm import tqdm

from core.config import (
    MODEL_PATHS, CAT_COLS, NUM_COLS, HIST_COLS, HIST_LEN,
    DAE_COLS, DAE_LEAKY_COLS, DAE_INPUT_DIM, DEVICE_STR
)

try:
    DEVICE = torch.device(DEVICE_STR)
except Exception:
    DEVICE = torch.device("cpu")

try:
    from core.models_nn import StrongDAE, RacingTransformer
except ImportError:
    pass


def _local_safe_transform(mapping, labels):
    """推論専用の安全なラベル変換"""
    unknown_idx = mapping.get('unknown', len(mapping))
    return [mapping.get(str(x), unknown_idx) for x in labels]


def load_all_models(device=DEVICE):
    """全ての学習済みモデルを一括ロードする"""
    REQUIRED_FILES = [
        'transformer_proc', 'transformer',
        'dae_model', 'dae_input_scaler', 'dae_target_scaler',
        'lgb_ranker', 'lgb_prob', 'grandmaster'
    ]
    missing = [k for k in REQUIRED_FILES if not os.path.exists(MODEL_PATHS.get(k, ''))]
    if missing:
        raise FileNotFoundError(f"❌ 必須モデルが不足しています: {missing}")

    with open(MODEL_PATHS['transformer_proc'], 'rb') as f:
        proc = pickle.load(f)

    if proc.get('hist_dim') != len(HIST_COLS):
        raise ValueError("❌ Transformer の hist_dim が config と不一致です。")

    trans = RacingTransformer(proc['cat_dims'], proc['num_dim'], proc['hist_dim'], seq_len=HIST_LEN).to(device)
    trans.load_state_dict(torch.load(MODEL_PATHS['transformer'], map_location=device, weights_only=True))
    trans.eval()

    dae_scaler = joblib.load(MODEL_PATHS['dae_input_scaler'])
    if dae_scaler.n_features_in_ != DAE_INPUT_DIM:
        raise ValueError("❌ DAE次元不一致。再学習が必要です。")

    dae = StrongDAE(DAE_INPUT_DIM).to(device)
    dae.load_state_dict(torch.load(MODEL_PATHS['dae_model'], map_location=device, weights_only=True))
    dae.eval()

    models = {
        'proc': proc, 'trans': trans, 'dae_scaler': dae_scaler, 'dae': dae,
        'ranker': lgb.Booster(model_file=MODEL_PATHS['lgb_ranker']),
        'clf': lgb.Booster(model_file=MODEL_PATHS['lgb_prob'])
    }

    if os.path.exists(MODEL_PATHS.get('meta_model', '')):
        models['meta_model'] = lgb.Booster(model_file=MODEL_PATHS['meta_model'])
    if os.path.exists(MODEL_PATHS.get('combo_model', '')):
        models['combo_model'] = lgb.Booster(model_file=MODEL_PATHS['combo_model'])

    with open(MODEL_PATHS['grandmaster'], 'rb') as f:
        models['gm_params'] = pickle.load(f)

    return models


def build_hist_tensor(df, hist_scaler, hist_len=HIST_LEN, df_history=None):
    """推論対象の過去成績テンソルを構築する (NumPyベースの超高速版)"""
    N, hist_dim = len(df), len(HIST_COLS)
    hist_tensor = np.zeros((N, hist_len, hist_dim), dtype=np.float32)

    if df_history is None or df_history.empty:
        return torch.tensor(hist_tensor)

    target_horses = df['horse_id'].unique()
    df_h = df_history[df_history['horse_id'].isin(target_horses)].copy()

    if df_h.empty:
        return torch.tensor(hist_tensor)

    df_h['race_date'] = pd.to_datetime(df_h['race_date'])
    df_h = df_h.sort_values(['horse_id', 'race_date', 'race_id'])

    # 🔥 FIX: 過去データスケーリング直後の inf / nan を完全に駆逐
    raw_hist_values = np.nan_to_num(df_h[HIST_COLS].fillna(0).values.astype(np.float32))
    scaled_hist = hist_scaler.transform(raw_hist_values)
    df_h[HIST_COLS] = np.nan_to_num(scaled_hist, nan=0.0, posinf=0.0, neginf=0.0)

    h_horses = df_h['horse_id'].values
    h_dates = df_h['race_date'].values
    h_feats = df_h[HIST_COLS].values

    horse_to_idx = defaultdict(list)
    for idx, (h, d) in enumerate(zip(h_horses, h_dates)):
        horse_to_idx[h].append((d, idx))

    df_dates = pd.to_datetime(df['race_date']).values
    df_horses = df['horse_id'].values

    for i in range(N):
        h, d_target = df_horses[i], df_dates[i]
        if h in horse_to_idx:
            valid_idx = [idx for d_hist, idx in horse_to_idx[h] if d_hist < d_target]
            if valid_idx:
                recent_idx = valid_idx[-hist_len:]
                seq_length = len(recent_idx)
                start_idx = hist_len - seq_length
                hist_tensor[i, start_idx:, :] = h_feats[recent_idx]

    return torch.tensor(hist_tensor)


def generate_dl_features(df, models, df_history=None, device=DEVICE, batch_size=2048, show_progress=False):
    """Transformer と DAE の推論を一括で行う (防弾仕様版)"""
    df = df.copy().reset_index(drop=True)
    proc = models['proc']

    x_c = torch.tensor(np.stack([
        _local_safe_transform(proc['encoders'][c], df[c].astype(str).fillna('unknown').tolist())
        for c in CAT_COLS
    ], axis=1), dtype=torch.long)

    # 🔥 FIX: 数値特徴量の変換時、ゼロ除算等で発生する inf も確実に 0 に潰す
    scaled_values = proc['num_scaler'].transform(df[NUM_COLS].values.astype(np.float32))
    x_n = torch.tensor(
        np.nan_to_num(scaled_values, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32
    )

    x_hist = build_hist_tensor(df, proc['hist_scaler'], hist_len=HIST_LEN, df_history=df_history)

    probs = []
    iterator = range(0, len(df), batch_size)
    if show_progress: iterator = tqdm(iterator, desc="Trans Infer")

    with torch.no_grad():
        for i in iterator:
            out = models['trans'](
                x_c[i:i + batch_size].to(device),
                x_n[i:i + batch_size].to(device),
                x_hist[i:i + batch_size].to(device)
            ).squeeze(-1)
            probs.append(torch.sigmoid(out).cpu().numpy().reshape(-1))

    # 🔥 FIX: Transformerモデル自体の重みが壊れてNaNを返した場合の防衛線
    transformer_preds = np.concatenate(probs)
    if np.isnan(transformer_preds).any():
        transformer_preds = np.nan_to_num(transformer_preds, nan=0.5)

    df['transformer_prob'] = transformer_preds

    df_dae = df[DAE_COLS].copy()
    df_dae[DAE_LEAKY_COLS] = 0

    # 🔥 FIX: DAEへの入力値の inf / nan も同様に完全ガード
    scaled_dae = models['dae_scaler'].transform(np.nan_to_num(df_dae.values.astype(np.float32)))
    inp = np.nan_to_num(scaled_dae, nan=0.0, posinf=0.0, neginf=0.0)

    dae_feats = []
    iterator = range(0, len(inp), batch_size)
    if show_progress: iterator = tqdm(iterator, desc="DAE Infer  ")

    with torch.no_grad():
        for i in iterator:
            enc, _ = models['dae'](torch.tensor(inp[i:i + batch_size], dtype=torch.float32).to(device))
            dae_feats.append(enc.cpu().numpy())

    dae_feats = np.concatenate(dae_feats, axis=0)
    for i in range(32):
        df[f'dae_{i}'] = dae_feats[:, i]

    del x_c, x_n, x_hist, probs, dae_feats, df_dae, inp
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return df