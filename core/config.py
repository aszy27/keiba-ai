# core/config.py
import os
import json
import warnings
from pathlib import Path

# =========================================================================
# パス定義 (pathlibによるモダンで軽量な管理)
# =========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR_TRAIN = DATA_DIR / "train"
DATA_DIR_VAL   = DATA_DIR / "val"
DATA_DIR_TEST  = DATA_DIR / "test"

DATA_DIRS     = [str(DATA_DIR_TRAIN), str(DATA_DIR_VAL)]
DATA_DIRS_ALL = [str(DATA_DIR_TRAIN), str(DATA_DIR_VAL), str(DATA_DIR_TEST)]

PEDIGREE_FILE     = str(DATA_DIR / "master_horse_data.csv")
HORSE_VECTOR_FILE = str(DATA_DIR / "horse_vectors.csv")
ODDS_DIR          = str(DATA_DIR / "odds_data")
OOF_CACHE_FILE    = str(DATA_DIR / "oof_features_cache.pkl")

MODEL_BASE  = BASE_DIR / "models"
PREDICT_DIR = BASE_DIR / "predict"

PARAM_FILE_RANKER     = str(MODEL_BASE / "best_params.json")
PARAM_FILE_CLF        = str(MODEL_BASE / "best_params_clf.json")

# =========================================================================
# 学習・モデル設定
# =========================================================================
try:
    import torch as _torch
    DEVICE_STR = "cuda" if _torch.cuda.is_available() else "cpu"
    del _torch
except ImportError:
    DEVICE_STR = "cpu"

TS_N_SPLITS  = 5
N_FOLDS      = TS_N_SPLITS
TS_GAP       = 36
META_GAP     = 12
TS_TEST_SIZE = None

TRANSFORMER_EPOCHS = 3
DAE_EPOCHS         = 5

MODEL_PATHS = {
    'lgb_ranker':        str(MODEL_BASE / 'keiba_strong_full_ranker.txt'),
    'lgb_prob':          str(MODEL_BASE / 'keiba_strong_full_classifier.txt'),
    'grandmaster':       str(MODEL_BASE / 'keiba_strong_full_grandmaster.pkl'),
    'dae_model':         str(MODEL_BASE / 'keiba_strong_full_dae.pth'),
    'dae_input_scaler':  str(MODEL_BASE / 'keiba_strong_full_dae_input_scaler.pkl'),
    'dae_target_scaler': str(MODEL_BASE / 'keiba_strong_full_dae_target_scaler.pkl'),
    'transformer':       str(MODEL_BASE / 'keiba_strong_full_transformer.pth'),
    'transformer_proc':  str(MODEL_BASE / 'keiba_strong_full_transformer_proc.pkl'),
    'meta_model':        str(MODEL_BASE / 'keiba_meta_confidence.txt'),
}

# =========================================================================
# 特徴量定義 (oikiri_last1f を完全抹消)
# =========================================================================
VEC_DIM = 64
assert VEC_DIM == 64, f"VEC_DIM={VEC_DIM} が train_graph_embedding.py の設定と一致しません。"

FEATURE_COLS = (
    [
        'bracket', 'horse_number', 'age', 'burden', 'weight', 'weight_diff', 'head_count',  # 🌟 2つを追加
        'run_count', 'interval', 'prev_is_win', 'jockey_change', 'avg_rank_5',
        'prev_rank_1', 'prev_rank_2', 'prev_rank_3', 'prev_rank_4', 'prev_rank_5',
        'prev_diff_1', 'prev_diff_2', 'prev_diff_3', 'prev_diff_4', 'prev_diff_5',
        'prev_3f_1',  'prev_3f_2',  'prev_3f_3',  'prev_3f_4',  'prev_3f_5',
        'is_nige', 'prev_pos_1',
        'jockey_win_rate', 'jockey_top3_rate', 'trainer_win_rate', 'trainer_top3_rate',
        'breeder_win_rate', 'breeder_top3_rate',
        'oikiri_score', 'oikiri_missing',
        'prev_pace_1', 'prev_first_3f_1', 'prev_last_3f_race_1',
        'jockey_course_win', 'trainer_course_win', 'sire_course_win',
        'race_senkou_count', 'race_senkou_ratio',
        'straight', 'elevation', 'has_slope', 'grade_score'
    ]
    + [f'dae_{i}' for i in range(32)]
    + [f'p_vec_{i}' for i in range(VEC_DIM)]
)

CAT_COLS  = ['place', 'type', 'weather', 'condition', 'horse_id', 'jockey_id', 'grade', 'breeder']
NUM_COLS  = ['burden', 'weight', 'weight_diff', 'length', 'horse_number', 'oikiri_score', 'straight', 'elevation', 'has_slope'] # 🌟 ここにも2つを追加
HIST_COLS = ['rank', 'last_3f', 'diff_time', 'popularity', 'grade_score', 'prize', 'pace_score', 'first_3f', 'last_3f_race']

DAE_COLS       = ['rank', 'bracket', 'horse_number', 'age', 'burden', 'prize', 'last_3f', 'interval']
DAE_LEAKY_COLS = ['rank', 'prize', 'last_3f']
DAE_INPUT_DIM  = len(DAE_COLS)
HIST_LEN       = 5

# =========================================================================
# LightGBM ハイパーパラメータ定義とロード (3連複 Sniper 最適化版)
# =========================================================================
DEFAULT_LABEL_GAIN = [0, 10, 30, 100, 200]

RANKER_PARAMS = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [3],
    'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 63,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_data_in_leaf': 30, 'max_depth': -1, 'verbose': -1, 'random_state': 42,
    'label_gain': DEFAULT_LABEL_GAIN,
}

if os.path.exists(PARAM_FILE_RANKER):
    try:
        with open(PARAM_FILE_RANKER, 'r') as f:
            p = json.load(f)
        if p.get('objective') == 'lambdarank':
            RANKER_PARAMS.update(p)
            # 🌟 古い3要素のパラメータを検知し、5要素に自動上書きするため「!= 5」に変更
            if not RANKER_PARAMS.get('label_gain') or len(RANKER_PARAMS['label_gain']) != 5:
                RANKER_PARAMS['label_gain'] = DEFAULT_LABEL_GAIN
    except Exception:
        pass

CLASS_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'is_unbalance': True,
    'verbose': -1, 'random_state': 42,
}

if os.path.exists(PARAM_FILE_CLF):
    try:
        with open(PARAM_FILE_CLF, 'r') as f:
            p = json.load(f)
        if p.get('objective') == 'binary':
            CLASS_PARAMS.update(p)
    except Exception:
        pass