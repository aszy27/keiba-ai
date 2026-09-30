# v2/model/pipeline.py
# 学習の共通処理。検証（v2/evaluate.py）・学習（v2/train.py）・実践（v2/predict.py）で同じものを使う。
#   基礎モデル: オッズを使わない LightGBM（objective = win: 1着だけ / pl: 1〜k着の順序）。3シードの平均を温度で合わせる
#   結合:       log(勝率) = a·x_mkt + b·u_base + 残差 LightGBM（オッズ＋基礎モデルからのずれ）
import lightgbm as lgb
import numpy as np
import pandas as pd

from v2.data import features as ft
from v2.data import features_extra as fx
from v2.model import trip as mt
from v2.model.base import PARAMS, eligible
from v2.model.combined import RESIDUAL_PARAMS
from v2.model.market import add_market_cols
from v2.model.plackett import PLGroups, lgb_objective_pl
from v2.model.softmax import RaceGroups, fit_logit, lgb_metric, lgb_objective
from v2.paths import V2_DIR, table_path

END = "2026-09-07"                       # 開発・学習に使うのはこの前日まで（以降は前向き検証の期間）
FEATS = ft.FEATURES_TRIP + fx.EXTRA_FEATURES   # 176列（C[all] の "all" と同じ）
MARKET_FEATS = ["x_mkt", "mkt_rank"]     # 残差の入力に足すオッズ由来の列（market_cols / --res-market）
SEEDS = mt.SEEDS                         # 基礎モデルは (42, 7, 2024) の3シード平均
BENCH_DIR = V2_DIR / "bench"
CURRENT_BASE = V2_DIR / "base_oos_all_2015_2026.parquet"   # C[all] の年ごとの基礎モデルの予測（experiments/walkforward.py --build-base）


def load():
    """結果の確定したレースの特徴量（前向き検証の期間は捨てる）"""
    df = eligible(pd.read_parquet(table_path("features")))
    return df[df["race_date"] < END].reset_index(drop=True)


def between(df, start, end):
    return df[(df["race_date"] >= start) & (df["race_date"] < end)].reset_index(drop=True)


def base_file(name):
    """年ごとの基礎モデルの予測の保存先。"a+b" は2つの予測（温度合わせ後の効用）の平均"""
    if "+" in name:
        path = BENCH_DIR / f"base_{name}.parquet"
        if not path.exists():
            parts = [pd.read_parquet(base_file(n)).set_index(["race_id", "horse_id"])["u_base"] for n in name.split("+")]
            pd.concat(parts, axis=1, join="inner").mean(axis=1).rename("u_base").reset_index().to_parquet(path, index=False)
        return path
    return CURRENT_BASE if name == "current" else BENCH_DIR / f"base_{name}.parquet"


def recency_weight(dates, end, half_life):
    if not half_life:
        return None
    age = (pd.Timestamp(end) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half_life)


def train(tr, va, feats, params, objective="win", k=3, lam=0.75, stage_w=None, base_tr=0.0, base_va=0.0, row_w=None):
    """objective=win は model/combined._train と同じ。pl は 1〜k着の順序で学習する（ES は両方とも勝ち馬の対数尤度）"""
    cats = [c for c in feats if c in ft.CATEGORICAL]
    dtr = lgb.Dataset(tr[feats], label=tr["win"], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["win"], categorical_feature=cats, reference=dtr)
    if objective == "win" and row_w is None:
        obj = lgb_objective(RaceGroups(tr["race_id"], tr["win"]), base_tr)
    else:
        kk = 1 if objective == "win" else k
        obj = lgb_objective_pl(PLGroups(tr["race_id"], tr["finish_pos"], kk, lam, stage_w), base_tr, row_w)
    return lgb.train(dict(params, objective=obj), dtr, num_boost_round=5000, valid_sets=[dva],
                     feval=lgb_metric(RaceGroups(va["race_id"], va["win"]), base_va),
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def fit_base(tr, va, objective="win", k=3, lam=0.75, stage_w=None, row_w=None):
    """基礎モデルを3シードで学習し、(モデルの一覧, 温度) を返す。予測は base_utility で出す"""
    models = [train(tr, va, FEATS, dict(PARAMS, seed=s), objective, k, lam, stage_w, row_w=row_w) for s in SEEDS]
    temp = fit_logit(raw_base(models, va), RaceGroups(va["race_id"], va["win"]))[0]
    return models, temp


def raw_base(models, df):
    return sum(m.predict(df[FEATS], num_iteration=m.best_iteration or None) for m in models) / len(models)


def base_utility(models, temp, df):
    return temp * raw_base(models, df)


def market(df, base):
    """確定オッズ・年ごとの基礎モデルの予測（u_base）・オッズ由来の列を付けた、評価に使えるレースだけの表"""
    orig = mt.base_path
    mt.base_path = lambda name: base_file(base)
    try:
        d, _ = mt.with_market(df, "all")
    finally:
        mt.base_path = orig
    return add_market_cols(d)


def fit_combined(tr, va, weight_end=None, half_life=None, market_cols=False, extra_cols=(), res_objective="win",
                 k=3, lam=0.75, stage_w=None, res_params=None):
    """A（オッズのみ）・B（オッズ＋基礎モデル）の係数と残差モデルを学習する。
    tr / va は market() の表（x_mkt, u_base 付き）。half_life は weight_end（学習期間の終わり）を基準にした直近重視。
    返り値: dict(beta_a, beta_b, residual, feats)"""
    g_tr = RaceGroups(tr["race_id"], tr["win"])
    beta_a = fit_logit(tr[["x_mkt"]].values, g_tr)
    beta_b = fit_logit(tr[["x_mkt", "u_base"]].values, g_tr)
    feats = FEATS + (MARKET_FEATS if market_cols else []) + list(extra_cols)
    row_w = recency_weight(tr["race_date"], weight_end, half_life)
    res = train(tr, va, feats, res_params or RESIDUAL_PARAMS, res_objective, k, lam, stage_w,
                tr[["x_mkt", "u_base"]].values @ beta_b, va[["x_mkt", "u_base"]].values @ beta_b, row_w)
    return dict(beta_a=beta_a, beta_b=beta_b, residual=res, feats=feats)


def predict_combined(fit, d):
    """p_a（オッズのみ）と p_c（オッズ＋基礎モデル＋残差）の効用を返す"""
    u_a = d[["x_mkt"]].values @ fit["beta_a"]
    u_c = d[["x_mkt", "u_base"]].values @ fit["beta_b"] + fit["residual"].predict(
        d[fit["feats"]], num_iteration=fit["residual"].best_iteration or None)
    return u_a, u_c
