# model/pipeline.py
# 学習の共通処理。検証（evaluate.py）・学習（train.py）・実践（predict.py）で同じものを使う。
#   基礎モデル: オッズを使わない LightGBM（objective = win: 1着だけ / pl: 1〜k着の順序）。3シードの平均を温度で合わせる
#   結合:       log(勝率) = a·x_mkt + b·u_base + 残差 LightGBM（オッズ＋基礎モデルからのずれ）
import lightgbm as lgb
import numpy as np
import pandas as pd

from prep import features as ft
from prep import features_extra as fx
from model import trip as mt
from model.base import PARAMS, eligible
from model.combined import RESIDUAL_PARAMS
from model.market import PLACE_COLS, add_market_cols, add_place_cols
from model.past_market import PAST_COLS, PAST_DIFF, add_past_market
from model.plackett import PLGroups, lgb_objective_pl
from model.softmax import RaceGroups, fit_logit, lgb_metric, lgb_objective
from paths import V2_DIR, table_path

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


EXTRA_COLS = {"place": PLACE_COLS, "past": PAST_COLS + PAST_DIFF}   # 組み合わせの版が残差の入力に足す列


def merge_final_place_odds(d):
    po = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "place_odds_min", "place_odds_max"])
    po["horse_number"] = po["horse_number"].astype(float)
    return d.merge(po, on=["race_id", "horse_number"], how="left")


def add_extra_cols(d, extras, final_place_odds=True):
    """版が使う追加の列を作る。final_place_odds=False なら d にある複勝オッズ（スナップショット）をそのまま使う"""
    if "place" in extras:
        d = add_place_cols(merge_final_place_odds(d) if final_place_odds else d)
    if "past" in extras:
        d = add_past_market(d)
    return d


def fit_combined(tr, va, weight_end=None, half_life=None, market_cols=False, extra_cols=(), res_objective="win",
                 k=3, lam=0.75, stage_w=None, res_params=None, res_seeds=None, b_cols=("x_mkt", "u_base"), temp=False,
                 feats_override=None, row_w_extra=None, restack=False):
    """A（オッズのみ）・B（オッズ＋基礎モデル）の係数と残差モデルを学習する。
    tr / va は market() の表（x_mkt, u_base 付き）。half_life は weight_end（学習期間の終わり）を基準にした直近重視。
    res_seeds を渡すと残差をシードごとに学習して予測を平均する（residual はモデルの一覧になる）。
    b_cols は B（出発点）の変数。temp=True なら最後に C の効用に掛ける温度を va で最尤推定する。
    feats_override は残差の入力列の置き換え、row_w_extra は行の重みの追加（直近重視と掛け合わせる）、
    restack=True なら va で「B の効用」と「残差の出力」の2つの係数を最尤推定し直す（技術の探索 T9）。
    返り値: dict(beta_a, beta_b, residual, feats)"""
    g_tr = RaceGroups(tr["race_id"], tr["win"])
    beta_a = fit_logit(tr[["x_mkt"]].values, g_tr)
    b_cols = list(b_cols)
    beta_b = fit_logit(tr[b_cols].values, g_tr)
    feats = list(feats_override) if feats_override is not None else FEATS + (MARKET_FEATS if market_cols else []) + list(extra_cols)
    row_w = recency_weight(tr["race_date"], weight_end, half_life)
    if row_w_extra is not None:
        row_w = row_w_extra if row_w is None else row_w * row_w_extra
    params = res_params or RESIDUAL_PARAMS
    args = (res_objective, k, lam, stage_w, tr[b_cols].values @ beta_b, va[b_cols].values @ beta_b, row_w)
    if res_seeds:
        res = [train(tr, va, feats, dict(params, seed=s), *args) for s in res_seeds]
    else:
        res = train(tr, va, feats, params, *args)
    fit = dict(beta_a=beta_a, beta_b=beta_b, residual=res, feats=feats, b_cols=b_cols, temp=1.0)
    if restack:
        base_va, res_va = split_utility(fit, va)
        fit["mix"] = fit_logit(np.c_[base_va, res_va], RaceGroups(va["race_id"], va["win"])).tolist()
    if temp:
        _, u_va = predict_combined(fit, va)
        fit["temp"] = float(fit_logit(u_va, RaceGroups(va["race_id"], va["win"]))[0])
    return fit


def residual_trees(fit):
    r = fit["residual"]
    return [m.best_iteration for m in r] if isinstance(r, list) else r.best_iteration


def split_utility(fit, d):
    """(B の効用, 残差の出力)"""
    res = fit["residual"] if isinstance(fit["residual"], list) else [fit["residual"]]
    base = d[fit.get("b_cols", ["x_mkt", "u_base"])].values @ fit["beta_b"]
    return base, sum(m.predict(d[fit["feats"]], num_iteration=m.best_iteration or None) for m in res) / len(res)


def predict_combined(fit, d):
    """p_a（オッズのみ）と p_c（オッズ＋基礎モデル＋残差）の効用を返す"""
    u_a = d[["x_mkt"]].values @ fit["beta_a"]
    base, res = split_utility(fit, d)
    mix = fit.get("mix", [1.0, 1.0])
    return u_a, (mix[0] * base + mix[1] * res) * fit.get("temp", 1.0)
