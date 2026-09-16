# v2/tune.py
# 基礎モデル（オッズなし）の学習設定を開発期間だけで調整する。
#   段階A: 学習 2013〜2016 / early stopping 2017 / 採点 2018 でランダム探索（報告年 2019〜2020 は使わない）
#   段階B: 見つかった設定を 2019〜2020 で現行設定と比べる（1回だけ）
# 探索するもの: LightGBM のパラメータ、直近重視の重み付け（半減期）、複数シードの平均。
# 使い方: python -m v2.tune --trials 24
import argparse
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2 import features as ft
from v2 import features_extra as fx
from v2.model_base import PARAMS, eligible
from v2.paths import table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

TUNE = {"train": ("2013-01-01", "2017-01-01"), "valid": ("2017-01-01", "2018-01-01"), "score": ("2018-01-01", "2019-01-01")}
CONFIRM = {"train": ("2013-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"), "score": ("2019-01-01", "2021-01-01")}
SPACE = {
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [31, 63, 127, 255],
    "min_data_in_leaf": [50, 100, 200, 400],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [1.0, 10.0, 50.0],
    "cat_smooth": [10, 20, 50],
}
HALF_LIVES = [None, 5.0, 3.0, 2.0]     # 直近重視の半減期（年）。None は重み付けなし
SEEDS = (42, 7, 2024)


def weights(df, end, half_life):
    if half_life is None:
        return None
    years = (pd.Timestamp(end) - df["race_date"]).dt.days / 365.25
    return np.power(0.5, years / half_life).to_numpy()


def fit(parts, feats, params, half_life, seeds, split):
    """seeds の数だけ学習して生スコアを平均し、valid で温度を合わせて score 期間の対数尤度を返す"""
    tr, va, sc = parts["train"], parts["valid"], parts["score"]
    cats = [c for c in feats if c in ft.CATEGORICAL]
    w = weights(tr, split["train"][1], half_life)
    u_va, u_sc, rounds = 0.0, 0.0, []
    for seed in seeds:
        dtr = lgb.Dataset(tr[feats], label=tr["win"], weight=w, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(va[feats], label=va["win"], categorical_feature=cats, reference=dtr)
        p = dict(PARAMS, **params, seed=seed, objective=lgb_objective(RaceGroups(tr["race_id"], tr["win"])))
        m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                      feval=lgb_metric(RaceGroups(va["race_id"], va["win"])),
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        u_va = u_va + m.predict(va[feats], num_iteration=m.best_iteration) / len(seeds)
        u_sc = u_sc + m.predict(sc[feats], num_iteration=m.best_iteration) / len(seeds)
        rounds.append(m.best_iteration)
    beta = fit_logit(u_va, RaceGroups(va["race_id"], va["win"]))[0]
    ll, _ = RaceGroups(sc["race_id"], sc["win"]).ll(beta * u_sc)
    return ll, rounds


def split_parts(df, split):
    return {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    args = ap.parse_args()
    feats = ft.FEATURES_TRIP + fx.EXTRA_FEATURES
    df = eligible(pd.read_parquet(table_path("features")))
    parts = split_parts(df, TUNE)
    print(f"段階A 学習{parts['train']['race_id'].nunique()}R / ES {parts['valid']['race_id'].nunique()}R "
          f"/ 採点 {parts['score']['race_id'].nunique()}R（2018年）", flush=True)

    base_ll, rounds = fit(parts, feats, {}, None, (42,), TUNE)
    print(f"現行設定: 対数尤度 {base_ll.mean():.4f} 木{rounds}本", flush=True)

    rng = random.Random(0)
    results = []
    for i in range(args.trials):
        params = {k: rng.choice(v) for k, v in SPACE.items()}
        half_life = rng.choice(HALF_LIVES)
        ll, rounds = fit(parts, feats, params, half_life, (42,), TUNE)
        results.append((ll.mean(), params, half_life))
        print(f"{i + 1:>3}/{args.trials} {ll.mean():.4f} 木{rounds[0]:>5}本 半減期={half_life} {params}", flush=True)

    results.sort(key=lambda x: -x[0])
    best_score, best_params, best_hl = results[0]
    print(f"\n最良: 対数尤度 {best_score:.4f}（現行 {base_ll.mean():.4f}） 半減期={best_hl} {best_params}", flush=True)

    print("\n段階B: 報告年 2019〜2020 で現行設定と比べる", flush=True)
    parts_c = split_parts(df, CONFIRM)
    ll_now, _ = fit(parts_c, feats, {}, None, (42,), CONFIRM)
    for label, params, hl, seeds in [("調整後", best_params, best_hl, (42,)),
                                     ("調整後＋3シード平均", best_params, best_hl, SEEDS)]:
        ll, rounds = fit(parts_c, feats, params, hl, seeds, CONFIRM)
        lo, hi = bootstrap_ci(ll - ll_now)
        print(f"  {label:<18} {ll.mean():.4f}（現行 {ll_now.mean():.4f}） 差 {np.mean(ll - ll_now):+.4f} [{lo:+.4f}, {hi:+.4f}] 木{rounds}本", flush=True)


if __name__ == "__main__":
    main()
