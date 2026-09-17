# v2/feature_lab.py
# 追加した特徴量グループがどれだけ効くかを、開発期間だけで調べる（オッズは使わない基礎モデルの当てやすさで比較）。
# 学習 2013〜2017 / early stopping 2018 / 報告 2019〜2020。最終テストの期間は触らない。
# 使い方: python -m v2.feature_lab            （基準 = 既存+展開。すべてのグループを1つずつ足す）
#         python -m v2.feature_lab --wave 5  （基準 = 第4弾までの全部入り。第5弾のグループだけを足す）
import argparse

import numpy as np
import pandas as pd

from v2 import features as ft
from v2 import features_extra as fx
from v2.model_base import eligible, train
from v2.paths import table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit

SPLIT = {"train": ("2013-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"),
         "report": ("2019-01-01", "2021-01-01")}


def feature_sets(wave=None):
    """wave を指定すると、基準を「その1つ前の弾までの全部入り」にして、その弾のグループだけを足す"""
    base, groups = ft.FEATURES_TRIP, fx.GROUPS
    if wave is not None:
        base = base + [c for w in sorted(fx.WAVES) if w < wave and w not in fx.INACTIVE_WAVES
                       for n in fx.WAVES[w] for c in fx.GROUPS[n]]
        groups = {n: fx.GROUPS[n] for n in fx.WAVES[wave]}
    sets = {f"基準（{len(base)}列）": base}
    for name, cols in groups.items():
        sets[f"＋{name}"] = base + cols
    sets["＋全部"] = base + [c for cols in groups.values() for c in cols]
    return sets, base


def report_ll(parts, feats):
    """2018年で温度を合わせ、2019〜2020年のレースごとの対数尤度を返す"""
    m = train({"train": parts["train"], "valid": parts["valid"]}, feats)
    u_va = m.predict(parts["valid"][feats], num_iteration=m.best_iteration)
    beta = fit_logit(u_va, RaceGroups(parts["valid"]["race_id"], parts["valid"]["win"]))[0]
    u = beta * m.predict(parts["report"][feats], num_iteration=m.best_iteration)
    ll, _ = RaceGroups(parts["report"]["race_id"], parts["report"]["win"]).ll(u)
    return ll, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, help="この弾のグループだけを、1つ前の弾までの全部入りを基準に測る")
    args = ap.parse_args()
    sets, base_feats = feature_sets(args.wave)
    df = eligible(pd.read_parquet(table_path("features")))
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in SPLIT.items()}
    for k, v in parts.items():
        print(f"{k:<6} {SPLIT[k][0]}〜{SPLIT[k][1]}: {v['race_id'].nunique()}R")

    base_ll = None
    for name, feats in sets.items():
        ll, m = report_ll(parts, feats)
        line = f"{name:<16} {len(feats):>3}列 木{m.best_iteration:>5}本 対数尤度 {ll.mean():.4f}"
        if base_ll is None:
            base_ll = ll
        else:
            d = ll - base_ll
            lo, hi = bootstrap_ci(d)
            line += f"  基準との差 {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            imp = pd.Series(m.feature_importance("gain"), index=feats)
            added = imp[[c for c in feats if c not in base_feats]].sort_values(ascending=False)
            if len(added):
                line += "  追加列の重要度: " + ", ".join(f"{k}={v / imp.sum():.1%}" for k, v in added.head(4).items())
        print(line, flush=True)


if __name__ == "__main__":
    main()
