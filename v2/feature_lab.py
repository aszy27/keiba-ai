# v2/feature_lab.py
# 追加した特徴量グループがどれだけ効くかを、開発期間だけで調べる（オッズは使わない基礎モデルの当てやすさで比較）。
# 学習 2013〜2017 / early stopping 2018 / 報告 2019〜2020。最終テストの期間は触らない。
# 使い方: python -m v2.feature_lab
import numpy as np
import pandas as pd

from v2 import features as ft
from v2 import features_extra as fx
from v2.model_base import eligible, train
from v2.paths import table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit

SPLIT = {"train": ("2013-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"),
         "report": ("2019-01-01", "2021-01-01")}


def feature_sets():
    base = ft.FEATURES_TRIP
    sets = {"既存+展開（基準）": base}
    for name, cols in fx.GROUPS.items():
        sets[f"＋{name}"] = base + cols
    sets["＋全部"] = base + fx.EXTRA_FEATURES
    return sets


def report_ll(parts, feats):
    """2018年で温度を合わせ、2019〜2020年のレースごとの対数尤度を返す"""
    m = train({"train": parts["train"], "valid": parts["valid"]}, feats)
    u_va = m.predict(parts["valid"][feats], num_iteration=m.best_iteration)
    beta = fit_logit(u_va, RaceGroups(parts["valid"]["race_id"], parts["valid"]["win"]))[0]
    u = beta * m.predict(parts["report"][feats], num_iteration=m.best_iteration)
    ll, _ = RaceGroups(parts["report"]["race_id"], parts["report"]["win"]).ll(u)
    return ll, m


def main():
    df = eligible(pd.read_parquet(table_path("features")))
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in SPLIT.items()}
    for k, v in parts.items():
        print(f"{k:<6} {SPLIT[k][0]}〜{SPLIT[k][1]}: {v['race_id'].nunique()}R")

    base_ll = None
    for name, feats in feature_sets().items():
        ll, m = report_ll(parts, feats)
        line = f"{name:<16} {len(feats):>3}列 木{m.best_iteration:>5}本 対数尤度 {ll.mean():.4f}"
        if base_ll is None:
            base_ll = ll
        else:
            d = ll - base_ll
            lo, hi = bootstrap_ci(d)
            line += f"  基準との差 {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            imp = pd.Series(m.feature_importance("gain"), index=feats)
            added = imp[[c for c in feats if c not in ft.FEATURES_TRIP]].sort_values(ascending=False)
            if len(added):
                line += "  追加列の重要度: " + ", ".join(f"{k}={v / imp.sum():.1%}" for k, v in added.head(4).items())
        print(line, flush=True)


if __name__ == "__main__":
    main()
