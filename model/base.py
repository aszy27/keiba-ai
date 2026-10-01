# model/base.py
# M3: オッズを使わない基礎モデル。LightGBM のレース内ソフトマックスで勝ち馬を予測する。
#  - 学習 2013〜2018 / early stopping 2019 / 評価 2020（2012年は過去成績が溜まっていないので学習に使わない）
#  - 2019年で係数（温度）を合わせてから 2020年の勝ち馬の対数尤度を出す
#  - 旧モデルの OOF スコア（experiments/legacy_scores.py）があれば、同じレースで比べる
#  - --ablation で特徴量グループを1つずつ外したときの差を出す
# 使い方: python -m model.base [--ablation]
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from prep import features as ft
from paths import V2_DIR, table_path
from model.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

SPLIT = {"train": ("2013-01-01", "2019-01-01"), "valid": ("2019-01-01", "2020-01-01"),
         "test": ("2020-01-01", "2021-01-01")}
PARAMS = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, cat_smooth=20, verbose=-1, seed=42, num_threads=0, metric="None")
FEATURE_GROUPS = {"レース条件": ft.CATEGORICAL + ft.PRE_RACE, "馬の過去走": ft.HISTORY_FEATURES,
                  "成績の集計": ft.ENTITY_FEATURES, "レース内の比較": ft.FIELD_FEATURES}
LEGACY = V2_DIR / "legacy_oof_scores.parquet"


def eligible(df):
    """結果が確定し、勝ち馬がちょうど1頭（同着なし）、5頭以上のレースだけを使う"""
    df = df[df["status"].isin(ft.HISTORY_STATUS)]
    g = df.groupby("race_id")
    ok = (g["win"].transform("sum") == 1) & (g["horse_id"].transform("size") >= 5)
    return df[ok].sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)


def split(df):
    return {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in SPLIT.items()}


def train(parts, feats):
    tr, va = parts["train"], parts["valid"]
    cats = [c for c in feats if c in ft.CATEGORICAL]
    dtr = lgb.Dataset(tr[feats], label=tr["win"], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["win"], categorical_feature=cats, reference=dtr)
    params = dict(PARAMS, objective=lgb_objective(RaceGroups(tr["race_id"], tr["win"])))
    return lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                     feval=lgb_metric(RaceGroups(va["race_id"], va["win"])),
                     callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)])


def calibrated_ll(u_valid, u_test, parts):
    """2019年で温度を合わせ、2020年のレースごとの対数尤度と勝率を返す"""
    beta = fit_logit(u_valid, RaceGroups(parts["valid"]["race_id"], parts["valid"]["win"]))
    return RaceGroups(parts["test"]["race_id"], parts["test"]["win"]).ll(np.asarray(u_test).reshape(len(u_test), -1) @ beta)


def show_diff(label, d):
    lo, hi = bootstrap_ci(d)
    print(f"  {label:<28} {d.mean():+.4f}  [95%区間 {lo:+.4f}, {hi:+.4f}]")


def calibration_table(p, win):
    bins = pd.cut(p, [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    t = pd.DataFrame({"p": p, "win": win}).groupby(bins, observed=True).agg(頭数=("p", "size"), 予測勝率=("p", "mean"),
                                                                             実際の勝率=("win", "mean"))
    print(t.round(3).to_string())


def legacy_comparison(parts, u_valid, u_test):
    if not LEGACY.exists():
        print("\n旧モデルの OOF スコアが無いため比較を省略（python -m experiments.legacy_scores で作成）")
        return
    old = pd.read_parquet(LEGACY)
    sub = {}
    for k, u in [("valid", u_valid), ("test", u_test)]:
        p = parts[k].assign(u_new=u).merge(old.drop(columns="race_date"), on=["race_id", "horse_id"], how="left")
        covered = p.groupby("race_id")["lgb_rank_score"].transform(lambda s: s.notna().all())
        sub[k] = p[covered].reset_index(drop=True)
    for k in sub:
        d = sub[k]
        rs = d.groupby("race_id")["lgb_rank_score"]
        d["old_rank_z"] = ((d["lgb_rank_score"] - rs.transform("mean")) / rs.transform("std").replace(0, 1)).fillna(0)
        for c in ["lgb_prob_score", "transformer_prob"]:
            q = d[c].clip(1e-4, 1 - 1e-4)
            d[f"old_{c}_logit"] = np.log(q / (1 - q))
    old_cols = ["old_rank_z", "old_lgb_prob_score_logit", "old_transformer_prob_logit"]
    print(f"\n=== 旧モデルとの比較（両方のスコアがある {sub['test']['race_id'].nunique()}R） ===")
    ll_new, _ = calibrated_ll(sub["valid"]["u_new"].values, sub["test"]["u_new"].values, sub)
    ll_old, _ = calibrated_ll(sub["valid"][old_cols].values, sub["test"][old_cols].values, sub)
    print(f"  v2 {ll_new.mean():.4f} / 旧モデル（LGB2種+Transformer）{ll_old.mean():.4f}")
    show_diff("v2 − 旧モデル", ll_new - ll_old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    df = eligible(pd.read_parquet(table_path("features")))
    parts = split(df)
    for k, v in parts.items():
        print(f"{k:<5} {SPLIT[k][0]}〜{SPLIT[k][1]}: {v['race_id'].nunique()}R / {len(v)}頭")

    model = train(parts, ft.FEATURES)
    u_valid = model.predict(parts["valid"][ft.FEATURES], num_iteration=model.best_iteration)
    u_test = model.predict(parts["test"][ft.FEATURES], num_iteration=model.best_iteration)
    ll, p = calibrated_ll(u_valid, u_test, parts)
    uniform = -np.log(parts["test"].groupby("race_id").size()).mean()
    print(f"\n=== 2020年の勝ち馬の対数尤度（1レースあたり） 木 {model.best_iteration}本 ===")
    print(f"  v2 {ll.mean():.4f} / 全馬同じ確率 {uniform:.4f}")
    calibration_table(p, parts["test"]["win"].values)

    imp = pd.Series(model.feature_importance("gain"), index=ft.FEATURES).sort_values(ascending=False)
    print("\n重要度(gain)上位20:", ", ".join(f"{k}={v / imp.sum():.1%}" for k, v in imp.head(20).items()))
    pd.DataFrame({"race_id": parts["test"]["race_id"], "horse_id": parts["test"]["horse_id"], "u": u_test, "p": p}) \
        .to_parquet(V2_DIR / "base_preds_2020.parquet", index=False)

    legacy_comparison(parts, u_valid, u_test)

    if args.ablation:
        print("\n=== 特徴量グループを外したときの差（外したモデル − 全部入り。負ならそのグループが効いている） ===")
        for name, cols in FEATURE_GROUPS.items():
            feats = [c for c in ft.FEATURES if c not in cols]
            m = train(parts, feats)
            ll_drop, _ = calibrated_ll(m.predict(parts["valid"][feats], num_iteration=m.best_iteration),
                                       m.predict(parts["test"][feats], num_iteration=m.best_iteration), parts)
            show_diff(f"{name}なし（{len(cols)}列）", ll_drop - ll)


if __name__ == "__main__":
    main()
