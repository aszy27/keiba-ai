# v2/model_combined.py
# M4: 基礎モデル（オッズなし）とオッズの結合。M5（最終テスト）もこのファイルの --final で行う。
#   段階1 (--build-base): 2021〜2026年の各年 Y について、2013〜Y-2年で学習・Y-1年で early stopping と温度合わせを
#                         した基礎モデルで Y 年を予測する（その年を学習していない予測）→ data/v2/base_oos.parquet
#   段階2: 報告年 Y で次の3つを比べる（学習 max(2021, Y-4)〜Y-2年 / early stopping Y-1年）
#          A. オッズのみ（条件付きロジット）
#          B. オッズ + 基礎モデル（条件付きロジット）
#          C. B を出発点に、残差を LightGBM で学習
#          指標: 1レースあたり対数尤度の A との差と、期待値ベースの単勝回収率（確定オッズ・割引後）
#   --final: 2026-09-07 以降を最終テストとして1回だけ評価する（docs/rebuild_plan.md の M5。設定は事前登録済み）
# 使い方: python -m v2.model_combined --build-base   # 段階1（新しいレースを予測に含めるときも再実行する）
#         python -m v2.model_combined --year 2024     # 段階2（開発。2024 か 2025）
#         python -m v2.model_combined --final         # 最終テスト（1回だけ）
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2 import features as ft
from v2.model_base import PARAMS, eligible
from v2.paths import V2_DIR, table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

BASE_OOS = V2_DIR / "base_oos.parquet"
BASE_YEARS = range(2021, 2027)
REPORT_YEARS = (2024, 2025)
EV_THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.5]
ODDS_HAIRCUT = [1.00, 0.95, 0.90]   # 購入時点のオッズは確定より不利にずれうる。スナップショットで実測するまでの目安
RESIDUAL_PARAMS = dict(PARAMS, learning_rate=0.02, num_leaves=15, min_data_in_leaf=500)

# M5（2026-09-15 事前登録。判定まで変えないこと）
FINAL_SPLIT = {"train": ("2022-01-01", "2026-01-01"), "valid": ("2026-01-01", "2026-09-07"),
               "report": ("2026-09-07", "2100-01-01")}
MIN_FINAL_RACES = 1000
FINAL_THRESHOLD = 1.1
FINAL_CI_LEVEL = 0.975


def make_split(year):
    """報告年 Y: 学習 max(2021, Y-4)〜Y-2年 / early stopping Y-1年 / 報告 Y年（オッズと基礎モデルの予測は2021年から）"""
    return {"train": (f"{max(2021, year - 4)}-01-01", f"{year - 1}-01-01"),
            "valid": (f"{year - 1}-01-01", f"{year}-01-01"), "report": (f"{year}-01-01", f"{year + 1}-01-01")}


def _train(tr, va, feats, base_tr=0.0, base_va=0.0, params=PARAMS):
    cats = [c for c in feats if c in ft.CATEGORICAL]
    dtr = lgb.Dataset(tr[feats], label=tr["win"], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["win"], categorical_feature=cats, reference=dtr)
    return lgb.train(dict(params, objective=lgb_objective(RaceGroups(tr["race_id"], tr["win"]), base_tr)), dtr,
                     num_boost_round=5000, valid_sets=[dva],
                     feval=lgb_metric(RaceGroups(va["race_id"], va["win"]), base_va),
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def build_base():
    df = eligible(pd.read_parquet(table_path("features")))
    out = []
    for year in BASE_YEARS:
        tr = df[(df["race_date"] >= "2013-01-01") & (df["race_date"] < f"{year - 1}-01-01")].reset_index(drop=True)
        va = df[(df["race_date"] >= f"{year - 1}-01-01") & (df["race_date"] < f"{year}-01-01")].reset_index(drop=True)
        te = df[(df["race_date"] >= f"{year}-01-01") & (df["race_date"] < f"{year + 1}-01-01")].reset_index(drop=True)
        if te.empty:
            continue
        m = _train(tr, va, ft.FEATURES)
        beta = fit_logit(m.predict(va[ft.FEATURES], num_iteration=m.best_iteration),
                         RaceGroups(va["race_id"], va["win"]))[0]
        u = beta * m.predict(te[ft.FEATURES], num_iteration=m.best_iteration)
        out.append(te[["race_id", "horse_id", "horse_number"]].assign(u_base=u))
        print(f"{year}: 学習 〜{year - 2}年 {tr['race_id'].nunique()}R / 木 {m.best_iteration}本 / 温度 {beta:.3f} / 予測 {te['race_id'].nunique()}R")
    pd.concat(out, ignore_index=True).to_parquet(BASE_OOS, index=False)
    print("保存:", BASE_OOS)


def load_market():
    """オッズと基礎モデルの予測を付けた、評価に使えるレースだけの表"""
    df = eligible(pd.read_parquet(table_path("features")))
    df = df.merge(pd.read_parquet(BASE_OOS)[["race_id", "horse_id", "u_base"]], on=["race_id", "horse_id"], how="inner")
    odds = pd.read_parquet(table_path("odds_final"))
    odds["horse_number"] = odds["horse_number"].astype(float)
    df = df.merge(odds[["race_id", "horse_number", "win_odds", "odds_status"]], on=["race_id", "horse_number"], how="left")

    pay = pd.read_parquet(table_path("payouts"))[["race_id", "win"]].rename(columns={"win": "win_pay"})
    pay["win_pay"] = pay["win_pay"].map(lambda v: v[0] if v is not None and len(v) == 1 else np.nan)
    df = df.merge(pay, on="race_id", how="left")
    winner = df[df["win"] == 1]
    mismatch = set(winner.loc[winner["win_pay"].notna() & ((winner["win_odds"] * 100).round() != winner["win_pay"]), "race_id"])

    g = df.groupby("race_id")
    ok = (g["win_odds"].transform(lambda s: (s > 0).all()) & (g["race_id"].transform("size") == g["u_base"].transform("count"))
          & ~df["race_id"].isin(mismatch))
    n_all = df["race_id"].nunique()
    df = df[ok].sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    print(f"オッズ・基礎モデルの予測がそろうレース: {df['race_id'].nunique()} / {n_all}（1着オッズと単勝払戻の不一致 {len(mismatch)}R を除外）")
    inv = 1.0 / df["win_odds"]
    df["x_mkt"] = np.log(inv / inv.groupby(df["race_id"]).transform("sum"))
    return df


def ev_table(part, p, label, thresholds=EV_THRESHOLDS):
    groups = RaceGroups(part["race_id"], part["win"])
    odds, win = part["win_odds"].values, part["win"].values
    print(f"-- {label}")
    for th in thresholds:
        sel = p * odds >= th
        if not sel.any():
            print(f"   期待値{th:.1f}以上: 0点")
            continue
        stake = np.add.reduceat(sel * 1.0, groups.starts)
        line = f"   期待値{th:.1f}以上: {int(sel.sum()):>5}点 的中{int((sel * win).sum()):>4}"
        for cut in ODDS_HAIRCUT:
            ret = np.add.reduceat(sel * win * odds * cut, groups.starts)
            idx = np.random.default_rng(0).integers(0, len(stake), (2000, len(stake)))
            boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
            line += (f" | ×{cut:.2f} 回収率 {ret.sum() / stake.sum() * 100:5.1f}%"
                     f" [{np.percentile(boots, 2.5):5.1f}〜{np.percentile(boots, 97.5):5.1f}]")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-base", action="store_true")
    ap.add_argument("--year", type=int, default=2025, choices=REPORT_YEARS, help="報告する年（開発）")
    ap.add_argument("--final", action="store_true", help="最終テスト（M5）。1回だけ実行する")
    args = ap.parse_args()
    if args.build_base:
        build_base()
        return

    split = FINAL_SPLIT if args.final else make_split(args.year)
    tag = "final" if args.final else str(args.year)
    if args.final:
        print("*** 最終テスト。この結果を見て特徴量・設定・閾値を変えないこと（docs/rebuild_plan.md の M5） ***")
    df = load_market()
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in split.items()}
    for k, v in parts.items():
        print(f"{k:<6} {split[k][0]}〜{split[k][1]}: {v['race_id'].nunique()}R")
    if args.final and parts["report"]["race_id"].nunique() < MIN_FINAL_RACES:
        print(f"最終テストの対象が {MIN_FINAL_RACES}R に達していないため評価しない")
        return
    g = {k: RaceGroups(v["race_id"], v["win"]) for k, v in parts.items()}
    rep = parts["report"]

    beta_a = fit_logit(parts["train"][["x_mkt"]].values, g["train"])
    ll_a, p_a = g["report"].ll(rep[["x_mkt"]].values @ beta_a)
    beta_b = fit_logit(parts["train"][["x_mkt", "u_base"]].values, g["train"])
    ll_b, p_b = g["report"].ll(rep[["x_mkt", "u_base"]].values @ beta_b)

    base = {k: v[["x_mkt", "u_base"]].values @ beta_b for k, v in parts.items()}
    m = _train(parts["train"], parts["valid"], ft.FEATURES, base["train"], base["valid"], params=RESIDUAL_PARAMS)
    ll_c, p_c = g["report"].ll(base["report"] + m.predict(rep[ft.FEATURES], num_iteration=m.best_iteration))
    rep[["race_id", "race_date", "horse_number", "win", "win_odds"]].assign(p_a=p_a, p_b=p_b, p_c=p_c).to_parquet(
        V2_DIR / f"combined_preds_{tag}.parquet", index=False)

    level = FINAL_CI_LEVEL if args.final else 0.95
    print(f"\n=== {tag} 1レースあたり対数尤度 ===")
    print(f"  A オッズのみ           {ll_a.mean():.4f}  係数 {beta_a.round(3).tolist()}")
    print(f"  B オッズ+基礎モデル    {ll_b.mean():.4f}  係数 {beta_b.round(3).tolist()}")
    print(f"  C B+残差LightGBM       {ll_c.mean():.4f}  木 {m.best_iteration}本")
    for label, ll in [("B − A", ll_b - ll_a), ("C − A", ll_c - ll_a)]:
        lo, hi = bootstrap_ci(ll, level=level)
        print(f"  {label}: {ll.mean():+.4f} [{level:.1%}区間 {lo:+.4f}, {hi:+.4f}]")

    print(f"\n=== {tag} 期待値ベースの単勝（1点1単位。×0.95 等はオッズを割り引いた場合） ===")
    thresholds = [FINAL_THRESHOLD] if args.final else EV_THRESHOLDS
    for label, p in [("A オッズのみ", p_a), ("B オッズ+基礎モデル", p_b), ("C B+残差", p_c)]:
        ev_table(rep, p, label, thresholds)

    if args.final:
        lo, _ = bootstrap_ci(ll_c - ll_a, level=level)
        sel = p_c * rep["win_odds"].values >= FINAL_THRESHOLD
        roi = (sel * rep["win"].values * rep["win_odds"].values).sum() / max(sel.sum(), 1) * 100
        print(f"\n判定: 対数尤度の差の{level:.1%}区間の下限 {lo:+.4f}（>0 が条件）/ 期待値{FINAL_THRESHOLD}以上の回収率 {roi:.1f}%（>100% が条件）"
              f" → {'合格' if lo > 0 and roi > 100 else '不合格'}")


if __name__ == "__main__":
    main()
