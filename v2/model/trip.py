# v2/model/trip.py
# 展開・不利の特徴量（TRIP_FEATURES）がオッズへの上積みを生むかの検証（docs/rebuild_plan.md「展開・不利の特徴量」）。
#   --build-base : 2015〜2023年の各年 Y を、学習 2013〜Y-2年 / early stopping・温度 Y-1年 の基礎モデルで予測する
#                  （特徴量セット base=FEATURES と trip=FEATURES+TRIP_FEATURES の2つ）
#   開発         : 報告2019年（学習 2015〜2017 / ES 2018）と報告2020年（学習 2015〜2018 / ES 2019）で
#                  A オッズのみ / C（base）/ C+展開（trip）を比べ、C+展開の期待値の閾値を機械的に1つ選ぶ
#   --final      : 学習 2016〜2019 / ES 2020 / 報告 2021〜2023 で1回だけ判定する
# 使い方: python -m v2.model.trip --build-base
#         python -m v2.model.trip
#         python -m v2.model.trip --final --threshold 1.1
import argparse

import numpy as np
import pandas as pd

from v2.data import features as ft
from v2.data import features_extra as fx
from v2.model.base import PARAMS, eligible
from v2.model.combined import RESIDUAL_PARAMS, _train, ev_table
from v2.paths import V2_DIR, table_path
from v2.model.softmax import RaceGroups, bootstrap_ci, fit_logit

FEATURE_SETS = {"base": ft.FEATURES, "trip": ft.FEATURES_TRIP, "all": ft.FEATURES_TRIP + fx.EXTRA_FEATURES}
BASE_YEARS = range(2015, 2024)
DEV_SPLITS = {
    2019: {"train": ("2015-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"), "report": ("2019-01-01", "2020-01-01")},
    2020: {"train": ("2015-01-01", "2019-01-01"), "valid": ("2019-01-01", "2020-01-01"), "report": ("2020-01-01", "2021-01-01")},
}
FINAL_SPLIT = {"train": ("2016-01-01", "2020-01-01"), "valid": ("2020-01-01", "2021-01-01"), "report": ("2021-01-01", "2024-01-01")}
THRESHOLDS = [1.0, 1.1, 1.2, 1.3]
MIN_BETS = 300
SEEDS = (42, 7, 2024)   # 基礎モデルはこの3シードの平均を使う
# 2021〜2023年は展開(trip)の判定で一度使っているため、2候補目は区間を97.5%にする（docs/rebuild_plan.md）
FINAL_CI_LEVEL = 0.975


def base_path(name):
    return V2_DIR / f"base_oos_{name}_2015_2023.parquet"


def build_base(df, names=None):
    for name, feats in FEATURE_SETS.items():
        if names and name not in names:
            continue
        out = []
        for year in BASE_YEARS:
            tr = df[(df["race_date"] >= "2013-01-01") & (df["race_date"] < f"{year - 1}-01-01")].reset_index(drop=True)
            va = df[(df["race_date"] >= f"{year - 1}-01-01") & (df["race_date"] < f"{year}-01-01")].reset_index(drop=True)
            te = df[(df["race_date"] >= f"{year}-01-01") & (df["race_date"] < f"{year + 1}-01-01")].reset_index(drop=True)
            # 複数シードの平均でブレを減らす（開発期間で +0.0038 [+0.0013, +0.0063]。docs/rebuild_plan.md「学習設定の調整」）
            u_va, u_te, rounds = 0.0, 0.0, []
            for seed in SEEDS:
                m = _train(tr, va, feats, params=dict(PARAMS, seed=seed))
                u_va = u_va + m.predict(va[feats], num_iteration=m.best_iteration) / len(SEEDS)
                u_te = u_te + m.predict(te[feats], num_iteration=m.best_iteration) / len(SEEDS)
                rounds.append(m.best_iteration)
            beta = fit_logit(u_va, RaceGroups(va["race_id"], va["win"]))[0]
            out.append(te[["race_id", "horse_id"]].assign(u_base=beta * u_te))
            print(f"[{name}] {year}: 学習 {tr['race_id'].nunique()}R / 木 {rounds}本 / 温度 {beta:.3f}", flush=True)
        pd.concat(out, ignore_index=True).to_parquet(base_path(name), index=False)


def with_market(df, name):
    """基礎モデルの予測とオッズを付け、出走馬全員にオッズがあり1着オッズ×100と単勝払戻が合うレースだけにする"""
    d = df.merge(pd.read_parquet(base_path(name)), on=["race_id", "horse_id"], how="inner")
    odds = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds"])
    odds["horse_number"] = odds["horse_number"].astype(float)
    d = d.merge(odds, on=["race_id", "horse_number"], how="left")
    pay = pd.read_parquet(table_path("payouts"), columns=["race_id", "win"])
    pay["win_pay"] = pay["win"].map(lambda v: v[0] if v is not None and len(v) == 1 else np.nan)
    w = d[d["win"] == 1].merge(pay[["race_id", "win_pay"]], on="race_id", how="left")
    mismatch = set(w.loc[w["win_pay"].notna() & ((w["win_odds"] * 100).round() != w["win_pay"]), "race_id"])
    g = d.groupby("race_id")
    ok = (g["win_odds"].transform(lambda s: (s > 0).all()) & (g["race_id"].transform("size") == g["u_base"].transform("count"))
          & ~d["race_id"].isin(mismatch))
    d = d[ok].sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    inv = 1.0 / d["win_odds"]
    d["x_mkt"] = np.log(inv / inv.groupby(d["race_id"]).transform("sum"))
    return d, len(mismatch)


def run_split(df, name, split, ci_level=0.95):
    """A（オッズのみ）と C（オッズ+基礎モデル+残差）を学習して、報告期間のレースごとの対数尤度と勝率を返す"""
    feats = FEATURE_SETS[name]
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True) for k, (s, e) in split.items()}
    g = {k: RaceGroups(v["race_id"], v["win"]) for k, v in parts.items()}
    rep = parts["report"]
    beta_a = fit_logit(parts["train"][["x_mkt"]].values, g["train"])
    ll_a, p_a = g["report"].ll(rep[["x_mkt"]].values @ beta_a)
    beta_b = fit_logit(parts["train"][["x_mkt", "u_base"]].values, g["train"])
    base = {k: v[["x_mkt", "u_base"]].values @ beta_b for k, v in parts.items()}
    m = _train(parts["train"], parts["valid"], feats, base["train"], base["valid"], params=RESIDUAL_PARAMS)
    ll_c, p_c = g["report"].ll(base["report"] + m.predict(rep[feats], num_iteration=m.best_iteration))
    lo, hi = bootstrap_ci(ll_c - ll_a, level=ci_level)
    print(f"  [{name}] 報告 {rep['race_id'].nunique()}R / A {ll_a.mean():.4f} / C {ll_c.mean():.4f} / 基礎モデルの係数 {beta_b[1]:+.3f} / 木 {m.best_iteration}本"
          f" / C − A {np.mean(ll_c - ll_a):+.4f} [{ci_level:.1%}区間 {lo:+.4f}, {hi:+.4f}]")
    return rep.assign(p_a=p_a, p_c=p_c), ll_c - ll_a


def pooled_threshold(preds, candidate):
    d = preds.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    starts = RaceGroups(d["race_id"], d["win"]).starts
    rng = np.random.default_rng(0)
    best = None
    print(f"  開発2年を合わせた期待値ベース単勝（C[{candidate}]）:")
    for th in THRESHOLDS:
        sel = (d["p_c"] * d["win_odds"] >= th).to_numpy()
        stake = np.add.reduceat(sel * 1.0, starts)
        ret = np.add.reduceat(sel * d["win"].to_numpy() * d["win_odds"].to_numpy(), starts)
        idx = rng.integers(0, len(stake), (2000, len(stake)))
        boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
        lo = np.percentile(boots, 2.5)
        print(f"    期待値{th:.1f}以上: {int(sel.sum()):>5}点 回収率 {ret.sum() / max(stake.sum(), 1) * 100:5.1f}% [{lo:5.1f}, {np.percentile(boots, 97.5):5.1f}]")
        if sel.sum() >= MIN_BETS and (best is None or lo > best[1]):
            best = (th, lo)
    print(f"  → 選ばれた閾値（{MIN_BETS}点以上で95%下限が最大）: {best[0] if best else 'なし'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-base", action="store_true")
    ap.add_argument("--sets", help="対象の特徴量セット（カンマ区切り。既定はすべて）")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()
    names = args.sets.split(",") if args.sets else list(FEATURE_SETS)
    df = eligible(pd.read_parquet(table_path("features")))
    if args.build_base:
        build_base(df, names)
        return

    markets = {}
    for name in names:
        markets[name], n_mismatch = with_market(df, name)
        print(f"[{name}] オッズ・基礎モデルの予測がそろうレース {markets[name]['race_id'].nunique()}R（1着オッズと単勝払戻の不一致 {n_mismatch}R を除外）")

    if args.final:
        if args.threshold is None:
            raise SystemExit("--final には登録した --threshold が必要")
        print("*** 最終テスト（2021〜2023年）。この結果を見て特徴量・設定・閾値を変えないこと ***")
        for name in names[:-1]:
            ref, _ = run_split(markets[name], name, FINAL_SPLIT, FINAL_CI_LEVEL)
            ev_table(ref, ref["p_c"].to_numpy(), f"C[{name}]（参考）", [args.threshold])
        preds, diff = run_split(markets[names[-1]], names[-1], FINAL_SPLIT, FINAL_CI_LEVEL)
        ev_table(preds, preds["p_c"].to_numpy(), f"C[{names[-1]}]（判定対象）", [args.threshold])
        lo, _ = bootstrap_ci(diff, level=FINAL_CI_LEVEL)
        sel = preds["p_c"] * preds["win_odds"] >= args.threshold
        roi = (sel * preds["win"] * preds["win_odds"]).sum() / max(sel.sum(), 1) * 100
        print(f"判定: C[{names[-1]}] − A の{FINAL_CI_LEVEL:.1%}区間の下限 {lo:+.4f}（>0 が条件）/ "
              f"期待値{args.threshold}以上の回収率 {roi:.1f}%（>100% が条件）"
              f" → {'合格' if lo > 0 and roi > 100 else '不合格'}")
        return

    cand_preds = []
    for year, split in DEV_SPLITS.items():
        print(f"\n=== 開発 報告{year}年 ===")
        for name in names[:-1]:
            run_split(markets[name], name, split)
        preds, _ = run_split(markets[names[-1]], names[-1], split)
        cand_preds.append(preds)
    pooled_threshold(pd.concat(cand_preds, ignore_index=True), names[-1])


if __name__ == "__main__":
    main()
