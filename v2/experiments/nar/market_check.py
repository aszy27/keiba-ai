# v2/experiments/nar/market_check.py
# 段階1: 地方競馬の単勝市場がどれだけ歪んでいるかを、同じ期間の中央競馬と並べて調べる。
#  1. 1レースの (1/オッズ) の合計 → 実質の払戻率
#  2. 人気別・オッズ帯別の単勝回収率と、オッズから見込まれる勝率に対する実際の勝率
#  3. log(オッズ由来の勝率) の係数 alpha（1から離れるほど、オッズの確率が一律にずれている）
#  4. 「オッズのみ」の勝率を alpha で直すと対数尤度がどれだけ良くなるか（係数は期間の前半で推定し、後半で評価）
# 使い方: python -m v2.experiments.nar.market_check [--start 2026-06-01 --end 2026-09-01]
import argparse

import numpy as np
import pandas as pd

from v2.experiments.nar.parse_results import OUT_DIR
from v2.paths import table_path
from v2.model.softmax import RaceGroups, bootstrap_ci, fit_logit

ODDS_BANDS = [0, 2, 3, 5, 10, 20, 50, 100, 10000]
POP_BANDS = [0, 1, 2, 3, 5, 8, 12, 18]


def nar_frame(start, end):
    races = pd.read_parquet(OUT_DIR / "races.parquet")
    run = pd.read_parquet(OUT_DIR / "runners.parquet")
    d = run.merge(races[["race_id", "race_date", "place", "surface"]], on="race_id")
    return d[(d["race_date"] >= start) & (d["race_date"] < end)]


def jra_frame(start, end):
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date", "place", "surface"])
    run = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos", "status"])
    odds = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds", "popularity"])
    d = run.merge(odds, on=["race_id", "horse_number"], how="left").merge(races, on="race_id")
    d["finish_pos"] = d["finish_pos"].astype(float)
    return d[(d["race_date"] >= start) & (d["race_date"] < end)]


def clean(d):
    """出走馬全員に単勝オッズがあり、勝ち馬がちょうど1頭のレースだけ"""
    d = d[d["status"] != "scratched"].copy()
    d["win"] = (d["finish_pos"] == 1).astype(float)
    g = d.groupby("race_id")
    ok = g["win_odds"].transform(lambda s: (s > 0).all()) & (g["win"].transform("sum") == 1) & (g["race_id"].transform("size") >= 5)
    d = d[ok].sort_values(["race_date", "race_id", "win_odds"]).reset_index(drop=True)
    inv = 1.0 / d["win_odds"]
    d["overround"] = inv.groupby(d["race_id"]).transform("sum")
    d["p_mkt"] = inv / d["overround"]
    d["x_mkt"] = np.log(d["p_mkt"])
    return d


def band_table(d, col, bands, label):
    """帯ごとの単勝回収率。95%区間は馬単位のブートストラップ（同じレースの馬どうしの相関は無視した目安）"""
    rng = np.random.default_rng(0)
    rows = []
    for iv, g in d.groupby(pd.cut(d[col], bands), observed=True):
        ret = (g["win"] * g["win_odds"]).to_numpy()
        boots = ret[rng.integers(0, len(ret), (1000, len(ret)))].mean(axis=1) * 100
        rows.append({label: f"{iv.left:g}〜{iv.right:g}", "頭数": len(g), "実際の勝率": g["win"].mean(),
                     "オッズの勝率": g["p_mkt"].mean(), "回収率%": ret.mean() * 100,
                     "95%下限": np.percentile(boots, 2.5), "95%上限": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows).round({"実際の勝率": 3, "オッズの勝率": 3, "回収率%": 1, "95%下限": 1, "95%上限": 1})


def summarize(name, d):
    races = d.drop_duplicates("race_id")
    print(f"\n#### {name}: {len(races)}R / {len(d)}頭 / 1レース平均 {len(d) / len(races):.1f}頭")
    print(f"実質の払戻率（1/Σ(1/オッズ) の平均）: {(1 / races['overround']).mean():.3f}")
    print(band_table(d, "popularity", POP_BANDS, "人気").to_string(index=False))
    print(band_table(d, "win_odds", ODDS_BANDS, "単勝オッズ").to_string(index=False))

    dates = np.sort(d["race_date"].unique())
    cut = dates[len(dates) // 2]
    first, second = d[d["race_date"] < cut].reset_index(drop=True), d[d["race_date"] >= cut].reset_index(drop=True)
    alpha = fit_logit(first[["x_mkt"]].values, RaceGroups(first["race_id"], first["win"]))[0]
    g2 = RaceGroups(second["race_id"], second["win"])
    ll_raw, _ = g2.ll(second["x_mkt"].values)
    ll_fit, _ = g2.ll(alpha * second["x_mkt"].values)
    lo, hi = bootstrap_ci(ll_fit - ll_raw)
    print(f"alpha（前半で推定）= {alpha:.3f} / 後半で alpha による補正の対数尤度の改善 {np.mean(ll_fit - ll_raw):+.4f} [{lo:+.4f}, {hi:+.4f}]"
          f" / オッズのみの対数尤度 {ll_raw.mean():.4f}")
    by_place = d.drop_duplicates("race_id").groupby("place").size().sort_values(ascending=False)
    print("競馬場別レース数:", by_place.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-09-01")
    args = ap.parse_args()
    summarize("地方", clean(nar_frame(args.start, args.end)))
    summarize("中央（同じ期間）", clean(jra_frame(args.start, args.end)))


if __name__ == "__main__":
    main()
