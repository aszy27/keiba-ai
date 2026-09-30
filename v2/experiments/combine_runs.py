# v2/experiments/combine_runs.py
# 技術の探索 T7: 予測の組み合わせ（対数プーリング）。v2/evaluate.py で保存した版の log(勝率) を等しい重みで平均し、
# レース内で正規化し直して、基準の版と同じレースで比べる（docs/rebuild_plan.md「技術の探索・第2弾」）。
# 需要予測などで「複雑な重み付けより単純平均が強い」（forecast combination puzzle）ことが知られているので、重みは推定しない。
# 使い方: python -m v2.experiments.combine_runs --tags res_mkt,res_seeds,mkt_place,past_mkt --ref res_mkt --tag t7_combine
import argparse

import numpy as np
import pandas as pd

from v2.evaluate import two_stage
from v2.model.pipeline import BENCH_DIR
from v2.model.softmax import RaceGroups, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ref", default="res_mkt")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tags = args.tags.split(",")
    keys = ["race_id", "horse_number"]
    d = None
    for t in tags:
        p = pd.read_parquet(BENCH_DIR / f"{t}_preds.parquet")[keys + ["race_date", "win", "win_odds", "p_a", "p_c"]]
        p[f"lp_{t}"] = np.log(p.pop("p_c"))
        d = p if d is None else d.merge(p[keys + [f"lp_{t}"]], on=keys)
    d = d.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    g = RaceGroups(d["race_id"], d["win"])
    ll_c, p_c = g.ll(d[[f"lp_{t}" for t in tags]].mean(axis=1).to_numpy())
    ll_a, _ = g.ll(np.log(d["p_a"].to_numpy()))
    races = pd.DataFrame({"race_id": d["race_id"].iloc[g.starts].to_numpy(), "year": d["race_date"].iloc[g.starts].dt.year.to_numpy(),
                          "ll_a": ll_a, "ll_c": ll_c})
    races.to_parquet(BENCH_DIR / f"{args.tag}.parquet", index=False)
    d.assign(p_c=p_c)[keys + ["race_date", "win", "win_odds", "p_a", "p_c"]].to_parquet(BENCH_DIR / f"{args.tag}_preds.parquet", index=False)
    both = races.merge(pd.read_parquet(BENCH_DIR / f"{args.ref}.parquet"), on=["race_id", "year"], suffixes=("", "_ref"))
    diff = both["ll_c"] - both["ll_c_ref"]
    print(f"[{args.tag}] {', '.join(tags)} の対数プーリング / {len(both)}R")
    for y, s in both.assign(d=diff).groupby("year"):
        lo, hi = bootstrap_ci(s["d"])
        print(f"  {y}年 {len(s):>5}R {s['d'].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    lo, hi = bootstrap_ci(diff)
    print(f"  通算  {len(both):>5}R {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for line in two_stage(both):
        print("  " + line)


if __name__ == "__main__":
    main()
