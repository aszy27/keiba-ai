# v2/experiments/stack_runs.py
# 技術の探索 T14: スタッキング。保存済みの版の log(勝率) を、選ぶ期間（2019〜2022）で条件付きロジットの重みを学習して組み合わせ、
# 確かめる期間（2023〜2026）だけで評価する。比較として同じ版の等しい重みの平均（対数プーリング）も出す。
# 使い方: python -m v2.experiments.stack_runs --ref res_mkt --tag t14_stack
import argparse

import numpy as np
import pandas as pd

from v2.evaluate import CONFIRM_YEARS, SELECT_YEARS
from v2.model.pipeline import BENCH_DIR
from v2.model.softmax import RaceGroups, bootstrap_ci, fit_logit

# 登録どおり: res_mkt と、それを基準に測った版（実験6・技術の探索の各版）。組み合わせの版・T1・動作確認の版は除く
TAGS = ["res_mkt", "res_seeds", "mkt_place", "mkt_race", "past_mkt", "t2_temp", "t5_base2", "t6_nobase", "t3_quarter",
        "t4_pl5", "t8_shin", "t9_restack", "t11_topk", "t12_split", "t13_adv", "t15_subspace"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="res_mkt")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    keys = ["race_id", "horse_number"]
    tags = [t for t in TAGS if (BENCH_DIR / f"{t}_preds.parquet").exists()]
    print("使う版:", ", ".join(tags), f"（登録した {len(TAGS)} 版のうち保存済みのもの）")
    d = None
    for t in tags:
        p = pd.read_parquet(BENCH_DIR / f"{t}_preds.parquet")[keys + ["race_date", "win", "p_c"]]
        p[t] = np.log(p.pop("p_c"))
        d = p if d is None else d.merge(p[keys + [t]], on=keys)
    d = d.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    d["year"] = d["race_date"].dt.year
    sel, con = d[d["year"].isin(SELECT_YEARS)].reset_index(drop=True), d[d["year"].isin(CONFIRM_YEARS)].reset_index(drop=True)
    g_sel, g_con = RaceGroups(sel["race_id"], sel["win"]), RaceGroups(con["race_id"], con["win"])
    w = fit_logit(sel[tags].to_numpy(), g_sel)
    print("学習した重み（選ぶ期間）:", ", ".join(f"{t} {x:+.3f}" for t, x in zip(tags, w)))
    ll_stack, _ = g_con.ll(con[tags].to_numpy() @ w)
    ll_eq, _ = g_con.ll(con[tags].mean(axis=1).to_numpy())
    ref = pd.read_parquet(BENCH_DIR / f"{args.ref}.parquet").set_index("race_id")["ll_c"]
    rids = con["race_id"].iloc[g_con.starts].to_numpy()
    ll_ref = ref.loc[rids].to_numpy()
    for name, ll in [("重みを学習（スタッキング）", ll_stack), ("等しい重みの平均", ll_eq)]:
        diff = ll - ll_ref
        lo, hi = bootstrap_ci(diff, level=0.90)
        print(f"  確かめる期間 2023〜2026 {name}: {args.ref} との差 {diff.mean():+.4f} [90% {lo:+.4f}, {hi:+.4f}]"
              f" → {'確認' if lo > 0 else '未確認'}（{len(diff)}R）")
    pd.DataFrame({"race_id": rids, "ll_stack": ll_stack, "ll_equal": ll_eq, "ll_ref": ll_ref}).to_parquet(
        BENCH_DIR / f"{args.tag}.parquet", index=False)


if __name__ == "__main__":
    main()
