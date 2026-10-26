# experiments/exotic_judge.py
# 組み合わせ券の検証の判定（docs/rebuild_plan.md「3連複・3連単の検証の手順」「馬連・馬単の検証の手順」、どちらも事前登録済み）。
#   開発（2020〜2023年）: 方式（直接 / 比の方式）と λ（0.75 / 1.0）を、組み合わせ券の市場の確率に対する対数尤度の差が大きいほうで選び、
#                         期待値の閾値を「300点以上で回収率の95%下限が最大」で選ぶ。
#   最終テスト（2024〜2025年）: 開発で選んだ設定で1回だけ。合格 = 対数尤度の差の95%下限 > 0 かつ 回収率 > 100%。
# 勝率は前進検証の t7_combine（候補2と同じ作り方）。回収率は確定オッズで計算するので甘い。
# 使い方: python -m experiments.exotic_judge --bet trio --phase dev
#         python -m experiments.exotic_judge --bet trio --phase final   （開発の結果ファイルから設定を読む。2回目は実行しない）
import argparse
import json

import numpy as np
import pandas as pd

from model.exotic import BET_NAMES, BET_TYPES, TAKEOUT, race_ev, winning_combo
from model.pipeline import BENCH_DIR
from model.softmax import bootstrap_ci
from paths import V2_DIR, table_path

THRESHOLDS = [1.0, 1.2, 1.5, 2.0, 3.0]
MIN_POINTS = 300
VARIANTS = [("direct", 0.75), ("direct", 1.0), ("ratio", 0.75), ("ratio", 1.0)]
YEARS = {"dev": [2020, 2021, 2022, 2023], "final": [2024, 2025]}
OUT_DIR = V2_DIR.parent.parent / "result" / "v2"


def race_rows(g, od, bet, method, lam):
    """1レースの (確率, オッズ, 期待値, 当たりの位置, 市場の確率)。確率は手に入る組み合わせの中で合計1に直す"""
    m = race_ev(g["horse_number"], g["p_c"].to_numpy(), od, bet, lam)
    if m.empty:
        return None
    inv = 1.0 / m["odds"].to_numpy()
    q_mkt = inv / inv.sum()
    if method == "ratio":
        a = race_ev(g["horse_number"], g["p_a"].to_numpy(), od, bet, lam)
        prob = q_mkt * (m["prob"].to_numpy() / a["prob"].to_numpy())
    else:
        prob = m["prob"].to_numpy()
    prob = prob / prob.sum()
    return m["combo"].to_numpy(), prob, m["odds"].to_numpy(), q_mkt


def evaluate(bet, years, variants):
    preds = pd.read_parquet(BENCH_DIR / "t7_combine_preds.parquet")
    preds = preds[preds["race_date"].dt.year.isin(years)]
    fin = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos"])
    fin["horse_number"] = fin["horse_number"].astype(float)
    preds = preds.merge(fin, on=["race_id", "horse_number"], how="left")
    # レースごとに、方式・λ ごとの [対数尤度の差, 閾値ごとの点数, 閾値ごとの払戻] を持つ
    res = {v: {"ll": [], "stake": [], "ret": []} for v in variants}
    n_skip = 0
    for y in years:                                       # 3連単のオッズは1年で約780万行あるので、1年ずつ読む
        o = pd.read_parquet(V2_DIR / "odds_exotic" / f"{y}_type{BET_TYPES[bet]}.parquet")
        odds = {rid: dict(zip(g["combo"], g["odds_min"].astype(float))) for rid, g in o.groupby("race_id", sort=False)}
        del o
        py = preds[preds["race_date"].dt.year == y]
        for rid, g in py.groupby("race_id", sort=False):
            od = odds.get(rid)
            g = g.sort_values("horse_number")
            win = winning_combo(g["finish_pos"], g["horse_number"], bet) if od is not None else None
            if win is None:
                n_skip += 1
                continue
            rows = {v: race_rows(g, od, bet, *v) for v in variants}
            if any(r is None or win not in set(r[0]) for r in rows.values()):
                n_skip += 1
                continue
            for v, (combo, prob, od_arr, q_mkt) in rows.items():
                hit = combo == win
                ev = prob * od_arr
                res[v]["ll"].append(np.log(prob[hit][0]) - np.log(q_mkt[hit][0]))
                res[v]["stake"].append([(ev >= th).sum() for th in THRESHOLDS])
                res[v]["ret"].append([(od_arr * hit)[ev >= th].sum() for th in THRESHOLDS])
        print(f"  {y}年 処理（累計 {len(res[variants[0]]['ll'])}R）", flush=True)
        del odds
    for v in variants:
        for k in res[v]:
            res[v][k] = np.asarray(res[v][k], dtype=float)
    return res, n_skip


def roi_table(stake, ret):
    n = len(stake)
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    out = []
    for t, th in enumerate(THRESHOLDS):
        s, r = stake[:, t], ret[:, t]
        boots = r[idx].sum(1) / np.maximum(s[idx].sum(1), 1) * 100
        out.append({"th": th, "points": int(s.sum()), "hits": int((r > 0).sum()),
                    "roi": float(r.sum() / max(s.sum(), 1) * 100),
                    "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))})
    return out


def show(v, d):
    lo, hi = bootstrap_ci(d["ll"])
    print(f"-- {v[0]} λ={v[1]}: {len(d['ll'])}R / 対数尤度 − 市場 {d['ll'].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    tab = roi_table(d["stake"], d["ret"])
    for r in tab:
        print(f"   期待値{r['th']}以上: {r['points']:>9,}点（{r['points'] / len(d['ll']):6.1f}点/R）的中 {r['hits']:>5}"
              f" | 回収率 {r['roi']:6.1f}% [{r['lo']:6.1f}, {r['hi']:6.1f}]")
    return lo, hi, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bet", choices=list(BET_TYPES), required=True)
    ap.add_argument("--phase", choices=["dev", "final"], required=True)
    args = ap.parse_args()
    dev_file = OUT_DIR / f"exotic_dev_{args.bet}.json"
    final_file = OUT_DIR / f"exotic_final_{args.bet}.json"
    if args.phase == "final":
        if final_file.exists():
            print(f"最終テストは実行済み（{final_file}）。2回目は実行しない")
            return
        if not dev_file.exists():
            print("開発の結果が無い。先に --phase dev を実行する")
            return
        chosen = json.loads(dev_file.read_text(encoding="utf-8"))["chosen"]
    years = YEARS[args.phase]
    print(f"[t7_combine] {BET_NAMES[args.bet]} {years}年 {args.phase}（控除率 {TAKEOUT[args.bet] * 100:.1f}%）", flush=True)
    variants = VARIANTS if args.phase == "dev" else [(chosen["method"], chosen["lam"])]
    res, n_skip = evaluate(args.bet, years, variants)
    print(f"使わなかったレース（オッズ無し・同着など）: {n_skip}R")
    if args.phase == "dev":
        summary = {}
        for v in variants:
            lo, hi, tab = show(v, res[v])
            summary[f"{v[0]}_{v[1]}"] = {"ll": float(res[v]["ll"].mean()), "ll_lo": float(lo), "ll_hi": float(hi), "roi": tab}
        best = max(variants, key=lambda v: res[v]["ll"].mean())
        tab = summary[f"{best[0]}_{best[1]}"]["roi"]
        ok = [r for r in tab if r["points"] >= MIN_POINTS]
        th = max(ok, key=lambda r: r["lo"])["th"] if ok else None
        chosen = {"method": best[0], "lam": best[1], "threshold": th}
        print(f"\n選んだ設定: 方式 {best[0]} / λ={best[1]} / 期待値の閾値 {th}（{MIN_POINTS}点以上で回収率の95%下限が最大）")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dev_file.write_text(json.dumps({"years": years, "chosen": chosen, "summary": summary}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"保存: {dev_file}")
        return
    v = (chosen["method"], chosen["lam"])
    print(f"開発で選んだ設定: 方式 {v[0]} / λ={v[1]} / 期待値の閾値 {chosen['threshold']}")
    lo, hi, tab = show(v, res[v])
    r = next((r for r in tab if r["th"] == chosen["threshold"]), None)
    pass_ll = lo > 0
    pass_roi = r is not None and r["roi"] > 100
    print(f"\n判定: 対数尤度の差の95%下限 {lo:+.4f} > 0 → {'○' if pass_ll else '×'} / "
          f"期待値{chosen['threshold']}以上の回収率 {r['roi'] if r else float('nan'):.1f}% > 100% → {'○' if pass_roi else '×'}"
          f" → **{'合格' if pass_ll and pass_roi else '不合格'}**")
    final_file.write_text(json.dumps({"years": years, "chosen": chosen, "ll": float(res[v]["ll"].mean()), "ll_lo": float(lo),
                                      "ll_hi": float(hi), "roi": tab, "pass": bool(pass_ll and pass_roi)},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {final_file}")


if __name__ == "__main__":
    main()
