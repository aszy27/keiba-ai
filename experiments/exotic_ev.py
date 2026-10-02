# experiments/exotic_ev.py
# 3連複・3連単の期待値ベースの購入の検証（docs/rebuild_plan.md「3連複・3連単の期待値」の X1）。
# 前進検証で保存した勝率（evaluate.py の <tag>_preds.parquet）と、組み合わせ券の確定オッズ（data/v2/odds_exotic/）を使う。
# 使い方: python -m experiments.exotic_ev --tag t7_combine --bet trio --years 2020
import argparse

import numpy as np
import pandas as pd

from model.exotic import BET_NAMES, BET_TYPES, TAKEOUT, race_ev, winning_combo
from model.pipeline import BENCH_DIR
from paths import V2_DIR, table_path

THRESHOLDS = [1.0, 1.2, 1.5, 2.0, 3.0]


def load_odds(bet, years):
    frames = [pd.read_parquet(V2_DIR / "odds_exotic" / f"{y}_type{BET_TYPES[bet]}.parquet") for y in years
              if (V2_DIR / "odds_exotic" / f"{y}_type{BET_TYPES[bet]}.parquet").exists()]
    o = pd.concat(frames, ignore_index=True)
    return {rid: dict(zip(g["combo"], g["odds_min"].astype(float))) for rid, g in o.groupby("race_id", sort=False)}


def build(preds, odds, bet, col, lam):
    rows, ll = [], []
    for rid, g in preds.groupby("race_id", sort=False):
        od = odds.get(rid)
        if od is None:
            continue
        g = g.sort_values("horse_number")
        win = winning_combo(g["finish_pos"], g["horse_number"], bet)
        if win is None:
            continue
        r = race_ev(g["horse_number"], g[col].to_numpy(), od, bet, lam)
        if r.empty or win not in set(r["combo"]):
            continue
        r["hit"] = r["combo"] == win
        r["race_id"] = rid
        rows.append(r)
        ll.append(np.log(r.loc[r["hit"], "prob"].iloc[0]))
    return pd.concat(rows, ignore_index=True), np.array(ll)


def build_ratio(preds, odds, bet, lam):
    """X2: 比の方式。期待値 = (モデルの確率 / 単勝オッズだけの確率) × (3連複の市場の確率 × オッズ)"""
    rows, ll = [], []
    for rid, g in preds.groupby("race_id", sort=False):
        od = odds.get(rid)
        if od is None:
            continue
        g = g.sort_values("horse_number")
        win = winning_combo(g["finish_pos"], g["horse_number"], bet)
        if win is None:
            continue
        m = race_ev(g["horse_number"], g["p_c"].to_numpy(), od, bet, lam)
        a = race_ev(g["horse_number"], g["p_a"].to_numpy(), od, bet, lam)
        if m.empty or win not in set(m["combo"]):
            continue
        inv = 1.0 / m["odds"]
        q_mkt = inv / inv.sum()                           # 3連複のオッズから見た確率
        prob = q_mkt * (m["prob"] / a["prob"])            # 市場の値付けを出発点に、モデルと単勝市場の比で直す
        prob = prob / prob.sum()
        r = pd.DataFrame({"combo": m["combo"], "prob": prob, "odds": m["odds"], "ev": prob * m["odds"]})
        r["hit"] = r["combo"] == win
        r["race_id"] = rid
        rows.append(r)
        ll.append((np.log(r.loc[r["hit"], "prob"].iloc[0]), np.log(q_mkt[m["combo"] == win].iloc[0])))
    return pd.concat(rows, ignore_index=True), np.array(ll)


def roi_lines(b):
    codes, _ = pd.factorize(b["race_id"])
    n = codes.max() + 1
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    out = []
    for th in THRESHOLDS:
        sel = (b["ev"] >= th).to_numpy()
        if not sel.any():
            out.append(f"   期待値{th}以上: 0点")
            continue
        stake = np.bincount(codes[sel], minlength=n).astype(float)
        ret = np.bincount(codes[sel], weights=(b["hit"].to_numpy()[sel] * b["odds"].to_numpy()[sel]), minlength=n)
        boots = ret[idx].sum(1) / np.maximum(stake[idx].sum(1), 1) * 100
        out.append(f"   期待値{th}以上: {int(sel.sum()):>7,}点（{sel.sum() / n:5.1f}点/R）的中 {int(b['hit'].to_numpy()[sel].sum()):>4}"
                   f" | 回収率 {ret.sum() / stake.sum() * 100:6.1f}% [{np.percentile(boots, 2.5):6.1f}, {np.percentile(boots, 97.5):6.1f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t7_combine")
    ap.add_argument("--bet", choices=list(BET_TYPES), default="trio")
    ap.add_argument("--years", default="2020")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]
    preds = pd.read_parquet(BENCH_DIR / f"{args.tag}_preds.parquet")
    preds = preds[preds["race_date"].dt.year.isin(years)]
    fin = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos"])
    fin["horse_number"] = fin["horse_number"].astype(float)
    preds = preds.merge(fin, on=["race_id", "horse_number"], how="left")
    odds = load_odds(args.bet, years)
    print(f"[{args.tag}] {BET_NAMES[args.bet]} {years}年（控除率 {TAKEOUT[args.bet] * 100:.1f}%）")
    res = {}
    for col, lam, who in [("p_c", 0.75, "モデル λ=0.75"), ("p_a", 0.75, "オッズだけ λ=0.75"), ("p_c", 1.0, "モデル λ=1.0（参考）")]:
        b, ll = build(preds, odds, args.bet, col, lam)
        res[who] = (b, ll)
        print(f"-- {who}: {b['race_id'].nunique()}R / 実際の組み合わせの対数尤度 {ll.mean():.4f}")
        print("\n".join(roi_lines(b)))
    from model.softmax import bootstrap_ci
    d = res["モデル λ=0.75"][1] - res["オッズだけ λ=0.75"][1]
    lo, hi = bootstrap_ci(d)
    print(f"\n実際の1〜3着の組み合わせの対数尤度 モデル − オッズだけ: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]（{len(d)}R）")
    for lam in (0.75, 1.0):
        b2, ll2 = build_ratio(preds, odds, args.bet, lam)
        d2 = ll2[:, 0] - ll2[:, 1]
        lo, hi = bootstrap_ci(d2)
        print()
        print(f"-- X2 比の方式 λ={lam}: {b2['race_id'].nunique()}R / 対数尤度 − {BET_NAMES[args.bet]}の市場だけ {d2.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        for line in roi_lines(b2):
            print(line)
    # 払戻の確認（払戻があるレースで「当たりのオッズ × 100」と実際の払戻を比べる）
    pay = pd.read_parquet(table_path("payouts"), columns=["race_id", args.bet]).dropna()
    pay["pay"] = pay[args.bet].map(lambda v: v[0] if v is not None and len(v) == 1 else np.nan)
    b = res["モデル λ=0.75"][0]
    hits = b[b["hit"]].drop_duplicates("race_id").merge(pay[["race_id", "pay"]], on="race_id")
    ok = (hits["odds"] * 100).round() == hits["pay"]
    print(f"払戻の確認: 当たりのオッズ×100 と実際の払戻が一致 {ok.sum()}/{len(hits)}R")


if __name__ == "__main__":
    main()
