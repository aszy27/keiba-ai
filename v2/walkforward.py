# v2/walkforward.py
# 年ごとの前進検証。各年 Y を「その年を学習していないモデル」で採点し、単勝・複勝の回収率を年別と通算で出す。
#   基礎モデル: 年ごとのOOS予測（学習 2013〜Y-2 / early stopping Y-1、3シード平均）
#   オッズ結合: 残差モデルを 学習 [Y-4, Y-1) / early stopping [Y-1, Y) で学習し、Y年で採点
#   券種: 単勝（確定単勝オッズ）と複勝（確定複勝オッズの下限。判断も回収も下限なので辛めに出る）
#   対照: オッズだけから作った確率（p_a）でも同じ買い方をして並べる
# 2026-09-07以降（前向き検証の期間）は使わない。
#
# 使い方: python -m v2.walkforward --build-base   # 2024〜2026年の基礎モデル予測を作る（初回だけ）
#         python -m v2.walkforward
import argparse

import numpy as np
import pandas as pd

from v2 import exotic_model as em
from v2 import model_trip as mt
from v2.model_base import PARAMS, eligible
from v2.paths import V2_DIR, table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit

YEARS = range(2019, 2027)          # 基礎モデルの予測が 2015年からあるので、残差の学習4年分を取ると2019年から
END = "2026-09-07"                 # 前向き検証の期間は触らない
EXTRA_BASE = V2_DIR / "base_oos_all_2024_2026.parquet"
COMBINED_BASE = V2_DIR / "base_oos_all_2015_2026.parquet"
THRESHOLDS = [1.0, 1.1, 1.2, 1.5]


def build_extra_base(df, years=(2024, 2025, 2026)):
    """model_trip.build_base と同じ作り方で 2024〜2026年分を足す"""
    feats = mt.FEATURE_SETS["all"]
    out = []
    for year in years:
        tr = df[(df["race_date"] >= "2013-01-01") & (df["race_date"] < f"{year - 1}-01-01")].reset_index(drop=True)
        va = df[(df["race_date"] >= f"{year - 1}-01-01") & (df["race_date"] < f"{year}-01-01")].reset_index(drop=True)
        te = df[(df["race_date"] >= f"{year}-01-01") & (df["race_date"] < f"{year + 1}-01-01")].reset_index(drop=True)
        u_va, u_te, rounds = 0.0, 0.0, []
        for seed in mt.SEEDS:
            m = mt._train(tr, va, feats, params=dict(PARAMS, seed=seed))
            u_va = u_va + m.predict(va[feats], num_iteration=m.best_iteration) / len(mt.SEEDS)
            u_te = u_te + m.predict(te[feats], num_iteration=m.best_iteration) / len(mt.SEEDS)
            rounds.append(m.best_iteration)
        beta = fit_logit(u_va, RaceGroups(va["race_id"], va["win"]))[0]
        out.append(te[["race_id", "horse_id"]].assign(u_base=beta * u_te))
        print(f"[all] {year}: 学習 {tr['race_id'].nunique()}R / 木 {rounds}本 / 温度 {beta:.3f}", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(EXTRA_BASE, index=False)
    combined = pd.concat([pd.read_parquet(mt.base_path("all")), pd.read_parquet(EXTRA_BASE)], ignore_index=True)
    combined.to_parquet(COMBINED_BASE, index=False)
    print(f"基礎モデルの予測 {len(combined)}行 → {COMBINED_BASE}", flush=True)


def place_bets(preds, odds_map, col):
    """複勝の（確率・オッズ下限・当たりか）"""
    rows = []
    for rid, g in preds.groupby("race_id", sort=False):
        odds = odds_map.get(rid)
        if odds is None:
            continue
        g = g.sort_values("horse_number")
        p = g[col].to_numpy()
        p = p / p.sum()
        combos, prob = em.race_combos(g["horse_number"].to_numpy(), p, "place")
        win_combo = em.winning_combo(g, "place")
        if win_combo is None:
            continue
        o = np.array([odds.get(c, np.nan) for c in combos])
        ok = np.isfinite(o) & (o > 0)
        rows.append(pd.DataFrame({"race_id": rid, "prob": prob[ok], "odds": o[ok],
                                  "hit": np.array([c in win_combo for c in combos])[ok]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def win_bets(preds, col):
    return pd.DataFrame({"race_id": preds["race_id"], "prob": preds[col], "odds": preds["win_odds"],
                         "hit": preds["win"] == 1})


def roi(bets, th, n_boot=2000):
    ev = (bets["prob"] * bets["odds"]).to_numpy()
    sel = ev >= th
    if sel.sum() == 0:
        return 0, 0, np.nan, (np.nan, np.nan)
    codes, _ = pd.factorize(bets["race_id"])
    n = codes.max() + 1
    stake = np.bincount(codes[sel], minlength=n).astype(float)
    ret = np.bincount(codes[sel], weights=(bets["hit"].to_numpy()[sel] * bets["odds"].to_numpy()[sel]), minlength=n)
    idx = np.random.default_rng(0).integers(0, n, (n_boot, n))
    boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
    return int(sel.sum()), int(bets["hit"].to_numpy()[sel].sum()), ret.sum() / stake.sum() * 100, \
        (np.percentile(boots, 2.5), np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-base", action="store_true")
    args = ap.parse_args()
    df = eligible(pd.read_parquet(table_path("features")))
    if args.build_base:
        build_extra_base(df)
        return

    orig = mt.base_path
    mt.base_path = lambda name: COMBINED_BASE if name == "all" else orig(name)
    market, _ = mt.with_market(df, "all")
    place_odds = em.load_odds("place", list(YEARS))

    all_preds, diffs = [], {}
    for year in YEARS:
        end = min(f"{year + 1}-01-01", END)
        split = {"train": (f"{year - 4}-01-01", f"{year - 1}-01-01"), "valid": (f"{year - 1}-01-01", f"{year}-01-01"),
                 "report": (f"{year}-01-01", end)}
        preds, diff = mt.run_split(market, "all", split)
        all_preds.append(preds.assign(year=year))
        diffs[year] = diff

    preds = pd.concat(all_preds, ignore_index=True)
    print(f"\n採点に使ったレース: {preds['race_id'].nunique():,}R（{min(YEARS)}〜{max(YEARS)}年）")

    for bet, maker in [("単勝", lambda d, c: win_bets(d, c)), ("複勝", lambda d, c: place_bets(d, place_odds, c))]:
        for col, who in [("p_c", "モデル"), ("p_a", "オッズのみ（対照）")]:
            print(f"\n=== {bet} / {who} ===")
            bets_all = []
            for year in YEARS:
                b = maker(preds[preds["year"] == year], col)
                bets_all.append(b)
                line = f"  {year}年 "
                for th in THRESHOLDS:
                    n, hit, r, _ = roi(b, th)
                    line += f"| {th:.1f}以上 {n:>5}点 {r:6.1f}% " if n else f"| {th:.1f}以上     0点   ---   "
                print(line, flush=True)
            pooled = pd.concat(bets_all, ignore_index=True)
            for th in THRESHOLDS:
                n, hit, r, (lo, hi) = roi(pooled, th)
                if n:
                    print(f"  通算 期待値{th:.1f}以上: {n:>7,}点 的中 {hit:>5} | 回収率 {r:6.1f}% [{lo:6.1f}, {hi:6.1f}]"
                          f" | ×0.95 {r * 0.95:6.1f}% | ×0.90 {r * 0.9:6.1f}%", flush=True)

    print("\n=== 対数尤度（C − A）の年別 ===")
    for year, d in diffs.items():
        lo, hi = bootstrap_ci(d)
        print(f"  {year}年 {len(d):>5}R  {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pooled = np.concatenate(list(diffs.values()))
    lo, hi = bootstrap_ci(pooled)
    print(f"  通算  {len(pooled):>5}R  {pooled.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
