# v2/model/market.py
# オッズから作る列。検証（確定オッズ）と実践（発走前のスナップショット）で同じ関数を使う。
import numpy as np
import pandas as pd


def add_market_cols(d):
    """x_mkt = log(オッズ由来の勝率。1/オッズ をレース内で合計1に直したもの)、mkt_rank = レース内の人気順（1が1番人気）。
    d は1行1頭で、同じレースの全馬に win_odds > 0 がそろっている前提"""
    inv = 1.0 / d["win_odds"]
    d["x_mkt"] = np.log(inv / inv.groupby(d["race_id"]).transform("sum"))
    d["mkt_rank"] = d.groupby("race_id")["x_mkt"].rank(ascending=False, method="min")
    return d


def add_shin(d, iters=60):
    """x_shin = log(Shin の確率)（技術の探索 T8）。Shin (1993) のインサイダーの割合 z をレースごとに二分法で求める
    （Jullien & Salanié 1994 の形）: p_i = (sqrt(z² + 4(1−z)·π_i²/B) − z) / (2(1−z))、π_i = 1/オッズ、B = Σπ、Σp = 1"""
    pi = 1.0 / d["win_odds"].to_numpy(dtype=float)
    rid = d["race_id"].to_numpy()
    codes, _ = pd.factorize(rid)
    B = np.bincount(codes, weights=pi)[codes]
    lo, hi = np.zeros(codes.max() + 1), np.full(codes.max() + 1, 0.5)
    for _ in range(iters):
        z = (lo + hi) / 2
        zr = z[codes]
        p = (np.sqrt(zr ** 2 + 4 * (1 - zr) * pi ** 2 / B) - zr) / (2 * (1 - zr))
        s = np.bincount(codes, weights=p)
        lo, hi = np.where(s > 1, z, lo), np.where(s > 1, hi, z)
    zr = ((lo + hi) / 2)[codes]
    p = (np.sqrt(zr ** 2 + 4 * (1 - zr) * pi ** 2 / B) - zr) / (2 * (1 - zr))
    p = p / np.bincount(codes, weights=p)[codes]
    d["x_shin"] = np.log(p)
    d["shin_z"] = zr
    return d


PLACE_COLS = ["x_place", "place_gap", "place_spread"]


def add_place_cols(d):
    """複勝オッズの列（実験5の mkt_place）。d に place_odds_min / place_odds_max と x_mkt があること。無い馬は NaN"""
    lo = d["place_odds_min"].where(d["place_odds_min"] > 0)
    hi = d["place_odds_max"].where(d["place_odds_max"] > 0)
    d["x_place"] = np.log(1.0 / lo)
    d["place_gap"] = d["x_place"] - d["x_mkt"]
    d["place_spread"] = np.log(hi / lo)
    return d
