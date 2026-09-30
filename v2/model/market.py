# v2/model/market.py
# オッズから作る列。検証（確定オッズ）と実践（発走前のスナップショット）で同じ関数を使う。
import numpy as np


def add_market_cols(d):
    """x_mkt = log(オッズ由来の勝率。1/オッズ をレース内で合計1に直したもの)、mkt_rank = レース内の人気順（1が1番人気）。
    d は1行1頭で、同じレースの全馬に win_odds > 0 がそろっている前提"""
    inv = 1.0 / d["win_odds"]
    d["x_mkt"] = np.log(inv / inv.groupby(d["race_id"]).transform("sum"))
    d["mkt_rank"] = d.groupby("race_id")["x_mkt"].rank(ascending=False, method="min")
    return d
