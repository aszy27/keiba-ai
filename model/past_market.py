# model/past_market.py
# その馬の過去の走の市場評価（確定オッズ）から作る列（docs/rebuild_plan.md「実験6」）。
# 過去の走はすべて「その日より前」のレースだけから作るので、レース前に分かる情報（同じ日に同じ馬は走らない）。
#   past_x1 / past_x2  : 前走・2走前のオッズ由来の勝率（対数。x_mkt と同じ作り方）
#   past_x_mean3       : 過去3走の past_x の平均
#   past_rank1         : 前走の人気順
#   past_gap1          : 前走の 人気順 − 着順（正 = 人気より上の着順に来た）
#   past_gap_mean3     : 過去3走の 人気順 − 着順 の平均
#   past_dx            : 今回の x_mkt − past_x1（今回のオッズが要るので残差の入力にだけ使う）
import numpy as np
import pandas as pd

from paths import table_path

PAST_COLS = ["past_x1", "past_x2", "past_x_mean3", "past_rank1", "past_gap1", "past_gap_mean3"]
PAST_DIFF = ["past_dx"]


def from_runs(runs):
    """runs: race_id, horse_id, race_date, finish_pos, win_odds（1行1頭。取消は除いておく）。
    レース内でオッズがそろわない馬は NaN のまま、その馬の過去の値として扱う"""
    r = runs.copy()
    odds = r["win_odds"].where(r["win_odds"] > 0)
    inv = 1.0 / odds
    r["x"] = np.log(inv / inv.groupby(r["race_id"]).transform("sum"))
    r["rank"] = r.groupby("race_id")["x"].rank(ascending=False, method="min")
    r["gap"] = r["rank"] - r["finish_pos"].astype(float)
    r = r.sort_values(["horse_id", "race_date", "race_id"]).reset_index(drop=True)
    g = r.groupby("horse_id", sort=False)
    prev = {c: [g[c].shift(i) for i in (1, 2, 3)] for c in ("x", "gap")}
    out = r[["race_id", "horse_id"]].copy()
    out["past_x1"], out["past_x2"] = prev["x"][0], prev["x"][1]
    out["past_x_mean3"] = pd.concat(prev["x"], axis=1).mean(axis=1)
    out["past_rank1"] = g["rank"].shift(1)
    out["past_gap1"] = prev["gap"][0]
    out["past_gap_mean3"] = pd.concat(prev["gap"], axis=1).mean(axis=1)
    return out


def load_table(extra_runs=None):
    """全レース（取消を除く）の過去の走の市場評価。extra_runs（race_id, horse_id, race_date）に発走前のレースを足すと、
    そのレースの行も作られる（値はその日より前の走だけから作るので、そのレース自身のオッズ・着順は要らない）"""
    r = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_id", "horse_number", "finish_pos", "status"])
    r = r[r["status"] != "scratched"]
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
    o = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds"])
    r = r.merge(races, on="race_id").merge(o, on=["race_id", "horse_number"], how="left")
    r = r[["race_id", "horse_id", "race_date", "finish_pos", "win_odds"]]
    if extra_runs is not None:
        e = extra_runs[["race_id", "horse_id", "race_date"]].assign(finish_pos=np.nan, win_odds=np.nan)
        r = pd.concat([r[~r["race_id"].isin(e["race_id"])], e], ignore_index=True)
    return from_runs(r)


def add_past_market(d, table=None):
    """d（1行1頭、x_mkt 付き）に過去の走の列を足す。行の並びは変えない"""
    table = load_table() if table is None else table
    out = d.merge(table, on=["race_id", "horse_id"], how="left")
    out["past_dx"] = out["x_mkt"] - out["past_x1"]
    return out
