# experiments/ev_odds_bands.py
# 期待値が同じなら、オッズの高低によらず回収率も同じになっているかを調べる（2026-10-20。記述的な点検で、候補の設定は変えない）。
#   候補2と同じ作り方の前進検証の予測（t7_combine、2019〜2026-09-06）で、期待値（勝率×確定オッズ）の帯 × オッズの帯ごとに
#   単勝の回収率と「回収率 ÷ 平均期待値」（1なら期待値どおり）を出す。比較にオッズのみ（p_a）も並べる。
#   区間はレース単位のブートストラップ。同着のレースは除く。
#
# 使い方: python -m experiments.ev_odds_bands
import numpy as np
import pandas as pd

from paths import V2_DIR

PREDS = V2_DIR / "bench" / "t7_combine_preds.parquet"
END = "2026-09-07"
EV_EDGES = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, np.inf]
ODDS_EDGES = [1.0, 3.0, 6.0, 12.0, 30.0, 100.0, np.inf]
PERIODS = {"2019〜2022": ("2019-01-01", "2023-01-01"), "2023〜2026-09-06": ("2023-01-01", END)}


def load():
    p = pd.read_parquet(PREDS)
    p = p[p["race_date"] < END]
    wins = p.groupby("race_id")["win"].sum()
    return p[p["race_id"].isin(wins[wins == 1].index)].reset_index(drop=True)


def bands(x, edges):
    labels = [f"{a:g}〜{b:g}" if np.isfinite(b) else f"{a:g}〜" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(x, edges, labels=labels, right=False)


def roi_ci(b, n=2000, seed=0):
    """買い目 b（1点1単位）の回収率と、レース単位で引き直した95%区間"""
    g = b.assign(ret=b["win"] * b["win_odds"]).groupby("race_id").agg(stake=("win", "size"), ret=("ret", "sum"))
    stake, ret = g["stake"].to_numpy(), g["ret"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    boots = ret[idx].sum(axis=1) / stake[idx].sum(axis=1) * 100
    return ret.sum() / stake.sum() * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def table(p, prob, ev_edges=EV_EDGES):
    d = p.assign(ev=p[prob] * p["win_odds"])
    d = d[d["ev"] >= ev_edges[0]]
    d["期待値"] = bands(d["ev"], ev_edges)
    d["オッズ"] = bands(d["win_odds"], ODDS_EDGES)
    rows = []
    for (e, o), b in d.groupby(["期待値", "オッズ"], observed=True):
        r, lo, hi = roi_ci(b)
        rows.append(dict(期待値=e, オッズ=o, 点数=len(b), 的中=int(b["win"].sum()), 平均期待値=b["ev"].mean(),
                         回収率=r, 下限=lo, 上限=hi, 実現率=r / 100 / b["ev"].mean()))
    return pd.DataFrame(rows)


def show(t, title):
    print(f"\n=== {title} ===")
    print(t.to_string(index=False, formatters={"平均期待値": "{:.3f}".format, "回収率": "{:.1f}".format,
                                               "下限": "{:.1f}".format, "上限": "{:.1f}".format, "実現率": "{:.2f}".format}))


def main():
    p = load()
    print(f"対象: {p['race_id'].nunique()}R / {len(p)}頭（{p['race_date'].min():%Y-%m-%d}〜{p['race_date'].max():%Y-%m-%d}、同着を除く）")
    pd.set_option("display.width", 200)
    show(table(p, "p_c"), "候補2の作り方（p_c）: 期待値の帯 × オッズの帯")
    show(table(p, "p_c", [1.0, np.inf]), "候補2の作り方（p_c）: 期待値1.0以上をまとめてオッズの帯ごと")
    show(table(p, "p_c", [1.3, np.inf]), "候補2の作り方（p_c）: 期待値1.3以上（候補2の買い方）をオッズの帯ごと")
    for name, (s, e) in PERIODS.items():
        q = p[(p["race_date"] >= s) & (p["race_date"] < e)]
        show(table(q, "p_c", [1.0, np.inf]), f"{name}: p_c 期待値1.0以上をオッズの帯ごと")
    show(table(p, "p_a"), "比較: オッズのみ（p_a）: 期待値の帯 × オッズの帯")


if __name__ == "__main__":
    main()
