# v2/experiments/exotic_edge.py
# 券種間の値付けのずれの検証。単勝オッズから各馬の勝率を出し、ハービル式で馬連・馬単・3連複・3連単の全組み合わせの確率を計算する。
# 「その確率の帯（または組み合わせの荒れ度合い）に入る組み合わせを全部100円ずつ買う」ルールの回収率を、
# 実際の払戻から計算する。回収率に要るのは的中した組み合わせの払戻だけなので、今あるデータで計算できる。
#
# 期間: 開発 2021〜2023年（払戻は一部のレースのみ）/ 最終テスト 2024〜2025年（docs/rebuild_plan.md「券種間の値付けのずれ」）。
# 使い方: python -m v2.experiments.exotic_edge                    # 開発期間
#         python -m v2.experiments.exotic_edge --period test --final   # 最終テスト（登録した1つのルールを決めてから1回だけ）
import argparse

import numpy as np
import pandas as pd

from v2.paths import table_path
from v2.model.softmax import RaceGroups, fit_logit

PERIODS = {"dev": ("2021-01-01", "2024-01-01"), "test": ("2024-01-01", "2026-01-01")}
BET_TYPES = {"quinella": "馬連", "exacta": "馬単", "trio": "3連複", "trifecta": "3連単"}
P_EDGES = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]           # 組み合わせの確率の帯の境目
POP_EDGES = [3, 6, 9, 12]                                    # 組み合わせ内で一番人気の低い馬の人気の境目
N_BOOT = 2000


def load(period):
    s, e = PERIODS[period]
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
    races = races[(races["race_date"] >= s) & (races["race_date"] < e)]
    run = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos", "status"])
    run = run[run["race_id"].isin(races["race_id"]) & (run["status"] != "scratched")]
    odds = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds"])
    d = run.merge(odds, on=["race_id", "horse_number"], how="left")
    d = d.merge(races, on="race_id").sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)

    g = d.groupby("race_id")
    ok = g["win_odds"].transform(lambda s: (s > 0).all()) & (g["horse_number"].transform("size") >= 5)
    top = d[d["finish_pos"].isin([1, 2, 3])].groupby("race_id")["finish_pos"].agg(lambda s: sorted(s.tolist()) == [1, 2, 3])
    ok &= d["race_id"].map(top).fillna(False).astype(bool)            # 1〜3着に同着が無いレースだけ
    d = d[ok].reset_index(drop=True)
    pay = pd.read_parquet(table_path("payouts"), columns=["race_id", "win"] + list(BET_TYPES))
    pay = pay[pay["race_id"].isin(d["race_id"])].set_index("race_id")
    return d, pay


def market_alpha():
    """開発期間の勝ち馬で log(オッズ由来の勝率) の係数を推定する（テストでも同じ値を使う）"""
    d, _ = load("dev")
    d = d[d.groupby("race_id")["finish_pos"].transform(lambda s: (s == 1).sum()) == 1]
    inv = 1.0 / d["win_odds"]
    x = np.log(inv / inv.groupby(d["race_id"]).transform("sum")).values
    return fit_logit(x, RaceGroups(d["race_id"], (d["finish_pos"] == 1).fillna(False).astype(float)))[0]


_COMBOS = {}


def combos(n):
    if n not in _COMBOS:
        r = np.arange(n)
        pairs = np.array(np.meshgrid(r, r, indexing="ij")).reshape(2, -1).T
        exacta = pairs[pairs[:, 0] != pairs[:, 1]]
        triples = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
        trifecta = triples[(triples[:, 0] != triples[:, 1]) & (triples[:, 0] != triples[:, 2]) & (triples[:, 1] != triples[:, 2])]
        _COMBOS[n] = {"quinella": exacta[exacta[:, 0] < exacta[:, 1]], "exacta": exacta,
                      "trio": trifecta[(trifecta[:, 0] < trifecta[:, 1]) & (trifecta[:, 1] < trifecta[:, 2])],
                      "trifecta": trifecta}
    return _COMBOS[n]


def harville(p, bet, idx):
    i, j = idx[:, 0], idx[:, 1]
    if bet == "exacta":
        return p[i] * p[j] / (1 - p[i])
    if bet == "quinella":
        return p[i] * p[j] / (1 - p[i]) + p[j] * p[i] / (1 - p[j])
    k = idx[:, 2]

    def ordered(a, b, c):
        return p[a] * p[b] / (1 - p[a]) * p[c] / (1 - p[a] - p[b])
    if bet == "trifecta":
        return ordered(i, j, k)
    return sum(ordered(*perm) for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)])


def single(v):
    return float(v[0]) if v is not None and len(v) == 1 else np.nan


def accumulate(d, pay, alpha):
    """レース × 帯 ごとの 購入点数・払戻・的中数・確率の合計 を券種ごとに作る"""
    n_p, n_pop = len(P_EDGES) + 1, len(POP_EDGES) + 1
    acc = {bet: {kind: np.zeros((0, 4, n)) for kind, n in [("prob", n_p), ("pop", n_pop)]} for bet in list(BET_TYPES) + ["win"]}
    rows = {bet: {"prob": [], "pop": []} for bet in acc}
    for rid, r in d.groupby("race_id", sort=False):
        if rid not in pay.index:
            continue
        odds = r["win_odds"].values
        q = (1.0 / odds) ** alpha
        p = q / q.sum()
        pop = pd.Series(odds).rank(method="min").values          # 確定オッズの順位＝人気
        pos = r["finish_pos"].fillna(0).to_numpy(dtype=int)     # 競走中止は着順なし（0）
        first, second, third = (int(np.flatnonzero(pos == k)[0]) for k in (1, 2, 3))
        winners = {"quinella": sorted([first, second]), "exacta": [first, second],
                   "trio": sorted([first, second, third]), "trifecta": [first, second, third]}
        for bet in acc:
            if bet == "win":
                idx, prob, target, payout = np.arange(len(p))[:, None], p, [first], odds[first] * 100
            else:
                payout = single(pay.at[rid, bet])
                if np.isnan(payout):
                    continue
                idx = combos(len(p))[bet]
                prob, target = harville(p, bet, idx), winners[bet]
            hit = (idx == np.array(target)).all(axis=1)
            for kind, cat in [("prob", np.digitize(prob, P_EDGES)), ("pop", np.digitize(pop[idx].max(axis=1), POP_EDGES, right=True))]:
                n_cat = len(P_EDGES) + 1 if kind == "prob" else len(POP_EDGES) + 1
                m = np.zeros((4, n_cat))
                m[0] = np.bincount(cat, minlength=n_cat)
                m[1] = np.bincount(cat, weights=hit * payout, minlength=n_cat)
                m[2] = np.bincount(cat, weights=hit.astype(float), minlength=n_cat)
                m[3] = np.bincount(cat, weights=prob, minlength=n_cat)
                rows[bet][kind].append(m)
    return {bet: {kind: np.stack(v) for kind, v in kinds.items() if v} for bet, kinds in rows.items()}


def labels(kind):
    if kind == "prob":
        e = ["0"] + [f"{x:g}" for x in P_EDGES] + ["1"]
        return [f"確率 {e[i]}〜{e[i + 1]}" for i in range(len(e) - 1)]
    e = [0] + POP_EDGES + [18]
    return [f"最低人気 {e[i] + 1}〜{e[i + 1]}番人気" for i in range(len(e) - 1)]


def report(acc):
    rng = np.random.default_rng(0)
    for bet, name in list(BET_TYPES.items()) + [("win", "単勝（参考）")]:
        for kind, m in acc.get(bet, {}).items():
            n_races = m.shape[0]
            idx = rng.integers(0, n_races, (N_BOOT, n_races))
            print(f"\n=== {name} {n_races}R / {'組み合わせの確率（ハービル式）' if kind == 'prob' else '組み合わせ内の一番人気の低い馬'}ごと ===")
            print(f"{'帯':<22} {'購入点数':>10} {'的中':>6} {'見込み的中':>9} {'回収率':>7}  95%区間")
            for b, label in enumerate(labels(kind)):
                stake, ret = m[:, 0, b], m[:, 1, b]
                if stake.sum() == 0:
                    continue
                boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1)
                print(f"{label:<22} {int(stake.sum()):>10} {int(m[:, 2, b].sum()):>6} {m[:, 3, b].sum():>9.1f} "
                      f"{ret.sum() / stake.sum():>6.1f}%  [{np.percentile(boots, 2.5):5.1f}, {np.percentile(boots, 97.5):5.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=list(PERIODS))
    ap.add_argument("--final", action="store_true", help="最終テスト期間を評価する（登録したルールを決めてから1回だけ）")
    args = ap.parse_args()
    if args.period == "test" and not args.final:
        print("最終テスト期間は --final を付けたときだけ評価する（docs/rebuild_plan.md「券種間の値付けのずれ」）")
        return

    alpha = market_alpha()
    d, pay = load(args.period)
    print(f"期間 {PERIODS[args.period][0]}〜{PERIODS[args.period][1]} / オッズと着順がそろうレース {d['race_id'].nunique()}R"
          f"（うち払戻あり {d['race_id'].isin(pay.index).groupby(d['race_id']).first().sum()}R）/ 勝率の係数 alpha={alpha:.3f}")
    report(accumulate(d, pay, alpha))


if __name__ == "__main__":
    main()
