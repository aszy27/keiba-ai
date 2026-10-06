# model/exotic.py
# 組み合わせ券（馬連・馬単・3連複・3連単）の確率と期待値。各馬の勝率（候補のモデルの p_c）から Plackett-Luce で着順の確率を作る。
#   P(1着 i, 2着 j, 3着 k) = p_i · q_j / (Σq − q_i) · q_k / (Σq − q_i − q_j)、q = p^λ（馬連・馬単は1・2着の部分だけ）
#   λ < 1 は、1着でない馬が2・3着に来る力を割り引く補正（Henery / Stern 型）。λ = 1 がハービル式。
#   ハービル式は強い馬が2・3着に来る確率を過大評価することが知られている（Benter 1994）。基礎モデルの学習と同じ λ = 0.75 を既定にする。
# 期待値 = 確率 × その組み合わせのオッズ。オッズは API（type 4 = 馬連、6 = 馬単、7 = 3連複、8 = 3連単）の全通りのもの。
import itertools

import numpy as np
import pandas as pd

LAMBDA = 0.75
BET_TYPES = {"quinella": 4, "exacta": 6, "trio": 7, "trifecta": 8}          # API の type
BET_NAMES = {"quinella": "馬連", "exacta": "馬単", "trio": "3連複", "trifecta": "3連単"}
TAKEOUT = {"quinella": 0.225, "exacta": 0.25, "trio": 0.25, "trifecta": 0.275}       # JRA の控除率


def order2_probs(p, lam=LAMBDA):
    """P[i, j] = 1着 i・2着 j の確率（同じ馬は 0）"""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    q = p ** lam
    prob = p[:, None] * q[None, :] / np.maximum(q.sum() - q[:, None], 1e-12)
    np.fill_diagonal(prob, 0.0)
    return prob


def order3_probs(p, lam=LAMBDA):
    """P[i, j, k] = 1着 i・2着 j・3着 k の確率（同じ馬を含む組み合わせは 0）"""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    q = p ** lam
    total = q.sum()
    d1 = np.maximum(total - q[:, None], 1e-12)
    d2 = np.maximum(total - q[:, None, None] - q[None, :, None], 1e-12)
    prob = p[:, None, None] * (q[None, :] / d1)[:, :, None] * (q[None, None, :] / d2)
    n = len(p)
    idx = np.arange(n)
    prob[idx, idx, :] = 0.0
    prob[idx, :, idx] = 0.0
    prob[:, idx, idx] = 0.0
    return prob


def combos(numbers, p, kind, lam=LAMBDA):
    """(組み合わせの文字列の配列, 確率の配列)。馬連・3連複は馬番の昇順、馬単・3連単は着順の順（API の表記と同じ 2桁ずつ）"""
    numbers = np.asarray(numbers).astype(int)
    if kind in ("exacta", "quinella"):
        prob2 = order2_probs(p, lam)
        n = len(numbers)
        i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
        if kind == "exacta":
            ok = i != j
            i, j = i[ok], j[ok]
            return np.array([f"{numbers[a]:02d}{numbers[b]:02d}" for a, b in zip(i, j)]), prob2[i, j]
        ok = i < j
        i, j = i[ok], j[ok]
        names = np.array(["".join(f"{x:02d}" for x in sorted((numbers[a], numbers[b]))) for a, b in zip(i, j)])
        return names, prob2[i, j] + prob2[j, i]
    prob = order3_probs(p, lam)
    n = len(numbers)
    if kind == "trifecta":
        i, j, k = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"))
        ok = (i != j) & (j != k) & (i != k)
        i, j, k = i[ok], j[ok], k[ok]
        return np.array([f"{numbers[a]:02d}{numbers[b]:02d}{numbers[c]:02d}" for a, b, c in zip(i, j, k)]), prob[i, j, k]
    if kind == "trio":
        trip = np.array(list(itertools.combinations(range(n), 3)))
        if not len(trip):
            return np.array([]), np.array([])
        perms = list(itertools.permutations(range(3)))
        pr = sum(prob[trip[:, a], trip[:, b], trip[:, c]] for a, b, c in perms)
        names = np.array(["".join(f"{x:02d}" for x in sorted(numbers[t])) for t in trip])
        return names, pr
    raise ValueError(kind)


def winning_combo(finish, numbers, kind):
    """着順（1〜3着、馬連・馬単は1〜2着がそれぞれ1頭）から当たりの組み合わせ。同着などで決まらなければ None"""
    s = pd.Series(np.asarray(numbers).astype(int), index=np.asarray(finish, dtype=float))
    top = [s.get(float(r)) for r in ((1, 2) if kind in ("exacta", "quinella") else (1, 2, 3))]
    if any(t is None or isinstance(t, pd.Series) for t in top):
        return None
    if kind in ("trifecta", "exacta"):
        return "".join(f"{x:02d}" for x in top)
    return "".join(f"{x:02d}" for x in sorted(top))


def race_ev(numbers, p, odds, kind, lam=LAMBDA):
    """1レースの全組み合わせの (組み合わせ, 確率, オッズ, 期待値)。odds は {組み合わせ: オッズ}（無い組み合わせは除く）"""
    names, pr = combos(numbers, p, kind, lam)
    o = np.array([odds.get(c, np.nan) for c in names], dtype=float)
    ok = np.isfinite(o) & (o > 0)
    return pd.DataFrame({"combo": names[ok], "prob": pr[ok], "odds": o[ok], "ev": pr[ok] * o[ok]})


def race_ev_ratio(numbers, p_model, p_market, odds, kind, lam=LAMBDA):
    """比の方式（docs/rebuild_plan.md の X2）: 期待値 = (モデルの確率 / 単勝オッズだけの確率) × (組み合わせ券の市場の確率 × オッズ)。
    分子・分母を同じ式・同じ λ で作るので式の偏りが打ち消し合い、組み合わせ券の市場の値付けを出発点にする"""
    m = race_ev(numbers, p_model, odds, kind, lam)
    a = race_ev(numbers, p_market, odds, kind, lam)
    if m.empty:
        return m
    inv = 1.0 / m["odds"]
    prob = (inv / inv.sum()) * (m["prob"] / a["prob"])
    prob = prob / prob.sum()
    return pd.DataFrame({"combo": m["combo"], "prob": prob, "odds": m["odds"], "ev": prob * m["odds"]})
