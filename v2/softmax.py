# v2/softmax.py
# レース内ソフトマックス（条件付きロジット）の共通処理。1行 = 1頭で、同じレースの行が連続している前提。
import numpy as np
from scipy.optimize import minimize


class RaceGroups:
    def __init__(self, race_ids, win):
        race_ids = np.asarray(race_ids)
        self.starts = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1]])
        self.ridx = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(race_ids)]))
        self.y = np.asarray(win, dtype=float)
        if not np.allclose(np.add.reduceat(self.y, self.starts), 1.0):
            raise ValueError("各レースの勝ち馬がちょうど1頭になっていない（行の並び順か同着の除外を確認）")

    def __len__(self):
        return len(self.starts)

    def ll(self, u):
        """レースごとの勝ち馬の対数尤度と、各馬の勝率"""
        u = np.asarray(u, dtype=float)
        m = np.maximum.reduceat(u, self.starts)
        lse = np.log(np.add.reduceat(np.exp(u - m[self.ridx]), self.starts)) + m
        return np.add.reduceat(self.y * u, self.starts) - lse, np.exp(u - lse[self.ridx])


def lgb_objective(groups, base=0.0):
    """LightGBM のカスタム目的関数。base は固定の出発点（オッズ由来の効用など）"""
    def objective(preds, data):
        _, p = groups.ll(base + preds)
        return p - groups.y, np.maximum(p * (1 - p), 1e-6)
    return objective


def lgb_metric(groups, base=0.0):
    def feval(preds, data):
        ll, _ = groups.ll(base + preds)
        return "race_ll", ll.mean(), True
    return feval


def fit_logit(X, groups):
    """条件付きロジットの係数を最尤推定する"""
    X = np.asarray(X, dtype=float).reshape(len(groups.y), -1)

    def nll(beta):
        ll, p = groups.ll(X @ beta)
        return -ll.sum(), -(X.T @ (groups.y - p))

    return minimize(nll, np.zeros(X.shape[1]), jac=True, method="BFGS").x


def bootstrap_ci(values, n=2000, seed=0, level=0.95):
    """レース単位の値の平均の区間（既定は95%）"""
    v = np.asarray(values)
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    means = v[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return np.percentile(means, tail), np.percentile(means, 100 - tail)
