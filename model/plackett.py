# model/plackett.py
# 1〜K着の順序を使うレース内の尤度（Plackett-Luce）。1着だけを使う softmax.RaceGroups の拡張で、K=1 なら同じになる。
#   s着目: まだ着順が決まっていない馬（着順 >= s。競走中止は最後まで残る）の中から s着の馬が選ばれる確率
#          = exp(λ_s·u_i) / Σ exp(λ_s·u_j)。λ_1 = 1、λ_2 以降 = lam（2・3着は1着ほど能力どおりに決まらない分の割引。
#          Henery/Stern 型。モデルの勝率で測った最良値は 0.75、ログ fit_lambda.log）
#   s着が同着・存在しないレースは、その段だけ使わない。
# 1行 = 1頭で、同じレースの行が連続している前提（softmax.py と同じ）。
import numpy as np


class PLGroups:
    def __init__(self, race_ids, finish_pos, k=3, lam=0.75, stage_w=None):
        race_ids = np.asarray(race_ids)
        self.starts = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1]])
        self.ridx = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(race_ids)]))
        fp = np.asarray(finish_pos, dtype=float)
        stage_w = [1.0] * k if stage_w is None else list(stage_w)
        self.stages = []
        for s in range(1, k + 1):
            avail = ~(fp < s)                       # 競走中止（NaN）は選ばれないまま残る
            chosen = (fp == s).astype(float)
            ok = (np.add.reduceat(chosen, self.starts) == 1) & (np.add.reduceat(avail * 1.0, self.starts) >= 2)
            mask = avail & ok[self.ridx]
            self.stages.append((mask, chosen * mask, 1.0 if s == 1 else lam, stage_w[s - 1], ok))
        if not self.stages[0][4].all():
            raise ValueError("1着がちょうど1頭でないレースがある（同着の除外か行の並び順を確認）")

    def __len__(self):
        return len(self.starts)

    def _stage(self, u, mask, y, scale, ok):
        v = np.where(mask, scale * u, -np.inf)
        m = np.maximum.reduceat(v, self.starts)
        m = np.where(ok, m, 0.0)
        e = np.exp(v - m[self.ridx])
        z = np.add.reduceat(e, self.starts)
        zr = np.where(ok, z, 1.0)
        p = e / zr[self.ridx]
        ll = np.where(ok, np.add.reduceat(np.where(mask, y * scale * u, 0.0), self.starts) - np.log(zr) - m, 0.0)
        return ll, p

    def ll(self, u):
        """レースごとの 1〜K着の順序の対数尤度（段の重み付きの和）"""
        u = np.asarray(u, dtype=float)
        return sum(w * self._stage(u, mask, y, sc, ok)[0] for mask, y, sc, w, ok in self.stages)

    def grad_hess(self, u):
        u = np.asarray(u, dtype=float)
        g = np.zeros_like(u)
        h = np.zeros_like(u)
        for mask, y, sc, w, ok in self.stages:
            _, p = self._stage(u, mask, y, sc, ok)
            g += w * sc * (p - y)
            h += w * sc * sc * p * (1 - p)
        return g, h


def lgb_objective_pl(groups, base=0.0, row_w=None):
    """LightGBM のカスタム目的関数（負の対数尤度の勾配）。row_w は行ごとの重み（直近重視など）"""
    def objective(preds, data):
        g, h = groups.grad_hess(base + preds)
        if row_w is not None:
            g, h = g * row_w, h * row_w
        return g, np.maximum(h, 1e-6)
    return objective
