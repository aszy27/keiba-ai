# evaluate_odds_edge.py
# モデルのスコアが確定単勝オッズ以上の情報を持つかを、レース内の条件付きロジットで検証する。
# 入力: result/backtest_horses.csv (evaluate_dump.py) と data/odds_data_progress.csv (scrape/scrape_odds.py)
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.config import BASE_DIR

FILE_HORSES = os.path.join(BASE_DIR, "result", "backtest_horses.csv")
FILE_ODDS = os.path.join(BASE_DIR, "data", "odds_data_progress.csv")
SPLIT_DATE = "2026-06-01"
N_BOOT = 2000
EV_THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.5]
SPECS = {
    'オッズのみ': ['x_mkt'],
    'モデルのみ': ['z_final_score'],
    'オッズ+final_score': ['x_mkt', 'z_final_score'],
    'オッズ+LGB2種+Transformer': ['x_mkt', 'z_lgb_rank_score', 'z_lgb_prob_score', 'z_transformer_prob'],
}
rng = np.random.default_rng(0)


def load():
    h = pd.read_csv(FILE_HORSES, dtype={'race_id': str, 'horse_id': str})
    o = pd.read_csv(FILE_ODDS, dtype=str)
    o = o[o['horse_number'] != 'NO_DATA'].copy()
    for c in ['horse_number', 'win_odds', 'popularity']:
        o[c] = pd.to_numeric(o[c], errors='coerce')
    h['horse_number'] = pd.to_numeric(h['horse_number'], errors='coerce')

    df = h.merge(o[['race_id', 'horse_number', 'horse_id', 'win_odds', 'popularity']],
                 on=['race_id', 'horse_number'], how='left', suffixes=('', '_odds'))
    matched = df['horse_id_odds'].notna()
    print(f"オッズと結合できた馬: {matched.mean() * 100:.1f}% "
          f"(うち馬IDも一致: {(df.loc[matched, 'horse_id'] == df.loc[matched, 'horse_id_odds']).mean() * 100:.1f}%)")

    df['bad'] = ~(df['win_odds'] > 0) | (df['horse_id'] != df['horse_id_odds'])
    g = df.groupby('race_id')
    ok = ~g['bad'].transform('any') & (g['rank'].transform(lambda s: (s == 1).sum()) == 1)
    n_all = df['race_id'].nunique()
    df = df[ok].copy()
    print(f"検証に使うレース: {df['race_id'].nunique()} / {n_all}（オッズ欠損・馬ID不一致・同着のレースを除外）")

    df['race_date'] = pd.to_datetime(df['race_date'])
    df['win'] = (df['rank'] == 1).astype(float)
    inv = 1.0 / df['win_odds']
    df['p_mkt'] = inv / inv.groupby(df['race_id']).transform('sum')
    df['x_mkt'] = np.log(df['p_mkt'])
    for c in ['final_score', 'lgb_rank_score', 'lgb_prob_score', 'transformer_prob']:
        grp = df.groupby('race_id')[c]
        df[f'z_{c}'] = ((df[c] - grp.transform('mean')) / grp.transform('std').replace(0, 1)).fillna(0)
    return df.sort_values(['race_id', 'horse_number']).reset_index(drop=True)


class Races:
    def __init__(self, df, cols):
        self.df = df
        self.X = df[cols].values.astype(float)
        self.y = df['win'].values
        codes = df['race_id'].values
        self.starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        self.ridx = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(codes)]))

    def race_ll(self, beta):
        u = self.X @ beta
        m = np.maximum.reduceat(u, self.starts)
        lse = np.log(np.add.reduceat(np.exp(u - m[self.ridx]), self.starts)) + m
        p = np.exp(u - lse[self.ridx])
        return np.add.reduceat(self.y * u, self.starts) - lse, p

    def nll_grad(self, beta):
        ll, p = self.race_ll(beta)
        return -ll.sum(), -(self.X.T @ (self.y - p))

    def fit(self):
        res = minimize(self.nll_grad, np.zeros(self.X.shape[1]), jac=True, method='BFGS')
        return res.x, np.sqrt(np.diag(res.hess_inv))


def boot_mean_ci(v):
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    s = v[idx].mean(axis=1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    df = load()
    tr, te = df[df['race_date'] < SPLIT_DATE], df[df['race_date'] >= SPLIT_DATE]
    print(f"学習: {tr['race_id'].nunique()}R (〜{SPLIT_DATE}) / 検証: {te['race_id'].nunique()}R ({SPLIT_DATE}〜)")

    print("\n=== 1. 勝ち馬の予測精度（検証期間の1レースあたり対数尤度。大きいほど良い） ===")
    fitted, test_ll = {}, {}
    for name, cols in SPECS.items():
        beta, se = Races(tr, cols).fit()
        ll, _ = Races(te, cols).race_ll(beta)
        fitted[name], test_ll[name] = beta, ll
        coef = ", ".join(f"{c}={b:+.3f}(±{s:.3f})" for c, b, s in zip(cols, beta, se))
        print(f"{name:<26} 対数尤度 {ll.mean():.4f} | 係数 {coef}")

    base = test_ll['オッズのみ']
    print("\nオッズのみ との差（正ならモデルがオッズ以上の情報を持つ。95%区間はレース単位のブートストラップ）")
    for name in SPECS:
        if name == 'オッズのみ':
            continue
        d = test_ll[name] - base
        lo, hi = boot_mean_ci(d)
        print(f"  {name:<26} {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print("\n=== 2. 期待値ベースの単勝購入（検証期間・確定オッズで計算するため実戦より甘く出る） ===")
    odds, win = te['win_odds'].values, te['win'].values
    for name in ['オッズのみ', 'オッズ+final_score', 'オッズ+LGB2種+Transformer']:
        rs = Races(te, SPECS[name])
        _, p = rs.race_ll(fitted[name])
        ev = p * odds
        print(f"-- {name}")
        for th in EV_THRESHOLDS:
            sel = ev >= th
            if sel.sum() == 0:
                continue
            stake = np.add.reduceat(sel * 100.0, rs.starts)
            ret = np.add.reduceat(sel * win * odds * 100.0, rs.starts)
            roi = ret.sum() / stake.sum() * 100
            idx = rng.integers(0, len(stake), (N_BOOT, len(stake)))
            boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
            print(f"   期待値{th:.1f}以上: {int(sel.sum()):>5}点 的中{int((sel * win).sum()):>4} 回収率 {roi:6.1f}% "
                  f"[95%区間 {np.percentile(boots, 2.5):5.1f}〜{np.percentile(boots, 97.5):5.1f}%]")

    print("\n=== 参考: 人気別の単勝回収率（検証期間・全馬を買った場合） ===")
    t = te.assign(ret=te['win'] * te['win_odds'] * 100).groupby(te['popularity'].clip(upper=10))
    print(pd.DataFrame({'頭数': t.size(), '勝率%': t['win'].mean() * 100,
                        '回収率%': t['ret'].sum() / (t.size() * 100) * 100}).round(1).to_string())


if __name__ == "__main__":
    main()
