# evaluate_residual_model.py
# オッズを出発点に固定し、市場が見落としている分だけを LightGBM（レース内ソフトマックス）で学習して、
# 「オッズのみ」を上回るかを検証する。最終テスト期間（HOLDOUT_START以降）は --final を付けたときだけ評価する。
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_edge_dataset import OUT_FILE, BASE_FEATURES, BIAS_FEATURES, CHANGE_FEATURES
from core.config import DATA_DIR
from evaluate_odds_edge import Races

FILE_ODDS = str(DATA_DIR / "odds_api_progress.csv")
FILE_RETURN = str(DATA_DIR / "return_data_progress.csv")
POOL_FEATURES = ['x_place', 'win_place_gap', 'place_spread']
HOLDOUT_START = "2026-06-01"
FINAL2_START = "2026-09-07"
MIN_FINAL2_RACES = 1000
SPLITS = {
    'dev':   {'train': ("2024-01-01", "2025-07-01"), 'valid': ("2025-07-01", "2026-01-01"),
              'test': ("2026-01-01", HOLDOUT_START)},
    'smoke': {'train': ("2026-01-01", "2026-04-01"), 'valid': ("2026-04-01", "2026-05-01"),
              'test': ("2026-05-01", HOLDOUT_START)},
    'final': {'train': ("2024-01-01", "2025-10-01"), 'valid': ("2025-10-01", HOLDOUT_START),
              'test': (HOLDOUT_START, "2100-01-01")},
    # 2026-09-15 に事前登録した2回目の最終テスト（CLAUDE.md 参照）。対象期間が MIN_FINAL2_RACES に達するまで評価しない
    'final2': {'train': ("2024-01-01", "2026-03-01"), 'valid': ("2026-03-01", FINAL2_START),
               'test': (FINAL2_START, "2100-01-01")},
    # 2026-09-15 に事前登録した3回目の最終テスト。オッズを一度も見ていない2021〜2023年を使う（CLAUDE.md 参照）
    'final3': {'train': ("2021-01-01", "2022-07-01"), 'valid': ("2022-07-01", "2023-01-01"),
               'test': ("2023-01-01", "2024-01-01")},
}
FEATURE_SETS = {
    '既存特徴量': BASE_FEATURES,
    '当日バイアス': BIAS_FEATURES,
    '既存+当日バイアス': BASE_FEATURES + BIAS_FEATURES,
    '条件変化': CHANGE_FEATURES,
    '全部': BASE_FEATURES + BIAS_FEATURES + CHANGE_FEATURES,
    'プール乖離': POOL_FEATURES,
    '全部+プール乖離': BASE_FEATURES + BIAS_FEATURES + CHANGE_FEATURES + POOL_FEATURES,
}
LGB_PARAMS = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42,
                  metric='None', num_threads=0)
EV_THRESHOLDS = [1.05, 1.1, 1.2, 1.3]
N_BOOT = 2000
rng = np.random.default_rng(0)


def load():
    df = pd.read_pickle(OUT_FILE)
    o = pd.read_csv(FILE_ODDS, dtype={'race_id': str}, on_bad_lines='skip')
    for c in ['horse_number', 'win_odds', 'place_min', 'place_max']:
        o[c] = pd.to_numeric(o[c], errors='coerce')
    o = o[o['horse_number'] > 0].drop_duplicates(['race_id', 'horse_number'])
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
    df = df.merge(o[['race_id', 'horse_number', 'win_odds', 'place_min', 'place_max']],
                  on=['race_id', 'horse_number'], how='left')

    # 1着馬のオッズ×100 が単勝払戻と一致しないレースは、レースか馬番の対応がずれているとみなして除外する
    r = pd.read_csv(FILE_RETURN, dtype=str).drop_duplicates('race_id', keep='last')
    r['tansho'] = pd.to_numeric(r['tansho'].str.split('|').str[0], errors='coerce')
    w = df[df['rank'] == 1].merge(r[['race_id', 'tansho']], on='race_id', how='left')
    # 2023年以前は払戻データが一部のレースにしかないため、払戻があるレースだけ照合する
    checked = w['tansho'].notna()
    pay_ng = set(w.loc[checked & ((w['win_odds'] * 100).round() != w['tansho']), 'race_id'])
    print(f"単勝払戻との照合: {w.loc[checked, 'race_id'].nunique()}R で実施 / 不一致 {len(pay_ng)}R（同着を含む）")
    df['bad'] = ~(df['win_odds'] > 0) | ~(df['place_min'] > 0) | df['race_id'].isin(pay_ng)
    g = df.groupby('race_id')
    ok = ~g['bad'].transform('any') & (g['rank'].transform(lambda s: (s == 1).sum()) == 1)
    n_all = df['race_id'].nunique()
    df = df[ok].copy()
    print(f"オッズ付きで使えるレース: {df['race_id'].nunique()} / {n_all}")
    df['win'] = (df['rank'] == 1).astype(float)
    inv = 1.0 / df['win_odds']
    df['x_mkt'] = np.log(inv / inv.groupby(df['race_id']).transform('sum'))
    # 複勝オッズから見た「3着以内（7頭以下は2着以内）に入る確率」と、単勝オッズから見た勝率とのずれ
    n_places = np.where(df.groupby('race_id')['horse_number'].transform('size') <= 7, 2, 3)
    q = 2.0 / (df['place_min'] + df['place_max'])
    df['x_place'] = np.log(n_places * q / q.groupby(df['race_id']).transform('sum'))
    df['win_place_gap'] = df['x_place'] - df['x_mkt']
    df['place_spread'] = np.log(df['place_max'] / df['place_min'])
    return df.sort_values(['race_date', 'race_id', 'horse_number']).reset_index(drop=True)


def softmax_ll(u, rs):
    m = np.maximum.reduceat(u, rs.starts)
    lse = np.log(np.add.reduceat(np.exp(u - m[rs.ridx]), rs.starts)) + m
    return np.add.reduceat(rs.y * u, rs.starts) - lse, np.exp(u - lse[rs.ridx])


def train_residual(tr, va, feats, alpha):
    rs_tr, rs_va = Races(tr, ['x_mkt']), Races(va, ['x_mkt'])
    base_tr, base_va = alpha * tr['x_mkt'].values, alpha * va['x_mkt'].values

    def objective(preds, data):
        _, p = softmax_ll(base_tr + preds, rs_tr)
        return p - rs_tr.y, np.maximum(p * (1 - p), 1e-6)

    def feval(preds, data):
        ll, _ = softmax_ll(base_va + preds, rs_va)
        return 'race_ll', ll.mean(), True

    dtr = lgb.Dataset(tr[feats].values.astype(np.float32), label=tr['win'].values, feature_name=feats)
    dva = lgb.Dataset(va[feats].values.astype(np.float32), label=va['win'].values, reference=dtr)
    params = dict(LGB_PARAMS, objective=objective)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], feval=feval,
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def ev_table(te, rs, p):
    odds, win = te['win_odds'].values, te['win'].values
    ev = p * odds
    for th in EV_THRESHOLDS:
        sel = ev >= th
        if sel.sum() == 0:
            print(f"   期待値{th:.2f}以上: 0点")
            continue
        stake = np.add.reduceat(sel * 100.0, rs.starts)
        ret = np.add.reduceat(sel * win * odds * 100.0, rs.starts)
        idx = rng.integers(0, len(stake), (N_BOOT, len(stake)))
        boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
        print(f"   期待値{th:.2f}以上: {int(sel.sum()):>5}点 的中{int((sel * win).sum()):>4} "
              f"回収率 {ret.sum() / stake.sum() * 100:6.1f}% [95%区間 {np.percentile(boots, 2.5):5.1f}〜"
              f"{np.percentile(boots, 97.5):5.1f}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=list(SPLITS))
    ap.add_argument('--sets', default=','.join(FEATURE_SETS), help="評価する特徴量セット（カンマ区切り）")
    args = ap.parse_args()
    if args.split in ('final', 'final2', 'final3'):
        print("*** 最終テスト期間を評価します。この結果を見て特徴量や設定を変えないこと ***")

    df = load()
    sp = SPLITS[args.split]
    part = {k: df[(df['race_date'] >= s) & (df['race_date'] < e)] for k, (s, e) in sp.items()}
    for k, v in part.items():
        print(f"{k:<5} {sp[k][0]}〜{sp[k][1]}: {v['race_id'].nunique()}R")
    if args.split == 'final2' and part['test']['race_id'].nunique() < MIN_FINAL2_RACES:
        print(f"最終テスト2の対象レースが {MIN_FINAL2_RACES}R に達していないため評価しません。")
        return

    alpha, _ = Races(part['train'], ['x_mkt']).fit()
    rs_te = Races(part['test'], ['x_mkt'])
    base_ll, base_p = softmax_ll(alpha[0] * part['test']['x_mkt'].values, rs_te)
    print(f"\nオッズのみ（係数 {alpha[0]:.3f}）: test 1レースあたり対数尤度 {base_ll.mean():.4f}")
    print("-- オッズのみで期待値ベースに単勝を買った場合")
    ev_table(part['test'], rs_te, base_p)

    for name in args.sets.split(','):
        feats = FEATURE_SETS[name]
        bst = train_residual(part['train'], part['valid'], feats, alpha[0])
        u = alpha[0] * part['test']['x_mkt'].values + bst.predict(part['test'][feats].values.astype(np.float32))
        ll, p = softmax_ll(u, rs_te)
        d = ll - base_ll
        idx = rng.integers(0, len(d), (N_BOOT, len(d)))
        boots = d[idx].mean(axis=1)
        print(f"\n=== {name}（{len(feats)}特徴量, 木 {bst.best_iteration}本） ===")
        print(f"対数尤度 {ll.mean():.4f} / オッズのみとの差 {d.mean():+.4f} "
              f"[95%区間 {np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")
        imp = pd.Series(bst.feature_importance('gain'), index=feats).sort_values(ascending=False)
        print("重要度(gain)上位:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items() if v > 0) or "なし")
        ev_table(part['test'], rs_te, p)


if __name__ == "__main__":
    main()
