# experiments/odds_from_features.py
# オッズを使わずに「確定オッズを予測するモデル」を作り、そのオッズで期待値を計算して単勝を買ったら回収率が100%を超えるかを見る（2026-10-09）。
#   買い目はレース前に分かる情報（特徴量）だけで決まるので、回収を実際の確定オッズで数えれば本番と同じ条件の検証になる。
#   オッズ予測: 176列 → 市場の勝率 log(p_mkt)（確定オッズから控除を除いて正規化）を LightGBM で回帰し、レース内で正規化。
#               予測オッズ = 1 / (p_mkt の予測 × 控除の倍率)。控除の倍率は学習期間の sum(1/オッズ) の平均。
#   勝率: 年ごとの基礎モデル（オッズを使わない C[all] の基礎、base_oos_all_2015_2026.parquet）。
#   各年 Y は 学習 2013〜Y-2 / early stopping Y-1 / 採点 Y（基礎モデルと同じ配置）。2026-09-07以降（前向き検証の期間）は使わない。
#
# 使い方: python -m experiments.odds_from_features
import lightgbm as lgb
import numpy as np
import pandas as pd

from model.pipeline import CURRENT_BASE, END, FEATS, load
from model.softmax import bootstrap_ci
from paths import table_path

YEARS = range(2019, 2027)
THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.5]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)


def race_softmax(u, race_id):
    u = pd.Series(u, index=race_id.index)
    e = np.exp(u - u.groupby(race_id).transform("max"))
    return (e / e.groupby(race_id).transform("sum")).to_numpy()


def prepare():
    df = load()
    odds = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds"])
    odds["horse_number"] = odds["horse_number"].astype(float)
    d = df.merge(odds, on=["race_id", "horse_number"], how="left")
    ok = d.groupby("race_id")["win_odds"].transform(lambda s: (s > 0).all())
    d = d[ok & (d["race_date"] >= "2013-01-01") & (d["race_date"] < END)].sort_values(["race_date", "race_id", "horse_number"])
    d = d.reset_index(drop=True)
    inv = 1.0 / d["win_odds"]
    d["overround"] = inv.groupby(d["race_id"]).transform("sum")
    d["x_mkt"] = np.log(inv / d["overround"])
    d = d.merge(pd.read_parquet(CURRENT_BASE), on=["race_id", "horse_id"], how="left")
    return d


def roi(sub):
    """レース単位で (払戻, 点数) を集め、回収率と95%区間（レース単位のブートストラップ）"""
    if sub.empty:
        return 0, np.nan, (np.nan, np.nan)
    pay = (sub["win"] * sub["win_odds"]).groupby(sub["race_id"]).sum()
    cnt = sub.groupby("race_id").size()
    r = pay.sum() / cnt.sum()
    v = (pay - cnt * r).to_numpy() / cnt.mean()     # 比の推定量の線形化
    lo, hi = bootstrap_ci(v)
    return len(sub), r * 100, ((r + lo) * 100, (r + hi) * 100)


def main():
    d = prepare()
    print(f"{d['race_id'].nunique()}R（2013-01-01〜{END} の前日、全頭に確定オッズのあるレース）", flush=True)
    out = []
    for year in YEARS:
        tr = d[d["race_date"] < f"{year - 1}-01-01"]
        va = d[(d["race_date"] >= f"{year - 1}-01-01") & (d["race_date"] < f"{year}-01-01")]
        te = d[(d["race_date"] >= f"{year}-01-01") & (d["race_date"] < f"{year + 1}-01-01")].copy()
        m = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr["x_mkt"]), 5000, valid_sets=[lgb.Dataset(va[FEATS], va["x_mkt"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        over = tr.groupby("race_id")["overround"].first().mean()
        te["p_mkt_hat"] = race_softmax(m.predict(te[FEATS], num_iteration=m.best_iteration), te["race_id"])
        te["odds_hat"] = 1.0 / (te["p_mkt_hat"] * over)
        te["p_model"] = race_softmax(te["u_base"].to_numpy(), te["race_id"])
        te["year"] = year
        out.append(te)
        print(f"{year}: 学習 {tr['race_id'].nunique()}R / 木 {m.best_iteration}本 / 控除の倍率 {over:.3f} / "
              f"log オッズの相関 {np.corrcoef(np.log(te['odds_hat']), np.log(te['win_odds']))[0, 1]:.3f}", flush=True)
    t = pd.concat(out, ignore_index=True)
    t = t[t["p_model"].notna()]
    t = t[t.groupby("race_id")["p_model"].transform("size") == t.groupby("race_id")["race_id"].transform("size")]
    n_race = t["race_id"].nunique()

    print(f"\n=== 採点 {YEARS.start}〜{END} の前日: {n_race}R ===")
    lo = np.log(t["win_odds"])
    lh = np.log(t["odds_hat"])
    print(f"予測オッズの精度: log オッズの相関 {np.corrcoef(lh, lo)[0, 1]:.3f} / 誤差の中央値 ×{np.exp(np.median(np.abs(lh - lo))):.2f}倍"
          f" / 2倍以上外れ {np.mean(np.abs(lh - lo) > np.log(2)) * 100:.0f}%")
    w = t[t["win"] == 1]
    for name, col in [("市場（確定オッズ）", np.exp(w["x_mkt"])), ("予測オッズ", w["p_mkt_hat"]), ("基礎モデル", w["p_model"])]:
        print(f"  勝ち馬の対数尤度 {name}: {np.log(col).mean():.4f}")

    print("\n単勝を1点同額で買ったときの回収率（回収は実際の確定オッズ）")
    print(f"  全馬: {roi(t)[1]:.1f}%")
    rows = []
    for th in THRESHOLDS:
        for label, ev in [("勝率×予測オッズ（本番と同じ条件）", t["p_model"] * t["odds_hat"]),
                          ("勝率×確定オッズ（参考: 買う時点では分からない）", t["p_model"] * t["win_odds"])]:
            n, r, (a, b) = roi(t[ev >= th])
            rows.append((th, label, n, r, a, b))
            print(f"  期待値{th:.1f}以上 {label}: {n:,}点 {r:.1f}% [{a:.1f}, {b:.1f}]")
    print("\n年別（勝率×予測オッズ、期待値1.0以上 / 1.2以上）")
    for y, g in t.groupby("year"):
        ev = g["p_model"] * g["odds_hat"]
        a, b = roi(g[ev >= 1.0]), roi(g[ev >= 1.2])
        print(f"  {y}: {g['race_id'].nunique()}R | 1.0以上 {a[0]:,}点 {a[1]:.1f}% | 1.2以上 {b[0]:,}点 {b[1]:.1f}%")


if __name__ == "__main__":
    main()
