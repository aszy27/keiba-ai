# v2/experiments/exotic_model.py
# モデルの勝率から組み合わせ券の確率を作り、期待値ベースで買ったときの回収率を測る
# （docs/rebuild_plan.md「券種を広げる」）。新しいモデルは作らない。
#   確率: Plackett-Luce（ハービル式）。1着の確率 p から「1着i・2着j・3着k」= p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)
#   オッズ: 複勝は odds_final の下限、その他は scrape_odds_exotic.py で取った全通りの確定オッズ
#   払戻: 当たった組み合わせのオッズ×100（ワイド・複勝は下限を使うので、実際より辛めに出る）
# 使い方: python -m v2.experiments.exotic_model --bets place,wide,trio,trifecta,quinella
import argparse

import numpy as np
import pandas as pd

from v2.model import trip as mt
from v2.model.base import eligible
from v2.paths import V2_DIR, table_path

PREDS_CACHE = V2_DIR / "dev_preds_all.parquet"
EXOTIC_DIR = V2_DIR / "odds_exotic"
THRESHOLDS = [1.0, 1.1, 1.2, 1.5, 2.0]
# 券種 → (API の type, 組み合わせの向き)
BETS = {"quinella": (4, "pair"), "wide": (5, "pair3"), "trio": (7, "triple"), "trifecta": (8, "order3")}
TAKEOUT = {"place": 0.20, "quinella": 0.225, "wide": 0.225, "trio": 0.25, "trifecta": 0.25}


def model_preds(df):
    """開発期間（報告2019年・2020年）の各馬の勝率 p_c。model_trip.py と同じ配置で残差モデルを学習する"""
    if PREDS_CACHE.exists():
        return pd.read_parquet(PREDS_CACHE)
    market, _ = mt.with_market(df, "all")
    out = []
    for year, split in mt.DEV_SPLITS.items():
        preds, _ = mt.run_split(market, "all", split)
        out.append(preds[["race_id", "race_date", "horse_number", "win", "finish_pos", "win_odds", "p_a", "p_c"]])
    preds = pd.concat(out, ignore_index=True)
    preds.to_parquet(PREDS_CACHE, index=False)
    return preds


def _triple_probs(p, lam=1.0):
    """P[i, j, k] = 1着i・2着j・3着k の確率。
    lam < 1 は「1着でない馬が2着・3着に来る力」を割り引く補正（Henery/Stern 型）。
    ハービル式（lam=1）は強い馬が2着・3着に来る確率を過大評価することが知られている"""
    n = len(p)
    q = p ** lam
    total = q.sum()
    pi = p[:, None, None]
    d1 = np.maximum(total - q[:, None, None], 1e-12)
    d2 = np.maximum(total - q[:, None, None] - q[None, :, None], 1e-12)
    prob = pi * (q[None, :, None] / d1) * (q[None, None, :] / d2)
    idx = np.arange(n)
    prob[idx, idx, :] = prob[idx, :, idx] = prob[:, idx, idx] = 0.0   # 同じ馬は使えない
    return prob


def race_combos(numbers, p, kind, lam=1.0):
    """組み合わせの文字列と確率を返す"""
    n = len(p)
    if kind == "place":
        prob = _triple_probs(p, lam)
        top3 = prob.sum(axis=(1, 2)) + prob.sum(axis=(0, 2)) + prob.sum(axis=(0, 1))
        return [f"{int(numbers[i]):02d}" for i in range(n)], top3
    if kind == "pair":      # 馬連（順不同の上位2頭）
        q = p ** lam
        pair = p[:, None] * (q[None, :] / np.maximum(q.sum() - q[:, None], 1e-12))
        np.fill_diagonal(pair, 0.0)
        combos, probs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                combos.append(f"{int(numbers[i]):02d}{int(numbers[j]):02d}")
                probs.append(pair[i, j] + pair[j, i])
        return combos, np.array(probs)
    prob = _triple_probs(p, lam)
    if kind == "pair3":     # ワイド（2頭とも3着以内）
        combos, probs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                probs.append(prob[i, j, :].sum() + prob[j, i, :].sum() + prob[i, :, j].sum()
                             + prob[j, :, i].sum() + prob[:, i, j].sum() + prob[:, j, i].sum())
                combos.append(f"{int(numbers[i]):02d}{int(numbers[j]):02d}")
        return combos, np.array(probs)
    if kind == "triple":    # 3連複（順不同の上位3頭）
        combos, probs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    probs.append(prob[[i, i, j, j, k, k], [j, k, i, k, i, j], [k, j, k, i, j, i]].sum())
                    combos.append(f"{int(numbers[i]):02d}{int(numbers[j]):02d}{int(numbers[k]):02d}")
        return combos, np.array(probs)
    if kind == "order3":    # 3連単
        combos, probs = [], []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len({i, j, k}) == 3:
                        combos.append(f"{int(numbers[i]):02d}{int(numbers[j]):02d}{int(numbers[k]):02d}")
                        probs.append(prob[i, j, k])
        return combos, np.array(probs)
    raise ValueError(kind)


def winning_combo(g, kind):
    """そのレースで当たりになる組み合わせの文字列"""
    top = g.sort_values("finish_pos")
    nums = [int(v) for v in top["horse_number"].tolist()]
    if len(nums) < 3 or top["finish_pos"].iloc[:3].tolist() != [1, 2, 3]:
        return None                       # 同着などで上位3頭が決まらないレースは使わない
    a, b, c = nums[:3]
    if kind == "place":
        return {f"{x:02d}" for x in (a, b, c)}
    if kind == "pair":
        return "".join(f"{x:02d}" for x in sorted((a, b)))
    if kind == "pair3":
        return {"".join(f"{x:02d}" for x in sorted(pair)) for pair in [(a, b), (a, c), (b, c)]}
    if kind == "triple":
        return "".join(f"{x:02d}" for x in sorted((a, b, c)))
    return f"{a:02d}{b:02d}{c:02d}"


def build_bets(preds, bet, odds_map, col="p_c", lam=1.0):
    """レースごとに（組み合わせ, 確率, オッズ, 当たりか）を作る"""
    kind = "place" if bet == "place" else BETS[bet][1]
    rows = []
    for rid, g in preds.groupby("race_id", sort=False):
        g = g.sort_values("horse_number")
        p = g[col].to_numpy()
        p = p / p.sum()
        combos, probs = race_combos(g["horse_number"].to_numpy(), p, kind, lam)
        odds = odds_map.get(rid)
        if odds is None:
            continue
        win_combo = winning_combo(g, kind)
        if win_combo is None:
            continue
        o = np.array([odds.get(c, np.nan) for c in combos])
        hit = np.array([c in win_combo if isinstance(win_combo, set) else c == win_combo for c in combos])
        ok = np.isfinite(o) & (o > 0)
        rows.append(pd.DataFrame({"race_id": rid, "prob": probs[ok], "odds": o[ok], "hit": hit[ok]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(bets, label, who="モデル"):
    """期待値の閾値ごとの点数・的中・回収率（レース単位のブートストラップ区間つき）"""
    print(f"-- {label} / {who}（{bets['race_id'].nunique()}R、控除率 {TAKEOUT[label] * 100:.1f}%）")
    codes, starts = pd.factorize(bets["race_id"])
    n_races = codes.max() + 1
    ev = (bets["prob"] * bets["odds"]).to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_races, (2000, n_races))
    for th in THRESHOLDS:
        sel = ev >= th
        if sel.sum() == 0:
            print(f"   期待値{th:.1f}以上: 0点")
            continue
        stake = np.bincount(codes[sel], minlength=n_races).astype(float)
        ret = np.bincount(codes[sel], weights=(bets["hit"].to_numpy()[sel] * bets["odds"].to_numpy()[sel]),
                          minlength=n_races)
        roi = ret.sum() / stake.sum() * 100
        boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        print(f"   期待値{th:.1f}以上: {int(sel.sum()):>7,}点 的中 {int(bets['hit'].to_numpy()[sel].sum()):>5} | "
              f"回収率 {roi:6.1f}% [{lo:6.1f}, {hi:6.1f}] | ×0.95 {roi * 0.95:6.1f}% | ×0.90 {roi * 0.9:6.1f}%", flush=True)


def load_odds(bet, years):
    """組み合わせ → オッズ の辞書（レースごと）"""
    if bet == "place":
        o = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "place_odds_min"])
        o = o[o["place_odds_min"] > 0]
        o["combo"] = o["horse_number"].astype(int).map("{:02d}".format)
        return {rid: dict(zip(g["combo"], g["place_odds_min"])) for rid, g in o.groupby("race_id", sort=False)}
    bet_type = BETS[bet][0]
    frames = [pd.read_parquet(EXOTIC_DIR / f"{y}_type{bet_type}.parquet") for y in years
              if (EXOTIC_DIR / f"{y}_type{bet_type}.parquet").exists()]
    if not frames:
        return {}
    o = pd.concat(frames, ignore_index=True)
    return {rid: dict(zip(g["combo"], g["odds_min"])) for rid, g in o.groupby("race_id", sort=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bets", default="place")
    ap.add_argument("--years", default="2019,2020")
    ap.add_argument("--lam", type=float, default=1.0, help="2着・3着の力を割り引く補正（1.0 はハービル式）")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]
    preds = model_preds(eligible(pd.read_parquet(table_path("features"))))
    preds = preds[preds["race_date"].dt.year.isin(years)]
    print(f"モデルの予測: {preds['race_id'].nunique()}R / {len(preds)}頭", flush=True)

    for bet in args.bets.split(","):
        odds_map = load_odds(bet, years)
        if not odds_map:
            print(f"-- {bet}: オッズが無いので飛ばす")
            continue
        target = preds[preds["race_id"].isin(odds_map)]
        for col, label in [("p_c", "モデル"), ("p_a", "オッズのみ（対照）")]:
            bets = build_bets(target, bet, odds_map, col, args.lam)
            if len(bets):
                report(bets, bet, label)


if __name__ == "__main__":
    main()
