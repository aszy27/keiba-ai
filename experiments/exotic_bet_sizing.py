# experiments/exotic_bet_sizing.py
# 組み合わせ券の前向き検証の買い方（docs/rebuild_plan.md「組み合わせ券の前向き検証と掛け金の決め方」）を、過去の確定オッズで模擬する。
# 判定で選ばれた設定（比の方式・λ=0.75・期待値2.0以上）で、1レースの点数の上限（1点 / 5点）ごとに
#   - 1点100円の回収率・的中
#   - 資金5万円から登録どおりの買い方（掛け金の単位は週ごと、1日5%まで、半分を割ったら止める）で動かした結果
#   - 1,000R（前向き検証の長さ）の区間を100Rずつずらしたときに、資金が半分を割る割合
# を出す。2024〜2025年は判定に使った期間なので参考（買い方を選ぶ根拠は2020〜2023年）。
# 使い方: python -m experiments.exotic_bet_sizing --bet trio
import argparse

import numpy as np
import pandas as pd

from experiments.exotic_judge import race_rows
from model.exotic import BET_NAMES, BET_TYPES, winning_combo
from model.pipeline import BENCH_DIR
from paths import V2_DIR, table_path

THRESHOLD, METHOD, LAM = 2.0, "ratio", 0.75
TOP = 5
START, UNIT_BANK, DAY_CAP, STOP = 50_000, 50_000, 0.05, 0.5
WINDOW, STEP = 1000, 100
CACHE = V2_DIR.parent.parent / "result" / "v2"


def top_bets(bet, years):
    """レースごとの期待値の上位 TOP 点（閾値以上）。期待値が閾値以上の組み合わせが無いレースも行（rank 0）を残す"""
    path = CACHE / f"exotic_top{TOP}_{bet}_{years[0]}_{years[-1]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    preds = pd.read_parquet(BENCH_DIR / "t7_combine_preds.parquet")
    preds = preds[preds["race_date"].dt.year.isin(years)]
    fin = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos"])
    fin["horse_number"] = fin["horse_number"].astype(float)
    preds = preds.merge(fin, on=["race_id", "horse_number"], how="left")
    out = []
    for y in years:
        o = pd.read_parquet(V2_DIR / "odds_exotic" / f"{y}_type{BET_TYPES[bet]}.parquet")
        odds = {rid: dict(zip(g["combo"], g["odds_min"].astype(float))) for rid, g in o.groupby("race_id", sort=False)}
        del o
        for rid, g in preds[preds["race_date"].dt.year == y].groupby("race_id", sort=False):
            od = odds.get(rid)
            g = g.sort_values("horse_number")
            win = winning_combo(g["finish_pos"], g["horse_number"], bet) if od is not None else None
            r = race_rows(g, od, bet, METHOD, LAM) if win is not None else None
            if r is None or win not in set(r[0]):
                continue
            combo, prob, od_arr, _ = r
            ev = prob * od_arr
            idx = [i for i in np.argsort(-ev)[:TOP] if ev[i] >= THRESHOLD]
            date = g["race_date"].iloc[0]
            out.append({"race_id": rid, "race_date": date, "rank": 0, "ev": np.nan, "odds": np.nan, "hit": False})
            out += [{"race_id": rid, "race_date": date, "rank": k + 1, "ev": ev[i], "odds": od_arr[i], "hit": combo[i] == win}
                    for k, i in enumerate(idx)]
        print(f"  {y}年 処理", flush=True)
        del odds
    df = pd.DataFrame(out)
    df.to_parquet(path)
    return df


def simulate(races, k):
    """races: レース順の [(日付, [(オッズ, 当たり), ...上位から])]。登録どおりの資金の動き → (最終資金, 最低資金, 止まったか, 点数)"""
    bank, low, n_pts = START, START, 0
    week, unit, cap_base = None, 1, START
    day, spent = None, 0
    for date, bets in races:
        wk = (date - pd.Timedelta(days=date.weekday())).date()
        if wk != week:                                     # 毎週月曜に資金を数え直す
            week, unit, cap_base = wk, max(1, int(bank // UNIT_BANK)), bank
        if date != day:
            day, spent = date, 0
        bets = bets[:k]
        room = int((cap_base * DAY_CAP - spent) // (100 * unit))
        bets = bets[:max(room, 0)]
        for o, h in bets:
            bank += (o * h - 1) * 100 * unit
            spent += 100 * unit
            n_pts += 1
        low = min(low, bank)
        if bank < START * STOP:
            return bank, low, True, n_pts
    return bank, low, False, n_pts


def report(df, label):
    df = df.sort_values(["race_date", "race_id", "rank"])
    races = [(d, list(zip(g.loc[g["rank"] > 0, "odds"], g.loc[g["rank"] > 0, "hit"])))
             for (d, _), g in df.groupby(["race_date", "race_id"], sort=False)]
    n = len(races)
    print(f"\n== {label}: {n}R")
    for k in (1, TOP):
        b = df[(df["rank"] > 0) & (df["rank"] <= k)]
        per = b.groupby("race_id").agg(st=("hit", "size"), ret=("odds", lambda s: (s * b.loc[s.index, "hit"]).sum()))
        per = per.reindex(df["race_id"].unique(), fill_value=0)
        idx = np.random.default_rng(0).integers(0, n, (2000, n))
        st, ret = per["st"].to_numpy(float), per["ret"].to_numpy(float)
        boots = ret[idx].sum(1) / np.maximum(st[idx].sum(1), 1) * 100
        bank, low, stopped, pts = simulate(races, k)
        wins = [simulate(races[s:s + WINDOW], k) for s in range(0, n - WINDOW + 1, STEP)]
        halved = np.mean([w[2] for w in wins])
        print(f"-- 1レース{k}点まで: {int(st.sum()):,}点（{st.sum() / n:.2f}点/R）的中 {int(b['hit'].sum())}"
              f" | 1点100円の回収率 {ret.sum() / st.sum() * 100:.1f}% [{np.percentile(boots, 2.5):.1f}, {np.percentile(boots, 97.5):.1f}]")
        print(f"   資金5万円・登録どおりの買い方（通し）: {'止まった' if stopped else '最後まで'} 最終 {bank:,.0f}円 / 最低 {low:,.0f}円 / {pts:,}点")
        print(f"   1,000R の区間（{len(wins)}個、100Rずつずらす）: 資金が半分を割った割合 {halved * 100:.0f}% /"
              f" 最終資金の中央値 {np.median([w[0] for w in wins]):,.0f}円")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bet", choices=["trio", "trifecta"], required=True)
    args = ap.parse_args()
    print(f"[t7_combine] {BET_NAMES[args.bet]} 比の方式・λ={LAM}・期待値{THRESHOLD}以上", flush=True)
    report(top_bets(args.bet, [2020, 2021, 2022, 2023]), "開発 2020〜2023年（買い方を選ぶ根拠）")
    report(top_bets(args.bet, [2024, 2025]), "2024〜2025年（判定に使った期間。参考）")


if __name__ == "__main__":
    main()
