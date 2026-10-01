# experiments/favorite_segments.py
# 「1番人気の単勝を、当てやすい条件のレースだけで買う」とプラスになるかの検証（中央競馬）。
# 条件は1つの軸ずつ（頭数・クラス・芝ダ・距離・馬場・競馬場・1番人気のオッズ・1/2番人気のオッズ比）で切る。
# 開発期間で、300R以上ある条件のうち回収率の95%区間（レース単位のブートストラップ）の下限が最も高いものを機械的に選び、
# 最終テスト期間でその1つだけを評価する（docs/rebuild_plan.md「当てやすいレースの選択」）。
# 使い方: python -m experiments.favorite_segments                                      # 開発 2021〜2023年
#         python -m experiments.favorite_segments --period test --final --segment "クラス=2勝"   # 最終テスト（1回だけ）
import argparse

import numpy as np
import pandas as pd

from paths import table_path

PERIODS = {"dev": ("2021-01-01", "2024-01-01"), "test": ("2024-01-01", "2026-01-01")}
MIN_DEV_RACES = 300
N_BOOT = 2000
CLASS_GROUP = {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝", "OP": "OP・L", "L": "OP・L",
               "G3": "重賞", "G2": "重賞", "G1": "重賞"}


def load(period):
    s, e = PERIODS[period]
    races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date", "place", "surface", "distance", "going", "race_class"])
    races = races[(races["race_date"] >= s) & (races["race_date"] < e)]
    run = pd.read_parquet(table_path("runners"), columns=["race_id", "horse_number", "finish_pos", "status"])
    run = run[run["race_id"].isin(races["race_id"]) & (run["status"] != "scratched")]
    odds = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "win_odds"])
    d = run.merge(odds, on=["race_id", "horse_number"], how="left")
    d["win"] = (d["finish_pos"] == 1).fillna(False).astype(float)
    g = d.groupby("race_id")
    ok = g["win_odds"].transform(lambda s: (s > 0).all()) & (g["race_id"].transform("size") >= 5) & (g["win"].transform("sum") == 1)
    d = d[ok].sort_values(["race_id", "win_odds"])

    top2 = d.groupby("race_id").head(2)
    r = top2.groupby("race_id").agg(o1=("win_odds", "first"), o2=("win_odds", "last"), fav_win=("win", "first"))
    r["n"] = d.groupby("race_id").size()
    r = r[r["o1"] < r["o2"]]                          # 1番人気が単独のレースだけ（オッズ同値は除く）
    r["ret"] = r["fav_win"] * r["o1"]
    r = r.join(races.set_index("race_id"))
    return r


def segments(r):
    return {
        "頭数": pd.cut(r["n"], [0, 8, 12, 16, 18], labels=["5〜8頭", "9〜12頭", "13〜16頭", "17〜18頭"]),
        "クラス": r["race_class"].astype(object).map(CLASS_GROUP),
        "芝ダ": r["surface"].astype(object),
        "距離": pd.cut(r["distance"].astype(float), [0, 1300, 1600, 2000, 2400, 5000],
                     labels=["〜1300m", "1400〜1600m", "1700〜2000m", "2100〜2400m", "2500m〜"]),
        "馬場": r["going"].astype(object),
        "競馬場": r["place"].astype(object),
        "1番人気のオッズ": pd.cut(r["o1"], [0, 1.5, 2, 3, 5, 1000], labels=["〜1.5倍", "1.6〜2.0倍", "2.1〜3.0倍", "3.1〜5.0倍", "5.1倍〜"]),
        "1/2番人気のオッズ比": pd.cut(r["o2"] / r["o1"], [1, 1.5, 2, 3, 1000], labels=["〜1.5", "1.5〜2", "2〜3", "3〜"]),
    }


def roi_row(sub, rng):
    ret = sub["ret"].to_numpy()
    boots = ret[rng.integers(0, len(ret), (N_BOOT, len(ret)))].mean(axis=1) * 100
    return {"レース数": len(ret), "的中率": sub["fav_win"].mean(), "平均オッズ": sub["o1"].mean(),
            "回収率%": ret.mean() * 100, "95%下限": np.percentile(boots, 2.5), "95%上限": np.percentile(boots, 97.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=list(PERIODS))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--segment", help="最終テストで評価する条件（例: クラス=2勝）")
    args = ap.parse_args()
    if args.period == "test" and not (args.final and args.segment):
        print("最終テスト期間は --final と、登録した --segment を付けたときだけ評価する")
        return

    rng = np.random.default_rng(0)
    r = load(args.period)
    segs = segments(r)
    print(f"期間 {PERIODS[args.period][0]}〜{PERIODS[args.period][1]} / 1番人気が単独のレース {len(r)}R")
    print("全レース:", {k: round(v, 3) for k, v in roi_row(r, rng).items()})

    if args.final:
        dim, label = args.segment.split("=", 1)
        sub = r[segs[dim].astype(str) == label]
        row = roi_row(sub, rng)
        print(f"\n最終テスト {args.segment}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        print(f"判定: 回収率の95%区間の下限 {row['95%下限']:.1f}%（>100% が条件）→ {'合格' if row['95%下限'] > 100 else '不合格'}")
        return

    rows = []
    for dim, labels in segs.items():
        for label, sub in r.groupby(labels.astype(str)):
            if label in ("nan", "None"):
                continue
            rows.append({"軸": dim, "条件": label} | roi_row(sub, rng))
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(t.round({"的中率": 3, "平均オッズ": 2, "回収率%": 1, "95%下限": 1, "95%上限": 1}).to_string(index=False))
    cand = t[t["レース数"] >= MIN_DEV_RACES].sort_values("95%下限", ascending=False).iloc[0]
    print(f"\n選ばれた条件（{MIN_DEV_RACES}R以上で95%下限が最大）: {cand['軸']}={cand['条件']} "
          f"回収率 {cand['回収率%']:.1f}% [{cand['95%下限']:.1f}, {cand['95%上限']:.1f}] {cand['レース数']}R")


if __name__ == "__main__":
    main()
