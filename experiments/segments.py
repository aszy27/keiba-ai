# experiments/segments.py
# 市場が弱いレースに絞る（事前登録: docs/rebuild_plan.md「市場が弱いレースに絞る」2026-10-09）。
#   候補2と同じ作り方の前進検証の予測（t7_combine）で、レースの種類ごとにオッズに対する上積み（C − A）を比べる。
#   選ぶ期間 2019〜2022 で1区分を機械的に選び、確かめる期間 2023〜2026-09-06 ではその1区分だけを計算する。
#
# 使い方: python -m experiments.segments
import numpy as np
import pandas as pd

from paths import V2_DIR, table_path

PREDS = V2_DIR / "bench" / "t7_combine_preds.parquet"
END = "2026-09-07"
SELECT = ("2019-01-01", "2023-01-01")
CONFIRM = ("2023-01-01", END)
MIN_RACES = 1500
EV_TH = 1.3


def cut(x, edges, labels):
    return pd.cut(x, [-np.inf] + edges + [np.inf], labels=labels, right=True).astype(str)


def race_table():
    p = pd.read_parquet(PREDS)
    p = p[p["race_date"] < END]
    w = p[p["win"] == 1]
    one = w.groupby("race_id").size()
    w = w[w["race_id"].isin(one[one == 1].index)]
    r = w[["race_id", "race_date"]].assign(d=np.log(w["p_c"]) - np.log(w["p_a"])).set_index("race_id")
    r["fav_odds"] = p.groupby("race_id")["win_odds"].min()
    races = pd.read_parquet(table_path("races")).set_index("race_id")
    r = r.join(races[["race_class", "surface", "distance", "n_starters", "place", "race_number", "going"]])
    cls = {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝", "OP": "OP・L", "L": "OP・L",
           "G1": "重賞", "G2": "重賞", "G3": "重賞"}
    seg = pd.DataFrame(index=r.index)
    seg["クラス"] = r["race_class"].map(cls)
    seg["芝ダ"] = r["surface"]
    seg["距離"] = cut(r["distance"], [1400, 1800, 2200], ["〜1400", "1500〜1800", "1900〜2200", "2300〜"])
    seg["頭数"] = cut(r["n_starters"], [10, 13, 16], ["〜10", "11〜13", "14〜16", "17〜"])
    seg["競馬場"] = r["place"]
    seg["レース番号"] = cut(r["race_number"].astype(float), [4, 8], ["1〜4R", "5〜8R", "9〜12R"])
    seg["馬場"] = r["going"].map({"良": "良", "稍": "稍重", "重": "重・不良", "不良": "重・不良"})
    seg["1番人気のオッズ"] = cut(r["fav_odds"], [2.0, 3.0, 5.0], ["〜2.0", "2.0〜3.0", "3.0〜5.0", "5.0〜"])
    return r, seg, p


def diff_ci(a, b, level, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), (n, len(b)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return a.mean() - b.mean(), np.percentile(ma - mb, tail), np.percentile(ma - mb, 100 - tail)


def roi(p, race_ids):
    b = p[p["race_id"].isin(race_ids) & (p["p_c"] * p["win_odds"] >= EV_TH)]
    return len(b), (b["win"] * b["win_odds"]).sum() / max(len(b), 1) * 100


def main():
    r, seg, p = race_table()
    sel = (r["race_date"] >= SELECT[0]) & (r["race_date"] < SELECT[1])
    con = (r["race_date"] >= CONFIRM[0]) & (r["race_date"] < CONFIRM[1])
    print(f"選ぶ期間 {SELECT[0]}〜{SELECT[1]} の前日: {sel.sum()}R / 全体の C − A {r.loc[sel, 'd'].mean():+.4f}", flush=True)
    rows = []
    for axis in seg.columns:
        for label in sorted(seg[axis].dropna().unique()):
            if label == "nan":
                continue
            inn = sel & (seg[axis] == label)
            out = sel & (seg[axis] != label)
            n = int(inn.sum())
            if n == 0:
                continue
            dm, lo, hi = diff_ci(r.loc[inn, "d"].to_numpy(), r.loc[out, "d"].to_numpy(), 0.95)
            rows.append(dict(軸=axis, 区分=label, R=n, 区分内=r.loc[inn, "d"].mean(), 差=dm, 下限=lo, 上限=hi,
                             対象=n >= MIN_RACES))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    ok = t[t["対象"] & (t["下限"] > 0)]
    if ok.empty:
        print("\n選ぶ期間で区間の下限が0を超える区分（1,500R以上）は無い → 候補は作らない（登録どおりここで終わり）")
        return
    best = ok.sort_values("下限", ascending=False).iloc[0]
    print(f"\n選んだ区分: {best['軸']} = {best['区分']}（差 {best['差']:+.4f} [{best['下限']:+.4f}, {best['上限']:+.4f}]）")

    inn = con & (seg[best["軸"]] == best["区分"])
    out = con & (seg[best["軸"]] != best["区分"])
    dm, lo, hi = diff_ci(r.loc[inn, "d"].to_numpy(), r.loc[out, "d"].to_numpy(), 0.90)
    n_in, roi_in = roi(p, r.index[inn])
    n_all, roi_all = roi(p, r.index[con])
    print(f"\n確かめる期間 {CONFIRM[0]}〜{CONFIRM[1]} の前日: 区分内 {inn.sum()}R")
    print(f"  1. 区分内 − 区分外 {dm:+.4f} [90%区間 {lo:+.4f}, {hi:+.4f}] → {'満たす' if lo > 0 else '満たさない'}")
    print(f"  2. 期待値{EV_TH}以上の単勝（確定オッズ）: 区分内 {n_in}点 {roi_in:.1f}% / 全レース {n_all}点 {roi_all:.1f}%"
          f" → {'満たす' if roi_in > roi_all else '満たさない'}")
    print(f"  判定: {'合格' if lo > 0 and roi_in > roi_all else '不合格'}")
    y26 = r["race_date"] >= "2026-01-01"
    i26, o26 = y26 & (seg[best["軸"]] == best["区分"]), y26 & (seg[best["軸"]] != best["区分"])
    print(f"  参考（採否に使わない）2026年: 区分内 {i26.sum()}R の C − A {r.loc[i26, 'd'].mean():+.4f} / 区分外 {r.loc[o26, 'd'].mean():+.4f}"
          f" / 期待値{EV_TH}以上 {roi(p, r.index[i26])[0]}点 {roi(p, r.index[i26])[1]:.1f}%")


if __name__ == "__main__":
    main()
