# v2/predict.py
# ⑤ 実践: 保存した候補のモデル（v2/train.py）で、発走前のオッズのスナップショットから各馬の勝率と買い目を出し、
#          前向き検証の採点をする（docs/rebuild_plan.md「前向き検証」「候補2以降の開発」）。
#   オッズ: data/v2/odds_snapshots/YYYYMMDD.csv（python -m scrape snapshot）の3分前。無ければ10分前、それも無ければ30分前
#   対象:   候補の start 以降で、結果を取り込み済み（python -m v2.data.ingest → features）のレース
#   買い目: 期待値（勝率 × スナップショットのオッズ）が候補の threshold 以上の単勝
#   回収率: 実際の払戻（確定オッズ）で計算したものと、スナップショットのオッズで計算したものの両方を出す。判定はどちらを使うかを候補ごとに登録（roi_basis）
# 判定（--judge）は1,000R以上たまってから候補ごとに1回だけ。一度判定すると result/v2/judged_<候補>.txt が残り、2回目は実行しない。
# ※ 発走前に買い目を出す（出馬表から特徴量を作る）処理はまだ無い。今は結果を取り込んだ後の採点だけ。
#
# 使い方: python -m v2.predict --candidate c_all            # これまでの集計（何Rたまったか・途中の成績）
#         python -m v2.predict --candidate c_all --judge    # 判定（1回だけ）
import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2.model.base import eligible
from v2.model.candidates import CANDIDATES
from v2.model.market import add_market_cols
from v2.model.pipeline import add_extra_cols, base_utility, predict_combined
from v2.model.softmax import RaceGroups, bootstrap_ci
from v2.paths import ROOT, V2_DIR, table_path
from v2.train import MODELS_DIR, variants_of

SNAPSHOT_DIR = V2_DIR / "odds_snapshots"
MINUTES_PRIORITY = [3, 10, 30]    # 登録どおり。判定まで変えない
MIN_RACES = 1000
OUT_DIR = ROOT / "result" / "v2"


def load_snapshots(snapshot_dir=SNAPSHOT_DIR):
    files = sorted(snapshot_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "minutes_used"])
    s = pd.concat([pd.read_csv(f, dtype={"race_id": str}, on_bad_lines="skip") for f in files], ignore_index=True)
    s["win_odds"] = pd.to_numeric(s["win_odds"], errors="coerce")
    s["minutes_before"] = pd.to_numeric(s["minutes_before"], errors="coerce")
    s["horse_number"] = pd.to_numeric(s["horse_number"], errors="coerce")
    for c in ("place_odds_min", "place_odds_max"):
        s[c] = pd.to_numeric(s[c], errors="coerce") if c in s else np.nan
    s = s[s["minutes_before"].isin(MINUTES_PRIORITY) & s["horse_number"].notna()]
    s = s.drop_duplicates(["race_id", "minutes_before", "horse_number"], keep="last")
    ok = s[s["win_odds"] > 0].groupby(["race_id", "minutes_before"]).size().rename("n").reset_index()
    ok["prio"] = ok["minutes_before"].map({m: i for i, m in enumerate(MINUTES_PRIORITY)})
    pick = ok.sort_values("prio").drop_duplicates("race_id")[["race_id", "minutes_before"]]
    s = s.merge(pick, on=["race_id", "minutes_before"])
    return s[["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "minutes_before"]].rename(
        columns={"minutes_before": "minutes_used"})


def load_models(name):
    """(設定, 基礎モデルの一覧, 版ごとの fit の一覧)"""
    d = MODELS_DIR / name
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    base = [lgb.Booster(model_file=str(d / f"base_{i}.txt")) for i in range(meta["n_base_models"])]
    fits = [dict(beta_a=np.array(v["beta_a"]), beta_b=np.array(v["beta_b"]), feats=v["feats"],
                 residual=[lgb.Booster(model_file=str(d / f"residual_{v['name']}_{i}.txt")) for i in range(v["n_models"])])
            for v in meta["variants"]]
    return meta, base, fits


def score(name, snapshot_dir=SNAPSHOT_DIR, out_dir=OUT_DIR, start=None):
    cfg = dict(CANDIDATES[name])
    cfg["start"] = start or cfg["start"]
    if not cfg["start"]:
        raise SystemExit(f"{name} はまだ登録されていない（v2/model/candidates.py の start が空）")
    meta, base, fits = load_models(name)
    df = eligible(pd.read_parquet(table_path("features")))
    df = df[df["race_date"] >= cfg["start"]].reset_index(drop=True)
    snap = load_snapshots(snapshot_dir)
    d = df.merge(snap, on=["race_id", "horse_number"], how="left")
    g = d.groupby("race_id")
    ok = g["win_odds"].transform(lambda s: (s > 0).all())
    n_all = d["race_id"].nunique()
    d = d[ok].sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    print(f"[{name}] {cfg['start']} 以降で結果を取り込んだレース {n_all}R のうち、スナップショットのオッズがそろう {d['race_id'].nunique()}R")
    if d.empty:
        return None
    d = add_market_cols(d)
    d = add_extra_cols(d, {e for v in variants_of(meta["config"]) for e in v["extras"]}, final_place_odds=False)
    d["u_base"] = base_utility(base, meta["base_temp"], d)
    us = [predict_combined(f, d) for f in fits]
    u_a, u_c = us[0][0], np.mean([u for _, u in us], axis=0)   # 版の効用の平均 = log(勝率) の対数プーリング
    grp = RaceGroups(d["race_id"], d["win"])
    (ll_a, p_a), (ll_c, p_c) = grp.ll(u_a), grp.ll(u_c)
    d["p_a"], d["p_c"] = p_a, p_c

    pay = pd.read_parquet(table_path("payouts"), columns=["race_id", "win"])
    pay["win_pay"] = pay["win"].map(lambda v: v[0] if v is not None and len(v) == 1 else np.nan)
    d = d.merge(pay[["race_id", "win_pay"]], on="race_id", how="left")
    th = cfg["threshold"]
    d["bet"] = (d["p_c"] * d["win_odds"] >= th) if th else False
    d["ret_actual"] = np.where(d["bet"] & (d["win"] == 1), d["win_pay"] / 100, 0.0)
    d["ret_snapshot"] = np.where(d["bet"] & (d["win"] == 1), d["win_odds"], 0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    d[["race_id", "race_date", "horse_number", "win", "win_odds", "minutes_used", "p_a", "p_c", "bet", "win_pay"]] \
        .to_csv(out_dir / f"forward_{name}.csv", index=False, encoding="utf-8-sig")

    diff = ll_c - ll_a
    lo, hi = bootstrap_ci(diff, level=cfg["ci_level"])
    n_races = len(diff)
    print(f"  使ったオッズ: " + " / ".join(f"{int(m)}分前 {n}R" for m, n in d.groupby("minutes_used")["race_id"].nunique().items()))
    print(f"  C − A {diff.mean():+.4f} [{cfg['ci_level']:.1%}区間 {lo:+.4f}, {hi:+.4f}]（{n_races}R）")
    bets = int(d["bet"].sum())
    roi_actual = d["ret_actual"].sum() / max(bets, 1) * 100
    roi_snap = d["ret_snapshot"].sum() / max(bets, 1) * 100
    if th:
        print(f"  期待値{th}以上の単勝: {bets}点 的中 {int((d['bet'] & (d['win'] == 1)).sum())} / "
              f"回収率 {roi_actual:.1f}%（実際の払戻）/ {roi_snap:.1f}%（スナップショットのオッズで計算）")
    print(f"  保存: {out_dir / f'forward_{name}.csv'}")
    return dict(n_races=n_races, diff=diff.mean(), lo=lo, hi=hi, bets=bets, roi_actual=roi_actual, roi_snapshot=roi_snap)


def judge(name, res):
    cfg = CANDIDATES[name]
    marker = OUT_DIR / f"judged_{name}.txt"
    if marker.exists():
        raise SystemExit(f"{name} は判定済み（{marker}）。判定は1回だけ")
    if cfg["threshold"] is None or cfg["roi_basis"] is None:
        raise SystemExit(f"{name} は閾値が未登録")
    if res is None or res["n_races"] < MIN_RACES:
        raise SystemExit(f"まだ {0 if res is None else res['n_races']}R。{MIN_RACES}R たまるまで判定しない")
    roi = res["roi_snapshot"] if cfg["roi_basis"] == "snapshot" else res["roi_actual"]
    basis = "スナップショットのオッズで計算" if cfg["roi_basis"] == "snapshot" else "実際の払戻"
    ok = res["lo"] > 0 and roi > 100
    text = (f"判定 {datetime.now():%Y-%m-%d %H:%M} / {name} / {res['n_races']}R / C − A {res['diff']:+.4f} "
            f"[{cfg['ci_level']:.1%}区間 {res['lo']:+.4f}, {res['hi']:+.4f}]（下限 > 0 が条件）/ "
            f"期待値{cfg['threshold']}以上 {res['bets']}点 回収率 {roi:.1f}%（{basis}。> 100% が条件）"
            f" / 参考: 実際の払戻 {res['roi_actual']:.1f}% ・スナップショット {res['roi_snapshot']:.1f}%"
            f" → {'合格' if ok else '不合格'}")
    marker.write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--judge", action="store_true", help="判定する（1,000R以上・候補ごとに1回だけ）")
    ap.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR, help="動作確認用。本番のスナップショット以外を読む")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="動作確認用の保存先")
    ap.add_argument("--start", help="動作確認用。対象の初日を上書きする（--snapshot-dir と一緒にだけ使える）")
    args = ap.parse_args()
    if args.start and args.snapshot_dir == SNAPSHOT_DIR:
        raise SystemExit("--start は動作確認（--snapshot-dir 指定）のときだけ使える")
    res = score(args.candidate, args.snapshot_dir, args.out_dir, args.start)
    if args.judge:
        if args.snapshot_dir != SNAPSHOT_DIR:
            raise SystemExit("判定は本番のスナップショットでだけ行う")
        judge(args.candidate, res)


if __name__ == "__main__":
    main()
