# experiments/live_checks.py
# 週末の運用の点検（結果を取り込んだ後に実行する）:
#   1. スナップショットの取れ具合（data/v2/odds_snapshots/YYYYMMDD.csv）と、判定に使えるレース数
#   2. 発走前の買い目（predict --live の result/live/*.csv）の勝率と、レース後に取り込んだデータから「同じオッズで」計算し直した勝率の差
#      （学習時と本番で特徴量がずれていないか。差が大きい馬は、発走前に分からなかった情報がある）
#   3. 発走前に取った追い切り評価（data/v2/live_checks/oikiri_pre_*.csv）と、レース後に取り込んだ評価の一致
# 候補の判定（前向き検証の成績）はここでは出さない。
# 使い方: python -m experiments.live_checks --dates 20261003,20261004
import argparse
import glob

import numpy as np
import pandas as pd

from model.market import add_market_cols
from model.pipeline import add_extra_cols, base_utility, predict_combined
from paths import ROOT, V2_DIR, table_path
from predict import load_models, load_snapshots, race_softmax
from train import variants_of


def snapshots(date):
    path = V2_DIR / "odds_snapshots" / f"{date}.csv"
    if not path.exists():
        print(f"  スナップショット {date}: ファイルが無い（取れていない）")
        return
    s = pd.read_csv(path, dtype={"race_id": str}, on_bad_lines="skip")
    s["minutes_before"] = pd.to_numeric(s["minutes_before"], errors="coerce")
    by_min = s.groupby("minutes_before")["race_id"].nunique().sort_index(ascending=False).to_dict()
    sec3 = pd.to_numeric(s.loc[s["minutes_before"] == 3, "seconds_to_post"], errors="coerce")
    print(f"  スナップショット {date}: {s['race_id'].nunique()}R / 何分前ごとのレース数 {by_min}")
    print(f"    status {s['api_status'].value_counts().to_dict()} / 3分前の実際の秒数 中央値 {sec3.median():.0f}・最小 {sec3.min():.0f}")


def usable(dates):
    snap = load_snapshots()
    snap = snap[snap["race_id"].isin(set(snap["race_id"]))]
    for date in dates:
        races = pd.read_parquet(table_path("races"), columns=["race_id", "race_date"])
        ids = set(races.loc[races["race_date"] == pd.Timestamp(date), "race_id"])
        s = snap[snap["race_id"].isin(ids)]
        print(f"  判定に使えるスナップショット {date}: {s['race_id'].nunique()}/{len(ids)}R（使う時点 "
              f"{s.drop_duplicates('race_id')['minutes_used'].value_counts().to_dict()}）")


def live_vs_ingest(date, name="cand2"):
    files = sorted(glob.glob(str(ROOT / "result" / "live" / f"{date}_*_{name}.csv")))
    if not files:
        print(f"  発走前の買い目 {date}: 記録が無い")
        return
    live = pd.concat([pd.read_csv(f, dtype={"race_id": str, "horse_id": str}).assign(file=f) for f in files], ignore_index=True)
    live = live.sort_values("file").drop_duplicates(["race_id", "horse_number"], keep="last")   # レースごとに最後の予測
    meta, base, fits = load_models(name)
    f = pd.read_parquet(table_path("features"))
    f = f[f["race_id"].isin(live["race_id"])]
    d = f.merge(live[["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "p_c", "bet"]]
                .rename(columns={"p_c": "p_live", "bet": "bet_live"}), on=["race_id", "horse_number"], how="inner")
    d = d[d.groupby("race_id")["win_odds"].transform(lambda x: (x > 0).all())]
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    d = add_market_cols(d)
    d = add_extra_cols(d, {e for v in variants_of(meta["config"]) for e in v["extras"]}, final_place_odds=False)
    d["u_base"] = base_utility(base, meta["base_temp"], d)
    d["p_ingest"] = race_softmax(np.mean([predict_combined(fi, d)[1] for fi in fits], axis=0), d["race_id"])
    diff = (d["p_live"] - d["p_ingest"]).abs()
    bet_ingest = d["p_ingest"] * d["win_odds"] >= meta["config"]["threshold"]
    print(f"  発走前の買い目 {date}: {d['race_id'].nunique()}R・{len(d)}頭 / 勝率の差（発走前 − 取り込み後、同じオッズ）"
          f" 平均 {diff.mean():.4f}・最大 {diff.max():.4f} / 0.01超 {int((diff > 0.01).sum())}頭")
    print(f"    買い目: 発走前 {int(d['bet_live'].sum())}点 / 取り込み後のデータなら {int(bet_ingest.sum())}点 / 共通 {int((d['bet_live'] & bet_ingest).sum())}点")
    top = d.assign(diff=diff).sort_values("diff", ascending=False).head(5)
    print(top[["race_id", "horse_number", "win_odds", "p_live", "p_ingest"]].round(4).to_string(index=False))


def oikiri(date_files):
    tr = pd.read_parquet(table_path("training"))
    for f in date_files:
        pre = pd.read_csv(f, dtype={"race_id": str, "horse_id": str})
        m = pre.merge(tr, on=["race_id", "horse_id"], how="left", suffixes=("_pre", "_post"))
        has = m["oikiri_rank_post"].notna()
        same = (m["oikiri_rank_pre"] == m["oikiri_rank_post"]) & has
        print(f"  追い切り評価 {f.split('/')[-1].split(chr(92))[-1]}: 発走前 {len(m)}頭 / 取り込み後にある {int(has.sum())}頭 / 一致 {int(same.sum())}頭"
              + (f"（不一致の例 {m.loc[has & ~same, ['race_id', 'horse_id', 'oikiri_rank_pre', 'oikiri_rank_post']].head(3).to_dict('records')}）"
                 if (has & ~same).any() else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    dates = ap.parse_args().dates.split(",")
    print("1. スナップショット")
    for date in dates:
        snapshots(date)
    usable(dates)
    print("2. 発走前の買い目と、取り込み後のデータの勝率（同じオッズ）")
    for date in dates:
        live_vs_ingest(date)
    print("3. 追い切り評価（発走前 vs 取り込み後）")
    oikiri(sorted(glob.glob(str(V2_DIR / "live_checks" / "oikiri_pre_*.csv"))))


if __name__ == "__main__":
    main()
