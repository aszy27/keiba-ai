# predict.py
# ⑤ 実践: 保存した候補のモデル（train.py）で、発走前のオッズのスナップショットから各馬の勝率と買い目を出し、
#          前向き検証の採点をする（docs/rebuild_plan.md「前向き検証」「候補2以降の開発」）。
#   オッズ: data/v2/odds_snapshots/YYYYMMDD.csv（python -m scrape snapshot）の3分前。無ければ10分前、それも無ければ30分前
#   対象:   候補の start 以降で、結果を取り込み済み（python -m prep.ingest → features）のレース
#   買い目: 期待値（勝率 × スナップショットのオッズ）が候補の threshold 以上の単勝
#   回収率: 実際の払戻（確定オッズ）で計算したものと、スナップショットのオッズで計算したものの両方を出す。判定はどちらを使うかを候補ごとに登録（roi_basis）
# 判定（--judge）は1,000R以上たまってから候補ごとに1回だけ。一度判定すると result/v2/judged_<候補>.txt が残り、2回目は実行しない。
# 発走前の買い目（--live）: 出馬表（live/entries.py）→ 特徴量（学習と同じ prep.features）→ 今のオッズ → 期待値が閾値以上の単勝。
#
# 使い方: python -m predict --candidate c_all            # これまでの集計（何Rたまったか・途中の成績）
#         python -m predict --candidate c_all --judge    # 判定（1回だけ）
#         python -m predict --candidate cand2 --live     # 今日のまだ発走していないレースの買い目（--within 30 で30分以内だけ）
#         python -m predict --candidate cand2 --live --watch   # 開催日の間ずっと、発走15分前になったレースから順に予測する
import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.base import eligible
from model.candidates import CANDIDATES
from model.market import add_market_cols
from model.pipeline import add_extra_cols, base_utility, predict_combined
from model.softmax import RaceGroups, bootstrap_ci
from paths import ROOT, V2_DIR, table_path
from train import MODELS_DIR, variants_of

SNAPSHOT_DIR = V2_DIR / "odds_snapshots"
MINUTES_PRIORITY = [3, 10, 30]    # 登録どおり。判定まで変えない
MIN_SECONDS_BEFORE = 60           # これより発走に近い（発走後を含む）スナップショットは使わない
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
    if "api_status" in s:   # 発売前の「予想オッズ」（status=yoso）は実際のオッズではないので使わない
        s = s[s["api_status"].astype(str).str.strip().str.lower() != "yoso"]
    if "seconds_to_post" in s:   # PC の復帰の遅れなどで発走の直前・後に取れた行は、買える時点のオッズではないので使わない（次の優先順位へ）
        s = s[pd.to_numeric(s["seconds_to_post"], errors="coerce") >= MIN_SECONDS_BEFORE]
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
        raise SystemExit(f"{name} はまだ登録されていない（model/candidates.py の start が空）")
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
    missing_pay = int((d["bet"] & (d["win"] == 1) & d["win_pay"].isna()).sum())
    if missing_pay:
        print(f"  ※ 当たった買い目のうち払戻が未取得 {missing_pay}点（その分だけ実際の払戻の回収率が低く出ている。python -m scrape weekly → prep.ingest）")
    print(f"  保存: {out_dir / f'forward_{name}.csv'}")
    return dict(n_races=n_races, diff=diff.mean(), lo=lo, hi=hi, bets=bets, roi_actual=roi_actual, roi_snapshot=roi_snap,
                missing_pay=missing_pay)


def judge(name, res):
    cfg = CANDIDATES[name]
    marker = OUT_DIR / f"judged_{name}.txt"
    if marker.exists():
        raise SystemExit(f"{name} は判定済み（{marker}）。判定は1回だけ")
    if cfg["threshold"] is None or cfg["roi_basis"] is None:
        raise SystemExit(f"{name} は閾値が未登録")
    if res is None or res["n_races"] < MIN_RACES:
        raise SystemExit(f"まだ {0 if res is None else res['n_races']}R。{MIN_RACES}R たまるまで判定しない")
    if res["missing_pay"]:
        raise SystemExit(f"当たった買い目の払戻が {res['missing_pay']}点 未取得。取り込んでから判定する")
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


LIVE_DIR = ROOT / "result" / "live"


def race_softmax(u, race_ids):
    """レース内で正規化した勝率（結果が未確定のレース用。RaceGroups は勝ち馬が要るので使えない）"""
    u = pd.Series(np.asarray(u, dtype=float))
    rid = pd.Series(np.asarray(race_ids))
    e = np.exp(u - u.groupby(rid).transform("max"))
    return (e / e.groupby(rid).transform("sum")).to_numpy()


def live(name, date, races=None, within=None, out_dir=LIVE_DIR, now=None):
    """発走前のレースの買い目を出す: 出馬表 → 特徴量（学習と同じ関数）→ 今のオッズ → 候補のモデル → 期待値が閾値以上の単勝"""
    import time
    from datetime import timedelta
    from live.entries import fetch
    from prep.features import build_features, load_tables
    from scrape.snapshot import fetch_odds, race_schedule

    cfg = CANDIDATES[name]
    if cfg["threshold"] is None:
        raise SystemExit(f"{name} は閾値が未登録")
    now = now or datetime.now()
    day = race_schedule(date)
    sched = [(r, p) for r, p in day if p >= now - timedelta(minutes=1)]   # まだ発走していないレース
    done = [r for r, p in day if p <= now - timedelta(minutes=10)]        # 終わったレース（当日バイアスに使う）
    if races:
        sched = [(r, p) for r, p in sched if r in races or str(int(r[10:12])) in races]
    if within:
        sched = [(r, p) for r, p in sched if p - now <= timedelta(minutes=within)]
    if not sched:
        print(f"{date}: 対象のレースが無い（発走済み・開催なし・条件に合わない）")
        return None
    ids = [r for r, _ in sched]
    post = dict(sched)
    print(f"[{name}] {date} の {len(ids)}R（{sched[0][1]:%H:%M}〜{sched[-1][1]:%H:%M}）/ 期待値{cfg['threshold']}以上の単勝", flush=True)

    from live.entries import fetch_finished
    fin_races, fin_runners, not_yet = fetch_finished(done) if done else (None, None, [])
    print(f"  今日終わったレース {len(done)}R のうち結果を取得 {0 if fin_races is None else len(fin_races)}R"
          + (f"（未取得 {len(not_yet)}R: 当日バイアスがその分だけ学習時より少ない）" if not_yet else ""), flush=True)
    races_t, runners, training, horses, warns = fetch(ids)
    if races_t is None:
        for rid in ids:
            print(f"  {rid}: {' / '.join(warns.get(rid, []))}")
        return None
    t = load_tables()
    if fin_races is not None:
        for key, new in (("races", fin_races), ("runners", fin_runners)):
            t[key] = pd.concat([t[key][~t[key]["race_id"].isin(new["race_id"])], new], ignore_index=True)
    for key, new in (("races", races_t), ("runners", runners), ("training", training), ("horses", horses)):
        if new is not None and len(new):
            old = t[key] if key == "horses" else t[key][~t[key]["race_id"].isin(ids)]
            t[key] = pd.concat([old, new], ignore_index=True)
    t["horses"] = t["horses"].drop_duplicates("horse_id", keep="first")
    print(f"  特徴量を作成中（全期間を作り直すので3分ほどかかる）", flush=True)
    f = build_features(t)
    f = f[f["race_id"].isin(ids) & (f["status"] == "entry")].reset_index(drop=True)

    odds = []
    for rid in ids:
        rows, status = fetch_odds(rid)
        if not rows:
            warns.setdefault(rid, []).append(f"オッズを取得できない（{status}）")
        elif str(status).lower() == "yoso":   # 発売前の予想オッズ
            warns.setdefault(rid, []).append("まだ発売前の予想オッズしか無い（予測しない）")
            continue
        odds += rows
        time.sleep(1.0)
    if not odds:
        for rid in ids:
            print(f"  {rid}: {' / '.join(warns.get(rid, []))}")
        return None
    o = pd.DataFrame(odds)
    for c in ("horse_number", "win_odds", "place_odds_min", "place_odds_max"):
        o[c] = pd.to_numeric(o.get(c), errors="coerce")
    d = f.merge(o[["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max"]],
                on=["race_id", "horse_number"], how="left")
    ok = d.groupby("race_id")["win_odds"].transform(lambda s: (s > 0).all()) & d["horse_number"].notna()
    for rid in d.loc[~ok, "race_id"].unique():
        warns.setdefault(rid, []).append("馬番かオッズがそろわないので予測しない")
    d = d[ok].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    if d.empty:
        for rid in ids:
            print(f"  {rid}: {' / '.join(warns.get(rid, []))}")
        return None

    meta, base, fits = load_models(name)
    d = add_market_cols(d)
    d = add_extra_cols(d, {e for v in variants_of(meta["config"]) for e in v["extras"]}, final_place_odds=False, live=True)
    d["u_base"] = base_utility(base, meta["base_temp"], d)
    us = [predict_combined(f_, d) for f_ in fits]
    d["p_a"] = race_softmax(us[0][0], d["race_id"])
    d["p_c"] = race_softmax(np.mean([u for _, u in us], axis=0), d["race_id"])
    d["ev"] = d["p_c"] * d["win_odds"]
    d["bet"] = d["ev"] >= cfg["threshold"]
    names = runners.set_index(["race_id", "horse_id"])["horse_name"]
    d["horse_name"] = [names.get((r, h), "") for r, h in zip(d["race_id"], d["horse_id"])]

    stamp = datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{date}_{stamp:%H%M}_{name}.csv"
    d.assign(post_time=d["race_id"].map(lambda r: f"{post[r]:%H:%M}"), odds_at=f"{stamp:%H:%M:%S}")[
        ["race_id", "post_time", "horse_number", "horse_name", "win_odds", "p_a", "p_c", "ev", "bet", "odds_at"]
    ].to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print(f"=== 買い目（{stamp:%H:%M} 時点のオッズ。発走が近いほどオッズは動くので、買う直前にもう一度実行する）===")
    for rid in ids:
        g = d[d["race_id"] == rid]
        w = warns.get(rid, [])
        head = f"{post[rid]:%H:%M} {rid}"
        if g.empty:
            print(f"{head}: 予測なし（{' / '.join(w)}）")
            continue
        b = g[g["bet"]].sort_values("ev", ascending=False)
        buy = " / ".join(f"{int(r.horse_number)}番 {r.horse_name} {r.win_odds:.1f}倍 勝率{r.p_c:.1%} 期待値{r.ev:.2f}" for r in b.itertuples())
        print(f"{head}: {'【買い】 ' + buy if len(b) else '見送り'}" + (f"  ※{' / '.join(w)}" if w else ""))
    print()
    print(f"保存: {out}")
    return d


def watch(name, date, lead=15, poll=60):
    """発走が近づいたレースから順に、直前に予測する（当日バイアスに、そのレースまでに終わった全レースの結果が入る）。
    発走の lead 分前を過ぎたレースを、まとめて1回予測する。特徴量の作成に3分ほどかかるので lead は10分以上にする"""
    import time
    from datetime import timedelta
    from scrape.snapshot import race_schedule
    day = race_schedule(date)
    if not day:
        print(f"{date}: 開催なし")
        return
    done = set()
    print(f"[{name}] {date} の {len(day)}R を見張る（発走{lead}分前になったレースから予測。Ctrl+C で終了）", flush=True)
    while True:
        now = datetime.now()
        due = [r for r, p in day if r not in done and now + timedelta(minutes=2) <= p <= now + timedelta(minutes=lead)]
        done |= {r for r, p in day if p < now + timedelta(minutes=2)}   # 間に合わなかったレースは飛ばす
        if due:
            live(name, date, races=due)
            done |= set(due)
        if all(p < datetime.now() for _, p in day):
            print("最終レースが発走したので終了")
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--judge", action="store_true", help="判定する（1,000R以上・候補ごとに1回だけ）")
    ap.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR, help="動作確認用。本番のスナップショット以外を読む")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="動作確認用の保存先")
    ap.add_argument("--start", help="動作確認用。対象の初日を上書きする（--snapshot-dir と一緒にだけ使える）")
    ap.add_argument("--live", action="store_true", help="発走前のレースの買い目を出す")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="--live の開催日（YYYYMMDD。既定は今日）")
    ap.add_argument("--races", type=lambda s: s.split(","), help="--live の対象（レースID か レース番号。カンマ区切り）")
    ap.add_argument("--within", type=int, help="--live で、発走まで N 分以内のレースだけ")
    ap.add_argument("--watch", action="store_true", help="--live を開催日の間ずっと続け、発走が近づいたレースから順に予測する")
    ap.add_argument("--lead", type=int, default=15, help="--watch で、発走の何分前に予測するか（既定15分）")
    ap.add_argument("--as-of", help="動作確認用。--live の「今」を上書きする（例 '2026-09-06 09:00'。2026-09-07 以降は前向き検証の期間なので使わない）")
    args = ap.parse_args()
    if args.live and args.watch:
        watch(args.candidate, args.date, args.lead)
        return
    if args.live:
        now = pd.Timestamp(args.as_of).to_pydatetime() if args.as_of else None
        if now and now >= datetime(2026, 9, 7):
            raise SystemExit("--as-of に前向き検証の期間（2026-09-07 以降）は使わない")
        live(args.candidate, args.date, args.races, args.within, LIVE_DIR if not now else LIVE_DIR / "test", now)
        return
    if args.start and args.snapshot_dir == SNAPSHOT_DIR:
        raise SystemExit("--start は動作確認（--snapshot-dir 指定）のときだけ使える")
    res = score(args.candidate, args.snapshot_dir, args.out_dir, args.start)
    if args.judge:
        if args.snapshot_dir != SNAPSHOT_DIR:
            raise SystemExit("判定は本番のスナップショットでだけ行う")
        judge(args.candidate, res)


if __name__ == "__main__":
    main()
