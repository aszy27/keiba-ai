# evaluate.py
# ④ 検証: 年ごとの前進検証（2019〜2026-09-06）で、モデルの変更を「現行」と同じレースで比べる（docs/rebuild_plan.md「候補2以降の開発」）。
# 各年 Y は 基礎モデル = 学習 2013〜Y-2 / ES Y-1（3シード平均、温度を Y-1年で合わせる）、残差 = 学習 [Y-4, Y-1) / ES [Y-1, Y)。
# 指標は C − A（1レースあたり、勝ち馬の対数尤度。A = オッズのみ、C = オッズ＋基礎モデル＋残差）。
# 結果はレース単位で data/v2/bench/<tag>.parquet に保存し、--ref の版と同じレースで比べる（対応のある差）。
#
# 使い方: python -m evaluate --tag current                          # 現行（C[all] と同じ作り方）。最初に1回
#         python -m evaluate --build-base base_pl3 --base-objective pl   # 年ごとの基礎モデルの予測を作る（1時間以上）
#         python -m evaluate --tag res_mkt --base base_pl3 --res-years 0 --half-life 730 --res-market --ref pl3_all_hl730
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.combined import RESIDUAL_PARAMS
from model.market import add_place_cols, add_shin
from model.past_market import PAST_COLS, PAST_DIFF, add_past_market
from model.pipeline import (BENCH_DIR, END, FEATS, MARKET_FEATS, SEEDS, base_file, between, fit_base, base_utility, fit_combined, load,
                               market, merge_final_place_odds, predict_combined, recency_weight, residual_trees)
from model.softmax import RaceGroups, bootstrap_ci
from paths import table_path

YEARS = range(2019, 2027)
BASE_YEARS = range(2015, 2027)
THRESHOLDS = [1.0, 1.1, 1.2, 1.3]
MARKET_EXTRA = {"place": ["x_place", "place_gap", "place_spread"],          # --res-market-extra で足す列（実験5・不採用）
                "race": ["fav_log_odds", "mkt_entropy", "n_runners"]}


def build_base(df, name, args):
    """1年ずつ base_NAME_YYYY.parquet に保存し、保存済みの年は飛ばす（途中で止まっても続きから）"""
    BENCH_DIR.mkdir(exist_ok=True)
    for year in BASE_YEARS:
        part = BENCH_DIR / f"base_{name}_{year}.parquet"
        if part.exists():
            print(f"[{name}] {year}: 保存済み", flush=True)
            continue
        tr = between(df, "2013-01-01", f"{year - 1}-01-01")
        va = between(df, f"{year - 1}-01-01", f"{year}-01-01")
        te = between(df, f"{year}-01-01", f"{year + 1}-01-01")
        row_w = recency_weight(tr["race_date"], f"{year - 1}-01-01", args.base_half_life)
        models, temp = fit_base(tr, va, args.base_objective, args.k, args.lam, args.stage_w, row_w)
        te[["race_id", "horse_id"]].assign(u_base=base_utility(models, temp, te)).to_parquet(part, index=False)
        print(f"[{name}] {year}: 学習 {tr['race_id'].nunique()}R / 木 {[m.best_iteration for m in models]}本 / 温度 {temp:.3f}", flush=True)
    parts = [pd.read_parquet(BENCH_DIR / f"base_{name}_{y}.parquet") for y in BASE_YEARS]
    pd.concat(parts, ignore_index=True).to_parquet(base_file(name), index=False)
    print("保存:", base_file(name))


def market_extra(d):
    """実験5で試した列（複勝オッズ・レース全体のオッズの形）"""
    d = add_place_cols(merge_final_place_odds(d))
    g = d.groupby("race_id")
    d["fav_log_odds"] = np.log(g["win_odds"].transform("min"))
    d["mkt_entropy"] = (-np.exp(d["x_mkt"]) * d["x_mkt"]).groupby(d["race_id"]).transform("sum")
    d["n_runners"] = g["race_id"].transform("size")
    return d


def adversarial_weight(tr, va, lo=0.2, hi=5.0):
    """技術の探索 T13: 学習期間の行と ES の年の行を見分ける分類器で、ES の年に似た行ほど重くする（重み p/(1−p)・行数の比、lo〜hi で切る）"""
    from model.pipeline import FEATS
    X = pd.concat([tr[FEATS], va[FEATS]], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(va))]
    cats = [c for c in FEATS if str(X[c].dtype) == "category"]
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
                       verbose=-1, seed=0), lgb.Dataset(X, y, categorical_feature=cats), num_boost_round=200)
    p = np.clip(m.predict(tr[FEATS]), 1e-6, 1 - 1e-6)
    return np.clip(p / (1 - p) * len(tr) / len(va), lo, hi)


def fit_by_surface(tr, va, rep, common, row_w_extra, args):
    """技術の探索 T12: 残差を芝（障害を含む）とダートで別々に学習する"""
    is_dirt = {k: (v["surface"].astype(str) == "ダート").to_numpy() for k, v in (("tr", tr), ("va", va), ("rep", rep))}
    u_a, u_c = np.zeros(len(rep)), np.zeros(len(rep))
    fit = None
    for flag in (False, True):
        t, v, r = tr[is_dirt["tr"] == flag], va[is_dirt["va"] == flag], rep[is_dirt["rep"] == flag]
        t, v, r = (x.reset_index(drop=True) for x in (t, v, r))
        c = (t, v) + common[2:]
        w = None if row_w_extra is None else row_w_extra[is_dirt["tr"] == flag]
        fit = fit_combined(*c, row_w_extra=w, restack=args.restack)
        a, cc = predict_combined(fit, r)
        u_a[is_dirt["rep"] == flag], u_c[is_dirt["rep"] == flag] = a, cc
    return fit, u_a, u_c


def fit_subspace(rep, common, row_w_extra, args, frac=0.5):
    """技術の探索 T15: 残差を列の半分を無作為に選んだ N 通りで学習し、効用を平均する（ランダム部分空間）"""
    pool = FEATS + (MARKET_FEATS if args.res_market else [])
    us, fit, u_a = [], None, None
    for s in range(args.res_subspace):
        sub = sorted(np.random.default_rng(s).choice(pool, size=int(len(pool) * frac), replace=False).tolist())
        fit = fit_combined(*common, feats_override=sub, row_w_extra=row_w_extra, restack=args.restack)
        u_a, u = predict_combined(fit, rep)
        us.append(u)
    return fit, u_a, np.mean(us, axis=0)


def periods(year, refit):
    """報告期間の区切り。year = 年1回（従来）、quarter = 四半期ごとに学習し直す（技術の探索 T3）"""
    if refit == "quarter":
        starts = [f"{year}-{m:02d}-01" for m in (1, 4, 7, 10)]
        ends = starts[1:] + [f"{year + 1}-01-01"]
        return [(s, min(e, END)) for s, e in zip(starts, ends) if s < END]
    return [(f"{year}-01-01", min(f"{year + 1}-01-01", END))]


def run_year(mk, year, args, res_params=None, quiet=False):
    params = res_params or dict(RESIDUAL_PARAMS, **({"num_leaves": args.res_leaves} if args.res_leaves else {}),
                                **({"min_data_in_leaf": args.res_min_data} if args.res_min_data else {}))
    x = "x_shin" if args.shin else "x_mkt"
    b_cols = [x] if args.no_base else [x, "u_base"] + (["u_base2"] if args.base2 else [])
    extra = ([c for k in args.res_market_extra for c in MARKET_EXTRA[k]] + (PAST_COLS + PAST_DIFF if args.past_mkt else [])
             + (["x_shin", "shin_z"] if args.shin else []))
    races, preds = [], []
    for rs, re_ in periods(year, args.refit):
        cut = pd.Timestamp(rs)
        es_start = f"{cut.year - 1}-{cut.month:02d}-01"
        start = f"{cut.year - 1 - args.res_years}-{cut.month:02d}-01" if args.res_years else "2015-01-01"   # 0 = 2015年から全部
        tr, va, rep = between(mk, start, es_start), between(mk, es_start, rs), between(mk, rs, re_)
        common = (tr, va, es_start, args.half_life, args.res_market, extra, args.residual, args.k, args.lam,
                  args.stage_w, params, SEEDS if args.res_seeds else None, b_cols, args.temp)
        row_w_extra = adversarial_weight(tr, va) if args.adv_weight else None
        if args.res_split:
            fit, u_a, u_c = fit_by_surface(tr, va, rep, common, row_w_extra, args)
        elif args.res_subspace:
            fit, u_a, u_c = fit_subspace(rep, common, row_w_extra, args)
        else:
            fit = fit_combined(*common, row_w_extra=row_w_extra, restack=args.restack)
            if args.res_topk:
                imp = pd.Series(fit["residual"].feature_importance("gain"), index=fit["feats"]).sort_values(ascending=False)
                fit = fit_combined(*common, feats_override=list(imp.index[:args.res_topk]), row_w_extra=row_w_extra,
                                   restack=args.restack)
            u_a, u_c = predict_combined(fit, rep)
        g = RaceGroups(rep["race_id"], rep["win"])
        (ll_a, p_a), (ll_c, p_c) = g.ll(u_a), g.ll(u_c)
        if not quiet:
            print(f"  {rs}〜 {len(ll_a):>5}R  C − A {np.mean(ll_c - ll_a):+.4f} / B の係数 {np.round(fit['beta_b'], 3).tolist()}"
                  f" / 木 {residual_trees(fit)}本 / 温度 {fit['temp']:.3f}", flush=True)
        races.append(pd.DataFrame({"race_id": rep["race_id"].iloc[g.starts].to_numpy(), "year": year, "ll_a": ll_a, "ll_c": ll_c}))
        preds.append(rep[["race_id", "race_date", "horse_number", "win", "win_odds"]].assign(p_a=p_a, p_c=p_c))
    return pd.concat(races, ignore_index=True), pd.concat(preds, ignore_index=True)


SELECT_YEARS, CONFIRM_YEARS = range(2019, 2023), range(2023, 2027)   # 技術の探索の2段階（docs/rebuild_plan.md）


def two_stage(both):
    """選ぶ期間（95%下限 > 0）と確かめる期間（90%区間の下限 > 0）での、同じレースの C の差"""
    d = both["ll_c"] - both["ll_c_ref"]
    sel, con = d[both["year"].isin(SELECT_YEARS)], d[both["year"].isin(CONFIRM_YEARS)]
    out = []
    if len(sel):
        lo, hi = bootstrap_ci(sel)
        out.append(f"選ぶ期間 2019〜2022 {sel.mean():+.4f} [95% {lo:+.4f}, {hi:+.4f}] → {'通過' if lo > 0 else '不通過'}")
    if len(con):
        lo, hi = bootstrap_ci(con, level=0.90)
        out.append(f"確かめる期間 2023〜2026 {con.mean():+.4f} [90% {lo:+.4f}, {hi:+.4f}] → {'確認' if lo > 0 else '未確認'}")
    return out


def tune(mk, args, ref):
    """技術の探索 T1: 残差のハイパーパラメータを無作為に試し、選ぶ期間だけで採点する"""
    rng = np.random.default_rng(0)
    out = BENCH_DIR / f"{args.tag}_tune.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []   # 途中で止まっても、終わった試行は飛ばす
    done = {r["trial"] for r in rows}
    for t in range(args.tune):
        p = dict(RESIDUAL_PARAMS, learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
                 num_leaves=int(rng.choice([7, 15, 31, 63])), min_data_in_leaf=int(rng.choice([100, 200, 500, 1000, 2000])),
                 feature_fraction=float(rng.choice([0.3, 0.5, 0.7, 0.9])), bagging_fraction=float(rng.choice([0.5, 0.7, 0.8, 1.0])),
                 lambda_l1=float(rng.choice([0.0, 1.0, 10.0])), lambda_l2=float(rng.choice([1.0, 10.0, 50.0, 100.0])),
                 boosting=str(rng.choice(["gbdt", "gbdt", "gbdt", "dart"])))
        if p["boosting"] == "dart":   # DART は early stopping が効かないので学習回数を固定する
            p.update(drop_rate=0.1, skip_drop=0.5, num_iterations=400)
        if t in done:
            continue
        races = pd.concat([run_year(mk, y, args, p, quiet=True)[0] for y in SELECT_YEARS], ignore_index=True)
        both = races.merge(ref, on=["race_id", "year"], suffixes=("", "_ref"))
        d = both["ll_c"] - both["ll_c_ref"]
        lo, hi = bootstrap_ci(d)
        keys = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
                "lambda_l1", "lambda_l2", "boosting")
        rows.append(dict(trial=t, diff=d.mean(), lo=lo, hi=hi, params=json.dumps({k: p[k] for k in keys})))
        print(f"  試行{t:>2} 選ぶ期間の差 {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {rows[-1]['params']}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    res = pd.DataFrame(rows).sort_values("diff", ascending=False)
    res.to_csv(out, index=False)
    print("最良:", res.iloc[0].to_dict())


def show(label, d):
    lo, hi = bootstrap_ci(d)
    return f"{label} {np.mean(d):+.4f} [{lo:+.4f}, {hi:+.4f}]"


def roi_table(preds):
    d = preds.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    starts = RaceGroups(d["race_id"], d["win"]).starts
    idx = np.random.default_rng(0).integers(0, len(starts), (2000, len(starts)))
    for th in THRESHOLDS:
        sel = (d["p_c"] * d["win_odds"] >= th).to_numpy()
        stake = np.add.reduceat(sel * 1.0, starts)
        ret = np.add.reduceat(sel * d["win"].to_numpy() * d["win_odds"].to_numpy(), starts)
        boots = ret[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1) * 100
        print(f"  期待値{th:.1f}以上: {int(sel.sum()):>6,}点 回収率 {ret.sum() / max(stake.sum(), 1) * 100:5.1f}%"
              f" [{np.percentile(boots, 2.5):5.1f}, {np.percentile(boots, 97.5):5.1f}]（確定オッズ）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="この版の名前（結果の保存名）")
    ap.add_argument("--ref", default="current", help="比べる相手の版")
    ap.add_argument("--base", default="current", help="基礎モデルの予測（current = C[all] と同じもの）")
    ap.add_argument("--residual", choices=["win", "pl"], default="win")
    ap.add_argument("--half-life", type=float, help="残差の学習を直近重視にする半減期（日）")
    ap.add_argument("--res-years", type=int, default=3, help="残差の学習に使う年数（0 = 2015年から全部）")
    ap.add_argument("--res-market", action="store_true", help="残差の入力にオッズ由来の列（MARKET_FEATS）を足す")
    ap.add_argument("--res-market-extra", type=lambda s: s.split(","), default=[], help="place / race（カンマ区切り）")
    ap.add_argument("--res-seeds", action="store_true", help="残差を3シードで学習して平均する（実験6）")
    ap.add_argument("--past-mkt", action="store_true", help="残差の入力に過去の走の市場評価を足す（実験6）")
    ap.add_argument("--years", help="採点する年（例 2019-2022）。既定は 2019〜2026")
    ap.add_argument("--temp", action="store_true", help="C の効用に掛ける温度を ES の期間で合わせる（T2）")
    ap.add_argument("--base2", help="2つ目の基礎モデルの予測を B の段に別の変数として足す（T5）")
    ap.add_argument("--no-base", action="store_true", help="基礎モデルを使わない1段階の構成（T6）")
    ap.add_argument("--refit", choices=["year", "quarter"], default="year", help="残差を学習し直す間隔（T3）")
    ap.add_argument("--tune", type=int, help="残差のハイパーパラメータを N 通り試す（T1。選ぶ期間だけで採点）")
    ap.add_argument("--res-params", type=json.loads, help="残差のハイパーパラメータ（JSON。RESIDUAL_PARAMS を上書き）")
    ap.add_argument("--shin", action="store_true", help="Shin の確率を出発点と残差の入力に使う（T8）")
    ap.add_argument("--restack", action="store_true", help="ES の期間で B と残差の係数を推定し直す（T9）")
    ap.add_argument("--res-topk", type=int, help="残差を重要度の上位 N 列だけで学習し直す（T11）")
    ap.add_argument("--res-split", action="store_true", help="残差を芝とダートで別々に学習する（T12）")
    ap.add_argument("--adv-weight", action="store_true", help="敵対的検証の重みで学習する（T13）")
    ap.add_argument("--res-subspace", type=int, help="残差を列の半分ずつ N 通りで学習して平均する（T15）")
    ap.add_argument("--res-leaves", type=int)
    ap.add_argument("--res-min-data", type=int)
    ap.add_argument("--build-base", metavar="NAME", help="基礎モデルの予測を作り直して base_NAME として保存")
    ap.add_argument("--base-objective", choices=["win", "pl"], default="win")
    ap.add_argument("--base-half-life", type=float)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--stage-w", type=lambda s: [float(x) for x in s.split(",")], help="各着の段の重み（例 1,0.5,0.5）")
    args = ap.parse_args()

    df = load()
    print(f"読み込み: {df['race_date'].min():%Y-%m-%d}〜{df['race_date'].max():%Y-%m-%d}（{END} 以降は使わない）", flush=True)
    if args.build_base:
        build_base(df, args.build_base, args)
        return
    if not args.tag:
        raise SystemExit("--tag が必要")

    mk = market(df, args.base)
    if args.base2:
        b2 = pd.read_parquet(base_file(args.base2)).rename(columns={"u_base": "u_base2"})
        mk = mk.merge(b2[["race_id", "horse_id", "u_base2"]], on=["race_id", "horse_id"], how="left")
        mk = mk[mk.groupby("race_id")["u_base2"].transform(lambda s: s.notna().all())].reset_index(drop=True)
    if args.res_market_extra:
        mk = market_extra(mk)
    if args.past_mkt:
        mk = add_past_market(mk)
    if args.shin:
        mk = add_shin(mk)
    print(f"[{args.tag}] 基礎={args.base} 残差={args.residual} k={args.k} λ={args.lam} 段の重み={args.stage_w} "
          f"半減期={args.half_life} 残差の学習年数={args.res_years or '2015〜'} オッズ列={args.res_market}{args.res_market_extra or ''} "
          f"葉={args.res_leaves} 最小={args.res_min_data} 残差3シード={args.res_seeds} 過去の市場評価={args.past_mkt} "
          f"温度={args.temp} 基礎2={args.base2} 基礎なし={args.no_base} 再学習={args.refit} 残差の設定={args.res_params} / {mk['race_id'].nunique():,}R", flush=True)
    years = YEARS
    if args.years:
        a, b = (int(x) for x in args.years.split("-"))
        years = range(a, b + 1)
    if args.tune:
        tune(mk, args, pd.read_parquet(BENCH_DIR / f"{args.ref}.parquet"))
        return
    res_params = dict(RESIDUAL_PARAMS, **args.res_params) if args.res_params else None
    results = [run_year(mk, y, args, res_params) for y in years]
    races = pd.concat([r for r, _ in results], ignore_index=True)
    preds = pd.concat([p for _, p in results], ignore_index=True)
    BENCH_DIR.mkdir(exist_ok=True)
    races.to_parquet(BENCH_DIR / f"{args.tag}.parquet", index=False)
    preds.to_parquet(BENCH_DIR / f"{args.tag}_preds.parquet", index=False)

    print(f"\n=== {args.tag}: C − A（1レースあたり、勝ち馬の対数尤度） ===")
    for y, d in races.groupby("year"):
        print(f"  {show(f'{y}年 {len(d):>5}R', d['ll_c'] - d['ll_a'])}")
    print(f"  {show(f'通算  {len(races):>5}R', races['ll_c'] - races['ll_a'])}")
    roi_table(preds)

    ref_path = BENCH_DIR / f"{args.ref}.parquet"
    if args.tag != args.ref and ref_path.exists():
        both = races.merge(pd.read_parquet(ref_path), on=["race_id", "year"], suffixes=("", "_ref"))
        d = both["ll_c"] - both["ll_c_ref"]
        print(f"\n=== {args.tag} − {args.ref}（同じレースの C の差。採用の条件は通算の95%下限 > 0） ===")
        for y, s in both.assign(d=d).groupby("year"):
            print(f"  {show(f'{y}年 {len(s):>5}R', s['d'])}")
        lo, _ = bootstrap_ci(d)
        print(f"  {show(f'通算  {len(both):>5}R', d)} → {'採用の条件を満たす' if lo > 0 else '満たさない'}")
        for line in two_stage(both):
            print("  " + line)


if __name__ == "__main__":
    main()
