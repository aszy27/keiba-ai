# v2/evaluate.py
# 候補2以降の開発用の採点（docs/rebuild_plan.md「候補2以降の開発」）。凍結した C[all] のコードは変えず、変更はここで試す。
# 採点は年ごとの前進検証（2019〜2026-09-06）の C − A。各年 Y は
#   基礎モデル: 学習 2013〜Y-2 / ES Y-1（3シード平均、温度を Y-1年で合わせる）
#   残差:       学習 [Y-4, Y-1) / ES [Y-1, Y)
# 結果はレース単位で data/v2/bench/<tag>.parquet に保存し、--ref の版と同じレースで比べる（対応のある差）。
#
# 使い方: python -m v2.evaluate --tag current                          # 現行（C[all] と同じ作り方）。最初に1回
#         python -m v2.evaluate --tag res_pl3 --residual pl            # 残差を 1〜3着の順序で学習
#         python -m v2.evaluate --build-base base_pl3 --base-objective pl   # 基礎モデルを 1〜3着で学習し直す（時間がかかる）
#         python -m v2.evaluate --tag base_pl3 --base base_pl3
#         python -m v2.evaluate --tag res_all_hl --res-years 0 --half-life 730   # 残差の学習期間を広げ、直近を重く
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2.model import trip as mt
from v2.model.base import PARAMS, eligible
from v2.model.combined import RESIDUAL_PARAMS
from v2.paths import V2_DIR, table_path
from v2.model.plackett import PLGroups, lgb_objective_pl
from v2.model.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

END = "2026-09-07"                       # 前向き検証の期間は読み込み直後に捨てる
YEARS = range(2019, 2027)
BASE_YEARS = range(2015, 2027)
BENCH_DIR = V2_DIR / "bench"
CURRENT_BASE = V2_DIR / "base_oos_all_2015_2026.parquet"   # C[all] の基礎モデルの予測（walkforward.py --build-base）
FEATS = mt.FEATURE_SETS["all"]
THRESHOLDS = [1.0, 1.1, 1.2, 1.3]
MARKET_FEATS = ["x_mkt", "mkt_rank"]     # --res-market で残差の入力に足すオッズ由来の列
MARKET_EXTRA = {"place": ["x_place", "place_gap", "place_spread"],          # --res-market-extra で足す列
                "race": ["fav_log_odds", "mkt_entropy", "n_runners"]}


def load():
    df = eligible(pd.read_parquet(table_path("features")))
    return df[df["race_date"] < END].reset_index(drop=True)


def base_file(name):
    """"a+b" は2つの基礎モデルの予測（温度合わせ後の効用）の平均"""
    if "+" in name:
        path = BENCH_DIR / f"base_{name}.parquet"
        if not path.exists():
            parts = [pd.read_parquet(base_file(n)).set_index(["race_id", "horse_id"])["u_base"] for n in name.split("+")]
            pd.concat(parts, axis=1, join="inner").mean(axis=1).rename("u_base").reset_index().to_parquet(path, index=False)
        return path
    return CURRENT_BASE if name == "current" else BENCH_DIR / f"base_{name}.parquet"


def recency_weight(dates, end, half_life):
    if not half_life:
        return None
    age = (pd.Timestamp(end) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half_life)


def train(tr, va, feats, params, objective="win", k=3, lam=0.75, stage_w=None, base_tr=0.0, base_va=0.0, row_w=None):
    """objective=win は model_combined._train と同じ。pl は 1〜k着の順序で学習する（ES は両方とも勝ち馬の対数尤度）"""
    cats = [c for c in feats if c in mt.ft.CATEGORICAL]
    dtr = lgb.Dataset(tr[feats], label=tr["win"], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["win"], categorical_feature=cats, reference=dtr)
    if objective == "win" and row_w is None:
        obj = lgb_objective(RaceGroups(tr["race_id"], tr["win"]), base_tr)
    else:
        kk = 1 if objective == "win" else k
        obj = lgb_objective_pl(PLGroups(tr["race_id"], tr["finish_pos"], kk, lam, stage_w), base_tr, row_w)
    return lgb.train(dict(params, objective=obj), dtr, num_boost_round=5000, valid_sets=[dva],
                     feval=lgb_metric(RaceGroups(va["race_id"], va["win"]), base_va),
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def build_base(df, name, args):
    """1年ずつ base_NAME_YYYY.parquet に保存し、保存済みの年は飛ばす（途中で止まっても続きから）"""
    BENCH_DIR.mkdir(exist_ok=True)
    for year in BASE_YEARS:
        part = BENCH_DIR / f"base_{name}_{year}.parquet"
        if part.exists():
            print(f"[{name}] {year}: 保存済み", flush=True)
            continue
        tr = df[(df["race_date"] >= "2013-01-01") & (df["race_date"] < f"{year - 1}-01-01")].reset_index(drop=True)
        va = df[(df["race_date"] >= f"{year - 1}-01-01") & (df["race_date"] < f"{year}-01-01")].reset_index(drop=True)
        te = df[(df["race_date"] >= f"{year}-01-01") & (df["race_date"] < f"{year + 1}-01-01")].reset_index(drop=True)
        row_w = recency_weight(tr["race_date"], f"{year - 1}-01-01", args.base_half_life)
        u_va, u_te, rounds = 0.0, 0.0, []
        for seed in mt.SEEDS:
            m = train(tr, va, FEATS, dict(PARAMS, seed=seed), args.base_objective, args.k, args.lam, args.stage_w,
                      row_w=row_w)
            u_va = u_va + m.predict(va[FEATS], num_iteration=m.best_iteration) / len(mt.SEEDS)
            u_te = u_te + m.predict(te[FEATS], num_iteration=m.best_iteration) / len(mt.SEEDS)
            rounds.append(m.best_iteration)
        beta = fit_logit(u_va, RaceGroups(va["race_id"], va["win"]))[0]
        te[["race_id", "horse_id"]].assign(u_base=beta * u_te).to_parquet(part, index=False)
        print(f"[{name}] {year}: 学習 {tr['race_id'].nunique()}R / 木 {rounds}本 / 温度 {beta:.3f}", flush=True)
    parts = [pd.read_parquet(BENCH_DIR / f"base_{name}_{y}.parquet") for y in BASE_YEARS]
    pd.concat(parts, ignore_index=True).to_parquet(base_file(name), index=False)
    print("保存:", base_file(name))


def market(df, base):
    orig = mt.base_path
    mt.base_path = lambda name: base_file(base)
    try:
        d, _ = mt.with_market(df, "all")
    finally:
        mt.base_path = orig
    d["mkt_rank"] = d.groupby("race_id")["x_mkt"].rank(ascending=False, method="min")
    # 複勝オッズ（スナップショットにもある列）。取消などで無い馬は NaN のまま LightGBM に任せる
    po = pd.read_parquet(table_path("odds_final"), columns=["race_id", "horse_number", "place_odds_min", "place_odds_max"])
    po["horse_number"] = po["horse_number"].astype(float)
    d = d.merge(po, on=["race_id", "horse_number"], how="left")
    lo = d["place_odds_min"].where(d["place_odds_min"] > 0)
    hi = d["place_odds_max"].where(d["place_odds_max"] > 0)
    d["x_place"] = np.log(1.0 / lo)
    d["place_gap"] = d["x_place"] - d["x_mkt"]
    d["place_spread"] = np.log(hi / lo)
    # レース全体のオッズの形
    g = d.groupby("race_id")
    d["fav_log_odds"] = np.log(g["win_odds"].transform("min"))
    q = np.exp(d["x_mkt"])
    d["mkt_entropy"] = (-q * d["x_mkt"]).groupby(d["race_id"]).transform("sum")
    d["n_runners"] = g["race_id"].transform("size")
    return d


def run_year(mk, year, args):
    start = f"{year - 1 - args.res_years}-01-01" if args.res_years else "2015-01-01"   # 0 = 基礎モデルの予測がある2015年から全部
    split = {"train": (start, f"{year - 1}-01-01"), "valid": (f"{year - 1}-01-01", f"{year}-01-01"),
             "report": (f"{year}-01-01", min(f"{year + 1}-01-01", END))}
    parts = {k: mk[(mk["race_date"] >= s) & (mk["race_date"] < e)].reset_index(drop=True) for k, (s, e) in split.items()}
    g = {k: RaceGroups(v["race_id"], v["win"]) for k, v in parts.items()}
    tr, rep = parts["train"], parts["report"]
    beta_a = fit_logit(tr[["x_mkt"]].values, g["train"])
    ll_a, p_a = g["report"].ll(rep[["x_mkt"]].values @ beta_a)
    beta_b = fit_logit(tr[["x_mkt", "u_base"]].values, g["train"])
    base = {k: v[["x_mkt", "u_base"]].values @ beta_b for k, v in parts.items()}
    row_w = recency_weight(tr["race_date"], split["train"][1], args.half_life)
    feats = FEATS + (MARKET_FEATS if args.res_market else []) + [c for k in args.res_market_extra for c in MARKET_EXTRA[k]]
    params = dict(RESIDUAL_PARAMS, **({"num_leaves": args.res_leaves} if args.res_leaves else {}),
                  **({"min_data_in_leaf": args.res_min_data} if args.res_min_data else {}))
    m = train(tr, parts["valid"], feats, params, args.residual, args.k, args.lam, args.stage_w,
              base["train"], base["valid"], row_w)
    ll_c, p_c = g["report"].ll(base["report"] + m.predict(rep[feats], num_iteration=m.best_iteration))
    print(f"  {year}年 {len(ll_a):>5}R  C − A {np.mean(ll_c - ll_a):+.4f} / 基礎モデルの係数 {beta_b[1]:+.3f} / 木 {m.best_iteration}本",
          flush=True)
    races = pd.DataFrame({"race_id": rep["race_id"].iloc[g["report"].starts].to_numpy(), "year": year, "ll_a": ll_a, "ll_c": ll_c})
    preds = rep[["race_id", "race_date", "horse_number", "win", "win_odds"]].assign(p_a=p_a, p_c=p_c)
    return races, preds


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
    print(f"[{args.tag}] 基礎={args.base} 残差={args.residual} k={args.k} λ={args.lam} 段の重み={args.stage_w} "
          f"半減期={args.half_life} 残差の学習年数={args.res_years or '2015〜'} オッズ列={args.res_market}{args.res_market_extra or ''} "
          f"葉={args.res_leaves} 最小={args.res_min_data} / {mk['race_id'].nunique():,}R", flush=True)
    results = [run_year(mk, y, args) for y in YEARS]
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


if __name__ == "__main__":
    main()
