# v2/model_family.py
# 基礎モデルの種類を比べる（docs/rebuild_plan.md「モデルの種類の比較」）。特徴量は176列で固定し、モデルだけ差し替える。
#   --stage1 : オッズ抜きの当てやすさ。学習 2013〜2017 / ES 2018 / 報告 2019〜2020 で
#              LightGBM・CatBoost・MLP と、その混合（較正した効用の平均）の対数尤度を比べる
#   --stage2 : オッズへの上積み。CatBoost・MLP の年ごとのOOS予測を作り、LightGBM(3シード) と混ぜた基礎モデルで
#              model_trip.py と同じ A / C の比較を 報告2019年・2020年 で行う
# 使い方: python -m v2.model_family --stage1
#         python -m v2.model_family --stage2
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2 import features as ft
from v2 import features_extra as fx
from v2 import model_trip as mt
from v2.model_base import PARAMS, eligible
from v2.paths import V2_DIR, table_path
from v2.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

FEATS = ft.FEATURES_TRIP + fx.EXTRA_FEATURES
STAGE1_SPLIT = {"train": ("2013-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"),
                "report": ("2019-01-01", "2021-01-01")}
BASE_YEARS = range(2015, 2021)          # 段階2で必要なのは 報告2020年の学習に使う 2015〜2020 まで
MIX_PATH = V2_DIR / "base_oos_mix_2015_2023.parquet"


# ---- それぞれの種類の学習（tr で学習し va で early stopping、predict(df) が生の効用を返す） ----

def fit_lgb(tr, va, seed=42):
    cats = [c for c in FEATS if c in ft.CATEGORICAL]
    dtr = lgb.Dataset(tr[FEATS], label=tr["win"], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[FEATS], label=va["win"], categorical_feature=cats, reference=dtr)
    params = dict(PARAMS, seed=seed, objective=lgb_objective(RaceGroups(tr["race_id"], tr["win"])))
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  feval=lgb_metric(RaceGroups(va["race_id"], va["win"])),
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return lambda d: m.predict(d[FEATS], num_iteration=m.best_iteration), m.best_iteration


def _cat_pool(d, cats):
    from catboost import Pool
    x = d[FEATS].copy()
    for c in cats:
        x[c] = x[c].astype(object).fillna("NA").astype(str)
    return Pool(x, label=d["win"].values, group_id=pd.factorize(d["race_id"])[0], cat_features=cats)


def fit_cat(tr, va, seed=42):
    from catboost import CatBoost
    cats = [c for c in FEATS if c in ft.CATEGORICAL]
    m = CatBoost(dict(loss_function="QuerySoftMax", iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=10.0,
                      random_seed=seed, od_type="Iter", od_wait=200, verbose=False, thread_count=-1))
    m.fit(_cat_pool(tr, cats), eval_set=_cat_pool(va, cats), use_best_model=True)
    return lambda d: m.predict(_cat_pool(d, cats)), m.tree_count_


def _mlp_matrix(d, stats, cats, cat_levels):
    x = d[[c for c in FEATS if c not in cats]].to_numpy(dtype=np.float32)
    med, iqr = stats
    x = np.clip((x - med) / iqr, -5, 5)
    x = np.nan_to_num(x, nan=0.0)
    dummies = [np.equal.outer(d[c].astype(object).fillna("NA").astype(str).to_numpy(), lv).astype(np.float32)
               for c, lv in zip(cats, cat_levels)]
    return np.hstack([x] + dummies)


def fit_mlp(tr, va, seed=42, epochs=40, hidden=(256, 128)):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cats = [c for c in FEATS if c in ft.CATEGORICAL]
    num = [c for c in FEATS if c not in cats]
    q = tr[num].quantile([0.25, 0.5, 0.75]).to_numpy(dtype=np.float32)
    stats = (q[1], np.maximum(q[2] - q[0], 1e-3))
    levels = [np.array(sorted(set(tr[c].astype(object).fillna("NA").astype(str)))) for c in cats]

    def to_tensor(d):
        g = RaceGroups(d["race_id"], d["win"])
        x = torch.as_tensor(_mlp_matrix(d, stats, cats, levels), device=dev)
        return x, g, torch.as_tensor(g.ridx, device=dev), torch.as_tensor(g.y, dtype=torch.float32, device=dev)

    xtr, gtr, ridx_tr, ytr = to_tensor(tr)
    xva, gva, ridx_va, yva = to_tensor(va)
    net = nn.Sequential(nn.Linear(xtr.shape[1], hidden[0]), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden[1], 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n_races = len(gtr)
    starts = torch.as_tensor(np.r_[gtr.starts, len(ytr)], device=dev)

    def race_nll(u, ridx, y, n):
        """レース内ソフトマックスの負の対数尤度（勝ち馬の確率）"""
        lse = torch.zeros(n, device=dev).index_reduce_(0, ridx, u, "amax", include_self=False)
        ex = torch.zeros(n, device=dev).index_add_(0, ridx, torch.exp(u - lse[ridx]))
        num = torch.zeros(n, device=dev).index_add_(0, ridx, y * u)
        return -(num - (torch.log(ex) + lse)).mean()

    best, best_state, bad = -np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(n_races)
        for i in range(0, n_races, 512):
            sel = order[i:i + 512]
            rows = torch.cat([torch.arange(starts[r], starts[r + 1], device=dev) for r in sel])
            sub_ridx = torch.repeat_interleave(torch.arange(len(sel), device=dev),
                                               torch.as_tensor([int(starts[r + 1] - starts[r]) for r in sel], device=dev))
            opt.zero_grad()
            race_nll(net(xtr[rows]).squeeze(1), sub_ridx, ytr[rows], len(sel)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ll = -race_nll(net(xva).squeeze(1), ridx_va, yva, len(gva)).item()
        if ll > best + 1e-5:
            best, best_state, bad = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(d):
        with torch.no_grad():
            x = torch.as_tensor(_mlp_matrix(d, stats, cats, levels), device=dev)
            return net(x).squeeze(1).cpu().numpy().astype(float)

    return predict, ep + 1


FITTERS = {"lgb": fit_lgb, "cat": fit_cat, "mlp": fit_mlp}


def calibrated(raw_va, raw_rep, va):
    """検証期間で温度を合わせた効用を返す"""
    beta = fit_logit(raw_va, RaceGroups(va["race_id"], va["win"]))[0]
    return beta * raw_va, beta * raw_rep, beta


# ---- 段階1: オッズ抜きの当てやすさ ----

def stage1(df):
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True)
             for k, (s, e) in STAGE1_SPLIT.items()}
    for k, v in parts.items():
        print(f"{k:<6} {STAGE1_SPLIT[k][0]}〜{STAGE1_SPLIT[k][1]}: {v['race_id'].nunique()}R", flush=True)
    g_rep = RaceGroups(parts["report"]["race_id"], parts["report"]["win"])

    cal = {}
    for name, fit in FITTERS.items():
        predict, rounds = fit(parts["train"], parts["valid"])
        u_va, u_rep, beta = calibrated(predict(parts["valid"]), predict(parts["report"]), parts["valid"])
        cal[name] = (u_va, u_rep)
        ll, _ = g_rep.ll(u_rep)
        print(f"{name:<12} 反復{rounds:>5} 温度{beta:+.3f} 対数尤度 {ll.mean():.4f}", flush=True)

    base_ll, _ = g_rep.ll(cal["lgb"][1])
    for names in [("lgb", "cat"), ("lgb", "mlp"), ("cat", "mlp"), ("lgb", "cat", "mlp")]:
        mix_va = np.mean([cal[n][0] for n in names], axis=0)
        mix_rep = np.mean([cal[n][1] for n in names], axis=0)
        u_va, u_rep, beta = calibrated(mix_va, mix_rep, parts["valid"])
        ll, _ = g_rep.ll(u_rep)
        lo, hi = bootstrap_ci(ll - base_ll)
        print(f"混合 {'+'.join(names):<20} 温度{beta:+.3f} 対数尤度 {ll.mean():.4f}  lgb との差 {np.mean(ll - base_ll):+.4f} [{lo:+.4f}, {hi:+.4f}]",
              flush=True)


# ---- 段階2: オッズへの上積み ----

def build_mix_base(df, families):
    """年ごとに「その年を学習していない予測」を作り、LightGBM(3シード) の既存の予測と混ぜる"""
    lgb_base = pd.read_parquet(mt.base_path("all"))
    out = []
    for year in BASE_YEARS:
        tr = df[(df["race_date"] >= "2013-01-01") & (df["race_date"] < f"{year - 1}-01-01")].reset_index(drop=True)
        va = df[(df["race_date"] >= f"{year - 1}-01-01") & (df["race_date"] < f"{year}-01-01")].reset_index(drop=True)
        te = df[(df["race_date"] >= f"{year}-01-01") & (df["race_date"] < f"{year + 1}-01-01")].reset_index(drop=True)
        parts = []
        for name in families:
            predict, rounds = FITTERS[name](tr, va)
            _, u_te, beta = calibrated(predict(va), predict(te), va)
            parts.append(te[["race_id", "horse_id"]].assign(**{f"u_{name}": u_te}))
            print(f"[{name}] {year}: 学習 {tr['race_id'].nunique()}R / 反復 {rounds} / 温度 {beta:+.3f}", flush=True)
        m = parts[0]
        for p in parts[1:]:
            m = m.merge(p, on=["race_id", "horse_id"])
        out.append(m)
    mix = pd.concat(out, ignore_index=True).merge(lgb_base, on=["race_id", "horse_id"], how="inner")
    cols = [f"u_{n}" for n in families] + ["u_base"]      # u_base は LightGBM 3シードの較正済み効用
    mix["u_base"] = mix[cols].mean(axis=1)
    mix[["race_id", "horse_id", "u_base"]].to_parquet(MIX_PATH, index=False)
    print(f"混合の基礎モデル {len(mix)}行 → {MIX_PATH}", flush=True)


def stage2(df, families, skip_build=False):
    if not skip_build:
        build_mix_base(df, families)
    mt.FEATURE_SETS["mix"] = FEATS
    orig = mt.base_path

    def patched(name):
        return MIX_PATH if name == "mix" else orig(name)

    mt.base_path = patched
    markets = {}
    for name in ["all", "mix"]:
        markets[name], n_mismatch = mt.with_market(df, name)
        print(f"[{name}] オッズ・基礎モデルの予測がそろうレース {markets[name]['race_id'].nunique()}R", flush=True)
    for year, split in mt.DEV_SPLITS.items():
        print(f"\n=== 開発 報告{year}年 ===")
        for name in ["all", "mix"]:
            mt.run_split(markets[name], name, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--families", default="cat,mlp", help="段階2で LightGBM に混ぜる種類")
    ap.add_argument("--skip-build", action="store_true", help="段階2で混合の基礎モデルを作り直さない")
    args = ap.parse_args()
    df = eligible(pd.read_parquet(table_path("features")))
    if args.stage1:
        stage1(df)
    if args.stage2:
        stage2(df, args.families.split(","), args.skip_build)


if __name__ == "__main__":
    main()
