# experiments/model_family.py
# 基礎モデルの種類を比べる（docs/rebuild_plan.md「モデルの種類の比較」）。特徴量は176列で固定し、モデルだけ差し替える。
#   --stage1 : オッズ抜きの当てやすさ。学習 2013〜2017 / ES 2018 / 報告 2019〜2020 で各種類と混合を比べる
#   --stage2 : オッズへの上積み。選んだ種類の年ごとのOOS予測を作り、LightGBM(3シード) と混ぜた基礎モデルで
#              model_trip.py と同じ A / C の比較を 報告2019年・2020年 で行う
# 種類:
#   lgb 現行の LightGBM / cat CatBoost(QuerySoftMax) / mlp 素の多層パーセプトロン
#   dae ノイズ除去オートエンコーダで作った表現を足した MLP（Porto Seguro 1位解法と同じ考え方）
#   emb 騎手・調教師のIDを埋め込みにした MLP
#   set 出走馬全体を1つの集合として見る Transformer（他馬との比較を直接学習する）
# 使い方: python -m experiments.model_family --stage1
#         python -m experiments.model_family --stage2 --families mlp,set
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from prep import features as ft
from prep import features_extra as fx
from model import trip as mt
from model.base import PARAMS, eligible
from paths import V2_DIR, table_path
from model.softmax import RaceGroups, bootstrap_ci, fit_logit, lgb_metric, lgb_objective

FEATS = ft.FEATURES_TRIP + fx.EXTRA_FEATURES
CATS = [c for c in FEATS if c in ft.CATEGORICAL]
NUMS = [c for c in FEATS if c not in ft.CATEGORICAL]
ID_FEATS = ["jockey_id", "trainer_id"]
STAGE1_SPLIT = {"train": ("2013-01-01", "2018-01-01"), "valid": ("2018-01-01", "2019-01-01"),
                "report": ("2019-01-01", "2021-01-01")}
BASE_YEARS = range(2015, 2021)          # 段階2で必要なのは 報告2020年の学習に使う 2015〜2020 まで
MIX_PATH = V2_DIR / "base_oos_mix_2015_2023.parquet"
MAX_FIELD = 18
MIXES = [("lgb", "mlp"), ("lgb", "dae"), ("lgb", "emb"), ("lgb", "set"), ("lgb", "cat"),
         ("lgb", "mlp", "set"), ("lgb", "dae", "set"), ("lgb", "mlp", "dae", "emb", "set"),
         ("lgb", "cat", "mlp", "dae", "emb", "set")]


# ---- 勾配ブースティング ----

def fit_lgb(tr, va, seed=42):
    dtr = lgb.Dataset(tr[FEATS], label=tr["win"], categorical_feature=CATS, free_raw_data=False)
    dva = lgb.Dataset(va[FEATS], label=va["win"], categorical_feature=CATS, reference=dtr)
    params = dict(PARAMS, seed=seed, objective=lgb_objective(RaceGroups(tr["race_id"], tr["win"])))
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  feval=lgb_metric(RaceGroups(va["race_id"], va["win"])),
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return lambda d: m.predict(d[FEATS], num_iteration=m.best_iteration), m.best_iteration


def _cat_pool(d):
    from catboost import Pool
    x = d[FEATS].copy()
    for c in CATS:
        x[c] = x[c].astype(object).fillna("NA").astype(str)
    return Pool(x, label=d["win"].values, group_id=pd.factorize(d["race_id"])[0], cat_features=CATS)


def fit_cat(tr, va, seed=42):
    from catboost import CatBoost
    m = CatBoost(dict(loss_function="QuerySoftMax", iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=10.0,
                      random_seed=seed, od_type="Iter", od_wait=200, verbose=False, thread_count=-1))
    m.fit(_cat_pool(tr), eval_set=_cat_pool(va), use_best_model=True)
    return lambda d: m.predict(_cat_pool(d)), m.tree_count_


# ---- ニューラルネット共通 ----

def _prep(tr):
    """数値列の中央値・四分位範囲と、カテゴリ・IDの水準（学習期間から作る）"""
    q = tr[NUMS].quantile([0.25, 0.5, 0.75]).to_numpy(dtype=np.float32)
    levels = [np.array(sorted(set(tr[c].astype(object).fillna("NA").astype(str)))) for c in CATS]
    ids = [{v: i + 1 for i, v in enumerate(sorted(set(tr[c].dropna())))} for c in ID_FEATS]   # 0 は未知
    return (q[1], np.maximum(q[2] - q[0], 1e-3)), levels, ids


def _matrix(d, prep):
    (med, iqr), levels, _ = prep
    x = np.clip((d[NUMS].to_numpy(dtype=np.float32) - med) / iqr, -5, 5)
    x = np.nan_to_num(x, nan=0.0)
    dummies = [np.equal.outer(d[c].astype(object).fillna("NA").astype(str).to_numpy(), lv).astype(np.float32)
               for c, lv in zip(CATS, levels)]
    return np.hstack([x] + dummies)


def _id_codes(d, prep):
    maps = prep[2]
    return np.stack([d[c].map(m).fillna(0).to_numpy(dtype=np.int64) for c, m in zip(ID_FEATS, maps)], axis=1)


def _pad_index(g):
    """レースごとの行番号（足りないところは -1）"""
    counts = np.diff(np.r_[g.starts, len(g.y)])
    idx = np.full((len(g.starts), MAX_FIELD), -1, dtype=np.int64)
    for i, (s, c) in enumerate(zip(g.starts, counts)):
        idx[i, :c] = np.arange(s, s + min(c, MAX_FIELD))
    return idx


def _fit_torch(tr, va, build, seed, epochs=60, patience=6, batch=256, lr=1e-3):
    """レース内ソフトマックスで学習する共通ループ。build(d_in, dev) が (net, forward) を返す。
    forward(net, x, ids, mask) は (レース数, 最大頭数) の効用を返す"""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = _prep(tr)

    def tensors(d):
        g = RaceGroups(d["race_id"], d["win"])
        x = torch.as_tensor(_matrix(d, prep), device=dev)
        ids = torch.as_tensor(_id_codes(d, prep), device=dev)
        idx = torch.as_tensor(_pad_index(g), device=dev)
        y = torch.as_tensor(g.y, dtype=torch.float32, device=dev)
        return g, x, ids, idx, y

    g_tr, x_tr, id_tr, idx_tr, y_tr = tensors(tr)
    g_va, x_va, id_va, idx_va, y_va = tensors(va)
    net, forward = build(x_tr.shape[1], dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def loss_on(x, ids, idx, y):
        mask = idx >= 0
        rows = idx.clamp(min=0)
        u = forward(net, x[rows], ids[rows], mask).masked_fill(~mask, -1e9)
        target = (y[rows] * mask).argmax(dim=1)
        return torch.nn.functional.cross_entropy(u, target)

    best, best_state, bad, ep = -np.inf, None, 0, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(idx_tr))
        for i in range(0, len(order), batch):
            sel = torch.as_tensor(order[i:i + batch], device=dev)
            opt.zero_grad()
            loss_on(x_tr, id_tr, idx_tr[sel], y_tr).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ll = -loss_on(x_va, id_va, idx_va, y_va).item()
        if ll > best + 1e-5:
            best, best_state, bad = ll, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(d):
        g = RaceGroups(d["race_id"], d["win"])
        x = torch.as_tensor(_matrix(d, prep), device=dev)
        ids = torch.as_tensor(_id_codes(d, prep), device=dev)
        idx = torch.as_tensor(_pad_index(g), device=dev)
        mask = idx >= 0
        with torch.no_grad():
            u = forward(net, x[idx.clamp(min=0)], ids[idx.clamp(min=0)], mask).cpu().numpy()
        out = np.zeros(len(d))
        out[idx.cpu().numpy()[mask.cpu().numpy()]] = u[mask.cpu().numpy()]
        return out

    return predict, ep + 1


def _mlp_head(torch, d_in, hidden=(256, 128), p=(0.2, 0.1)):
    from torch import nn
    return nn.Sequential(nn.Linear(d_in, hidden[0]), nn.ReLU(), nn.Dropout(p[0]),
                         nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(p[1]),
                         nn.Linear(hidden[1], 1))


def fit_mlp(tr, va, seed=42):
    import torch

    def build(d_in, dev):
        net = _mlp_head(torch, d_in).to(dev)
        return net, lambda net, x, ids, mask: net(x).squeeze(-1)

    return _fit_torch(tr, va, build, seed)


def fit_emb(tr, va, seed=42, dim=16):
    """騎手・調教師のIDを埋め込みにして足す"""
    import torch
    from torch import nn
    n_ids = [len(m) + 1 for m in _prep(tr)[2]]

    class Net(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(n, dim) for n in n_ids])
            self.head = _mlp_head(torch, d_in + dim * len(n_ids))

        def forward(self, x, ids):
            e = [emb(ids[..., i]) for i, emb in enumerate(self.emb)]
            return self.head(torch.cat([x] + e, dim=-1)).squeeze(-1)

    def build(d_in, dev):
        net = Net(d_in).to(dev)
        return net, lambda net, x, ids, mask: net(x, ids)

    return _fit_torch(tr, va, build, seed)


def fit_dae(tr, va, seed=42, dim=512, layers=3, noise=0.15, epochs=25, batch=1024):
    """ノイズ除去オートエンコーダ（入れ替えノイズ）で表現を作り、その中間層を足した MLP を学習する"""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = _prep(tr)
    x_all = torch.as_tensor(_matrix(tr, prep), device=dev)
    d_in = x_all.shape[1]

    enc = nn.ModuleList([nn.Linear(d_in if i == 0 else dim, dim) for i in range(layers)]).to(dev)
    dec = nn.Linear(dim * layers, d_in).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)

    def encode(x):
        hs, h = [], x
        for layer in enc:
            h = torch.relu(layer(h))
            hs.append(h)
        return torch.cat(hs, dim=-1)

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(len(x_all))
        total = 0.0
        for i in range(0, len(order), batch):
            b = x_all[torch.as_tensor(order[i:i + batch], device=dev)]
            swap = b[torch.randint(0, len(b), b.shape, device=dev), torch.arange(b.shape[1], device=dev)]
            noisy = torch.where(torch.rand(b.shape, device=dev) < noise, swap, b)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(dec(encode(noisy)), b)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if ep % 5 == 0:
            print(f"    [dae] epoch {ep}: 復元誤差 {total / len(x_all):.4f}", flush=True)
    for p in enc.parameters():
        p.requires_grad_(False)

    def build(d_in_, dev_):
        head = _mlp_head(torch, d_in_ + dim * layers).to(dev_)

        def forward(net, x, ids, mask):
            with torch.no_grad():
                h = encode(x)
            return net(torch.cat([x, h], dim=-1)).squeeze(-1)

        return head, forward

    return _fit_torch(tr, va, build, seed)


def fit_set(tr, va, seed=42, d_model=128, heads=4, layers=2):
    """出走馬全体を1つの集合として見る Transformer（他馬との相互作用を直接学習する）"""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.LayerNorm(d_model))
            layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, dropout=0.1, batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.out = nn.Linear(d_model, 1)

        def forward(self, x, mask):
            h = self.enc(self.inp(x), src_key_padding_mask=~mask)
            return self.out(h).squeeze(-1)

    def build(d_in, dev):
        net = Net(d_in).to(dev)
        return net, lambda net, x, ids, mask: net(x, mask)

    return _fit_torch(tr, va, build, seed, lr=5e-4)


FITTERS = {"lgb": fit_lgb, "cat": fit_cat, "mlp": fit_mlp, "dae": fit_dae, "emb": fit_emb, "set": fit_set}


def calibrated(raw_va, raw_rep, va):
    """検証期間で温度を合わせた効用を返す"""
    beta = fit_logit(raw_va, RaceGroups(va["race_id"], va["win"]))[0]
    return beta * raw_va, beta * raw_rep, beta


# ---- 段階1: オッズ抜きの当てやすさ ----

def stage1(df, names):
    parts = {k: df[(df["race_date"] >= s) & (df["race_date"] < e)].reset_index(drop=True)
             for k, (s, e) in STAGE1_SPLIT.items()}
    for k, v in parts.items():
        print(f"{k:<6} {STAGE1_SPLIT[k][0]}〜{STAGE1_SPLIT[k][1]}: {v['race_id'].nunique()}R", flush=True)
    g_va = RaceGroups(parts["valid"]["race_id"], parts["valid"]["win"])
    g_rep = RaceGroups(parts["report"]["race_id"], parts["report"]["win"])

    cal = {}
    for name in names:
        predict, rounds = FITTERS[name](parts["train"], parts["valid"])
        u_va, u_rep, beta = calibrated(predict(parts["valid"]), predict(parts["report"]), parts["valid"])
        cal[name] = (u_va, u_rep)
        ll, _ = g_rep.ll(u_rep)
        print(f"{name:<12} 反復{rounds:>5} 温度{beta:+.3f} 対数尤度 {ll.mean():.4f}", flush=True)

    base_ll, _ = g_rep.ll(cal["lgb"][1])
    print(f"\n単体の lgb を基準にした差:")
    for mix in [m for m in MIXES if all(n in cal for n in m)]:
        u_va, u_rep, beta = calibrated(np.mean([cal[n][0] for n in mix], axis=0),
                                       np.mean([cal[n][1] for n in mix], axis=0), parts["valid"])
        ll, _ = g_rep.ll(u_rep)
        lo, hi = bootstrap_ci(ll - base_ll)
        print(f"混合 {'+'.join(mix):<28} 対数尤度 {ll.mean():.4f}  差 {np.mean(ll - base_ll):+.4f} [{lo:+.4f}, {hi:+.4f}]",
              flush=True)

    if len(cal) > 1:   # 検証期間で重みを推定する（等分平均ではなく）
        order = [n for n in names if n in cal]
        w = fit_logit(np.column_stack([cal[n][0] for n in order]), g_va)
        ll, _ = g_rep.ll(np.column_stack([cal[n][1] for n in order]) @ w)
        lo, hi = bootstrap_ci(ll - base_ll)
        print(f"重み付き混合 {dict(zip(order, w.round(3)))} 対数尤度 {ll.mean():.4f}  "
              f"差 {np.mean(ll - base_ll):+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)


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
    mt.base_path = lambda name: MIX_PATH if name == "mix" else orig(name)
    markets = {}
    for name in ["all", "mix"]:
        markets[name], _ = mt.with_market(df, name)
        print(f"[{name}] オッズ・基礎モデルの予測がそろうレース {markets[name]['race_id'].nunique()}R", flush=True)
    for year, split in mt.DEV_SPLITS.items():
        print(f"\n=== 開発 報告{year}年 ===")
        for name in ["all", "mix"]:
            mt.run_split(markets[name], name, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--families", default="lgb,cat,mlp,dae,emb,set")
    ap.add_argument("--skip-build", action="store_true", help="段階2で混合の基礎モデルを作り直さない")
    args = ap.parse_args()
    df = eligible(pd.read_parquet(table_path("features")))
    names = args.families.split(",")
    if args.stage1:
        stage1(df, names)
    if args.stage2:
        stage2(df, [n for n in names if n != "lgb"], args.skip_build)


if __name__ == "__main__":
    main()
