# v2/train.py
# ③ 学習: 候補（v2/model/candidates.py）の設定どおりに本番用のモデルを学習し、models/v2/<候補>/ に保存する。
# 保存するもの: 基礎モデル（3シード）・温度、A/B の係数、残差モデル、設定（config.json）。実践（v2/predict.py）はこれだけを読む。
#
# 前提: 残差の学習に使う「年ごとの基礎モデルの予測」ができていること
#   c_all → data/v2/base_oos_all_2015_2026.parquet（python -m v2.experiments.walkforward --build-base）
#   cand2 → data/v2/bench/base_base_pl3.parquet（python -m v2.evaluate --build-base base_pl3 --base-objective pl）
# 確認: 基礎モデルは「年ごとの予測」の2026年分と同じ作り方なので、両者の予測が一致するかを表示する。
#
# 使い方: python -m v2.train --candidate cand2
import argparse
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from v2.model.candidates import CANDIDATES
from v2.model.pipeline import base_file, base_utility, between, fit_base, fit_combined, load, market, predict_combined
from v2.model.softmax import RaceGroups
from v2.paths import ROOT

MODELS_DIR = ROOT / "models" / "v2"


def train_candidate(name):
    cfg = CANDIDATES[name]
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    df = load()

    # 基礎モデル（前向き検証の期間はすべてこのモデルで予測する）
    tr, va = between(df, *cfg["base_train"]), between(df, *cfg["base_valid"])
    print(f"[{name}] 基礎モデル: 学習 {cfg['base_train']} {tr['race_id'].nunique()}R / ES {cfg['base_valid']} "
          f"目的関数 {cfg['base_objective']}", flush=True)
    models, temp = fit_base(tr, va, cfg["base_objective"], cfg["k"], cfg["lam"])
    for i, m in enumerate(models):
        m.save_model(str(out / f"base_{i}.txt"), num_iteration=m.best_iteration)
    print(f"  木 {[m.best_iteration for m in models]}本 / 温度 {temp:.3f}", flush=True)

    # 確認: 年ごとの予測の最終年（学習・ES が同じ期間）と一致するか
    year = cfg["base_valid"][1][:4]
    oos = pd.read_parquet(base_file(cfg["base_oos"]))
    te = between(df, f"{year}-01-01", f"{int(year) + 1}-01-01")
    chk = te[["race_id", "horse_id"]].assign(u_new=base_utility(models, temp, te)).merge(oos, on=["race_id", "horse_id"])
    diff = (chk["u_new"] - chk["u_base"]).abs()
    print(f"  確認: {year}年の年ごとの予測との差 最大 {diff.max():.2e}（{len(chk):,}頭）", flush=True)

    # 残差（オッズ＋基礎モデルからのずれ）。学習には年ごとの基礎モデルの予測を使う
    mk = market(df, cfg["base_oos"])
    rtr, rva = between(mk, *cfg["res_train"]), between(mk, *cfg["res_valid"])
    fit = fit_combined(rtr, rva, cfg["res_train"][1], cfg["half_life"], cfg["market_cols"])
    fit["residual"].save_model(str(out / "residual.txt"), num_iteration=fit["residual"].best_iteration)
    u_a, u_c = predict_combined(fit, rva)
    g = RaceGroups(rva["race_id"], rva["win"])
    print(f"[{name}] 残差: 学習 {cfg['res_train']} {rtr['race_id'].nunique()}R / ES {cfg['res_valid']} {rva['race_id'].nunique()}R"
          f" / 木 {fit['residual'].best_iteration}本 / 係数 A {fit['beta_a'][0]:.3f} B {np.round(fit['beta_b'], 3).tolist()}"
          f" / ES期間の C − A {np.mean(g.ll(u_c)[0] - g.ll(u_a)[0]):+.4f}（学習に使った期間なので参考）", flush=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = dict(candidate=name, config=cfg, trained_at=datetime.now().isoformat(timespec="seconds"), git_commit=commit,
                base_temp=temp, n_base_models=len(models), beta_a=fit["beta_a"].tolist(), beta_b=fit["beta_b"].tolist(),
                residual_feats=fit["feats"])
    (out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("保存:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    train_candidate(ap.parse_args().candidate)


if __name__ == "__main__":
    main()
