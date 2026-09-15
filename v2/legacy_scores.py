# v2/legacy_scores.py
# 旧システムの OOF キャッシュから、各馬のスコア（その期間を学習していない分割モデルの予測）だけを取り出して保存する。
# M3 で「旧モデルより勝ち馬をよく当てるか」を同じ期間で比べるために使う。キャッシュは大きいので1回だけ実行する。
# 使い方: python -m v2.legacy_scores
import pandas as pd

from v2.paths import LEGACY_DIR, V2_DIR

SRC = LEGACY_DIR / "oof_features_cache.pkl"
OUT = V2_DIR / "legacy_oof_scores.parquet"
COLS = ["race_id", "horse_id", "race_date", "lgb_rank_score", "lgb_prob_score", "transformer_prob"]


def main():
    df = pd.read_pickle(SRC)
    out = df.loc[df["has_oof"].astype(bool), COLS].copy()
    del df
    for c in ["race_id", "horse_id"]:
        out[c] = out[c].astype(str).str.replace(r"\.0$", "", regex=True)
    out["race_date"] = pd.to_datetime(out["race_date"])
    out.to_parquet(OUT, index=False)
    print(f"{len(out)}行 {out['race_date'].min():%Y-%m-%d}〜{out['race_date'].max():%Y-%m-%d} → {OUT}")


if __name__ == "__main__":
    main()
