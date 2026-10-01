# v2/model/candidates.py
# 前向き検証の候補の設定。学習（v2/train.py）と実践（v2/predict.py）はここだけを見る。
# 登録した候補の値は判定まで変えない。変えたくなったら新しい名前の候補として足す（docs/rebuild_plan.md「候補2以降の開発」）。
#
#   base_*      : 本番の基礎モデル。前向き検証の期間のレースはすべてこのモデルで予測する
#   base_oos    : 残差の学習に使う「その年を学習していない」基礎モデルの予測（v2/evaluate.py の保存名。current = C[all] のもの）
#   res_*       : 残差（オッズ＋基礎モデルからのずれ）の学習期間。half_life は res_train の終わりを基準にした直近重視の半減期（日）
#   market_cols : 残差の入力にオッズ由来の列（x_mkt・人気順）を足すか
#   threshold   : 期待値（勝率 × オッズ）がこれ以上の単勝を買う。None = 未登録
#   start       : 前向き検証に数えるレースの初日（登録の翌日以降）
#   variants    : 残差の版の一覧。複数あれば各版の効用を平均してからレース内で正規化する（= log(勝率) の対数プーリング。技術の探索 T7）。
#                 extras = 残差の入力に足す列（place = 複勝オッズ、past = 過去の走の市場評価）、seeds = 残差を複数シードで学習して平均
#   roi_basis   : 判定の回収率の数え方。snapshot = スナップショットのオッズで払戻を計算 / actual = 実際の払戻（確定オッズ）

CANDIDATES = {
    # 2026-09-17 登録（docs/rebuild_plan.md「前向き検証」）。凍結タグ forward-c-all と同じ作り方
    "c_all": dict(
        registered="2026-09-17", start="2026-09-07",
        base_objective="win", k=1, lam=1.0,
        base_train=("2013-01-01", "2025-01-01"), base_valid=("2025-01-01", "2026-01-01"),
        base_oos="current",
        res_train=("2022-01-01", "2026-01-01"), res_valid=("2026-01-01", "2026-09-07"),
        half_life=None, market_cols=False,
        threshold=1.2, ci_level=0.95,
        roi_basis="snapshot",   # 登録の文面「回収率（スナップショットのオッズで計算）」どおり
    ),
    # 2026-10-01 登録（docs/rebuild_plan.md「候補2（事前登録）」、凍結タグ forward-cand2）。
    # 基礎モデルを1〜3着で学習＋残差を2015年〜・半減期730日・オッズ列あり（実験2〜4で採用）
    # ＋4版の対数プーリング（技術の探索 T7 で採用。res_mkt / res_seeds / mkt_place / past_mkt）
    "cand2": dict(
        registered="2026-10-01", start="2026-10-02",
        base_objective="pl", k=3, lam=0.75,
        base_train=("2013-01-01", "2025-01-01"), base_valid=("2025-01-01", "2026-01-01"),
        base_oos="base_pl3",
        res_train=("2015-01-01", "2026-01-01"), res_valid=("2026-01-01", "2026-09-07"),
        half_life=730, market_cols=True,
        variants=[dict(name="res_mkt", extras=[], seeds=None),
                  dict(name="res_seeds", extras=[], seeds=[42, 7, 2024]),
                  dict(name="mkt_place", extras=["place"], seeds=None),
                  dict(name="past_mkt", extras=["past"], seeds=None)],
        threshold=1.3, ci_level=0.975,
        roi_basis="actual",     # 実際の払戻（確定オッズ）。実際に賭けたときの収支と同じ数え方
    ),
}
