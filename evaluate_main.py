# evaluate_main.py
import pandas as pd
import numpy as np
import torch
import os
import warnings
import random
from tqdm import tqdm
import re

from core.config import (
    MODEL_PATHS, FEATURE_COLS, DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST, BASE_DIR, DEVICE_STR
)
from core.features import feature_engineering
from core.inference import load_all_models, generate_dl_features

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

FILE_RETURN = os.path.join(BASE_DIR, "data", "return_data_progress.csv")
CONFIDENCE_THRESHOLDS = [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]

try:
    DEVICE = torch.device(DEVICE_STR)
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
except Exception:
    DEVICE = torch.device("cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prep_return_data(filepath):
    if not os.path.exists(filepath): return None
    df = pd.read_csv(filepath, dtype={'race_id': str})

    for col in ['tansho', 'sanrenpuku', 'sanrentan']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.split('|').str[0].str.replace(r'[^0-9]', '', regex=True),
                                    errors='coerce').fillna(0).astype(int)

    if 'fukusho' in df.columns:
        df['fukusho_list'] = df['fukusho'].astype(str).apply(
            lambda x: [int(re.sub(r'[^0-9]', '', p)) if re.sub(r'[^0-9]', '', p) else 0 for p in x.split('|')]
        )
    df = df.drop_duplicates(subset=['race_id'], keep='last')
    return df.set_index('race_id')


def main():
    seed_everything(42)
    print("🚀 [FAST EVAL] Strong Full 実力検証シミュレーション (V7 1点勝負 Sniper Custom)")

    if not os.path.exists(MODEL_PATHS['meta_model']):
        print("❌ メタモデルが見つかりません。train_main.py を実行してください。")
        return

    print("\n1️⃣ モデルをロード中...")
    try:
        models = load_all_models()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    gm_params = models['gm_params']
    weight_mean, burden_mean = gm_params.get('weight_mean', 470.0), gm_params.get('burden_mean', 55.0)

    print("\n2️⃣ データをロード中...")
    from core.data_loader import load_and_merge_all_data
    try:
        df_history = load_and_merge_all_data([DATA_DIR_TRAIN, DATA_DIR_VAL])
    except FileNotFoundError:
        df_history = pd.DataFrame()

    try:
        df_test = load_and_merge_all_data([DATA_DIR_TEST])
    except FileNotFoundError as e:
        print(f"❌ Testデータが見つかりません: {e}")
        return

    if df_test.empty:
        print("❌ Testデータが空です。")
        return

    df_test['race_date'] = pd.to_datetime(df_test['race_date'])
    for c in ['prize', 'popularity', 'grade', 'rank', 'last_3f', 'diff_time']:
        df_test[c] = df_test.get(c, 0 if c != 'grade' else 'OP')

    test_race_ids = set(df_test['race_id'].astype(str))

    if not df_history.empty:
        df_history['race_date'] = pd.to_datetime(df_history['race_date'])
        for c in ['prize', 'popularity', 'grade', 'rank', 'last_3f', 'diff_time']:
            df_history[c] = df_history.get(c, 0 if c != 'grade' else 'OP')
        df_all = pd.concat([df_history, df_test], ignore_index=True)
    else:
        df_all = df_test.copy()

    print("\n3️⃣ 特徴量エンジニアリング...")
    df_all = df_all.copy()

    # 🔴 FIX: 出走取消・除外・競走中止等（rankが数値でない馬）は、本番の出馬表
    #          （predict_main はキャンセル馬をスキップする）と条件を揃えるため
    #          予測対象から外し、履歴(history)としてのみ扱う。
    #          旧実装ではこれらの馬が fillna(0) で rank=0 となり、「rank<=3」の
    #          判定で実際の3着以内と誤集計され、的中率が水増しされる余地があった。
    _rank_num = pd.to_numeric(df_all['rank'], errors='coerce')
    _is_test = df_all['race_id'].astype(str).isin(test_race_ids)
    df_all['split_tag'] = np.where(_is_test & _rank_num.notna(), 'target', 'history')
    n_excluded = int((_is_test & _rank_num.isna()).sum())
    if n_excluded:
        print(f"   -> ℹ️ 出走取消・中止等で着順が無い {n_excluded} 頭を予測対象から除外（履歴としては使用）")

    df_all = feature_engineering(df_all, weight_mean=weight_mean, burden_mean=burden_mean)
    df_all = df_all.sort_values(['race_date', 'race_id']).reset_index(drop=True)
    # 🔴 FIX: rank は fillna(0) の対象から守る（0着=3着以内と誤判定されるのを防ぐ）
    df_all['rank'] = pd.to_numeric(df_all['rank'], errors='coerce').fillna(99)
    df_all = df_all.fillna(0).replace([np.inf, -np.inf], 0)

    print("\n4️⃣ 過去成績履歴の隠蔽（カンニング防止）＆ DL推論...")
    df_history_dl = df_all.copy()
    df_targets_dl = df_all[df_all['split_tag'] == 'target'].copy()

    true_labels = df_targets_dl[['race_id', 'horse_id', 'rank']].copy()
    leaky_cols = ['prize', 'last_3f', 'diff_time', 'popularity', 'pace_score', 'first_3f', 'last_3f_race']
    for c in leaky_cols:
        if c in df_targets_dl.columns: df_targets_dl[c] = 0.0
    if 'rank' in df_targets_dl.columns: df_targets_dl['rank'] = np.nan

    df_targets_dl = generate_dl_features(df_targets_dl, models, df_history=df_history_dl, show_progress=True)
    df_test_final = df_targets_dl.copy()
    df_test_final = df_test_final.drop(columns=['rank']).merge(true_labels, on=['race_id', 'horse_id'], how='left')

    print("\n5️⃣ LightGBM 推論...")
    for feat in FEATURE_COLS:
        if feat not in df_test_final.columns: df_test_final[feat] = 0

    X_test = df_test_final[FEATURE_COLS].values
    df_test_final['lgb_rank_score'] = models['ranker'].predict(X_test)
    df_test_final['lgb_prob_score'] = models['clf'].predict(X_test)

    df_test_final['norm_rank_score'] = df_test_final.groupby('race_id')['lgb_rank_score'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5)

    _w_rank = gm_params.get('w_rank', 0.6)
    _w_prob = gm_params.get('w_prob', 0.4)
    _w_trans = max(0.0, 1.0 - _w_rank - _w_prob)

    df_test_final['transformer_prob'] = df_test_final['transformer_prob'].fillna(0.5)
    df_test_final['final_score'] = (
            _w_rank * df_test_final['norm_rank_score'].fillna(0.5)
            + _w_prob * df_test_final['lgb_prob_score'].fillna(0.5)
            + _w_trans * df_test_final['transformer_prob']
    )

    print("\n6️⃣ 監督AI (Meta-Model) 一括予測...")
    meta_cols = ['score_1st', 'score_2nd', 'score_3rd', 'gap_1_2', 'gap_3_4', 'score_std', 'score_mean', 'head_count',
                 'transformer_prob_1st']
    df_test_final['rank'] = pd.to_numeric(df_test_final['rank'], errors='coerce').fillna(99).astype(int)

    meta_rows, race_rids = [], []
    for rid, grp in tqdm(df_test_final.groupby('race_id'), desc="Meta Prep"):
        grp = grp.sort_values('final_score', ascending=False)
        if len(grp) < 3: continue
        top1, top2, top3 = grp.iloc[0]['final_score'], grp.iloc[1]['final_score'], grp.iloc[2]['final_score']
        top4 = grp.iloc[3]['final_score'] if len(grp) > 3 else 0.0
        meta_rows.append({
            'score_1st': top1,
            'score_2nd': top2,
            'score_3rd': top3,
            'gap_1_2': top1 - top2,
            'gap_3_4': top3 - top4,
            'score_std': grp['final_score'].std() if len(grp) > 1 else 0.0,
            'score_mean': grp['final_score'].mean(),
            'head_count': len(grp), 'transformer_prob_1st': grp.iloc[0]['transformer_prob'],
        })
        race_rids.append(rid)

    df_meta_eval = pd.DataFrame(meta_rows, columns=meta_cols)
    confidences = models['meta_model'].predict(df_meta_eval) * 100 if (
                models.get('meta_model') and not df_meta_eval.empty) else np.full(len(df_meta_eval), -1.0)
    conf_dict = dict(zip(race_rids, confidences))

    print("\n7️⃣ レース結果の超高速集計...")
    df_ret_indexed = prep_return_data(FILE_RETURN)
    race_results = []

    for rid, grp in tqdm(df_test_final.groupby('race_id'), desc="Result Check"):
        if rid not in conf_dict: continue

        grp = grp.sort_values('final_score', ascending=False)
        ai_top3 = grp.iloc[:3]['horse_id'].astype(str).values
        # 🔴 FIX: rank>=1 のガードを追加（欠損rankが0埋めされた場合の誤的中を根絶）
        actual_top3 = set(grp[(grp['rank'] >= 1) & (grp['rank'] <= 3)]['horse_id'].astype(str).values)

        # 👑 【完全先祖返り】 3頭完全的中フラグ
        is_3puku_hit = 1 if (len(actual_top3) >= 3 and set(ai_top3).issubset(actual_top3)) else 0

        valid_ranks = grp[(grp['rank'] >= 1) & (grp['rank'] <= 3)]
        actual_order = valid_ranks.sort_values('rank')['horse_id'].astype(str).values
        is_3tan_hit = 1 if (len(actual_order) >= 3 and np.array_equal(actual_order[:3], ai_top3)) else 0

        payouts = {'tansho': 0, 'fukusho': 0, 's3': 0, 't3': 0}
        has_ret = False

        if df_ret_indexed is not None and rid in df_ret_indexed.index:
            ret_row = df_ret_indexed.loc[rid]
            has_ret = True
            payouts['tansho'] = ret_row.get('tansho', 0)
            payouts['s3'] = ret_row.get('sanrenpuku', 0)
            payouts['t3'] = ret_row.get('sanrentan', 0)

            actual_rank = int(grp.iloc[0]['rank'])
            if 1 <= actual_rank <= 3 and 'fukusho_list' in ret_row:
                flist = ret_row['fukusho_list']
                if isinstance(flist, list) and len(flist) >= actual_rank:
                    payouts['fukusho'] = flist[actual_rank - 1]

        race_results.append({
            'race_date': grp['race_date'].iloc[0],
            'confidence': conf_dict[rid],
            'is_win': 1 if grp.iloc[0]['rank'] == 1 else 0,
            'is_place': 1 if grp.iloc[0]['rank'] <= 3 else 0,
            'is_3puku_hit': is_3puku_hit,
            'is_3tan_hit': is_3tan_hit,
            'has_return': has_ret,
            'tansho_payout': payouts['tansho'], 'fukusho_payout': payouts['fukusho'],
            's3_payout': payouts['s3'], 't3_payout': payouts['t3']
        })

    if not race_results:
        print("❌ 集計できるレースがありませんでした。")
        return

    df_res = pd.DataFrame(race_results)

    print("\n" + "=" * 115)
    print("📊 [FAST EVAL] 3連複1点勝負シミュレーション (1点100円均等・実戦リアル収支版)")
    print("=" * 115)
    print(
        f"{'自信度閾値':<10} | {'購入R数':<5} | {'勝率':<6} | {'複勝率':<6} | {'3複的中':<6} | {'3単的中':<6} | {'単勝回収':<8} | {'複勝回収':<8} | {'3複回収':<8} | {'3単回収':<8}")
    print("-" * 115)

    for th in CONFIDENCE_THRESHOLDS:
        tgt = df_res[df_res['confidence'] >= th]
        bets = len(tgt)
        if bets == 0: continue

        wr, pr, p3r, t3r = (tgt[col].sum() / bets * 100 for col in
                            ['is_win', 'is_place', 'is_3puku_hit', 'is_3tan_hit'])

        ret_tgt = tgt[tgt['has_return']]
        ret_bets = len(ret_tgt)

        if ret_bets > 0:
            # 👑 【完全先祖返り】 すべて1R=100円投資（分母）としてROIを算出
            w_ret = (ret_tgt['tansho_payout'] * ret_tgt['is_win']).sum() / (ret_bets * 100) * 100
            p_ret = (ret_tgt['fukusho_payout'] * ret_tgt['is_place']).sum() / (ret_bets * 100) * 100
            s3_ret = (ret_tgt['s3_payout'] * ret_tgt['is_3puku_hit']).sum() / (ret_bets * 100) * 100
            t3_ret = (ret_tgt['t3_payout'] * ret_tgt['is_3tan_hit']).sum() / (ret_bets * 100) * 100

            note = f"(確定:{ret_bets}/{bets}R)" if ret_bets < bets else ""
            print(
                f"{th:>3.0f}% 以上    | {bets:<5} | {wr:>6.1f}% | {pr:>6.1f}% | {p3r:>6.1f}% | {t3r:>6.1f}% | {w_ret:>8.1f}% | {p_ret:>8.1f}% | {s3_ret:>8.1f}% | {t3_ret:>8.1f}% {note}")
        else:
            print(
                f"{th:>3.0f}% 以上    | {bets:<5} | {wr:>6.1f}% | {pr:>6.1f}% | {p3r:>6.1f}% | {t3r:>6.1f}% | --- 払戻データなし ---")

    # =========================================================================
    # 🕒 月別ROI推移（モデルの鮮度劣化＝コンセプトドリフトの検証用）
    # =========================================================================
    print("\n" + "=" * 115)
    print("🕒 [DRIFT CHECK] 月別ROI推移シミュレーション (学習データからの経過時間による性能劣化を確認)")
    print("=" * 115)

    df_res['race_month'] = pd.to_datetime(df_res['race_date']).dt.to_period('M')

    # 検証したい自信度の閾値をここで指定（0%=全レース, 50%, 75%=バックテストの目玉ライン）
    DRIFT_CHECK_THRESHOLDS = [0.0, 50.0, 75.0]

    for th in DRIFT_CHECK_THRESHOLDS:
        print(f"\n--- 自信度 {th:.0f}% 以上のみを対象 ---")
        print(f"{'年月':<10} | {'購入R数':<5} | {'3複的中':<6} | {'3複回収':<8} | {'的中数':<6}")
        print("-" * 60)

        tgt_th = df_res[df_res['confidence'] >= th]
        if tgt_th.empty:
            print("   (該当レースなし)")
            continue

        for month, grp_m in tgt_th.groupby('race_month'):
            bets_m = len(grp_m)
            hits_m = int(grp_m['is_3puku_hit'].sum())
            p3r_m = hits_m / bets_m * 100 if bets_m > 0 else 0.0

            ret_m = grp_m[grp_m['has_return']]
            ret_bets_m = len(ret_m)
            if ret_bets_m > 0:
                s3_ret_m = (ret_m['s3_payout'] * ret_m['is_3puku_hit']).sum() / (ret_bets_m * 100) * 100
                note_m = f"(確定:{ret_bets_m}/{bets_m}R)" if ret_bets_m < bets_m else ""
                print(f"{str(month):<10} | {bets_m:<5} | {p3r_m:>6.1f}% | {s3_ret_m:>8.1f}% | {hits_m:>6} {note_m}")
            else:
                print(f"{str(month):<10} | {bets_m:<5} | {p3r_m:>6.1f}% | --- 払戻データなし --- | {hits_m:>6}")

    print("\n" + "-" * 115)
    print("💡 見方: 学習データの最終月（DATA_DIR_VALの終端）から時間が経つほど3複回収率が")
    print("   右肩下がりに劣化していく場合、モデルの鮮度劣化（コンセプトドリフト）が疑われます。")
    print("   横ばい〜安定していれば、直近の性能悪化は主にサンプル数不足による分散が原因と考えられます。")


if __name__ == "__main__":
    main()