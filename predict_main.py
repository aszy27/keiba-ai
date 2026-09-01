# predict_main.py
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
import warnings
import time

from core.config import FEATURE_COLS, DATA_DIRS, DATA_DIRS_ALL, PREDICT_DIR
from core.features import feature_engineering
from core.inference import load_all_models, generate_dl_features
from core.data_loader import load_and_merge_all_data, extract_grade_from_name, FILE_BREEDER

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

# ==========================================
# ★設定: 予測したいレースID（12桁）を指定
# ==========================================
TARGET_IDS = [
    "2026040206",
    "2026070206",
    "2026010106"
]

# ==========================================
# ★設定: 実際に賭ける自信度の閾値（%）
# ==========================================
# 🔴 実データ検証の結果、閾値でフィルタせず全レースに賭けると回収率が
#    大きく下がることが確認されている。ここで指定した値未満のレースは
#    レポート上で「⚠️ 見送り推奨」として明示的に警告され、以上のレースには
#    「〇 購入対象」マークが付く。実際に賭ける前に必ずこのマークを確認すること。
BET_THRESHOLD = 60.0

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}

# 🔴 FIX: 旧マップの "Icon_GradeType5": "L" は誤り（Type5はオープン系アイコンで、
#          OP特別が誤ってL扱い＝grade_score 7 になっていた）。確実なG1/G2/G3のみ
#          アイコンから判定し、L等はページタイトル（例:「白富士Ｓ(L)」）から
#          data_loader.extract_grade_from_name で抽出する（学習データと同一ロジック）。
GRADE_CLASS_MAP = {
    "Icon_GradeType1": "G1", "Icon_GradeType2": "G2", "Icon_GradeType3": "G3",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive"
}


def _get_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.encoding = 'EUC-JP'
            if r.status_code == 200: return r
            if r.status_code == 404:
                print(f"   ❌ 404 Not Found: レースが存在しません。")
                return None
            if r.status_code == 403:
                print(f"   ❌ 403 Forbidden: Netkeibaにアクセスをブロックされました。")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"   ❌ リトライ失敗({max_retries}回): {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def extract_grade_from_tag(race_name_tag):
    if not race_name_tag: return "OP"
    for span in race_name_tag.find_all("span", class_=lambda c: c and "Icon_GradeType" in c):
        for cls in span.get("class", []):
            if cls in GRADE_CLASS_MAP: return GRADE_CLASS_MAP[cls]
    return "OP"


def scrape_race_date(race_id: str) -> pd.Timestamp:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    r = _get_with_retry(url)
    if r:
        soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')
        title = soup.title.text if soup.title else ""
        m_title = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
        if m_title:
            return pd.Timestamp(f"{m_title.group(1)}-{m_title.group(2).zfill(2)}-{m_title.group(3).zfill(2)}")

    today = pd.Timestamp('today').normalize()
    print(f"   ⚠️ 日付を取得できませんでした。本日 ({today.date()}) を使用します。")
    return today


def parse_target_ids(target_ids: list) -> list:
    result = []
    for tid in target_ids:
        tid = str(tid).strip()
        if len(tid) == 10:
            result.extend([(tid + str(i).zfill(2), tid) for i in range(1, 14)])
        elif len(tid) == 12:
            result.append((tid, tid))
    return result


def scrape_oikiri(race_id: str) -> dict:
    """ ★ 修正：タイムのパースロジックを完全抹消。評価（ABC）のみを確実に回収する """
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    result = {}
    r = _get_with_retry(url)
    if not r: return result

    soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')
    table = soup.find("table", class_=re.compile(r"OikiriTable|race_table_01"))
    if not table: return result

    rows = table.find_all("tr")
    for row in rows:
        try:
            horse_a = row.find("a", href=re.compile(r"/horse/\d+"))
            if not horse_a: continue
            horse_id = re.search(r'/horse/(\d+)', horse_a['href']).group(1)

            # 🔴 FIX: 学習データ収集側(scrape_other_data.parse_training)と完全に同じ
            #          生テキストを保持する（正規化は features.py が一元的に行う）
            oikiri_rank = "C"
            rank_elem = row.find("td", class_=re.compile(r"Oikiri_Rank|Rank"))
            if rank_elem:
                oikiri_rank = rank_elem.get_text(strip=True) or "C"

            result[horse_id] = {'oikiri_rank': oikiri_rank}
        except Exception:
            continue

    return result


def scrape_shutuba(race_id, race_date: pd.Timestamp):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    print(f"   🔍 Accessing: {url}")
    r = _get_with_retry(url)
    if not r:
        print("   ❌ レスポンスが空でした。")
        return None

    try:
        soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')
        if not soup.find("div", class_="RaceData01"):
            print("   ❌ RaceData01 が見つかりません。")
            return None

        place_name = PLACE_MAP.get(race_id[4:6], "unknown")
        race_name_tag = soup.find("h1", class_="RaceName")
        race_name = race_name_tag.get_text(strip=True) if race_name_tag else "不明"
        grade = extract_grade_from_tag(race_name_tag)
        if grade == "OP":
            # アイコンで判別できない場合はタイトル（例: 根岸Ｓ(G3) / 白富士Ｓ(L)）から
            # 学習データと同一ロジック(extract_grade_from_name)で抽出する
            title_text = soup.title.get_text(strip=True) if soup.title else ""
            grade = extract_grade_from_name(title_text)

        info_text = soup.find("div", class_="RaceData01").get_text().replace("\n", "")
        race_type = "障害" if "障" in info_text else ("芝" if "芝" in info_text else "ダート")

        m_len = re.search(r'(\d+)m', info_text)
        length = int(m_len.group(1)) if m_len else 1600

        weather = next((w for w in ["小雪", "雪", "小雨", "雨", "曇", "晴"] if w in info_text), "晴")
        # 🔴 FIX: 学習データ(keibascraper)の馬場表記は「稍」。旧実装は「稍重」を探すため
        #          「稍」表記のページで「良」に誤判定され、「稍重」表記でも学習語彙に無い
        #          カテゴリとなって unknown 化していた。「稍」に統一する。
        condition = next((c for c in ["不良", "稍重", "稍", "重", "良"] if c in info_text), "良")
        if condition == "稍重":
            condition = "稍"

        rows = soup.find_all("tr", class_="HorseList")
        if not rows:
            print("   ❌ HorseList が見つかりません。")
            return None

        data = []
        for row in rows:
            if "Cancel" in row.get("class", []): continue
            cols = row.find_all("td")
            if len(cols) < 5: continue

            try:
                horse_td = row.find("td", class_="HorseInfo")
                horse_a = horse_td.find("a") if horse_td else None
                horse_id = re.search(r'/horse/(\d+)', horse_a['href']).group(
                    1) if horse_a and 'href' in horse_a.attrs else "unknown"
                horse_name = horse_a.get_text(strip=True) if horse_a else "unknown"

                jockey_td = row.find("td", class_="Jockey")
                jockey_a = jockey_td.find("a") if jockey_td else None
                jockey_id = re.search(r'/jockey/(?:result/recent/)?(\d+)', jockey_a['href']).group(1).zfill(
                    5) if jockey_a and 'href' in jockey_a.attrs else "unknown"

                trainer_td = row.find("td", class_="Trainer")
                trainer_a = trainer_td.find("a") if trainer_td else None
                trainer_id = re.search(r'/trainer/(?:result/recent/)?(\d+)', trainer_a['href']).group(1).zfill(
                    5) if trainer_a and 'href' in trainer_a.attrs else "unknown"

                barei_td = row.find("td", class_="Barei")
                age_m = re.search(r'(\d+)', barei_td.get_text()) if barei_td else None
                age = int(age_m.group(1)) if age_m else 3

                # ★ 修正: "Baden" というクラスは実ページに存在せず、斤量は常に55.0固定になっていた。
                #   斤量tdは Barei の直後（クラス無しの Txt_C）に来るため、兄弟要素から取得する。
                burden_td = barei_td.find_next_sibling("td") if barei_td else None
                burden_text = burden_td.get_text(strip=True) if burden_td else ""
                burden = float(burden_text) if burden_text else 55.0

                weight, weight_diff = np.nan, 0
                w_td = row.find("td", class_="Weight")
                if w_td:
                    w_text = w_td.get_text(strip=True)
                    m_w = re.search(r'(\d+)\(([+-]?\d+)\)', w_text)
                    if m_w:
                        weight, weight_diff = int(m_w.group(1)), int(m_w.group(2))
                    elif re.search(r'(\d+)', w_text):
                        weight = int(re.search(r'(\d+)', w_text).group(1))

                data.append({
                    "race_id": race_id, "horse_id": horse_id, "jockey_id": jockey_id, "trainer_id": trainer_id,
                    "bracket": int(cols[0].get_text(strip=True)), "horse_number": int(cols[1].get_text(strip=True)),
                    "horse_name": horse_name, "age": age, "burden": burden, "weight": weight,
                    "weight_diff": weight_diff,
                    "place": place_name, "type": race_type, "length": length, "race_name": race_name,
                    "weather": weather, "condition": condition, "grade": grade, "head_count": len(rows),
                    "race_date": race_date, "split_tag": "target",
                })
            except Exception as e:
                print(f"   ⚠️ パースエラー (馬番: {cols[1].get_text(strip=True)}): {e}")
                continue

        if not data: return None
        df_shutuba = pd.DataFrame(data)

        print(f"   🏇 追い切りデータを取得中...")
        time.sleep(1)
        oikiri_map = scrape_oikiri(race_id)
        # 🔴 FIX: 欠損馬を ''(空文字) にすると oikiri_missing の判定が学習時(NaN=1)と
        #          食い違うため、NaN に統一する
        df_shutuba['oikiri_rank'] = df_shutuba['horse_id'].map(
            lambda hid: oikiri_map.get(hid, {}).get('oikiri_rank', np.nan))
        # ★ 修正：oikiri_last1f 列への割当を完全抹消

        return df_shutuba
    except Exception as e:
        print(f"   ❌ Scraping Error: {e}")
        return None


# 👑 【完全先祖返り】 3連複1点買い仕様のアドバイステキスト
def conf_label(c):
    if c >= 85: return "👉 【超激アツ】 3連複1点大勝負！バックテスト最高水準の信頼度。"
    if c >= 70: return "👉 【勝負レース】 上位3頭で決着する可能性が高い。"
    if c >= 50: return "👉 【有力】 狙える水準。資金管理しながら参加。"
    if c >= 40: return "👉 【様子見】 ギリギリ許容圏。少点数で。"
    return "👉 【見送り】 混戦模様. 3連複1点は危険。"


def bet_decision_mark(c, threshold=BET_THRESHOLD):
    """
    🔴 追加: BET_THRESHOLD を基準に「実際に賭けるべきか」を明示するマーク。
    conf_label は参考コメントに過ぎず読み飛ばされやすいため、
    レポートの最も目立つ位置（自信度の直後）に判定結果を出す。
    """
    if c >= threshold:
        return f"〇 購入対象 (閾値{threshold:.0f}%以上)"
    return f"⚠️ 見送り推奨 (閾値{threshold:.0f}%未満)"


if __name__ == "__main__":
    if not TARGET_IDS:
        print("⚠️ TARGET_IDS を設定してください")
        exit()

    race_list = parse_target_ids(TARGET_IDS)
    if not race_list: exit()

    print(f"🎯 予想対象: {len(race_list)} レース")
    date_cache = {}
    for race_id, tid in race_list:
        if race_id[:10] not in date_cache:
            time.sleep(1)
            date_cache[race_id[:10]] = scrape_race_date(race_id)

    print("\n🧠 モデルをロード中...")
    models = load_all_models()
    gm_params = models['gm_params']

    print("📚 過去データをロード中...")

    # ⚠️ 修正: inference.py の d_hist < d_target によって未来リークは既に防がれているため、
    # 最新のテスト期間(2025〜)も含めた全データを読み込み、本日のレースだけを除外する。
    df_past = load_and_merge_all_data(DATA_DIRS_ALL)
    today = pd.Timestamp('today').normalize()
    if df_past is not None and not df_past.empty:
        df_past['race_date'] = pd.to_datetime(df_past['race_date'])
        # 🔴 FIX: 予測対象レース自体が過去データ(data/test等)に確定結果として存在する場合
        #          （過去レースの再予測時）、同一レースが履歴とtargetに二重計上され
        #          head_count 等のレース内特徴量が破壊されるため、必ず除外する。
        target_rid_set = {rid for rid, _ in race_list}
        df_past = df_past[
            (df_past['race_date'] < today) &
            (~df_past['race_id'].astype(str).isin(target_rid_set))
        ].copy()
        df_past['split_tag'] = "history"
    else:
        df_past = pd.DataFrame()

    print("\n📋 全レースの出馬表をスクレイプ中...")
    scraped_frames = []
    for race_id, tid in race_list:
        df_tgt = scrape_shutuba(race_id, date_cache[race_id[:10]])
        if df_tgt is not None and len(df_tgt) > 0:
            df_tgt['tid'] = tid
            scraped_frames.append(df_tgt)
            time.sleep(2)

    if not scraped_frames:
        print("❌ 取得できたレースがありません。")
        exit()

    df_targets = pd.concat(scraped_frames, ignore_index=True)
    df_targets['split_tag'] = "target"

    # 🔴 FIX: 学習・バックテストでは breeder(生産者) が data_loader 経由で全行に結合される
    #          のに対し、スクレイプした出馬表には無いため本番だけ全馬 'unknown' となり、
    #          CAT特徴量 breeder と breeder_win_rate / breeder_top3_rate が学習時と
    #          食い違っていた。同じ breeder マスタをtarget行にも結合して条件を揃える。
    if os.path.exists(FILE_BREEDER):
        try:
            df_b = pd.read_csv(FILE_BREEDER, usecols=['horse_id', 'breeder'], dtype={'horse_id': str})
            df_b = df_b.drop_duplicates(subset=['horse_id'], keep='last')
            df_targets = pd.merge(df_targets, df_b, on='horse_id', how='left')
        except Exception as e:
            print(f"   ⚠️ Breederデータ結合失敗: {e}")

    for c in ['prize', 'popularity', 'rank', 'last_3f', 'diff_time']:
        val = 0 if c != 'rank' else np.nan
        if not df_past.empty and c not in df_past.columns: df_past[c] = val
        if c not in df_targets.columns: df_targets[c] = val

    df_combined = pd.concat([df_past, df_targets], ignore_index=True) if not df_past.empty else df_targets.copy()

    print("\n🛠️ 特徴量生成・推論 (全レース一括処理)...")
    df_combined = feature_engineering(df_combined, weight_mean=gm_params.get('weight_mean', 470.0),
                                      burden_mean=gm_params.get('burden_mean', 55.0))
    df_combined = df_combined.fillna(0).replace([np.inf, -np.inf], 0)

    df_hist = df_combined[df_combined['split_tag'] == 'history'].copy()
    df_pred = df_combined[df_combined['split_tag'] == 'target'].copy()

    df_pred = generate_dl_features(df_pred, models, df_history=df_hist, show_progress=True)

    for feat in FEATURE_COLS:
        if feat not in df_pred.columns: df_pred[feat] = 0

    df_pred['lgb_rank_score'] = models['ranker'].predict(df_pred[FEATURE_COLS].values)
    df_pred['lgb_prob_score'] = models['clf'].predict(df_pred[FEATURE_COLS].values)

    df_pred['norm_rank_score'] = df_pred.groupby('race_id', sort=False)['lgb_rank_score'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    )

    _w_rank = gm_params.get('w_rank', 0.6)
    _w_prob = gm_params.get('w_prob', 0.4)
    _w_trans = max(0.0, 1.0 - _w_rank - _w_prob)

    df_pred['transformer_prob'] = df_pred['transformer_prob'].fillna(0.5).replace([np.inf, -np.inf], 0.5)
    df_pred['norm_rank_score'] = df_pred['norm_rank_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5)
    df_pred['lgb_prob_score'] = df_pred['lgb_prob_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5)

    df_pred['final_score'] = (
            _w_rank * df_pred['norm_rank_score'] +
            _w_prob * df_pred['lgb_prob_score'] +
            _w_trans * df_pred['transformer_prob']
    )
    df_pred['final_score'] = df_pred['final_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5)

    df_pred['norm_trans_score'] = df_pred.groupby('race_id', sort=False)['transformer_prob'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    ).fillna(0.5)

    # メタモデル（監督AI）一括バッチ推論
    meta_rows, rids_ordered = [], []
    for rid, grp in df_pred.groupby('race_id', sort=False):
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
            'head_count': len(grp),
            'transformer_prob_1st': grp.iloc[0]['transformer_prob'],
        })
        rids_ordered.append(rid)

    if not models.get('meta_model'):
        print("⚠️ 警告: meta_model が見つかりません。自信度は 50.0% 固定で動作します。")

    confidences = (
        models['meta_model'].predict(pd.DataFrame(meta_rows)) * 100
        if (models.get('meta_model') and meta_rows)
        else np.full(len(rids_ordered), 50.0)
    )
    conf_dict = dict(zip(rids_ordered, confidences))

    # 🔴 FIX: カレントディレクトリ依存の "predict" をやめ、config の絶対パスに統一
    out_dir = str(PREDICT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    total_races, total_horses = 0, 0

    base_cols = ['race_id', 'bracket', 'horse_number', 'horse_name', 'horse_id', 'final_score', 'lgb_rank_score',
                 'lgb_prob_score', 'norm_trans_score']

    for tid, grp_tid in df_pred.groupby('tid'):
        texts = []
        n_bet_target = 0
        n_skip = 0
        for rid, grp_rid in grp_tid.groupby('race_id'):
            grp_rid = grp_rid.sort_values('final_score', ascending=False)
            top1 = grp_rid.iloc[0]
            conf = conf_dict.get(rid, 50.0)

            if conf >= BET_THRESHOLD:
                n_bet_target += 1
            else:
                n_skip += 1

            lines = ["=" * 100, f"🤖 監督AI診断 (Race: {rid} {top1['race_name']})",
                     f"   本命馬: {top1['horse_name']} (Score: {top1['final_score']:.4f})",
                     f"   AI自信度(3頭完全的中率): {conf:.1f}%  {conf_label(conf)}",
                     f"   >>> 判定: {bet_decision_mark(conf)} <<<", "-" * 100,
                     f"{'枠':<3} {'番':<3} {'馬名':<20} {'スコア':<8} {'評価'}", "-" * 100]
            for rank_i, (_, row) in enumerate(grp_rid.iterrows()):
                mark = "👑 本命" if rank_i == 0 else (
                    "✨ 対抗" if row['final_score'] >= top1['final_score'] * 0.8 else ("🔺 紐" if rank_i < 6 else ""))
                lines.append(
                    f"{int(row['bracket']):<3} {int(row['horse_number']):<3} {row['horse_name']:<20} {row['final_score']:.4f}   {mark}")
            lines.extend(["=" * 100, ""])
            texts.append("\n".join(lines))

        df_dump = grp_tid.sort_values(['race_id', 'horse_number'])
        exist_base = [c for c in base_cols if c in df_dump.columns]
        other_cols = [c for c in df_dump.columns if c not in exist_base]
        df_dump = df_dump[exist_base + other_cols]

        tid_str = str(tid)
        if len(tid_str) >= 10:
            place_code = tid_str[4:6]
            kai = int(tid_str[6:8])
            nichi = int(tid_str[8:10])
            place_name = PLACE_MAP.get(place_code, "不明")
            file_prefix = f"{kai}回{place_name}{nichi}日目"

            # 12桁（単一レース指定）の場合は「_11R」のようにレース番号も付与
            if len(tid_str) == 12:
                race_num = int(tid_str[10:12])
                file_prefix += f"_{race_num}R"
        else:
            file_prefix = tid_str

        df_dump.to_csv(os.path.join(out_dir, f"{file_prefix}_features.csv"), index=False, encoding='utf-8-sig')
        with open(os.path.join(out_dir, f"{file_prefix}.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(texts))

        total_races += len(grp_tid['race_id'].unique())
        total_horses += len(df_dump)
        print(f"📁 保存完了: {file_prefix}_features.csv & {file_prefix}.txt")
        print(f"   >>> 自信度{BET_THRESHOLD:.0f}%以上(〇購入対象): {n_bet_target}R / "
              f"見送り推奨: {n_skip}R <<<")

    print(f"\n✅ 完了: 合計 {total_races} レース / {total_horses} 頭")