# repair_race_info.py
"""
race_data_*.csv のうち、レース情報（芝ダ・距離・天候・馬場・発走時刻・競馬場など）が
欠損しているレースを netkeiba のレース結果ページから補完する。

背景: keibascraper のスクレイプ失敗により、2026年8月以降の中京や 9/5〜9/6 の全レースなどで
      type / length / weather / condition が空欄になっていた。これらはコース特徴量
      （直線長・高低差）、Transformer の距離・馬場入力、騎手/調教師/種牡馬のコース別成績に
      直接効くため、学習・バックテスト・本番予測の履歴として条件がズレる原因になる。

使い方:
    python repair_race_info.py            # data/test のみ対象（通常はこれ）
    python repair_race_info.py --all      # train / val / test すべて対象
    python repair_race_info.py --dry-run  # 欠損レースの一覧だけ表示して何もしない

・別レースの出走馬・着順がそのままコピーされた「偽レース」（keibascraper の取得ミス）は削除する。
  削除後に scrape/scrape_main_data.py を実行すると、そのレースだけ取り直される
・既に値が入っている欄は上書きしない（空欄だけを埋める）
・書き込み前に元ファイルを race_data_YYYY.csv.bak_日時 としてバックアップする
"""
import argparse
import glob
import os
import random
import re
import shutil
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}
GRADE_ICON = {"Icon_GradeType1": "GI", "Icon_GradeType2": "GII", "Icon_GradeType3": "GIII"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
# 空欄なら「壊れている」とみなす列（特徴量に効くもの）
KEY_COLS = ['type', 'length', 'weather', 'condition', 'place']


def length_class(length: int) -> str:
    # 既存データ(keibascraper)の区分に合わせる
    if length < 1400: return "Sprint"
    if length < 1800: return "Mile"
    if length < 2200: return "Intermediate"
    if length < 2850: return "Long"
    return "Extended"


def fetch(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.encoding = 'EUC-JP'
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                print(f"   ❌ HTTP {r.status_code}: {url}")
                return None
        except Exception as e:
            print(f"   ⚠️ 通信エラー({attempt + 1}/{max_retries}): {e}")
        time.sleep(2 ** attempt * 3)
    return None


def parse_race_info(race_id: str) -> dict:
    """レース結果ページから、レース単位の情報を取り出す（predict_main の出馬表パースと同じ表記に揃える）"""
    info = {
        'place': PLACE_MAP.get(race_id[4:6]),
        'round': str(int(race_id[6:8])),
        'days': str(int(race_id[8:10])),
        'race_number': str(int(race_id[10:12])),
    }
    r = fetch(f"https://race.netkeiba.com/race/result.html?race_id={race_id}")
    if r is None:
        return info
    soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')

    data01 = soup.find("div", class_="RaceData01")
    if data01:
        text = data01.get_text().replace("\n", "").replace(" ", "")
        info['type'] = "障害" if "障" in text else ("芝" if "芝" in text else "ダート")
        m_len = re.search(r'(\d{3,4})m', text)
        if m_len:
            info['length'] = str(int(m_len.group(1)))
            info['length_class'] = length_class(int(m_len.group(1)))
        m_time = re.search(r'(\d{1,2}:\d{2})発走', text)
        if m_time:
            info['race_time'] = m_time.group(1)
        m_hand = re.search(r'\((左|右)', text)
        if m_hand:
            info['handed'] = m_hand.group(1)
        m_w = re.search(r'天候:([^/]+)', text)
        if m_w:
            w = m_w.group(1)
            info['weather'] = next((x for x in ["小雪", "雪", "小雨", "雨", "曇", "晴"] if x in w), None)
        m_c = re.search(r'馬場:([^/]+)', text)
        if m_c:
            c = m_c.group(1)
            cond = next((x for x in ["不良", "稍重", "稍", "重", "良"] if x in c), None)
            info['condition'] = "稍" if cond == "稍重" else cond

    name_tag = soup.find("h1", class_="RaceName")
    if name_tag:
        name = name_tag.get_text(strip=True)
        grade = None
        for span in name_tag.find_all("span", class_=lambda c: c and "Icon_GradeType" in c):
            for cls in span.get("class", []):
                grade = GRADE_ICON.get(cls, grade)
        if grade is None and soup.title:
            m_l = re.search(r'[\(（](L|Ｌ)[\)）]', soup.title.get_text())
            if m_l:
                grade = "L"
        # data_loader.extract_grade_from_name が読めるよう「名前(GIII)」形式にする
        if grade and not re.search(r'[\(（][^\)）]+[\)）]', name):
            name = f"{name}({grade})"
        info['race_name'] = name

    if data01 is not None and ('condition' not in info or 'weather' not in info or 'length' not in info):
        info['_raw'] = data01.get_text(" ", strip=True)[:120]

    if info.get('place') and info.get('type') and info.get('length'):
        info['course'] = f"{info['place']}{info['type']}{info['length']}"
    return {k: v for k, v in info.items() if v is not None}


def find_phantom_races(df: pd.DataFrame) -> list:
    """同じ日に全く同じ出走馬の組み合わせを持つレースを探し、最初の1レース以外（コピー）を返す"""
    sets = df.groupby('race_id').agg(date=('race_date', 'first'), horses=('horse_id', lambda s: frozenset(s)))
    phantoms = []
    for _, grp in sets.groupby(['date', 'horses']):
        if len(grp) > 1:
            phantoms += sorted(grp.index.astype(str))[1:]
    return sorted(phantoms)


def fill_from_same_day(df: pd.DataFrame, rid: str, cols=('condition', 'weather')) -> None:
    """パースできなかった馬場・天候を、同じ日・同じ競馬場・同じ芝ダの直前（なければ直後）のレースから借りる"""
    mask = df['race_id'] == rid
    row = df.loc[mask].iloc[0]
    same = df[(df['race_date'] == row['race_date']) & (df['place'] == row['place']) & (df['race_id'] != rid)]
    for col in cols:
        if col not in df.columns or pd.notna(row.get(col)):
            continue
        cand = same[same[col].notna()]
        if 'type' in df.columns and pd.notna(row.get('type')):
            typed = cand[cand['type'] == row['type']]
            cand = typed if not typed.empty else (cand if col == 'weather' else typed)
        if cand.empty:
            continue
        before = cand[cand['race_id'] < rid].sort_values('race_id')
        src = before.iloc[-1] if not before.empty else cand.sort_values('race_id').iloc[0]
        df.loc[mask & df[col].isna(), col] = src[col]


def find_broken(df: pd.DataFrame) -> list:
    cols = [c for c in KEY_COLS if c in df.columns]
    g = df.groupby('race_id')[cols].agg(lambda s: s.isna().all() or (s.astype(str).str.strip() == '').all())
    return sorted(g.index[g.any(axis=1)].astype(str))


def repair_file(path: str, dry_run: bool):
    df = pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=True)
    name = os.path.basename(path)
    backup_done = False

    phantoms = find_phantom_races(df)
    if phantoms:
        print(f"🗑️ {name}: 他レースのコピーになっている偽レース {len(phantoms)}R: {', '.join(phantoms)}")
        if not dry_run:
            backup = f"{path}.bak_{datetime.now():%Y%m%d_%H%M}"
            shutil.copy2(path, backup)
            backup_done = True
            print(f"   💾 バックアップ: {os.path.basename(backup)}")
            df = df[~df['race_id'].isin(phantoms)].copy()
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"   ✅ 削除しました。scrape/scrape_main_data.py を実行して正しいデータを取り直してください。")

    broken = find_broken(df)
    if not broken:
        print(f"✅ {name}: 欠損レースなし")
        return
    dates = df[df['race_id'].isin(broken)].groupby('race_id')['race_date'].first()
    print(f"🔧 {name}: 欠損レース {len(broken)}R  （{dates.min()} 〜 {dates.max()}）")
    if dry_run:
        print("   " + ", ".join(broken[:30]) + (" ..." if len(broken) > 30 else ""))
        return

    if not backup_done:
        backup = f"{path}.bak_{datetime.now():%Y%m%d_%H%M}"
        shutil.copy2(path, backup)
        print(f"   💾 バックアップ: {os.path.basename(backup)}")

    fixed, failed, raw_texts = 0, [], {}
    for i, rid in enumerate(broken, 1):
        info = parse_race_info(rid)
        if '_raw' in info:
            raw_texts[rid] = info.pop('_raw')
        mask = df['race_id'] == rid
        for col, val in info.items():
            if col not in df.columns:
                continue
            empty = mask & (df[col].isna() | (df[col].astype(str).str.strip().isin(['', 'None', 'nan'])))
            df.loc[empty, col] = val
        # 'NoneNoneNone' のような壊れた course 文字列も置き換える
        if 'course' in df.columns and 'course' in info:
            bad_course = mask & df['course'].astype(str).str.contains('None|nan', regex=True)
            df.loc[bad_course, 'course'] = info['course']

        fill_from_same_day(df, rid)
        ok = all(pd.notna(df.loc[mask, c]).all() for c in KEY_COLS if c in df.columns)
        fixed += ok
        if not ok:
            failed.append(rid)
        print(f"\r   {i}/{len(broken)} 修復中... (成功 {fixed})", end="")
        time.sleep(random.uniform(1.5, 3.0))

    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n   ✅ {fixed}/{len(broken)}R を修復して保存しました")
    if failed:
        print(f"   ⚠️ 修復できなかったレース: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
        for rid in failed[:5]:
            if rid in raw_texts:
                print(f"      {rid} のページ表記: {raw_texts[rid]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='train / val / test すべてを対象にする')
    ap.add_argument('--dry-run', action='store_true', help='欠損レースの一覧表示のみ')
    args = ap.parse_args()

    dirs = ['train', 'val', 'test'] if args.all else ['test']
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(DATA_DIR, d, "race_data_*.csv")))
    if not files:
        print("❌ race_data_*.csv が見つかりません")
        return
    for f in files:
        repair_file(f, args.dry_run)
    if not args.dry_run:
        print("\n💡 修復後は evaluate_main.py / evaluate_dump.py を再実行するとバックテストに反映されます。")


if __name__ == "__main__":
    main()
