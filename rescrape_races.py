# rescrape_races.py
"""
指定したレースを netkeiba のレース結果ページ（db.netkeiba.com）から直接取り直し、
race_data_YYYY.csv に書き戻す。

背景: keibascraper が特定のレースで「同じ開催日の別レースの出走馬・着順」をそのまま返すこと
      （偽レース）があり、再取得しても同じ結果になる。そのため keibascraper を経由せず、
      結果ページを直接パースして同じ列構成で保存する。

使い方:
    python rescrape_races.py --auto                 # 抜けているレースを自動検出して取得
    python rescrape_races.py --ids 202605010308,202605010309
    python rescrape_races.py --auto --dry-run       # 取得して中身を表示するだけ（保存しない）

・db.netkeiba が別レースの内容を返す場合は race.netkeiba の結果ページから取り直す
・どちらのページでも同じ日の別レースと出走馬が同一だった場合は保存しない
・--auto は「抜けているレース」と「他レースのコピーになっているレース」の両方を対象にする
  （1日12Rを前提に探すため、中止等で元々存在しないレースは取得時に404になり、そのまま飛ばされる）
・既存の同じ race_id の行は置き換える
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

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
COLS = ['race_id', 'rank', 'bracket', 'horse_number', 'horse_id', 'horse_name',
        'gender', 'age', 'burden', 'jockey_id', 'jockey_name', 'rap_time',
        'diff_time', 'passage_rank', 'last_3f', 'weight', 'weight_diff',
        'trainer_id', 'trainer_name', 'prize', 'id', 'race_number', 'race_name',
        'race_date', 'race_time', 'type', 'length', 'length_class', 'handed',
        'weather', 'condition', 'place', 'course', 'round', 'days',
        'head_count', 'max_prize']


def length_class(length: int) -> str:
    if length < 1400: return "Sprint"
    if length < 1800: return "Mile"
    if length < 2200: return "Intermediate"
    if length < 2850: return "Long"
    return "Extended"


def to_seconds(text: str):
    """'1:26.5' → 86.5"""
    m = re.match(r'^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$', str(text))
    if not m:
        return np.nan
    minutes = int(m.group(1)) if m.group(1) else 0
    return round(minutes * 60 + float(m.group(2)), 1)


def to_float(text):
    try:
        return float(re.sub(r'[^0-9.\-]', '', str(text)))
    except Exception:
        return np.nan


def fetch(url, max_retries=3, quiet=False):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.encoding = 'EUC-JP'
            if r.status_code == 200:
                return r
            if quiet:
                return None
            if r.status_code in (403, 404):
                print(f"   ❌ HTTP {r.status_code}: {url}")
                return None
        except Exception as e:
            print(f"   ⚠️ 通信エラー({attempt + 1}/{max_retries}): {e}")
        time.sleep(3 * 2 ** attempt)
    return None


def _id_from(cell, kind):
    a = cell.find("a", href=re.compile(rf'/{kind}/'))
    if not a:
        return None, cell.get_text(strip=True)
    m = re.search(rf'/{kind}/(?:result/recent/)?([0-9a-zA-Z]+)', a['href'])
    return (m.group(1) if m else None), a.get_text(strip=True)


def parse_db_page(race_id: str, verbose: bool = False) -> pd.DataFrame:
    """db.netkeiba のレース結果ページ（旧デザイン）を、race_data_*.csv と同じ列構成にする"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    r = fetch(url)
    if r is None:
        return pd.DataFrame()
    if race_id not in str(r.url):
        if verbose:
            print(f"   ↪ 転送されました: {r.url}")
        return pd.DataFrame()
    soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')

    table = soup.find("table", class_=re.compile(r'race_table_01|nk_tb_common'))
    if table is None:
        print(f"   ❌ 結果テーブルが見つかりません: {race_id}")
        return pd.DataFrame()

    rows = table.find_all("tr")
    header = [re.sub(r'\s+', '', th.get_text()) for th in rows[0].find_all(["th", "td"])]

    def col(*names):
        for n in names:
            if n in header:
                return header.index(n)
        return None

    idx = {k: col(*v) for k, v in {
        'rank': ('着順',), 'bracket': ('枠番',), 'horse_number': ('馬番',), 'horse': ('馬名',),
        'sexage': ('性齢',), 'burden': ('斤量',), 'jockey': ('騎手',), 'time': ('タイム',),
        'passage': ('通過',), 'last_3f': ('上り', '後3F'), 'weight': ('馬体重',),
        'trainer': ('調教師', '厩舎'), 'prize': ('賞金(万円)', '賞金'),
    }.items()}

    # レース情報（例: 「ダ右1200m / 天候 : 雨 / ダート : 重 / 発走 : 11:40」）
    info_text, race_name, date_text = "", "", ""
    racedata = soup.find("dl", class_="racedata")
    if racedata:
        h1 = racedata.find("h1")
        race_name = h1.get_text(strip=True) if h1 else ""
        snap = racedata.find("diary_snap_cut") or racedata.find("div", class_="diary_snap_cut")
        info_text = (snap or racedata).get_text(" ", strip=True)
    small = soup.find("p", class_="smalltxt")
    if small:
        date_text = small.get_text(" ", strip=True)

    info_c = info_text.replace(" ", "")
    race_type = "障害" if "障" in info_c else ("芝" if "芝" in info_c else "ダート")
    m_len = re.search(r'(\d{3,4})m', info_c)
    length = int(m_len.group(1)) if m_len else np.nan
    m_hand = re.search(r'[芝ダ障]\s*(右|左)', info_c)
    handed = m_hand.group(1) if m_hand else None
    m_time = re.search(r'発走:?(\d{1,2}:\d{2})', info_c)
    race_time = m_time.group(1) if m_time else None
    m_w = re.search(r'天候:(\S)', info_c)
    weather = m_w.group(1) if m_w else None
    if weather and weather in "小":
        m_w2 = re.search(r'天候:(小雨|小雪)', info_c)
        weather = m_w2.group(1) if m_w2 else weather
    m_c = re.search(r'(?:芝|ダート|障害):(不良|稍重|重|良)', info_c)
    condition = m_c.group(1) if m_c else None
    if condition == "稍重":
        condition = "稍"

    m_date = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
    race_date = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}" if m_date else None
    if race_name and re.search(r'[\(（][^\)）]+[\)）]', race_name) is None:
        m_grade = re.search(r'[\(（](G[IⅠ1]{1,3}|JpnI{1,3}|L)[\)）]', date_text)
        if m_grade:
            race_name = f"{race_name}({m_grade.group(1)})"

    return build_records_and_finalize(rows[1:], idx, race_id, dict(
        race_name=race_name, race_date=race_date, race_time=race_time, race_type=race_type,
        length=length, handed=handed, weather=weather, condition=condition))


def build_records(rows, idx, race_id: str, info: dict) -> list:
    """結果テーブルの各行を race_data_*.csv の1行に変換する（新旧どちらのページでも共通）"""
    records = []
    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        def cell(key):
            i = idx.get(key)
            return tds[i] if (i is not None and i < len(tds)) else None

        horse_cell = cell('horse')
        if horse_cell is None:
            continue
        horse_id, horse_name = _id_from(horse_cell, 'horse')
        if not horse_id:
            continue
        jockey_cell, trainer_cell = cell('jockey'), cell('trainer')
        jockey_id, jockey_name = _id_from(jockey_cell, 'jockey') if jockey_cell else (None, None)
        trainer_id, trainer_name = _id_from(trainer_cell, 'trainer') if trainer_cell else (None, None)

        rank_text = cell('rank').get_text(strip=True) if cell('rank') else ''
        rank = float(rank_text) if re.fullmatch(r'\d+', rank_text) else np.nan

        sexage = cell('sexage').get_text(strip=True) if cell('sexage') else ''
        gender = sexage[:1] if sexage else None
        age = to_float(sexage[1:]) if len(sexage) > 1 else np.nan

        w_text = cell('weight').get_text(strip=True) if cell('weight') else ''
        m_wt = re.match(r'(\d+)\(([+\-±]?\d+)\)', w_text)
        if m_wt:
            weight = float(m_wt.group(1))
            weight_diff = float(m_wt.group(2).replace('±', ''))
        else:
            weight = to_float(w_text) if re.match(r'^\d+$', w_text) else np.nan
            weight_diff = 0.0

        prize_cell = cell('prize')
        prize = to_float(prize_cell.get_text(strip=True).replace(',', '')) if prize_cell else np.nan
        if prize_cell is not None and np.isnan(prize):
            prize = 0.0
        if prize_cell is None:
            prize = 0.0  # 新デザインのページには賞金欄が無い

        length = info.get('length')
        records.append({
            'race_id': race_id,
            'rank': rank,
            'bracket': to_float(cell('bracket').get_text(strip=True)) if cell('bracket') else np.nan,
            'horse_number': to_float(cell('horse_number').get_text(strip=True)) if cell('horse_number') else np.nan,
            'horse_id': horse_id, 'horse_name': horse_name,
            'gender': gender, 'age': age,
            'burden': to_float(cell('burden').get_text(strip=True)) if cell('burden') else np.nan,
            'jockey_id': jockey_id, 'jockey_name': jockey_name,
            'rap_time': to_seconds(cell('time').get_text(strip=True)) if cell('time') else np.nan,
            'passage_rank': cell('passage').get_text(strip=True) if cell('passage') else None,
            'last_3f': to_float(cell('last_3f').get_text(strip=True)) if cell('last_3f') else np.nan,
            'weight': weight, 'weight_diff': weight_diff,
            'trainer_id': trainer_id, 'trainer_name': trainer_name,
            'prize': prize,
            'id': race_id, 'race_number': int(race_id[10:12]), 'race_name': info.get('race_name'),
            'race_date': info.get('race_date'), 'race_time': info.get('race_time'),
            'type': info.get('race_type'), 'length': length,
            'length_class': length_class(int(length)) if pd.notna(length) else None,
            'handed': info.get('handed'), 'weather': info.get('weather'), 'condition': info.get('condition'),
            'place': PLACE_MAP.get(race_id[4:6]),
            'round': int(race_id[6:8]), 'days': int(race_id[8:10]),
        })
    return records


def build_records_and_finalize(rows, idx, race_id: str, info: dict) -> pd.DataFrame:
    return finalize(build_records(rows, idx, race_id, info))


def finalize(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    winner = df.loc[df['rank'] == 1, 'rap_time']
    base = winner.iloc[0] if not winner.empty else df['rap_time'].min()
    df['diff_time'] = (df['rap_time'] - base).round(1)
    df['head_count'] = len(df)
    df['max_prize'] = df['prize'].max()
    df['course'] = df['place'].astype(str) + df['type'].astype(str) + df['length'].fillna(0).astype(int).astype(str)
    return df.reindex(columns=COLS)


def parse_modern_page(race_id: str, verbose: bool = False) -> pd.DataFrame:
    """race.netkeiba.com の結果ページ（新デザイン）から取り出す。
       db.netkeiba 側が別レースの内容を返す場合の代替。賞金欄が無いため prize は 0 になる。"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = fetch(url)
    if r is None:
        return pd.DataFrame()
    soup = BeautifulSoup(r.content, "html.parser", from_encoding='EUC-JP')

    table = soup.find("table", class_=re.compile(r'RaceTable01|ResultTableWrap'))
    if table is None:
        if verbose:
            print(f"   ❌ 結果テーブルが見つかりません（新デザイン）: {race_id}")
        return pd.DataFrame()

    rows = table.find_all("tr")
    header = [re.sub(r'\s+', '', c.get_text()) for c in rows[0].find_all(["th", "td"])]

    def col(*names):
        for n in names:
            for i, h in enumerate(header):
                if h == n or h.startswith(n):
                    return i
        return None

    idx = {k: col(*v) for k, v in {
        'rank': ('着順',), 'bracket': ('枠',), 'horse_number': ('馬番',), 'horse': ('馬名',),
        'sexage': ('性齢',), 'burden': ('斤量',), 'jockey': ('騎手',), 'time': ('タイム',),
        'passage': ('コーナー通過順', '通過'), 'last_3f': ('後3F', '上り'), 'weight': ('馬体重',),
        'trainer': ('厩舎', '調教師'), 'prize': ('賞金',),
    }.items()}

    # レース情報（例: 「10:05発走 / ダ1400m (左) / 天候:晴 / 馬場:良」）
    info_c = ""
    data01 = soup.find("div", class_="RaceData01")
    if data01:
        info_c = data01.get_text().replace("\n", "").replace(" ", "")
    race_type = "障害" if "障" in info_c else ("芝" if "芝" in info_c else "ダート")
    m_len = re.search(r'(\d{3,4})m', info_c)
    length = int(m_len.group(1)) if m_len else np.nan
    m_hand = re.search(r'\((左|右)', info_c)
    handed = m_hand.group(1) if m_hand else None
    m_time = re.search(r'(\d{1,2}:\d{2})発走', info_c)
    race_time = m_time.group(1) if m_time else None
    m_w = re.search(r'天候:([^/]+)', info_c)
    weather = next((x for x in ["小雪", "雪", "小雨", "雨", "曇", "晴"] if m_w and x in m_w.group(1)), None)
    m_c = re.search(r'馬場:([^/]+)', info_c)
    condition = next((x for x in ["不良", "稍重", "稍", "重", "良"] if m_c and x in m_c.group(1)), None)
    if condition == "稍重":
        condition = "稍"

    name_tag = soup.find("h1", class_="RaceName")
    race_name = name_tag.get_text(strip=True) if name_tag else ""
    title = soup.title.get_text(strip=True) if soup.title else ""
    m_date = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
    race_date = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}" if m_date else None
    if race_name and not re.search(r'[\(（][^\)）]+[\)）]', race_name):
        m_grade = re.search(r'[\(（](G[IⅠ1]{1,3}|JpnI{1,3}|L)[\)）]', title)
        if m_grade:
            race_name = f"{race_name}({m_grade.group(1)})"

    return build_records_and_finalize(rows[1:], idx, race_id, dict(
        race_name=race_name, race_date=race_date, race_time=race_time, race_type=race_type,
        length=length, handed=handed, weather=weather, condition=condition))


def phantom_match(new_df: pd.DataFrame, existing: pd.DataFrame):
    """取得したレースが同じ日の別レースと出走馬まで同一なら、その別レースのIDを返す"""
    date = new_df['race_date'].iloc[0]
    rid = str(new_df['race_id'].iloc[0])
    horses = frozenset(new_df['horse_id'].astype(str))
    same_day = existing[(existing['race_date'].astype(str) == str(date)) & (existing['race_id'].astype(str) != rid)]
    for other, g in same_day.groupby('race_id'):
        if frozenset(g['horse_id'].astype(str)) == horses:
            return str(other)
    return None


def fetch_race(race_id: str, existing: pd.DataFrame):
    """db.netkeiba → race.netkeiba の順に試し、別レースのコピーでない結果を返す"""
    attempts = []
    for label, parser in [("db.netkeiba", parse_db_page), ("race.netkeiba", parse_modern_page)]:
        df = parser(race_id, verbose=True)
        if df.empty:
            attempts.append((label, "取得できず"))
            time.sleep(random.uniform(1.0, 2.0))
            continue
        dup = phantom_match(df, existing)
        if dup is None:
            return df, label, attempts
        attempts.append((label, f"{dup} と出走馬が同一"))
        print(f"   ⚠️ {label}: {dup} と同じ出走馬が返ってきました")
        time.sleep(random.uniform(1.0, 2.0))
    return pd.DataFrame(), None, attempts


def find_missing_ids(df: pd.DataFrame, races_per_day: int = 12) -> list:
    """開催日ごとに 1R〜12R を見て、抜けているレースIDを返す（中止等で元々無い場合は取得時に404になる）"""
    have = set(df['race_id'].astype(str))
    missing = []
    for base in sorted({rid[:10] for rid in have}):
        for n in range(1, races_per_day + 1):
            rid = f"{base}{n:02d}"
            if rid not in have:
                missing.append(rid)
    return missing


def find_phantom_ids(df: pd.DataFrame) -> list:
    """同じ日に出走馬が完全に同じレースが複数あれば、2件目以降を取り直し対象として返す"""
    sets = df.groupby('race_id').agg(date=('race_date', 'first'), horses=('horse_id', lambda s: frozenset(s)))
    out = []
    for _, grp in sets.groupby(['date', 'horses']):
        if len(grp) > 1:
            out += sorted(grp.index.astype(str))[1:]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', help='取得するレースID（カンマ区切り）')
    ap.add_argument('--auto', action='store_true', help='race_data_*.csv で抜けているレースを自動検出する')
    ap.add_argument('--year', type=int, default=None, help='対象年（省略時はレースIDから判定）')
    ap.add_argument('--dry-run', action='store_true', help='取得内容を表示するだけで保存しない')
    args = ap.parse_args()

    files = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*", "race_data_*.csv"))):
        year = re.search(r'race_data_(\d{4})\.csv', path).group(1)
        files[year] = path

    targets = []
    if args.ids:
        targets = [x.strip() for x in args.ids.split(',') if x.strip()]
    if args.auto:
        for year, path in files.items():
            if args.year and int(year) != args.year:
                continue
            df = pd.read_csv(path, dtype=str, encoding='utf-8-sig',
                             usecols=['race_id', 'race_date', 'horse_id'])
            miss = find_missing_ids(df)
            phantom = find_phantom_ids(df)
            if miss:
                print(f"🔎 {os.path.basename(path)}: 抜けているレース {len(miss)}R -> {', '.join(miss[:20])}"
                      f"{' ...' if len(miss) > 20 else ''}")
            if phantom:
                print(f"🔎 {os.path.basename(path)}: 他レースのコピーになっているレース {len(phantom)}R -> "
                      f"{', '.join(phantom[:20])}{' ...' if len(phantom) > 20 else ''}")
            targets += miss + phantom
    if not targets:
        print("取得対象がありません。--ids か --auto を指定してください。")
        return

    by_year = {}
    for rid in sorted(set(targets)):
        by_year.setdefault(rid[:4], []).append(rid)

    for year, rids in by_year.items():
        path = files.get(year)
        if path is None:
            print(f"❌ {year}年の race_data_{year}.csv が見つかりません")
            continue
        base = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
        fetched, skipped = [], []

        for i, rid in enumerate(rids, 1):
            print(f"\n[{i}/{len(rids)}] {rid} を取得中...")
            new_df, source, attempts = fetch_race(rid, base)
            if new_df.empty:
                why = " / ".join(f"{lab}: {msg}" for lab, msg in attempts) or "取得失敗"
                skipped.append((rid, why))
            else:
                top = new_df.sort_values('rank').head(3)
                length_txt = int(new_df['length'].iloc[0]) if pd.notna(new_df['length'].iloc[0]) else '?'
                print(f"   ✅ [{source}] {new_df['race_date'].iloc[0]} {new_df['place'].iloc[0]} "
                      f"{new_df['race_name'].iloc[0]} {new_df['type'].iloc[0]}{length_txt}m "
                      f"{new_df['condition'].iloc[0]} / {len(new_df)}頭")
                print(f"      1〜3着: {', '.join(top['horse_name'].tolist())}")
                if source == "race.netkeiba":
                    print("      ※ このページには賞金欄が無いため prize は 0 になります")
                fetched.append(new_df)
            time.sleep(random.uniform(2.0, 4.0))

        if fetched and not args.dry_run:
            backup = f"{path}.bak_{datetime.now():%Y%m%d_%H%M}"
            shutil.copy2(path, backup)
            print(f"\n💾 バックアップ: {os.path.basename(backup)}")
            add = pd.concat(fetched, ignore_index=True).astype(str)
            keep = base[~base['race_id'].isin(add['race_id'].unique())]
            out = pd.concat([keep, add.reindex(columns=base.columns)], ignore_index=True)
            out.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ {len(fetched)}R を race_data_{year}.csv に保存しました（{len(add)}行）")

        if skipped:
            print("\n⚠️ 取得できなかったレース:")
            for rid, why in skipped:
                print(f"   {rid}: {why}")

    if args.dry_run:
        print("\n（--dry-run のため保存していません）")
    else:
        print("\n💡 この後 repair_race_info.py → evaluate_dump.py の順で実行してください。")


if __name__ == "__main__":
    main()
