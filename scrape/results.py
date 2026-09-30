# scrape/results.py
# レース結果（db.netkeiba.com/race/<race_id>/）を取得して race_data_YYYY.csv に追記する。
#
# 以前は keibascraper（外部ライブラリ）を使っていた。出力の列・表記・型が既存データと変わると
# 学習済みモデルや前向き検証の特徴量がずれるため、keibascraper 3.1.4 の CSS セレクタ・正規表現・変換を
# そのまま移植し、同じ HTML から同じ値が出ることを確かめてある（scrape/tests/test_results.py）。
# 表記を直したくなっても、ここは変えずに v2/data/normalize.py 側で吸収すること。
#
# keibascraper との違い（出力の値は変えていない）:
#  - 存在しないレース番号（11Rしかない日の12Rなど）を指定すると、db.netkeiba はその日の1Rのページを返す。
#    これが「偽レース」の原因だったので、ページが表示しているレースIDが指定と違えば保存しない
#  - 念のため、同じ日の別レースと出走馬が完全に同じレースも保存せずに警告する（以前は保存してから repair で削除していた）
#  - レースが無いページでは待たずに次へ進む。アクセス制限を検知したら保存して止まる
#  - 保存先は既存の race_data_YYYY.csv の場所に合わせる（2025年は data/val）
import datetime
import os
import random
import re
import sys
import time

import pandas as pd

from scrape.common import DATA_DIR, create_session, get_headers, is_block_title, race_data_files
from bs4 import BeautifulSoup

URL = "https://db.netkeiba.com/race/{}/"
SAVE_INTERVAL = 20
MIN_SLEEP, MAX_SLEEP = 2.0, 5.0
COLS = ['race_id', 'rank', 'bracket', 'horse_number', 'horse_id', 'horse_name',
        'gender', 'age', 'burden', 'jockey_id', 'jockey_name', 'rap_time',
        'diff_time', 'passage_rank', 'last_3f', 'weight', 'weight_diff',
        'trainer_id', 'trainer_name', 'prize', 'id', 'race_number', 'race_name',
        'race_date', 'race_time', 'type', 'length', 'length_class', 'handed',
        'weather', 'condition', 'place', 'course', 'round', 'days',
        'head_count', 'max_prize']
PRIZE_RE = r"\d{1,3},\d{3}\.\d{1}|\d{1,3}\.\d{1}"
# ページから取らずに計算する列（空行の判定から外す）
DERIVED = {'race_id', 'diff_time', 'id', 'length_class', 'course'}


class NoRace(Exception):
    """ページにレース結果が無い（開催が無い・まだ確定していない・日付が読めない）"""


class Blocked(Exception):
    """アクセス制限・通信異常"""


# ---------- keibascraper の helper と同じ変換 ----------

def _fmt(pattern, target, var_type):
    """正規表現で取り出して型変換する。取れない・変換できないときは None"""
    if target is None:
        return None
    m = re.search(pattern, target)
    if not m:
        return None
    value = m.group(1) if m.groups() else m.group(0)
    if var_type in ('integer', 'real'):
        value = value.replace(',', '')
    try:
        if var_type == 'integer':
            return int(value)
        if var_type == 'real':
            return float(value)
        return value
    except (ValueError, TypeError):
        return None


def _text(tag):
    return tag.get_text(strip=True) if tag is not None else None


def _attr(tag, name):
    return tag.get(name) if tag is not None else None


def _time_to_seconds(tag):
    """'1:26.5' → '86.5'（文字列で返し、後で正規表現にかける）"""
    if not tag:
        return None
    parts = tag.text.split(':')
    try:
        if len(parts) == 3:
            h, m, s = parts
            return str(int(h) * 3600 + int(m) * 60 + float(s))
        if len(parts) == 2:
            m, s = parts
            return str(int(m) * 60 + float(s))
        if len(parts) == 1:
            return parts[0]
        return None
    except (ValueError, TypeError):
        return None


def _length_class(length):
    if length is None:
        return None
    if length < 1400:
        return 'Sprint'
    if length < 1800:
        return 'Mile'
    if length < 2200:
        return 'Intermediate'
    if length < 2800:
        return 'Long'
    return 'Extended'


def _has_value(record):
    for k, v in record.items():
        if k in DERIVED or v is None:
            continue
        if isinstance(v, str) and v.strip() == '':
            continue
        return True
    return False


# 1着のタイム。keibascraper と同じく、1着の行が無いレースでは直前に読んだレースの値が残る
_winner_time = 0.0


def _diff_time(rank, rap_time):
    global _winner_time
    if rank == 1 and rap_time is not None:
        _winner_time = rap_time
        return 0.0
    if rap_time is not None:
        return rap_time - _winner_time
    return None


# ---------- パース ----------

def _check_page(soup, validator):
    if soup.find(string=lambda s: isinstance(s, str) and "該当するデータはありません" in s):
        raise NoRace("該当するデータはありません")
    if not soup.select_one(validator):
        raise NoRace(f"{validator} がありません")


def parse_race_info(soup, race_id):
    """レース単位の情報（keibascraper の race_db と同じ）"""
    _check_page(soup, "div.mainrace_data")
    records = []
    for page in soup.select("div#page"):
        s = page.select_one
        span = _text(s("dl.racedata dd p diary_snap_cut span"))
        small = _text(s("div.mainrace_data.fc > div.data_intro > p.smalltxt"))
        date = _fmt(r"(\d{8})", _attr(s("div.race_num p a:first-child"), "href"), "text")
        if date is None:
            raise NoRace("開催日が読めません")
        race_type = _fmt("^芝|^ダ|^障", span, "text")
        race_type = {'芝': '芝', 'ダ': 'ダート', '障': '障害'}.get(race_type, race_type)
        length = _fmt(r"\d{3,4}", span, "integer")
        place = _fmt("函館|札幌|福島|新潟|中山|東京|中京|京都|阪神|小倉|帯広|門別|盛岡|水沢|浦和|船橋|大井|川崎|"
                     "金沢|笠松|名古屋|園田|姫路|高知|佐賀", small, "text")
        table = s("table.race_table_01.nk_tb_common")
        rec = {
            'id': race_id,
            'race_number': _fmt(r"(\d+)R", _text(s("div.race_num a.active")), "integer"),
            'race_name': _fmt(".+", _text(s("dl.racedata dd h1")), "text"),
            'race_date': datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
            'race_time': _fmt(r"発走 : (\d+:\d+)", span, "text"),
            'type': race_type,
            'length': length,
            'length_class': _length_class(length),
            'handed': _fmt("(右|左)", span, "text"),
            'weather': _fmt("晴|曇|小雨|雨|小雪|雪", span, "text"),
            'condition': _fmt("良|稍|稍重|重|不良", span, "text"),
            'place': place,
            'course': ''.join(map(str, (place, race_type, length))),
            'round': _fmt(r"(\d+)回", small, "integer"),
            'days': _fmt(r"(\d+)日目", small, "integer"),
            'head_count': len(table.select("tr")) - 1 if table is not None else None,
            'max_prize': _fmt(PRIZE_RE, _text(s("table.race_table_01.nk_tb_common tr:nth-of-type(2) "
                                                "td.txt_r:last-child")), "real"),
        }
        if _has_value(rec):
            records.append(rec)
    if not records:
        raise NoRace("レース情報がありません")
    return records


def parse_result_rows(soup, race_id):
    """出走馬ごとの結果（keibascraper の result と同じ）"""
    _check_page(soup, "table.race_table_01")
    records = []
    for tr in soup.select("table.race_table_01.nk_tb_common tr:not([class='txt_c'])"):
        s = tr.select_one
        sexage = _text(s("td:nth-of-type(5)"))
        weight = _text(s("td:nth-of-type(12)"))
        horse_a, jockey_a, trainer_a = s("td:nth-of-type(4) a"), s("td:nth-of-type(7) a"), s("td:nth-of-type(13) a")
        rank = _fmt(r"\d+", _text(s("td:nth-of-type(1)")), "integer")
        horse_number = _fmt(r"\d+", _text(s("td:nth-of-type(3)")), "integer")
        rap_time = _fmt(r"(\d+.\d+)", _time_to_seconds(s("td:nth-of-type(8)")), "real")
        prize = _fmt(PRIZE_RE, _text(s("td.txt_r:last-child")), "real")
        rec = {
            'race_id': race_id,
            'rank': rank,
            'bracket': _fmt(r"\d+", _text(s("td:nth-of-type(2)")), "integer"),
            'horse_number': horse_number,
            'horse_id': _fmt(r"/horse/(\w+)", _attr(horse_a, "href"), "text"),
            'horse_name': _attr(horse_a, "title"),
            'gender': _fmt("[牡牝騸セ]", sexage, "text"),
            'age': _fmt(r"\d+", sexage, "integer"),
            'burden': _fmt(r"\d+.?\d?", _text(s("td:nth-of-type(6)")), "real"),
            'jockey_id': _fmt(r"/jockey/result/recent/(\w+)/", _attr(jockey_a, "href"), "text"),
            'jockey_name': _fmt(r"\D+", _attr(jockey_a, "title"), "text"),
            'rap_time': rap_time,
            'diff_time': _diff_time(rank, rap_time),
            'passage_rank': _fmt(".*", _text(s("td:not([class])")), "text"),
            'last_3f': _fmt(r"(\d+.\d+)", _text(s("td.txt_c span")), "real"),
            'weight': _fmt(r"(\d+)\([+-]?\d*\)", weight, "integer"),
            'weight_diff': _fmt(r"\d+\(([+-]?\d+)\)", weight, "integer"),
            'trainer_id': _fmt(r"/trainer/result/recent/(\w+)/", _attr(trainer_a, "href"), "text"),
            'trainer_name': _fmt(r"\D+", _attr(trainer_a, "title"), "text"),
            'prize': prize if prize is not None else 0,
            'id': f"{race_id}{str(horse_number).zfill(2)}",
        }
        if _has_value(rec):
            records.append(rec)
    if not records:
        raise NoRace("結果の行がありません")
    return records


def parse_page(html, race_id):
    """keibascraper.load('result', race_id) と同じ (レース情報のリスト, 結果のリスト) を返す"""
    soup = BeautifulSoup(html, 'html.parser')
    return parse_race_info(soup, race_id), parse_result_rows(soup, race_id)


def shown_race_id(html):
    """ページが実際に表示しているレースのID（レース番号の選択欄で選ばれているもの）。読めなければ None"""
    soup = BeautifulSoup(html, 'html.parser')
    return _fmt(r"/race/(\d{12})/", _attr(soup.select_one("div.race_num a.active"), "href"), "text")


# ---------- 取得 ----------

def fetch_page(session, race_id, max_retries=2):
    """ページの HTML を返す。レースが無ければ None、制限・通信異常が続けば Blocked"""
    for attempt in range(max_retries):
        try:
            r = session.get(URL.format(race_id), headers=get_headers("https://db.netkeiba.com/"), timeout=20)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                r.encoding = r.apparent_encoding
                html = r.text
                if not is_block_title(BeautifulSoup(html[:5000], 'html.parser')):
                    return html
                why = "制限画面"
            else:
                why = f"HTTP {r.status_code}"
        except Exception as e:
            why = str(e)
        if attempt < max_retries - 1:
            print(f"\n🚨 通信エラー（{why}）。120秒待ってリトライします")
            time.sleep(120)
    raise Blocked(why)


def load_race(session, race_id):
    """1レース分の DataFrame（race_data と同じ列）。レースが無ければ None"""
    html = fetch_page(session, race_id)
    if html is None:
        return None
    shown = shown_race_id(html)
    if shown is not None and shown != race_id:
        return None     # 存在しないレース番号で別のレース（その日の1R）が返ってきた
    try:
        info, rows = parse_page(html, race_id)
    except NoRace:
        return None
    df = pd.DataFrame(rows)
    for key, val in info[0].items():
        df[key] = val
    df['race_id'] = race_id
    return df


def phantom_of(df, day_horses):
    """同じ日に出走馬がまったく同じレースがあれば、そのレースID（偽レースの検出）"""
    horses = frozenset(df['horse_id'].astype(str))
    rid = str(df['race_id'].iloc[0])
    for other, hs in day_horses.get(str(df['race_date'].iloc[0]), {}).items():
        if other != rid and hs == horses:
            return other
    return None


def output_path(year):
    files = race_data_files()
    if year in files:
        return files[year]
    folder = "train" if year <= 2023 else ("val" if year <= 2025 else "test")
    return str(DATA_DIR / folder / f"race_data_{year}.csv")


def save_to_csv(data_list, filename):
    if not data_list:
        return
    df_new = pd.concat(data_list, ignore_index=True)

    if os.path.exists(filename):
        try:
            existing_cols = pd.read_csv(filename, nrows=0, encoding='utf-8-sig').columns.tolist()
            for c in existing_cols:
                if c not in df_new.columns:
                    df_new[c] = None
            df_new = df_new[existing_cols]
            df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='a', header=False)
            return
        except Exception:
            pass

    for c in COLS:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new[COLS]
    df_new.to_csv(filename, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(filename))


def scrape_year(year):
    filename = output_path(year)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"\n🚀 {year}年のレース結果を取得します")
    print(f"   💾 保存先: {filename}")

    existing_ids, day_horses = set(), {}
    if os.path.exists(filename):
        df_exist = pd.read_csv(filename, dtype=str, encoding='utf-8-sig', usecols=['race_id', 'race_date', 'horse_id'])
        existing_ids = set(df_exist['race_id'].dropna())
        for (date, rid), g in df_exist.groupby(['race_date', 'race_id']):
            day_horses.setdefault(date, {})[rid] = frozenset(g['horse_id'].astype(str))
        print(f"📂 既存 {len(existing_ids)} レース -> 続きから取得します")

    session = create_session()
    session_data, phantoms = [], []

    def get(race_id):
        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
        return load_race(session, race_id)

    try:
        for place in range(1, 11):
            for kai in range(1, 7):
                for day in range(1, 13):
                    base = f"{year}{place:02d}{kai:02d}{day:02d}"
                    first = None
                    if f"{base}01" not in existing_ids:
                        first = get(f"{base}01")
                        if first is None:
                            continue
                    print(f"\n📅 開催確認: {base}")

                    for r in range(1, 13):
                        race_id = f"{base}{r:02d}"
                        if race_id in existing_ids:
                            continue
                        sys.stdout.write(f"\r    Running: {race_id} ... ")
                        df = first if r == 1 else get(race_id)
                        if df is None or df.empty:
                            sys.stdout.write("Skip (No Data)\n")
                            continue
                        dup = phantom_of(df, day_horses)
                        if dup:
                            sys.stdout.write(f"⚠️ {dup} と出走馬が同じ（偽レース）なので保存しません\n")
                            phantoms.append(race_id)
                            continue
                        session_data.append(df)
                        existing_ids.add(race_id)
                        day_horses.setdefault(str(df['race_date'].iloc[0]), {})[race_id] = \
                            frozenset(df['horse_id'].astype(str))
                        sys.stdout.write(f"✅ OK ({len(df)}頭)\n")

                        if len(session_data) >= SAVE_INTERVAL:
                            save_to_csv(session_data, filename)
                            session_data = []
                            print(f"    💾 保存 (計 {len(existing_ids)} レース)")
    except Blocked as e:
        print(f"\n🚨 アクセス制限または通信異常を検知したため停止します（{e}）。時間をおいて再実行すれば続きから取得します。")
        save_to_csv(session_data, filename)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 中断。取得済みの分を保存します。")

    save_to_csv(session_data, filename)
    print(f"\n🎉 終了。総レース数: {len(existing_ids)}")
    if phantoms:
        print(f"⚠️ 偽レースのため保存しなかったレース: {', '.join(phantoms)}")
        print(f"   python -m scrape rescrape --ids {','.join(phantoms)} で race.netkeiba から取り直せます。")
