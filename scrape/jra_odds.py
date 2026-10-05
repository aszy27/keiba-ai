# scrape/jra_odds.py
# JRA 公式サイトの単勝・複勝オッズ（発走前のスナップショットと --live 用）。
# netkeiba のオッズの API は、ログインしていない人の閲覧回数を日ごとのクッキーで数え、5回で制限する（2026-10-03 に判明）ため、
# 発売中のオッズは JRA 公式から取る。JRA のサイトはプログラムからのアクセスを禁止しておらず（robots.txt は全許可、
# 「ご利用に際して」に禁止の記載なし。2026-10-05 確認）、オッズは確定後に netkeiba の確定オッズと一致する（毎日王冠 2026-10-04 の17頭で確認）。
# 通信は scrape.common.http_post（間隔の制御・再試行・ブロックの検知と冷却期間）を通す。
#
# ページのたどり方（JRA のサイトはフォームの POST で、cname の末尾の2文字は検査用なので自分では作れない）:
#   オッズの一覧（cname = pw15oli00/6D） → 開催ごとのページ（pw15orl10{場}{年}{回}{日}{日付}/XX）
#   → レースごとの単勝・複勝（pw151ou10{場}{年}{回}{日}{R}{日付}Z/XX）
#   {場}{年}{回}{日}{R} は netkeiba のレース ID（年・場・回・日・R の順）と1対1で対応する。
import re

import numpy as np
from bs4 import BeautifulSoup

from scrape.common import http_post

URL = "https://www.jra.go.jp/JRADB/accessO.html"
LIST_CNAME = "pw15oli00/6D"
_ACTION = re.compile(r"doAction\('/JRADB/accessO\.html',\s*'([^']+)'\)")
_MEETING = re.compile(r"^pw15orl10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/")
_RACE = re.compile(r"^pw151ou10(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z/")
_cnames = {}   # netkeiba のレース ID → 単勝・複勝のページの cname（日ごとに一度たどれば足りる）


def _post(cname, honor_cooldown=True):
    r, status = http_post(URL, {"cname": cname}, referer=URL, honor_cooldown=honor_cooldown)
    return (r.content.decode("cp932", errors="replace") if r is not None else None), status


def race_links(meeting_html, date):
    """開催のページの HTML → {netkeiba のレース ID: cname}（その日付のものだけ）"""
    out = {}
    for c in _ACTION.findall(meeting_html):
        m = _RACE.match(c)
        if m and m.group(6) == date:
            place, year, kai, day, r = m.group(1, 2, 3, 4, 5)
            out[f"{year}{place}{kai}{day}{r}"] = c
    return out


def meeting_links(list_html, date):
    return sorted({c for c in _ACTION.findall(list_html) if (m := _MEETING.match(c)) and m.group(5) == date})


def discover(date, honor_cooldown=False):
    """その日の全レースの cname をたどって覚える。戻り値は {レース ID: cname}"""
    html, status = _post(LIST_CNAME, honor_cooldown)
    if html is None:
        return {}
    found = {}
    for mc in meeting_links(html, date):
        mh, _ = _post(mc, honor_cooldown)
        if mh:
            found.update(race_links(mh, date))
    _cnames.update(found)
    return found


def _num(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return np.nan            # 取消・除外・発売前（「----」など）


def parse_odds(html, race_id):
    """単勝・複勝のページ → (行の一覧, 状態)。行の形は scrape.snapshot.fetch_odds と同じ。
    状態: jra_final（最終オッズ）/ jra（発売中）/ jra_presale（数字が1つも無い）/ jra_no_table"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tanpuku")
    if table is None:
        return [], "jra_no_table"
    rows = []
    for tr in table.find_all("tr"):
        num, tan, fuku = (tr.find("td", class_=c) for c in ("num", "odds_tan", "odds_fuku"))
        if not (num and tan and num.get_text(strip=True).isdigit()):
            continue
        f = re.findall(r"[\d,]+\.\d+|[\d,]+", fuku.get_text(" ", strip=True)) if fuku else []
        rows.append({"race_id": race_id, "horse_number": int(num.get_text(strip=True)),
                     "win_odds": _num(tan.get_text(strip=True)),
                     "place_odds_min": _num(f[0]) if len(f) > 0 else np.nan,
                     "place_odds_max": _num(f[1]) if len(f) > 1 else np.nan})
    rows = [r for r in rows if np.isfinite(r["win_odds"]) and r["win_odds"] > 0]
    if not rows:
        return [], "jra_presale"
    order = sorted(rows, key=lambda r: r["win_odds"])
    for i, r in enumerate(order, 1):
        r["popularity"] = i
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(\d{1,2})時(\d{1,2})分現在", text)
    for r in rows:
        r["official_datetime"] = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}" if m else ""
    return rows, ("jra_final" if "最終オッズ" in text else "jra")


def fetch_odds(race_id, date, honor_cooldown=False):
    """1レースの単勝・複勝（JRA 公式）→ (行, 状態)。date は YYYYMMDD。cname を知らなければその日の分をたどる"""
    if race_id not in _cnames:
        discover(date, honor_cooldown)
    cname = _cnames.get(race_id)
    if cname is None:
        return [], "error: jra_no_link"
    html, status = _post(cname, honor_cooldown)
    if html is None:
        return [], f"error: {status}"
    return parse_odds(html, race_id)
