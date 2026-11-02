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
#   orl・ou の後の2文字は変わる（2026-10-10 に orl00・ouS3 になった）ので、何でも受け付ける。
#   {場}{年}{回}{日}{R} は netkeiba のレース ID（年・場・回・日・R の順）と1対1で対応する。
#   3連複は pw157ou…Z99/XX（全通り）、3連単は pw158ou…Z/XX（全通り）。単勝・複勝のページと、3連複・3連単のページ自身に
#   その日のレースの分のリンクがあるので、見つけたものを覚えておく（2026-10-26 確認。3連単の全通りのページは18頭で約0.7MB）。
#   馬連は pw154ou…Z/XX、馬単は pw156ou…Z/XX（どちらも全通りが1ページ、17頭で約0.1MB。2026-11-02 確認）。
import re

import numpy as np
from bs4 import BeautifulSoup

from scrape.common import http_post

URL = "https://www.jra.go.jp/JRADB/accessO.html"
LIST_CNAME = "pw15oli00/6D"
_ACTION = re.compile(r"doAction\('/JRADB/accessO\.html',\s*'([^']+)'\)")
_MEETING = re.compile(r"^pw15orl[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/")
_RACE = re.compile(r"^pw151ou[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z/")
_EXOTIC = {"trio": re.compile(r"^pw157ou[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z99/"),
           "trifecta": re.compile(r"^pw158ou[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z/"),
           "quinella": re.compile(r"^pw154ou[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z/"),
           "exacta": re.compile(r"^pw156ou[0-9A-Z]{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z/")}
_PAIR_TABLE = {"quinella": "umaren", "exacta": "umatan"}   # 馬連・馬単のページの表（見出しが1頭目、行が2頭目）
_cnames = {}   # netkeiba のレース ID → 単勝・複勝のページの cname（日ごとに一度たどれば足りる）
_exotic_cnames = {}   # (レース ID, 券種) → 組み合わせ券のページの cname


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
    # 発売中のページも注記に「最終オッズ」の語を含む（2026-10-10 確認）ので、「○時○分現在」があれば発売中とみなす
    return rows, ("jra" if m else "jra_final" if "最終オッズ" in text else "jra")


def _status(text):
    """「○時○分現在」があれば発売中（jra）、無くて「最終オッズ」があれば jra_final"""
    m = re.search(r"(\d{1,2})時(\d{1,2})分現在", text)
    official = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}" if m else ""
    return official, ("jra" if m else "jra_final" if "最終オッズ" in text else "jra")


def harvest_exotic(html):
    """ページの中の組み合わせ券（馬連・馬単・3連複・3連単）のリンクを覚える"""
    for c in _ACTION.findall(html):
        for bet, pat in _EXOTIC.items():
            m = pat.match(c)
            if m:
                place, year, kai, day, r = m.group(1, 2, 3, 4, 5)
                _exotic_cnames[(f"{year}{place}{kai}{day}{r}", bet)] = c


def parse_exotic(html, race_id, bet):
    """組み合わせ券（全通り）のページ → (行の一覧, 状態)。行は {race_id, combo, odds, official_datetime}。
    combo は netkeiba の API と同じ表記（馬番2桁ずつ。馬連・3連複は昇順、馬単・3連単は着順）。取消などで空欄の組み合わせは含めない"""
    soup = BeautifulSoup(html, "html.parser")
    official, status = _status(soup.get_text(" ", strip=True))
    rows = []

    def add(nums, td):
        v = _num(td.get_text(strip=True)) if td else np.nan
        if np.isfinite(v) and v > 0:
            rows.append({"race_id": race_id, "combo": "".join(f"{n:02d}" for n in nums), "odds": v,
                         "official_datetime": official})

    if bet == "trio":
        for t in soup.find_all("table", class_="fuku3"):
            cap = t.find("caption")
            m = re.fullmatch(r"(\d+)-(\d+)", cap.get_text(strip=True)) if cap else None
            if not m:
                continue
            for tr in t.find_all("tr"):
                th = tr.find("th")
                if th and th.get_text(strip=True).isdigit():
                    add(sorted((int(m.group(1)), int(m.group(2)), int(th.get_text(strip=True)))), tr.find("td"))
    elif bet == "trifecta":
        for li in soup.select("ul.tan3_list > li"):
            heads = [d.get_text(strip=True) for d in li.select("div.p_line div.num")]
            t = li.find("table", class_="tan3")
            if len(heads) != 2 or not all(h.isdigit() for h in heads) or t is None:
                continue
            for tr in t.find_all("tr"):
                th = tr.find("th")
                if th and th.get_text(strip=True).isdigit():
                    add((int(heads[0]), int(heads[1]), int(th.get_text(strip=True))), tr.find("td"))
    elif bet in _PAIR_TABLE:
        for t in soup.find_all("table", class_=_PAIR_TABLE[bet]):
            cap = t.find("caption")
            if not (cap and cap.get_text(strip=True).isdigit()):
                continue
            for tr in t.find_all("tr"):
                th = tr.find("th")
                if th and th.get_text(strip=True).isdigit():
                    nums = (int(cap.get_text(strip=True)), int(th.get_text(strip=True)))
                    add(sorted(nums) if bet == "quinella" else nums, tr.find("td"))
    else:
        raise ValueError(bet)
    if not rows:
        table = {"trio": "fuku3", "trifecta": "tan3"}.get(bet) or _PAIR_TABLE[bet]
        return [], "jra_presale" if soup.find("table", class_=table) else "jra_no_table"
    return rows, status


def fetch_exotic(race_id, date, bet, honor_cooldown=False):
    """1レースの組み合わせ券（全通り、JRA 公式）→ (行, 状態)。リンクを知らなければ、そのレースの単勝・複勝のページから探す"""
    if (race_id, bet) not in _exotic_cnames:
        if race_id not in _cnames:
            discover(date, honor_cooldown)
        if race_id not in _cnames:
            return [], "error: jra_no_link"
        html, status = _post(_cnames[race_id], honor_cooldown)
        if html is None:
            return [], f"error: {status}"
        harvest_exotic(html)
    cname = _exotic_cnames.get((race_id, bet))
    if cname is None:
        return [], "error: jra_no_link"
    html, status = _post(cname, honor_cooldown)
    if html is None:
        return [], f"error: {status}"
    harvest_exotic(html)
    return parse_exotic(html, race_id, bet)


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
    harvest_exotic(html)
    return parse_odds(html, race_id)
