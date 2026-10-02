# scrape/common.py
# スクレイパー共通の設定と通信処理（保存先・通信の窓口 http_get・ブロックの検知と冷却期間・race_data の場所）。
import glob
import json
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_SEARCH_DIRS = [str(DATA_DIR / "train"), str(DATA_DIR / "val"), str(DATA_DIR / "test")]

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]
# ページのタイトルにこれが含まれていたらアクセス制限の画面とみなす
BLOCK_WORDS = ["アクセス", "お手数ですが", "Error", "Cloudflare", "Block", "制限", "大変混み合って"]
# netkeiba のオッズAPI。type: 1=単勝・複勝 / 4=馬連 / 5=ワイド / 7=3連複 / 8=3連単。
# action は init（確定オッズの一括取得で使ってきた値）と update（スナップショット・組み合わせ券）の2種類
ODDS_API = ("https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&input=UTF-8&output=json"
            "&race_id={race_id}&type={bet_type}&action={action}&sort=odds&compress=0")


# ---------- 通信の窓口（サーバーに負担をかけない取り方と、ブロックされかけたら止まる仕組み） ----------
# すべてのスクレイパーはここを通す。入れているもの:
#   1. 間隔の制御: 同じホストへのリクエストは、どの処理からでも MIN_INTERVAL〜2倍 の間隔をあける（同じプロセス内）
#   2. 429（混雑）・5xx は Retry-After（無ければ倍々に延ばした待ち時間）に従って再試行する
#   3. ブロックの兆候（403・API の 400・制限画面）を数え、BLOCK_WINDOW 秒以内に BLOCK_LIMIT 回あれば
#      COOLDOWN_HOURS 時間の冷却期間に入る。冷却期間はファイルに残すので、別の処理・翌日の実行もその間は取りに行かない
#   4. User-Agent は1つに固定する（毎回変えると、かえって機械的なアクセスに見える）
# ブロックされた後に IP や User-Agent を変えて取り続けるような回避はしない（相手の制限をすり抜けることになるため）。
USER_AGENT = USER_AGENTS[0]
MIN_INTERVAL = 1.0          # 同じホストへの最短間隔（秒）。実際は 1〜2倍のランダムな間隔になる
BLOCK_LIMIT = 3             # BLOCK_WINDOW 秒以内にこの回数のブロックの兆候 → 冷却期間
BLOCK_WINDOW = 15 * 60
COOLDOWN_HOURS = 6
COOLDOWN_FILE = DATA_DIR / "scrape_cooldown.json"

_session = None
_last_request = {}          # ホスト → 最後にリクエストした時刻（time.monotonic）
_block_times = deque()
_sleep = time.sleep         # テストで置き換えられるように
_now = time.monotonic


class Blocked(Exception):
    """アクセス制限の兆候が続いた（冷却期間に入った）"""


def shared_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"})
    return _session


def cooldown_until():
    """冷却期間の終わり（無ければ None）"""
    try:
        until = datetime.fromisoformat(json.loads(COOLDOWN_FILE.read_text(encoding="utf-8"))["until"])
    except (OSError, ValueError, KeyError):
        return None
    return until if until > datetime.now() else None


def start_cooldown(reason, hours=COOLDOWN_HOURS):
    until = datetime.now() + timedelta(hours=hours)
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOLDOWN_FILE.write_text(json.dumps({"until": until.isoformat(timespec="seconds"), "reason": reason,
                                         "started": datetime.now().isoformat(timespec="seconds")},
                                        ensure_ascii=False), encoding="utf-8")
    print(f"\n🛑 アクセス制限の兆候が続いたので、{until:%m/%d %H:%M} まで取得を止めます（{reason}）", flush=True)


def _block_signal(reason):
    """ブロックの兆候を1回数える。続いていれば冷却期間に入る"""
    t = _now()
    _block_times.append(t)
    while _block_times and t - _block_times[0] > BLOCK_WINDOW:
        _block_times.popleft()
    if len(_block_times) >= BLOCK_LIMIT and cooldown_until() is None:
        start_cooldown(reason)


def _pace(host):
    last = _last_request.get(host)
    if last is not None:
        wait = MIN_INTERVAL * random.uniform(1.0, 2.0) - (_now() - last)
        if wait > 0:
            _sleep(wait)
    _last_request[host] = _now()


def http_get(url, referer=None, timeout=20, tries=3, honor_cooldown=True):
    """(Response または None, 状態)。状態: ok / not_found / block / cooldown / network_error / http_NNN。
    honor_cooldown=False（発走前オッズのスナップショット用）は冷却期間中でも取りに行く（ブロックの兆候は数える）"""
    if honor_cooldown and cooldown_until() is not None:
        return None, "cooldown"
    host = urlparse(url).netloc
    status = "network_error"
    for attempt in range(tries):
        _pace(host)
        try:
            r = shared_session().get(url, headers={"Referer": referer or f"https://{host}/"}, timeout=timeout)
        except requests.RequestException as e:
            status = "network_error"
            print(f"   ⚠️ 通信エラー（{type(e).__name__}）。{5 * 3 ** attempt}秒待って再試行", flush=True)
            _sleep(5 * 3 ** attempt + random.uniform(0, 2))
            continue
        code = r.status_code
        if code == 200:
            return r, "ok"
        if code == 404:
            return None, "not_found"
        if code in (429, 503):
            retry_after = r.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else 60 * 2 ** attempt
            _block_signal(f"HTTP {code}")
            if honor_cooldown and cooldown_until() is not None:
                return None, "cooldown"
            print(f"   ⏳ HTTP {code}（混雑）。{wait:.0f}秒待って再試行", flush=True)
            _sleep(wait)
            status = "block"
            continue
        if code in (400, 403):
            _block_signal(f"HTTP {code}")
            return None, "block"
        if code >= 500:
            status = f"http_{code}"
            _sleep(10 * 2 ** attempt + random.uniform(0, 3))
            continue
        return None, f"http_{code}"
    return None, status


def create_session():
    """互換のため残している。すべての通信は共通のセッション（shared_session）と窓口（http_get）を通る"""
    return shared_session()


def get_headers(referer="https://race.netkeiba.com/"):
    return {"User-Agent": USER_AGENT, "Referer": referer, "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}


def is_block_title(soup):
    return bool(soup.title) and any(w in soup.title.get_text() for w in BLOCK_WORDS)


def get_soup(session, url):
    """race.netkeiba / db.netkeiba のページを取得する（session は互換のための引数で使わない）。
       失敗時は文字列を返す: "NO_DATA"（404）/ "BLOCK"（制限・冷却期間中・想定外の応答）/ "NETWORK_ERROR" """
    r, status = http_get(url)
    if status == "not_found":
        return "NO_DATA"
    if r is None:
        return "NETWORK_ERROR" if status == "network_error" else "BLOCK"
    try:
        html = r.content.decode('euc-jp')
    except UnicodeDecodeError:
        try:
            html = r.content.decode('shift_jis')
        except UnicodeDecodeError:
            html = r.content.decode('utf-8', errors='replace')
    soup = BeautifulSoup(html, "html.parser")
    if is_block_title(soup):
        _block_signal("制限画面")
        return "BLOCK"
    if not soup.find(id=re.compile("header|container|main")) and not soup.find(class_=re.compile("Race|Header|Layout")):
        _block_signal("想定外のページ")
        return "BLOCK"
    return soup


def race_data_files():
    """{年: race_data_YYYY.csv のパス}（train / val / test のどこにあっても拾う）"""
    files = {}
    for d in DATA_SEARCH_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "race_data_*.csv"))):
            m = re.search(r'race_data_(\d{4})\.csv$', path)
            if m:
                files[int(m.group(1))] = path
    return files
