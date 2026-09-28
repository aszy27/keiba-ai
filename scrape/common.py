# scrape/common.py
# スクレイパー共通の設定と通信処理（保存先・User-Agent・セッション・ブロック検知・race_data の場所）。
import glob
import os
import random
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def get_headers(referer="https://race.netkeiba.com/"):
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": referer,
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    }


def is_block_title(soup):
    return bool(soup.title) and any(w in soup.title.get_text() for w in BLOCK_WORDS)


def get_soup(session, url):
    """race.netkeiba / db.netkeiba のページを取得する。
       失敗時は文字列を返す: "NO_DATA"（404）/ "BLOCK"（制限・想定外の応答）/ "NETWORK_ERROR" """
    try:
        res = session.get(url, headers=get_headers(), timeout=20)

        if res.status_code == 404:
            return "NO_DATA"

        if res.status_code != 200:
            return "BLOCK"

        try:
            html = res.content.decode('euc-jp')
        except UnicodeDecodeError:
            try:
                html = res.content.decode('shift_jis')
            except UnicodeDecodeError:
                html = res.content.decode('utf-8', errors='replace')

        soup = BeautifulSoup(html, "html.parser")

        if is_block_title(soup):
            return "BLOCK"

        if not soup.find(id=re.compile("header|container|main")) and not soup.find(
                class_=re.compile("Race|Header|Layout")):
            return "BLOCK"

        return soup
    except Exception:
        return "NETWORK_ERROR"


def race_data_files():
    """{年: race_data_YYYY.csv のパス}（train / val / test のどこにあっても拾う）"""
    files = {}
    for d in DATA_SEARCH_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "race_data_*.csv"))):
            m = re.search(r'race_data_(\d{4})\.csv$', path)
            if m:
                files[int(m.group(1))] = path
    return files
