# v2/nar/fetch_results.py
# 地方競馬（netkeiba 地方版）の結果ページを日付の範囲で取得し、HTML をそのまま data/nar/raw/<race_id>.html.gz に保存する。
# 結果ページ1枚に 着順・馬番・騎手・タイム・人気・確定単勝オッズ・全券種の払戻 がそろっているため、1レース1リクエスト。
# 保存済みのレースは飛ばすので、止まっても再実行で続きから取れる。ばんえい（帯広）は除く。
# 使い方: python -m v2.nar.fetch_results --start 2026-06-01 --end 2026-08-31
import argparse
import gzip
import random
import re
import sys
import time
from datetime import date, timedelta

import requests

from v2.paths import LEGACY_DIR

RAW_DIR = LEGACY_DIR / "nar" / "raw"
LIST_URL = "https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date={}"
RESULT_URL = "https://nar.netkeiba.com/race/result.html?race_id={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36",
           "Referer": "https://nar.netkeiba.com/", "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
BANEI_PLACE = "65"   # 帯広（ばんえい）
BLOCK_WORDS = ("アクセス", "お手数ですが", "Cloudflare", "制限", "大変混み合って")


def decode(content):
    m = re.search(rb'charset=["\']?([\w-]+)', content[:3000])
    return content.decode(m.group(1).decode() if m else "utf-8", errors="replace")


def fetch(session, url):
    """(bytes, None) か (None, 理由)。ブロックの疑いは理由が 'BLOCK' で始まる"""
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        return None, f"通信エラー: {e}"
    if r.status_code != 200:
        return None, f"BLOCK: HTTP {r.status_code}"
    title = re.search(r"<title>(.*?)</title>", decode(r.content), re.S)
    if title and any(w in title.group(1) for w in BLOCK_WORDS):
        return None, f"BLOCK: {title.group(1).strip()[:40]}"
    return r.content, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    day, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    n_req = n_saved = 0
    while day <= end:
        content, err = fetch(session, LIST_URL.format(day.strftime("%Y%m%d")))
        if err:
            print(f"{day} レース一覧の取得に失敗: {err}")
            if err.startswith("BLOCK"):
                sys.exit(1)
            day += timedelta(days=1)
            continue
        ids = sorted({rid for rid in re.findall(r"race_id=(\d{12})", decode(content)) if rid[4:6] != BANEI_PLACE})
        todo = [rid for rid in ids if not (RAW_DIR / f"{rid}.html.gz").exists()]
        failed = []
        for rid in todo:
            time.sleep(random.uniform(1.0, 2.0))
            content, err = fetch(session, RESULT_URL.format(rid))
            n_req += 1
            if n_req % 100 == 0:
                time.sleep(10)
            if err:
                if err.startswith("BLOCK"):
                    print(f"\n{rid}: {err} → 停止します。時間をおいて再実行すれば続きから取得します。")
                    sys.exit(1)
                failed.append(rid)
                continue
            if b"RaceTable01" not in content:     # 結果が未確定・中止のレース
                failed.append(rid)
                continue
            (RAW_DIR / f"{rid}.html.gz").write_bytes(gzip.compress(content))
            n_saved += 1
        print(f"{day}: {len(ids)}R（取得済み {len(ids) - len(todo)} / 今回保存 {len(todo) - len(failed)} / 結果なし {len(failed)}）累計保存 {n_saved}",
              flush=True)
        day += timedelta(days=1)
        time.sleep(random.uniform(1.0, 2.0))
    print("完了:", RAW_DIR)


if __name__ == "__main__":
    main()
