# v2/experiments/nar/parse_results.py
# data/nar/raw/*.html.gz（fetch_results.py で保存した結果ページ）をパースして、
# data/nar/{races,runners,payouts}.parquet に保存する。
# 使い方: python -m v2.experiments.nar.parse_results
import gzip
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from v2.experiments.nar.fetch_results import RAW_DIR, decode
from v2.normalize import normalize_going

OUT_DIR = RAW_DIR.parent
BET_NAMES = {"単勝": "win", "複勝": "place", "枠連": "bracket_quinella", "馬連": "quinella", "ワイド": "wide",
             "枠単": "bracket_exacta", "馬単": "exacta", "3連複": "trio", "3連単": "trifecta"}
N_HORSES_IN_COMBO = {"win": 1, "place": 1, "bracket_quinella": 2, "quinella": 2, "wide": 2, "bracket_exacta": 2,
                     "exacta": 2, "trio": 3, "trifecta": 3}


def _num(text):
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else np.nan


def _seconds(text):
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text.strip())
    return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else np.nan


def _link_id(cell, kind):
    a = cell.find("a", href=re.compile(rf"/{kind}/"))
    m = re.search(rf"/{kind}/(?:result/recent/)?([0-9a-zA-Z]+)", a["href"]) if a else None
    return m.group(1) if m else None


def parse(race_id, html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    m_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\S+?)(\d{1,2})R", title)
    d1 = soup.find("div", class_="RaceData01")
    d2 = soup.find("div", class_="RaceData02")
    info = d1.get_text(" ", strip=True).replace(" ", "") if d1 else ""
    info2 = d2.get_text(" ", strip=True) if d2 else ""
    m_len = re.search(r"(ダ|芝|障)(\d{3,4})m", info)
    m_going = re.search(r"馬場:(\S)", info)
    going_map = {"良": "良", "稍": "稍", "重": "重", "不": "不良"}
    race = {
        "race_id": race_id,
        "race_date": pd.Timestamp(f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}") if m_date else pd.NaT,
        "place": m_date.group(4) if m_date else None,
        "race_number": int(race_id[10:12]),
        "race_name": title.split(" 結果")[0] if title else None,
        "post_time": (re.search(r"(\d{1,2}:\d{2})発走", info) or [None, None])[1],
        "surface": {"ダ": "ダート", "芝": "芝", "障": "障害"}.get(m_len.group(1)) if m_len else None,
        "distance": float(m_len.group(2)) if m_len else np.nan,
        "direction": (re.search(r"\((左|右|直)", info) or [None, None])[1],
        "weather": (re.search(r"天候:([^/]+)", info) or [None, None])[1],
        "going": going_map.get(m_going.group(1)) if m_going else None,
        "condition_text": info2,
        "first_prize": _num(info2.split("本賞金:")[1]) if "本賞金:" in info2 else np.nan,
    }

    table = soup.find("table", class_=re.compile("RaceTable01"))
    rows = table.find_all("tr")
    header = [re.sub(r"\s+", "", c.get_text()) for c in rows[0].find_all(["th", "td"])]
    col = {name: header.index(name) for name in header}
    runners = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < len(header):
            continue
        cell = lambda name: tds[col[name]] if name in col else None
        text = lambda name: cell(name).get_text(" ", strip=True) if cell(name) is not None else ""
        sexage = text("性齢")
        weight = re.match(r"(\d+)\s*\(([+\-]?\d+)\)", text("馬体重(増減)"))
        finish = text("着順")
        runners.append({
            "race_id": race_id,
            "finish_text": finish,
            "finish_pos": float(finish) if finish.isdigit() else np.nan,
            "bracket": _num(text("枠")),
            "horse_number": _num(text("馬番")),
            "horse_id": _link_id(cell("馬名"), "horse"),
            "horse_name": text("馬名"),
            "sex": sexage[:1] or None,
            "age": _num(sexage[1:]) if len(sexage) > 1 else np.nan,
            "burden": _num(text("斤量")),
            "jockey_id": _link_id(cell("騎手"), "jockey"),
            "jockey_name": re.sub(r"^[▲△☆◇★]+", "", text("騎手")),
            "time_sec": _seconds(text("タイム")),
            "margin_text": text("着差"),
            "popularity": _num(text("人気")),
            "win_odds": _num(text("単勝オッズ")),
            "last_3f": _num(text("後3F")),
            "trainer_id": _link_id(cell("厩舎"), "trainer"),
            "trainer_name": text("厩舎"),
            "weight": float(weight.group(1)) if weight else np.nan,
            "weight_diff": float(weight.group(2)) if weight else np.nan,
        })

    payouts = []
    for pt in soup.find_all("table", class_=re.compile("Payout")):
        for tr in pt.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 3 or cells[0].get_text(strip=True) not in BET_NAMES:
                continue
            bet = BET_NAMES[cells[0].get_text(strip=True)]
            nums = re.findall(r"\d+", cells[1].get_text(" ", strip=True))
            yen = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+(?=円)", cells[2].get_text(" ", strip=True))]
            k = N_HORSES_IN_COMBO[bet]
            if len(nums) != k * len(yen):
                continue   # 組み合わせと払戻の数が合わない行（表記の想定外）は捨てる
            for i, y in enumerate(yen):
                payouts.append({"race_id": race_id, "bet_type": bet, "combination": "-".join(nums[i * k:(i + 1) * k]), "payout": y})
    return race, runners, payouts


def main():
    races, runners, payouts, failed = [], [], [], []
    files = sorted(RAW_DIR.glob("*.html.gz"))
    for i, f in enumerate(files, 1):
        rid = f.name.split(".")[0]
        try:
            r, ru, pa = parse(rid, decode(gzip.decompress(f.read_bytes())))
        except Exception as e:  # 1ページの想定外で全体を止めない。件数は最後に出す
            failed.append((rid, repr(e)[:80]))
            continue
        races.append(r)
        runners += ru
        payouts += pa
        if i % 1000 == 0:
            print(f"{i}/{len(files)}", flush=True)
    races, runners, payouts = pd.DataFrame(races), pd.DataFrame(runners), pd.DataFrame(payouts)
    races["going"] = normalize_going(races["going"])
    runners["status"] = np.select([runners["finish_pos"].notna(), runners["finish_text"].str.contains("中止|失格", na=False)],
                                  ["finished", "dnf"], "scratched")
    for name, df in [("races", races), ("runners", runners), ("payouts", payouts)]:
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name:<8} {len(df):>7}行")
    print(f"パース失敗 {len(failed)}件", failed[:5])


if __name__ == "__main__":
    main()
