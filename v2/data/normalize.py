# v2/data/normalize.py
# 旧データ（keibascraper 由来の CSV）の表記ゆれを v2 の値にそろえる関数群。副作用なし。
import re
import unicodedata

import numpy as np
import pandas as pd

PLACE_CODES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
               "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
SURFACES = ("芝", "ダート", "障害")
GOINGS = ("良", "稍", "重", "不良")
WEATHERS = ("晴", "曇", "小雨", "雨", "小雪", "雪")
CLASSES = ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "G3", "G2", "G1")

# EUC-JP のバイト列を1バイト文字コードで読んだ文字化け（例: '3КаЬЄОЁЭј'）を戻すための候補
_MOJIBAKE_ENCODINGS = ("cp1251", "iso8859_5", "koi8_r", "mac_cyrillic", "cp1253", "iso8859_7", "mac_greek",
                       "cp1257", "iso8859_13", "cp1250", "cp1252", "latin-1", "cp1254")
_MOJIBAKE_CHARS = re.compile(r"[ -ɏͰ-ӿ–-≥]")
# 戻した結果が正しいかの判定に使う、レース名によく出る語
_NAME_WORDS = ("歳", "勝", "新馬", "未勝利", "クラス", "以上", "万下", "オープン", "障害", "ステークス", "特別",
               "賞", "杯", "記念", "カップ", "ジャンプ", "第", "回")


def fix_mojibake(name):
    """レース名の文字化けを戻す。戻り値は (名前, 'ok' | 'fixed' | 'broken')"""
    if not isinstance(name, str):
        return name, "ok"
    if not _MOJIBAKE_CHARS.search(name):
        return name, "ok"
    best, best_score = None, 0
    for enc in _MOJIBAKE_ENCODINGS:
        try:
            cand = name.encode(enc).decode("euc-jp")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if _MOJIBAKE_CHARS.search(cand):
            continue
        score = sum(w in cand for w in _NAME_WORDS)
        if score > best_score:
            best, best_score = cand, score
    return (best, "fixed") if best else (name, "broken")


_GRADE_RE = re.compile(r"[(（]\s*J?[・.]?\s*(?:G|JPN)\s*(III|II|I|3|2|1)\s*[)）]")
_GRADE_MAP = {"III": "G3", "3": "G3", "II": "G2", "2": "G2", "I": "G1", "1": "G1"}


def parse_grade(name, grade_col=None):
    """重賞・リステッドの格付け（G1/G2/G3/L）。平場・特別は None"""
    n = unicodedata.normalize("NFKC", name).upper() if isinstance(name, str) else ""
    m = _GRADE_RE.search(n)
    if m:
        return _GRADE_MAP[m.group(1)]
    if re.search(r"[(（]\s*L\s*[)）]", n):
        return "L"
    if "(重賞)" in n:  # 格付け前の新設重賞
        return "G3"
    if isinstance(grade_col, str) and grade_col in ("G1", "G2", "G3", "L"):
        return grade_col
    return None


def parse_class(name, grade=None):
    """レース名から競走条件のクラスを返す。読み取れなければ None（賞金から推定する）"""
    if isinstance(grade, str) and grade:
        return grade
    n = unicodedata.normalize("NFKC", name) if isinstance(name, str) else ""
    if "新馬" in n:
        return "新馬"
    if "未勝利" in n:
        return "未勝利"
    if re.search(r"1勝|(?<!\d)500万", n):
        return "1勝"
    if re.search(r"2勝|1000万", n):
        return "2勝"
    if re.search(r"3勝|1600万", n):
        return "3勝"
    if re.search(r"オープン|OP", n):
        return "OP"
    return None


def class_from_prize(prize, reference):
    """1着賞金に最も近いクラスを返す。reference は {クラス: その年の1着賞金の中央値}"""
    if not reference or not (prize > 0):
        return None
    return min(reference, key=lambda c: abs(np.log(prize) - np.log(reference[c])))


def parse_age_condition(name):
    n = unicodedata.normalize("NFKC", name) if isinstance(name, str) else ""
    m = re.search(r"(2歳|3歳以上|4歳以上|3歳)", n)
    return m.group(1) if m else None


def pad_id(s):
    """騎手・調教師IDを5桁にそろえる（2014〜2025年のデータは先頭の0が落ちている）"""
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    digits = s.str.fullmatch(r"\d+").fillna(False)
    return s.where(~digits, s.str.zfill(5))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_going(s):
    s = s.astype("string").str.strip().replace({"稍重": "稍"})
    return s.where(s.isin(GOINGS))


def corner_positions(passage):
    """'5-5-3-3' → (最初のコーナーの順位, 最後のコーナーの順位)"""
    parts = passage.astype("string").str.split("-")
    first = pd.to_numeric(parts.str[0], errors="coerce")
    last = pd.to_numeric(parts.str[-1], errors="coerce")
    return first, last


def runner_status(finish_pos, passage):
    """着順あり=finished / 着順なしで通過順あり=dnf（競走中止等）/ どちらもなし=scratched（取消・除外）"""
    return pd.Series(np.select([finish_pos.notna(), passage.notna()], ["finished", "dnf"], "scratched"),
                     index=finish_pos.index)


def parse_payouts(s):
    """'170|260|800' → [170, 260, 800]。空欄は None"""
    def one(v):
        if not isinstance(v, str) or not v.strip():
            return None
        vals = [int(x) for x in v.split("|") if x.strip().isdigit()]
        return vals or None
    return s.map(one)
