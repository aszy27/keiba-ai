# live/entries.py
# 発走前のレースの出馬表を netkeiba から取り、学習データと同じ表（races / runners / training / horses）の形にする。
# 出馬表を「スクレイパーが書く CSV と同じ列」の行に直してから prep.ingest の build_races / build_runners に通すので、
# 馬場・クラス・ID の書式などの表記のそろえ方は学習データとまったく同じになる。
# 取れなかった項目（枠順・天候・馬場・馬体重）は既定値で埋めず、空欄のまま警告する（v1 は 1600m・晴・良を黙って入れていた）。
import re
import unicodedata

import numpy as np
import pandas as pd

from prep import ingest
from prep import normalize as nz
from paths import table_path
from scrape.common import create_session, get_soup
from scrape.extras import parse_training
from scrape.pedigree import scrape_pedigree_page
from scrape.repair import parse_race_info
from scrape.results import load_race

SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={}"
GRADE_ICON = {"Icon_GradeType1": "G1", "Icon_GradeType2": "G2", "Icon_GradeType3": "G3"}


def _id(a, kind):
    m = re.search(rf"/{kind}/(?:result/recent/)?(\w+)", a.get("href", "")) if a else None
    return m.group(1) if m else None


def parse_shutuba(soup, race_id):
    """出馬表のページ → (スクレイパーの CSV と同じ列の行の一覧, 警告の一覧)"""
    warns = []
    title = soup.title.get_text(strip=True) if soup.title else ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
    race_date = f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}" if m else None
    name_tag = soup.find(class_="RaceName")
    # レース名はページのタイトルから取る（「オパールステークス(L)」「毎日王冠(G2)」のように格付けが付いていて、学習データの名前の形に近い）
    race_name = title.split(" 出馬表")[0].strip() if " 出馬表" in title else (name_tag.get_text(strip=True) if name_tag else None)
    grade = None
    if name_tag:
        for span in name_tag.find_all("span"):
            grade = next((GRADE_ICON[c] for c in span.get("class", []) if c in GRADE_ICON), grade)
    d1 = soup.find(class_="RaceData01")
    d1 = re.sub(r"\s+", " ", d1.get_text(" ", strip=True)) if d1 else ""
    d2 = soup.find(class_="RaceData02")
    d2 = re.sub(r"\s+", " ", d2.get_text(" ", strip=True)) if d2 else ""
    t = re.search(r"(\d{1,2}:\d{2})発走", d1)
    course = re.search(r"(芝|ダ|障)\D*?(\d{3,4})m", d1)
    handed = re.search(r"\((右|左|直)", d1)
    weather = re.search(r"天候\s*:\s*(\S+)", d1)
    going = re.search(r"馬場\s*:\s*(\S+)", d1)
    prize = re.search(r"本賞金\s*:\s*([\d.]+)", d2)
    # 障害レースはこの欄に「ダ3100m」としか書かれないことがあるので、レース名の「障害」でも判定する
    is_jump = "障" in d1 or "障害" in unicodedata.normalize("NFKC", race_name or "")
    surface = "障害" if is_jump else ({"芝": "芝", "ダ": "ダート"}.get(course[1]) if course else None)
    if not course:
        warns.append("芝ダ・距離が読めない")
    if not weather or not going:
        warns.append("天候・馬場が未発表（当日の朝に出る）")
    going_s = going[1] if going else None
    going_s = {"不": "不良", "稍重": "稍"}.get(going_s, going_s)   # race.netkeiba は不良を「不」1文字で書く
    race = dict(race_id=race_id, race_date=race_date, race_time=t[1] if t else None, race_name=race_name, grade=grade,
                type=surface, length=course[2] if course else None, handed=handed[1] if handed else None,
                weather=weather[1] if weather else None, condition=going_s, max_prize=prize[1] if prize else None)

    rows = []
    for tr in soup.find_all("tr", class_="HorseList"):
        tds = [(td.get("class") or [], td) for td in tr.find_all("td")]
        # 枠順が決まると「Waku1」「Umaban1」のように番号付きの名前になるので前方一致で探す
        cell = lambda name: next((td for cls, td in tds if any(c == name or c.startswith(name) and c[len(name):].isdigit()
                                                                for c in cls)), None)
        if cell("HorseInfo") is None:   # 同じページの別の表（データ分析など）の行は飛ばす
            continue
        horse_a = cell("HorseInfo").find("a") if cell("HorseInfo") else None
        jockey_a = cell("Jockey").find("a") if cell("Jockey") else None
        trainer_a = cell("Trainer").find("a") if cell("Trainer") else None
        barei = cell("Barei")
        sex_age = re.match(r"(牡|牝|セ)(\d+)", barei.get_text(strip=True)) if barei else None
        burden = barei.find_next_sibling("td").get_text(strip=True) if barei and barei.find_next_sibling("td") else None
        w = re.match(r"(\d+)\s*\(([+\-±]?\d+)\)", cell("Weight").get_text(strip=True)) if cell("Weight") else None
        cancelled = "Cancel" in tr.get("class", [])
        rows.append(dict(race, rank=None, bracket=(cell("Waku").get_text(strip=True) if cell("Waku") else "") or None,
                         horse_number=(cell("Umaban").get_text(strip=True) if cell("Umaban") else "") or None,
                         horse_id=_id(horse_a, "horse"), horse_name=horse_a.get_text(strip=True) if horse_a else None,
                         gender=sex_age[1] if sex_age else None, age=sex_age[2] if sex_age else None, burden=burden,
                         jockey_id=_id(jockey_a, "jockey"), jockey_name=jockey_a.get_text(strip=True) if jockey_a else None,
                         trainer_id=_id(trainer_a, "trainer"), trainer_name=trainer_a.get_text(strip=True) if trainer_a else None,
                         weight=w[1] if w else None, weight_diff=w[2] if w else None,
                         rap_time=None, diff_time=None, passage_rank=None, last_3f=None, prize=None,
                         cancelled=cancelled))
    if rows and all(r["horse_number"] is None for r in rows):
        warns.append("枠順が未確定（馬番が無いのでオッズと対応づけられない）")
    elif rows and all(r["weight"] is None for r in rows if not r["cancelled"]):
        warns.append("馬体重が未発表（発走の約70分前に出る。学習時は全馬に実測値がある）")
    return rows, warns


def class_reference(year, jump):
    """クラス名が読めないレース（名前だけの特別戦）用: 前の年のクラス別1着賞金の中央値（prep.ingest と同じ考え方）"""
    r = pd.read_parquet(table_path("races"), columns=["race_id", "surface", "race_class", "class_source", "first_prize"])
    r = r[(r["race_id"].str[:4] == str(year - 1)) & (r["class_source"] == "name") & ((r["surface"] == "障害") == jump)]
    return r.groupby("race_class")["first_prize"].median().to_dict()


def fetch(race_ids, session=None, with_training=True):
    """発走前のレース → (races, runners, training, horses, 警告 {race_id: [...]})。runners の status は entry / scratched"""
    session = session or create_session()
    raw, warns, train_rows = [], {}, []
    for rid in race_ids:
        soup = get_soup(session, SHUTUBA_URL.format(rid))
        if isinstance(soup, str) or soup is None:   # get_soup はブロック・取得失敗を文字列で返す
            warns[rid] = [f"出馬表を取得できない（{soup}）"]
            continue
        rows, w = parse_shutuba(soup, rid)
        warns[rid] = w
        raw += rows
        if with_training:
            t = parse_training(session, rid)
            if isinstance(t, pd.DataFrame):
                train_rows.append(t)
            else:
                warns[rid].append(f"追い切り評価を取得できない（{t}）")
    if not raw:
        return None, None, None, None, warns
    raw = pd.DataFrame(raw)
    notes = []
    races = ingest.build_races(raw, notes)
    # 名前だけの特別戦は賞金からクラスを推定する（学習データと同じ考え方）。build_races は渡したレースの中だけで比べるので、
    # 出馬表の数レースだと比べる相手が足りない（オープン特別が1勝クラスになる）。前の年の全レースと比べ直す
    for i in races.index[races["race_class"].isna() | (races["class_source"] == "prize")]:
        ref = class_reference(races.at[i, "race_date"].year, races.at[i, "surface"] == "障害")
        c = nz.class_from_prize(races.at[i, "first_prize"], ref)
        if c:
            races.at[i, "race_class"], races.at[i, "class_source"] = c, "prize"
    # 取消・除外の馬は特徴量の計算から外れる（学習データでも status = scratched は使わない）ので、ここで外す
    for rid, n in raw[raw["cancelled"]].groupby("race_id").size().items():
        warns[rid].append(f"取消・除外 {n}頭")
    no_id = raw["horse_id"].isna() & ~raw["cancelled"]
    for rid, n in raw[no_id].groupby("race_id").size().items():
        warns[rid].append(f"馬のIDが読めない {n}頭（そのレースは予測しない）")
    raw = raw[~raw["cancelled"] & ~raw["race_id"].isin(raw.loc[no_id, "race_id"])]
    runners = ingest.build_runners(raw, races)
    runners["status"] = pd.array(["entry"] * len(runners), dtype="string")
    g = runners[runners["status"] == "entry"].groupby("race_id")["horse_id"]
    races = races.merge(g.size().rename("n_entries"), on="race_id", how="left")

    training = pd.DataFrame(columns=["race_id", "horse_id", "oikiri_rank"])
    if train_rows:
        t = pd.concat(train_rows, ignore_index=True)
        t = t[t["horse_id"] != "NO_TRAIN"]
        grade = t["oikiri_rank"].astype(str).str.upper().str.extract(r"([SABCD])", expand=False)   # prep.ingest.build_training と同じ
        training = pd.DataFrame({"race_id": t["race_id"].astype("string"), "horse_id": t["horse_id"].astype("string"),
                                 "oikiri_rank": grade.astype("string")})

    # 血統マスタに無い馬（新馬など）は血統ページから父・母を取る（学習データと同じスクレイパー）
    known = set(pd.read_parquet(table_path("horses"), columns=["horse_id"])["horse_id"])
    missing = [h for h in runners["horse_id"].dropna().unique() if h not in known]
    peds = [p for p in (scrape_pedigree_page(h, session) for h in missing) if isinstance(p, dict)]
    horses = pd.DataFrame(peds).reindex(columns=["horse_id", "horse_name", "sire_id", "sire_name", "dam_id", "dam_name",
                                                 "breeder", "owner"]).astype("string") if peds else None
    if missing:
        for rid in runners.loc[runners["horse_id"].isin(missing), "race_id"].unique():
            warns[rid].append(f"血統マスタに無い馬 {int(runners.loc[runners['race_id'] == rid, 'horse_id'].isin(missing).sum())}頭"
                              f"（血統ページから取得: {len(peds)}/{len(missing)}頭）")
    return races, runners, training, horses, warns


_FINISHED_CACHE = {}   # race_id → その日の結果（--watch で何度も呼ばれるので、取得済みのレースは取り直さない）


def fetch_finished(race_ids, session=None):
    """今日すでに終わったレースの結果（学習データと同じスクレイパー scrape.results で db.netkeiba から取る）→ (races, runners)。
    特徴量の「当日バイアス」（同じ日・同じ場・同じ芝ダで先に終わったレースの上位馬の枠・位置取り）は、これが無いと作れない"""
    session = session or create_session()
    raw = []
    for rid in race_ids:
        if rid in _FINISHED_CACHE:
            raw.append(_FINISHED_CACHE[rid])
            continue
        df = load_race(session, rid)
        if df is None or not len(df):
            continue
        # db.netkeiba の結果ページからは芝ダ・距離・天候・馬場が取れず空欄になる（既知）。毎週の取り込みでは
        # python -m scrape repair が race.netkeiba から埋めているので、ここでも同じ関数で空欄だけ埋める
        info = parse_race_info(rid)
        for key in ("type", "length", "handed", "weather", "condition", "race_time"):
            if key in info:
                blank = df[key].isna() | (df[key].astype(str).str.strip().isin(["", "None", "nan"])) if key in df else True
                df[key] = df[key].where(~blank, info[key]) if key in df else info[key]
        _FINISHED_CACHE[rid] = df
        raw.append(df)
    if not raw:
        return None, None, []
    raw = pd.concat(raw, ignore_index=True)
    races = ingest.build_races(raw, [])
    runners = ingest.build_runners(raw, races)
    return races, runners, sorted(set(race_ids) - set(races["race_id"]))
