# v2/checks.py
# v2 の表の検査。ingest.py が保存前に実行し、ERROR が1件でもあれば保存しない。
# 件数は特に断りがなければレース数。例にはレースIDなどを最大8件載せる。
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from v2.normalize import CLASSES

LEVELS = ("ERROR", "WARN", "INFO")


@dataclass
class Issue:
    level: str
    name: str
    count: int
    examples: list = field(default_factory=list)
    note: str = ""


def _add(issues, level, name, values, note=""):
    values = list(dict.fromkeys(str(v) for v in values))
    if values:
        issues.append(Issue(level, name, len(values), values[:8], note))


def _rate(issues, name, bad, n_checked, error_rate, note=""):
    """同着など正当な例外があり得る検査。割合が error_rate を超えたら ERROR、それ以下なら WARN"""
    bad = list(dict.fromkeys(str(v) for v in bad))
    if not bad:
        return
    rate = len(bad) / max(n_checked, 1)
    issues.append(Issue("ERROR" if rate > error_rate else "WARN", name, len(bad), bad[:8],
                        f"{note}（{n_checked}件中 {rate:.2%}、ERROR の基準 {error_rate:.1%}）"))


def _first(values):
    """払戻リストが1つだけならその値、同着などで複数あれば NaN"""
    def one(v):
        return v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else np.nan
    return values.map(one).astype(float)


def check_races(races, runners):
    out = []
    rid = races["race_id"]
    _add(out, "ERROR", "races: race_id の重複", rid[rid.duplicated()])
    _add(out, "ERROR", "races: race_id が12桁の数字でない", rid[~rid.str.fullmatch(r"\d{12}").fillna(False)])
    _add(out, "ERROR", "races: 日付の欠損", rid[races["race_date"].isna()])
    year_ng = races["race_date"].notna() & (races["race_date"].dt.year.astype("Int64").astype("string") != rid.str[:4])
    _add(out, "ERROR", "races: 日付の年と race_id の年が違う", rid[year_ng])
    _add(out, "ERROR", "races: 競馬場コードが不正", rid[races["place"].isna()])
    for col, label in [("surface", "芝ダ"), ("distance", "距離"), ("going", "馬場状態"), ("weather", "天候")]:
        _add(out, "WARN", f"races: {label}の欠損", rid[races[col].isna()], "patches/race_info.csv で補完する")
    _add(out, "WARN", "races: 距離が範囲外(800〜4300m)", rid[races["distance"].notna() & ~races["distance"].between(800, 4300)])
    _add(out, "INFO", "races: 文字化けを戻したレース名", rid[races["name_status"] == "fixed"])
    _add(out, "WARN", "races: 文字化けを戻せなかったレース名", rid[races["name_status"] == "broken"], "クラスは1着賞金から推定")
    _add(out, "INFO", "races: クラスを1着賞金から推定", rid[races["class_source"] == "prize"])
    _add(out, "ERROR", "races: クラス不明", rid[races["race_class"].isna()])
    _add(out, "ERROR", "races: クラスが想定外の値", rid[races["race_class"].notna() & ~races["race_class"].isin(CLASSES)])

    n = races.groupby(["race_date", "place"]).size()
    small = n[n < 11]
    _add(out, "WARN", "races: 1開催日のレース数が11未満（開催日数）",
         [f"{d:%Y-%m-%d} {p} {c}R" for (d, p), c in small.items()], "取得漏れ・偽レース削除の跡。patches/race_rows で補う")
    _add(out, "ERROR", "races: 出走馬のいないレース", rid[~rid.isin(runners["race_id"])])
    return out


def check_runners(runners, races):
    out = []
    r = runners
    _add(out, "ERROR", "runners: (race_id, horse_id) の重複", r.loc[r.duplicated(["race_id", "horse_id"]), "race_id"])
    _add(out, "ERROR", "runners: (race_id, 馬番) の重複", r.loc[r.duplicated(["race_id", "horse_number"]), "race_id"])
    _add(out, "ERROR", "runners: races に無いレース", r.loc[~r["race_id"].isin(races["race_id"]), "race_id"])
    _add(out, "ERROR", "runners: 馬番が1〜18でない", r.loc[~r["horse_number"].between(1, 18).fillna(False), "race_id"])
    _add(out, "ERROR", "runners: 枠番が1〜8でない", r.loc[~r["bracket"].between(1, 8).fillna(False), "race_id"])
    _add(out, "ERROR", "runners: 馬IDが10桁でない", r.loc[~r["horse_id"].str.fullmatch(r"[0-9a-zA-Z]{10}").fillna(False), "race_id"])
    for col in ["jockey_id", "trainer_id"]:
        ng = r[col].notna() & ~r[col].str.fullmatch(r"[0-9a-zA-Z]{5}").fillna(False)
        _add(out, "ERROR", f"runners: {col} が5桁でない", r.loc[ng, "race_id"])

    d = r.merge(races[["race_id", "race_date", "surface", "distance"]], on="race_id", how="left")
    _add(out, "ERROR", "runners: 同じ馬が同じ日に2回出走", d.loc[d.duplicated(["race_date", "horse_id"], keep=False), "race_id"],
         "偽レース（同じ日の別レースのコピー）の疑い")
    sets = d.groupby("race_id").agg(date=("race_date", "first"), horses=("horse_id", frozenset))
    _add(out, "ERROR", "runners: 同じ日に出走馬の組み合わせが同一のレース", sets.index[sets.duplicated(["date", "horses"], keep=False)])

    fin = d[d["status"] == "finished"]
    top = fin.groupby("race_id")["finish_pos"].min()
    _add(out, "ERROR", "runners: 完走馬がいるのに1着がいないレース", top.index[top != 1])
    n_start = d[d["status"] != "scratched"].groupby("race_id").size().rename("n_start")
    over = fin.join(n_start, on="race_id")
    _add(out, "ERROR", "runners: 着順が出走頭数を超える", over.loc[over["finish_pos"] > over["n_start"], "race_id"])

    _add(out, "WARN", "runners: 完走馬の馬体重の欠損", fin.loc[fin["weight"].isna(), "race_id"])
    _add(out, "WARN", "runners: 完走馬の走破タイムの欠損", fin.loc[fin["time_sec"].isna(), "race_id"])
    flat = fin[fin["surface"].isin(["芝", "ダート"])]
    # 2026-09-15 に確認: 範囲外はほぼ大差の最下位など実在の記録。特徴量側で値を丸める
    _add(out, "INFO", "runners: 上がり3Fが範囲外(平地30〜60秒)",
         flat.loc[flat["last_3f"].notna() & ~flat["last_3f"].between(30, 60), "race_id"], "大差の入線など。特徴量で丸める")
    per100 = flat["time_sec"] / flat["distance"] * 100
    _add(out, "INFO", "runners: 100mあたりの走破タイムが範囲外(平地5.0〜9.0秒)",
         flat.loc[per100.notna() & ~per100.between(5.0, 9.0), "race_id"], "大差の入線など。特徴量で丸める")
    _add(out, "WARN", "runners: 斤量が範囲外(45〜66kg)", r.loc[r["burden"].notna() & ~r["burden"].between(45, 66), "race_id"])
    _add(out, "WARN", "runners: 馬体重が範囲外(300〜650kg)", r.loc[r["weight"].notna() & ~r["weight"].between(300, 650), "race_id"])
    _add(out, "WARN", "runners: 出走頭数が5頭未満", n_start.index[n_start < 5])
    return out


def check_payouts(payouts, races):
    out = []
    p = payouts
    _add(out, "ERROR", "payouts: race_id の重複", p.loc[p["race_id"].duplicated(), "race_id"])
    _add(out, "ERROR", "payouts: races に無いレース", p.loc[~p["race_id"].isin(races["race_id"]), "race_id"])

    trio, trifecta = _first(p["trio"]), _first(p["trifecta"])
    ok = trio.notna() & trifecta.notna()
    _rate(out, "payouts: 3連複 > 3連単", p.loc[ok & (trio > trifecta), "race_id"], int(ok.sum()), 0.005, "列ずれの疑い")
    quinella, exacta = _first(p["quinella"]), _first(p["exacta"])
    ok = quinella.notna() & exacta.notna()
    _rate(out, "payouts: 馬連 > 馬単", p.loc[ok & (quinella > exacta), "race_id"], int(ok.sum()), 0.02, "列ずれの疑い")
    n_place = p["place"].map(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0)
    _add(out, "WARN", "payouts: 複勝の払戻の数が2〜5でない", p.loc[~n_place.between(2, 5), "race_id"])

    held = races[races["n_finishers"] > 0]
    miss = held.loc[~held["race_id"].isin(p["race_id"]), ["race_id", "race_date"]]
    _add(out, "WARN", "payouts: 2024年以降で払戻の無いレース", miss.loc[miss["race_date"] >= "2024-01-01", "race_id"])
    old = miss[miss["race_date"] < "2024-01-01"]
    if len(old):
        by_year = old["race_id"].str[:4].value_counts().sort_index()
        out.append(Issue("INFO", "payouts: 2023年以前で払戻の無いレース", len(old), [],
                         "年別 " + ", ".join(f"{y}:{c}" for y, c in by_year.items())))
    return out


def check_odds(odds, runners, payouts, races):
    out = []
    o = odds
    _add(out, "ERROR", "odds: (race_id, 馬番) の重複", o.loc[o.duplicated(["race_id", "horse_number"]), "race_id"])
    m = o.merge(runners[["race_id", "horse_number", "status", "finish_pos"]], on=["race_id", "horse_number"],
                how="left", indicator=True)
    _rate(out, "odds: 出走馬に無い馬番", m.loc[m["_merge"] == "left_only", "race_id"], o["race_id"].nunique(), 0.005,
          "旧データに取消馬の行が無いレースがある")
    _add(out, "ERROR", "odds: 取消・除外なのに完走している", m.loc[(m["odds_status"] != "ok") & (m["status"] == "finished"), "race_id"])

    fin = runners[(runners["status"] == "finished") & runners["race_id"].isin(o["race_id"])]
    f = fin.merge(o[["race_id", "horse_number", "win_odds"]], on=["race_id", "horse_number"], how="left")
    _add(out, "WARN", "odds: 完走馬のオッズの欠損", f.loc[f["win_odds"].isna(), "race_id"])

    winners = f[f["finish_pos"] == 1]
    winners = winners[~winners["race_id"].duplicated(keep=False)]  # 1着同着のレースは除く
    w = winners.merge(payouts[["race_id", "win"]], on="race_id", how="inner")
    w["win_pay"] = _first(w["win"])
    ok = w["win_odds"].notna() & w["win_pay"].notna()
    ng = ok & ((w["win_odds"] * 100).round() != w["win_pay"])
    _rate(out, "odds: 1着馬のオッズ×100 と単勝払戻が違う", w.loc[ng, "race_id"], int(ok.sum()), 0.01,
          "オッズと馬番の対応ずれの疑い")

    held = races[(races["n_finishers"] > 0) & (races["race_date"] >= "2021-01-01")]
    _add(out, "WARN", "odds: 2021年以降でオッズの無いレース", held.loc[~held["race_id"].isin(o["race_id"]), "race_id"])
    return out


def check_others(laps, training, horses, courses, races, runners):
    out = []
    held = races.loc[races["n_finishers"] > 0, "race_id"]
    has_lap = laps.loc[laps["lap_times"].notna(), "race_id"]
    _add(out, "INFO", "laps: ラップの無いレース", held[~held.isin(has_lap)])

    _add(out, "ERROR", "training: (race_id, horse_id) の重複", training.loc[training.duplicated(["race_id", "horse_id"]), "race_id"])
    _add(out, "INFO", "training: 追い切り評価の無いレース", held[~held.isin(training["race_id"])])
    t = training.merge(runners[["race_id", "horse_id"]], on=["race_id", "horse_id"], how="left", indicator=True)
    _add(out, "WARN", "training: 出走馬に無い馬", t.loc[t["_merge"] == "left_only", "race_id"])

    _add(out, "ERROR", "horses: horse_id の重複", horses.loc[horses["horse_id"].duplicated(), "horse_id"])
    ids = runners["horse_id"].drop_duplicates()
    _add(out, "WARN", "horses: 血統マスタに無い出走馬（頭数）", ids[~ids.isin(horses["horse_id"])],
         "python -m scrape pedigree で取得する")

    flat = races[races["surface"].isin(["芝", "ダート"])]
    key = flat["place"] + flat["surface"]
    _add(out, "ERROR", "courses: コースマスタに無い競馬場×芝ダ", key[~key.isin(courses["place"] + courses["surface"])])
    return out


def run_checks(t):
    return (check_races(t["races"], t["runners"]) + check_runners(t["runners"], t["races"])
            + check_payouts(t["payouts"], t["races"]) + check_odds(t["odds_final"], t["runners"], t["payouts"], t["races"])
            + check_others(t["laps"], t["training"], t["horses"], t["courses"], t["races"], t["runners"]))


def format_report(issues):
    lines = []
    for level in LEVELS:
        items = [i for i in issues if i.level == level]
        lines.append(f"==== {level} {len(items)}件 ====")
        for i in items:
            ex = f"  例: {', '.join(i.examples)}" if i.examples else ""
            note = f"  ※{i.note}" if i.note else ""
            lines.append(f"- {i.name}: {i.count}{ex}{note}")
    return "\n".join(lines)
