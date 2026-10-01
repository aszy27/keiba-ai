# prep/ingest.py
# 旧データの CSV を正規化して data/v2/*.parquet に保存する。
# 保存前に checks.py で検査し、ERROR があれば保存しない（レポートだけ data/v2/check_report.txt に書く）。
# 旧 CSV は書き換えず、修正は data/v2/patches/ に置いて取り込み時に当てる。
#
# 使い方: python -m prep.ingest               # 検査して、ERROR が無ければ保存
#         python -m prep.ingest --allow-errors  # ERROR があっても保存（中身の確認用）
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prep import normalize as nz
from prep.checks import Issue, format_report, run_checks
from paths import LEGACY_DIR, PATCH_DIR, TABLES, V2_DIR, table_path

FILE_RETURN = LEGACY_DIR / "return_data_progress.csv"
FILE_ODDS = LEGACY_DIR / "odds_api_progress.csv"
FILE_LAP = LEGACY_DIR / "lap_data_progress.csv"
FILE_TRAIN = LEGACY_DIR / "training_data_progress.csv"
FILE_HORSE = LEGACY_DIR / "master_horse_data.csv"
FILE_BREEDER = LEGACY_DIR / "breeder_data_progress.csv"
FILE_COURSE = LEGACY_DIR / "master_course_data.csv"

PAYOUT_COLS = {"tansho": "win", "fukusho": "place", "wakuren": "bracket_quinella", "umaren": "quinella",
               "wide": "wide", "umatan": "exacta", "sanrenpuku": "trio", "sanrentan": "trifecta"}
# patches/race_info.csv で上書きできる列（旧 CSV の列名）
PATCHABLE_INFO = ["race_name", "race_time", "type", "length", "handed", "weather", "condition"]


def read_csv(path, **kw):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False, **kw)


def race_data_files():
    return sorted(glob.glob(str(LEGACY_DIR / "*" / "race_data_*.csv")))


def apply_patches(raw, notes):
    """patches/ の修正を旧 CSV の行に当てる
    - race_delete.csv  : race_id, reason      → そのレースの行を削除
    - race_rows/*.csv  : 旧 CSV と同じ列       → 同じ race_id の行を丸ごと差し替え（取り直したレース）
    - race_info.csv    : race_id + PATCHABLE_INFO の一部 → 空欄でない値でレース情報を上書き。
                         ただし race_name は、元の名前が空か文字化けを戻せない場合だけ置き換える
                         （結果ページの名前には「(2勝)」などのクラス表記が無いため）
    """
    f = PATCH_DIR / "race_delete.csv"
    if f.exists():
        ids = set(read_csv(f)["race_id"])
        raw = raw[~raw["race_id"].isin(ids)]
        notes.append(Issue("INFO", "patches: 削除したレース", len(ids), sorted(ids)[:8]))

    rows = sorted((PATCH_DIR / "race_rows").glob("*.csv")) if (PATCH_DIR / "race_rows").exists() else []
    if rows:
        new = pd.concat([read_csv(p) for p in rows], ignore_index=True)
        raw = pd.concat([raw[~raw["race_id"].isin(new["race_id"])], new.reindex(columns=raw.columns)], ignore_index=True)
        notes.append(Issue("INFO", "patches: 差し替えたレース", new["race_id"].nunique(), sorted(new["race_id"].unique())[:8]))

    f = PATCH_DIR / "race_info.csv"
    if f.exists():
        info = read_csv(f).drop_duplicates("race_id", keep="last").set_index("race_id")
        cols = [c for c in PATCHABLE_INFO if c in info.columns]
        raw = raw.copy()
        for c in cols:
            val = raw["race_id"].map(info[c])
            use = val.notna() & (val.astype(str).str.strip() != "")
            if c == "race_name":
                use &= raw[c].isna() | raw[c].map(lambda n: nz.fix_mojibake(n)[1] == "broken")
            raw[c] = val.where(use, raw[c])
        notes.append(Issue("INFO", "patches: レース情報を上書きしたレース", len(info), list(info.index[:8]), ", ".join(cols)))
    return raw


def build_races(raw, notes):
    r = raw.drop_duplicates("race_id", keep="last").reset_index(drop=True)
    names = r["race_name"].map(nz.fix_mojibake)
    out = pd.DataFrame({
        "race_id": r["race_id"].astype("string"),
        "race_date": pd.to_datetime(r["race_date"], errors="coerce"),
        "post_time": r["race_time"].astype("string"),
        "place": r["race_id"].str[4:6].map(nz.PLACE_CODES).astype("string"),
        "kai": nz.to_num(r["race_id"].str[6:8]).astype("Int8"),
        "day": nz.to_num(r["race_id"].str[8:10]).astype("Int8"),
        "race_number": nz.to_num(r["race_id"].str[10:12]).astype("Int8"),
        "race_name": names.str[0].astype("string"),
        "name_status": names.str[1].astype("string"),
        "surface": r["type"].where(r["type"].isin(nz.SURFACES)).astype("string"),
        "distance": nz.to_num(r["length"]).astype("Int16"),
        "direction": r["handed"].where(r["handed"].isin(["右", "左"])).astype("string"),
        "weather": r["weather"].where(r["weather"].isin(nz.WEATHERS)).astype("string"),
        "going": nz.normalize_going(r["condition"]),
        "first_prize": nz.to_num(r["max_prize"]).astype("float32"),
    })

    # 馬場状態・天候が取れなかったレースは、同じ日・同じ競馬場（馬場状態は同じ芝ダ）の直前のレース（無ければ直後）から補う
    out = out.sort_values(["race_date", "place", "race_number"]).reset_index(drop=True)
    for col, keys in [("going", ["race_date", "place", "surface"]), ("weather", ["race_date", "place"])]:
        missing = out[col].isna()
        if missing.any():
            filled = out.groupby(keys, dropna=False)[col].transform(lambda s: s.ffill().bfill())
            out[col] = out[col].fillna(filled)
            notes.append(Issue("INFO", f"races: {col} を同じ日の前後のレースから補完", int((missing & out[col].notna()).sum()),
                               list(out.loc[missing & out[col].notna(), "race_id"][:8])))

    grade_col = r.set_index("race_id")["grade"] if "grade" in r.columns else pd.Series(dtype=object)
    grade_col = out["race_id"].map(grade_col)
    out["grade"] = pd.array([nz.parse_grade(n, g) for n, g in zip(out["race_name"], grade_col)], dtype="string")
    out["race_class"] = pd.array([nz.parse_class(n, g) for n, g in zip(out["race_name"], out["grade"])], dtype="string")
    out["class_source"] = pd.array(np.where(out["race_class"].notna(), "name", None), dtype="string")
    out["age_condition"] = out["race_name"].map(nz.parse_age_condition).astype("string")

    # クラス名が読めないレース（文字化け・「春風S」のような名前だけの特別戦）は、同じ年・同じ平地/障害の
    # クラス別1着賞金の中央値に最も近いクラスとする
    year = out["race_id"].str[:4]
    jump = (out["surface"] == "障害").fillna(False)
    known = out[out["race_class"].notna()]
    ref = known.groupby([known["race_id"].str[:4], jump[known.index], "race_class"])["first_prize"].median()
    for i in out.index[out["race_class"].isna()]:
        key = (year[i], bool(jump[i]))
        reference = ref.loc[key].to_dict() if key in ref.index.droplevel(2) else {}
        c = nz.class_from_prize(out.at[i, "first_prize"], reference)
        if c:
            out.at[i, "race_class"], out.at[i, "class_source"] = c, "prize"
    return out


def build_runners(raw, races):
    d = raw.reset_index(drop=True)
    out = pd.DataFrame({
        "race_id": d["race_id"].astype("string"),
        "horse_id": d["horse_id"].astype("string").str.strip(),
        "horse_name": d["horse_name"].astype("string"),
        "horse_number": nz.to_num(d["horse_number"]).astype("Int8"),
        "bracket": nz.to_num(d["bracket"]).astype("Int8"),
        "sex": d["gender"].where(d["gender"].isin(["牡", "牝", "セ"])).astype("string"),
        "age": nz.to_num(d["age"]).astype("Int8"),
        "burden": nz.to_num(d["burden"]).astype("float32"),
        "jockey_id": nz.pad_id(d["jockey_id"]),
        "jockey_name": d["jockey_name"].astype("string"),
        "trainer_id": nz.pad_id(d["trainer_id"]),
        "trainer_name": d["trainer_name"].astype("string"),
        "weight": nz.to_num(d["weight"]).astype("float32"),
        "weight_diff": nz.to_num(d["weight_diff"].astype("string").str.replace("±", "")).astype("float32"),
        "finish_pos": nz.to_num(d["rank"]).astype("Int8"),
        "time_sec": nz.to_num(d["rap_time"]).round(1).astype("float32"),
        "margin_sec": nz.to_num(d["diff_time"]).round(1).astype("float32"),
        "last_3f": nz.to_num(d["last_3f"]).round(1).astype("float32"),
        "passage": d["passage_rank"].astype("string"),
        "prize": nz.to_num(d["prize"]).astype("float32"),
    })
    out["corner_first"], out["corner_last"] = (s.astype("Int8") for s in nz.corner_positions(out["passage"]))
    out["status"] = nz.runner_status(out["finish_pos"], out["passage"]).astype("string")

    # 障害レースの last_3f は平地と単位が違う（1F平均に近い値）ため使わない
    jump = (out["race_id"].map(races.set_index("race_id")["surface"]) == "障害").fillna(False)
    out.loc[jump, "last_3f"] = np.nan
    return out.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def add_race_counts(races, runners):
    g = runners.groupby("race_id")["status"]
    counts = pd.DataFrame({
        "n_entries": g.size(),
        "n_starters": g.apply(lambda s: (s != "scratched").sum()),
        "n_finishers": g.apply(lambda s: (s == "finished").sum()),
    }).astype("Int8")
    return races.join(counts, on="race_id")


def build_payouts(notes):
    p = read_csv(FILE_RETURN)
    p = p[~p.drop(columns="race_id").isna().all(axis=1)]  # 払戻ページが取れなかったときの空行
    dup = p[p["race_id"].duplicated(keep=False)]
    if len(dup):
        differ = dup.groupby("race_id").apply(lambda g: len(g.drop_duplicates()) > 1, include_groups=False)
        notes.append(Issue("WARN" if differ.any() else "INFO", "payouts: 旧 CSV で race_id が重複（最後の行を使用）",
                           dup["race_id"].nunique(), list(differ.index[:8]), f"内容が異なるもの {int(differ.sum())}件"))
    p = p.drop_duplicates("race_id", keep="last")
    out = pd.DataFrame({"race_id": p["race_id"].astype("string")})
    for src, dst in PAYOUT_COLS.items():
        out[dst] = nz.parse_payouts(p[src]) if src in p.columns else None
    return out.reset_index(drop=True)


def build_odds(notes):
    o = pd.read_csv(FILE_ODDS, dtype=str, on_bad_lines="skip")
    o["horse_number"] = nz.to_num(o["horse_number"])
    no_data = o.loc[o["horse_number"] <= 0, "race_id"]
    if len(no_data):
        notes.append(Issue("INFO", "odds: API がオッズを返さなかったレース", no_data.nunique(), list(no_data[:8])))
    o = o[o["horse_number"] > 0]
    if o.duplicated(["race_id", "horse_number"]).any():
        n = o.loc[o.duplicated(["race_id", "horse_number"]), "race_id"]
        notes.append(Issue("INFO", "odds: 旧 CSV で (race_id, 馬番) が重複（最後の行を使用）", n.nunique(), list(n[:8])))
        o = o.drop_duplicates(["race_id", "horse_number"], keep="last")
    win = nz.to_num(o["win_odds"])
    # 1レースの全馬が負の値（取消扱い）で返るのは、そのレースのオッズが API に無い場合。本物の取消と区別して捨てる
    has_odds = (win > 0).groupby(o["race_id"]).transform("any")
    if not has_odds.all():
        ids = o.loc[~has_odds, "race_id"]
        notes.append(Issue("INFO", "odds: 全馬が取消扱いで返るレース（オッズ無しとして除外）", ids.nunique(),
                           list(ids.unique()[:8])))
        o, win = o[has_odds], win[has_odds]
    pop = nz.to_num(o["popularity"])
    place_min, place_max = nz.to_num(o["place_min"]), nz.to_num(o["place_max"])
    # API は取消を -3.0、除外を -2.0 で返す
    status = np.select([win > 0, win == -3, win == -2], ["ok", "cancelled", "excluded"], "unknown")
    out = pd.DataFrame({
        "race_id": o["race_id"].astype("string"),
        "horse_number": o["horse_number"].astype("Int8"),
        "win_odds": win.where(win > 0).astype("float32"),
        "popularity": pop.where((pop > 0) & (pop < 999)).astype("Int8"),
        "place_odds_min": place_min.where(place_min > 0).astype("float32"),
        "place_odds_max": place_max.where(place_max > 0).astype("float32"),
        "odds_status": pd.array(status, dtype="string"),
        "fetched_at": pd.to_datetime(o["official_datetime"], errors="coerce"),
    })
    return out.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def build_laps():
    lap = read_csv(FILE_LAP)
    has = lap["lap_times"].notna() & (lap["lap_times"] != "NO_LAP")
    times = lap["lap_times"].where(has).map(
        lambda s: [float(x) for x in s.split("-")] if isinstance(s, str) else None)
    first_seg = nz.to_num(lap["lap_headers"].where(has).str.extract(r"^(\d+)m", expand=False))
    pace = lap["pace_type"].where(lap["pace_type"].isin(["H", "M", "S"]))
    return pd.DataFrame({
        "race_id": lap["race_id"].astype("string"),
        "lap_times": times,
        "first_segment_m": first_seg.astype("Int16"),   # 最初の区間の距離（100m なら端数距離のコース）
        "pace_type": pace.astype("string"),
    })


def build_training():
    t = read_csv(FILE_TRAIN)
    t = t[t["horse_id"] != "NO_TRAIN"]
    grade = t["oikiri_rank"].str.upper().str.extract(r"([SABCD])", expand=False)
    # 注意: 旧スクレイパーは評価欄が空の馬を「C」で保存しているため、C には「評価なし」が混ざる
    return pd.DataFrame({"race_id": t["race_id"].astype("string"), "horse_id": t["horse_id"].astype("string"),
                         "oikiri_rank": grade.astype("string")}).reset_index(drop=True)


def build_horses():
    h = read_csv(FILE_HORSE)
    b = read_csv(FILE_BREEDER).drop_duplicates("horse_id", keep="last")
    h = h.drop_duplicates("horse_id", keep="last").merge(b, on="horse_id", how="left")
    out = h[["horse_id", "horse_name", "sire_id", "sire_name", "dam_id", "dam_name", "breeder", "owner"]].astype("string")
    for c in ["breeder", "owner"]:
        out[c] = out[c].where(out[c] != "unknown")
    return out.reset_index(drop=True)


def build_courses():
    c = read_csv(FILE_COURSE)
    surface = c["course_id"].str.extract(r"(芝|ダート)$", expand=False)
    return pd.DataFrame({
        "place": c["course_id"].str.replace(r"(芝|ダート)$", "", regex=True).astype("string"),
        "surface": surface.astype("string"),
        "straight_m": nz.to_num(c["straight_len"]).astype("float32"),
        "elevation_m": nz.to_num(c["slope_height"]).astype("float32"),
        "has_slope": nz.to_num(c["has_slope"]).astype("Int8"),
    })


def build_all():
    notes = []
    raw = pd.concat([read_csv(f) for f in race_data_files()], ignore_index=True)
    raw = apply_patches(raw, notes)
    races = build_races(raw, notes)
    runners = build_runners(raw, races)
    races = add_race_counts(races, runners)
    tables = {
        "races": races.sort_values(["race_date", "race_id"]).reset_index(drop=True),
        "runners": runners,
        "payouts": build_payouts(notes),
        "odds_final": build_odds(notes),
        "laps": build_laps(),
        "training": build_training(),
        "horses": build_horses(),
        "courses": build_courses(),
    }
    return tables, notes


def source_manifest():
    files = race_data_files() + [str(p) for p in [FILE_RETURN, FILE_ODDS, FILE_LAP, FILE_TRAIN, FILE_HORSE,
                                                   FILE_BREEDER, FILE_COURSE]]
    if PATCH_DIR.exists():
        files += [str(p) for p in sorted(PATCH_DIR.rglob("*.csv"))]
    out = {}
    for f in map(Path, files):
        st = f.stat()
        out[f.relative_to(LEGACY_DIR).as_posix()] = {
            "bytes": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-errors", action="store_true", help="ERROR があっても保存する")
    args = ap.parse_args()

    print("1. 旧データを読み込んで正規化...")
    tables, notes = build_all()
    for name in TABLES:
        print(f"   {name:<11} {len(tables[name]):>8}行")

    print("2. 検査...")
    issues = notes + run_checks(tables)
    report = format_report(issues)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    (V2_DIR / "check_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)

    n_err = sum(i.level == "ERROR" for i in issues)
    if n_err and not args.allow_errors:
        print(f"\nERROR が {n_err}件あるため保存しません（レポート: {V2_DIR / 'check_report.txt'}）")
        sys.exit(1)

    print("3. 保存...")
    for name in TABLES:
        tables[name].to_parquet(table_path(name), index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": {name: len(tables[name]) for name in TABLES},
        "issues": {lv: sum(i.level == lv for i in issues) for lv in ("ERROR", "WARN", "INFO")},
        "sources": source_manifest(),
    }
    (V2_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   保存先: {V2_DIR}")


if __name__ == "__main__":
    main()
