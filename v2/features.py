# v2/features.py
# 特徴量を作る唯一の場所。学習・評価・本番のすべてで build_features を使う。
#
# 原則: 日付 D のレースの特徴量は「D より前の日のレース結果」と「そのレース自身の発走前に分かる情報
#       （出走馬・馬番・斤量・馬体重・馬場状態・追い切り評価など）」だけから作る。
#       過去成績はすべて「各日の成績を累積した表」から race_date より前の最後の行を取り出す形で求めるため、
#       同じ日の他レースの結果や自分自身の結果は構造上入らない。
#       検証は tests/test_features.py のリークテスト（未来を消しても・当日の結果を消しても値が変わらない）。
#
# 予測するレース（結果が未確定）は runners に status="entry" の行として入れれば、同じ関数で特徴量が作られる。
import numpy as np
import pandas as pd

from v2.paths import table_path

CLASS_ORDER = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 6, "G3": 7, "G2": 8, "G1": 9}
GOING_ORDER = {"良": 0, "稍": 1, "重": 2, "不良": 3}
OIKIRI_ORDER = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
RESULT_COLS = ["finish_pos", "time_sec", "margin_sec", "last_3f", "passage", "corner_first", "corner_last", "prize"]
HISTORY_STATUS = ("finished", "dnf")
BASE_RATE = {"win": 0.075, "top3": 0.22}   # 縮小推定の事前値（1レース約14頭の平均的な勝率・複勝率）
FIG_CLIP, L3_CLIP = 15.0, 10.0              # スピード指数（秒/1000m）・上がりの相対値（秒）を丸める幅

ID_COLS = ["race_id", "race_date", "horse_id", "horse_number", "jockey_id", "trainer_id"]
LABEL_COLS = ["status", "finish_pos", "win", "top3"]
CATEGORICAL = ["place", "surface", "going", "weather", "sex", "direction", "age_condition"]
PRE_RACE = ["bracket", "age", "burden", "weight", "weight_diff", "distance", "class_ord", "going_ord", "n_starters",
            "race_number", "first_prize", "straight_m", "elevation_m", "has_slope", "oikiri", "month"]

# (名前, キー, 集計する日数（None は全期間）, 縮小の強さ)
ENTITY_STATS = [
    ("horse", ["horse_id"], None, 3),
    ("horse_surface", ["horse_id", "surface"], None, 3),
    ("horse_dist", ["horse_id", "dist_bucket"], None, 3),
    ("horse_place", ["horse_id", "place"], None, 3),
    ("jockey", ["jockey_id"], 365, 30),
    ("trainer", ["trainer_id"], 365, 30),
    ("jockey_course", ["jockey_id", "place", "surface"], 1095, 20),
    ("jockey_trainer", ["jockey_id", "trainer_id"], 1095, 10),
    ("sire_surface", ["sire_id", "surface"], None, 50),
    ("sire_dist", ["sire_id", "dist_bucket"], None, 50),
    ("dam", ["dam_id"], None, 5),
]
ENTITY_FEATURES = [f"{name}_{s}" for name, *_ in ENTITY_STATS for s in ("n", "win", "top3")]
# 過去走の指標（名前, 平均をとる走数）
RUN_METRICS = [("fig", 3), ("l3_rel", 3), ("rel_pos", 3), ("margin", 3), ("corner_rel", 5)]
HISTORY_FEATURES = (["h_runs", "h_dnf", "h_prize", "fig_best5", "days_since", "dist_change", "class_change",
                     "burden_change", "surface_change", "place_change"]
                    + [f"{c}_last" for c, _ in RUN_METRICS] + [f"{c}_mean{n}" for c, n in RUN_METRICS])
# レース内で相対化する特徴量
FIELD_RELATIVE = ["fig_mean3", "fig_best5", "l3_rel_mean3", "rel_pos_mean3", "horse_top3", "jockey_top3",
                  "trainer_top3", "burden"]
FIELD_FEATURES = (["gate_rel", "field_front_share", "style_vs_field"]
                  + [f"{c}_{s}" for c in FIELD_RELATIVE for s in ("vs_field", "rank")])
FEATURES = CATEGORICAL + PRE_RACE + HISTORY_FEATURES + ENTITY_FEATURES + FIELD_FEATURES


def load_tables(names=("races", "runners", "horses", "courses", "training")):
    return {n: pd.read_parquet(table_path(n)) for n in names}


def asof_before(state, keys, query, date_col="race_date"):
    """state（keys + date_col + 値の列）から、query の各行について date_col より前の最後の行の値を返す"""
    q = query[keys + [date_col]].copy()
    q["_row"] = np.arange(len(q))
    q = q[q[keys].notna().all(axis=1)].sort_values(date_col, kind="stable")
    s = state.dropna(subset=keys).rename(columns={date_col: "_state_date"}).sort_values("_state_date", kind="stable")
    m = pd.merge_asof(q, s, left_on=date_col, right_on="_state_date", by=keys,
                      allow_exact_matches=False, direction="backward")
    values = [c for c in state.columns if c not in keys and c != date_col]
    return m.set_index("_row")[values].reindex(np.arange(len(query))).set_axis(query.index)


def cum_before(events, keys, query, cols, window_days=None, date_col="race_date"):
    """events の cols を keys ごとに合計する。query の各行について、日付より前（window_days があれば
    [日付 - window_days, 日付) の範囲）の合計"""
    daily = events.groupby(keys + [date_col], observed=True)[cols].sum().reset_index().sort_values(keys + [date_col])
    daily[cols] = daily.groupby(keys, observed=True)[cols].cumsum()
    total = asof_before(daily, keys, query, date_col).fillna(0.0)
    if window_days is None:
        return total
    shifted = query[keys + [date_col]].copy()
    shifted[date_col] = shifted[date_col] - pd.Timedelta(days=window_days)
    return total - asof_before(daily, keys, shifted, date_col).fillna(0.0)


def _changed(now, before):
    return pd.Series(np.where(before.isna(), np.nan, (now.astype(object) != before.astype(object)).astype(float)),
                     index=now.index)


def base_frame(t):
    races = t["races"].drop(columns=["n_entries", "n_starters", "n_finishers", "name_status", "class_source"],
                            errors="ignore")
    df = t["runners"][t["runners"]["status"] != "scratched"].merge(races, on="race_id", how="left",
                                                                     validate="many_to_one")
    df = df.merge(t["horses"][["horse_id", "sire_id", "dam_id"]], on="horse_id", how="left")
    df = df.merge(t["courses"], on=["place", "surface"], how="left")
    tr = t["training"].assign(oikiri=lambda x: x["oikiri_rank"].map(OIKIRI_ORDER).astype(float))
    df = df.merge(tr[["race_id", "horse_id", "oikiri"]], on=["race_id", "horse_id"], how="left")

    # 数値の nullable 型（Int8 等）は計算・LightGBM 用に float にそろえる
    for c in df.columns:
        if pd.api.types.is_extension_array_dtype(df[c].dtype) and pd.api.types.is_numeric_dtype(df[c].dtype):
            df[c] = df[c].astype(float)
    df["status"] = df["status"].astype(object)
    df["is_hist"] = df["status"].isin(HISTORY_STATUS).to_numpy()
    df["win"] = (df["finish_pos"] == 1).astype(float)
    df["top3"] = (df["finish_pos"] <= 3).astype(float)
    df["class_ord"] = df["race_class"].astype(object).map(CLASS_ORDER).astype(float)
    df["going_ord"] = df["going"].astype(object).map(GOING_ORDER).astype(float)
    df["dist_bucket"] = pd.cut(df["distance"], [0, 1300, 1600, 2000, 2400, 5000], labels=False).astype(float)
    df["n_starters"] = df.groupby("race_id")["horse_id"].transform("size").astype(float)
    df["month"] = df["race_date"].dt.month.astype(float)
    return df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)


def past_run_metrics(df):
    """結果が確定した各出走について、スピード指数などレース後に分かる指標を計算する（未確定の行は NaN）"""
    hist = df["is_hist"].to_numpy()
    fin_mask = hist & (df["status"] == "finished").to_numpy()
    per100 = df["time_sec"] / df["distance"] * 100

    # 基準タイム: そのレースより前の日の、同じ条件のレースの勝ち時計（100mあたり）の平均。件数が少なければ条件を粗くする
    winners = df[fin_mask & (df["finish_pos"] == 1).to_numpy()].drop_duplicates("race_id")
    races = df.drop_duplicates("race_id")[["race_id", "race_date", "place", "surface", "distance", "going"]]
    ev = races.merge(pd.DataFrame({"race_id": winners["race_id"].values, "t": per100[winners.index].values, "n": 1.0}),
                     on="race_id")
    par = pd.Series(np.nan, index=races.index)
    for keys, min_n in [(["surface", "distance"], 1), (["surface", "distance", "going"], 8),
                        (["place", "surface", "distance", "going"], 8)]:
        s = cum_before(ev, keys, races, ["t", "n"])
        par = par.where(s["n"] < min_n, s["t"] / s["n"])
    df["par100"] = df["race_id"].map(pd.Series(par.values, index=races["race_id"].values))

    g = df[fin_mask].groupby("race_id")
    n_fin = df["race_id"].map(g.size())
    # 基準より1000mあたり何秒速いか。大差の最下位（1200mで140秒など実在する記録）が平均を振り回さないよう丸める
    df["fig"] = ((df["par100"] - per100) * 10).clip(-FIG_CLIP, FIG_CLIP).where(fin_mask)
    df["l3_rel"] = (df["race_id"].map(g["last_3f"].median()) - df["last_3f"]).clip(-L3_CLIP, L3_CLIP).where(fin_mask)
    df["rel_pos"] = ((df["finish_pos"] - 1) / (n_fin - 1).clip(lower=1)).where(fin_mask)
    df["margin"] = df["margin_sec"].clip(upper=5).where(fin_mask)
    df["corner_rel"] = ((df["corner_first"] - 1) / (df["n_starters"] - 1).clip(lower=1)).where(hist)
    df["dnf"] = (df["status"] == "dnf").astype(float)
    return df


def horse_history(df):
    """各馬の出走ごとに「そのレース後の状態」を作り、各行の日付より前の最後の状態を結合する"""
    h = df[df["is_hist"]].sort_values(["horse_id", "race_date"])
    g = h.groupby("horse_id", sort=False)
    state = h[["horse_id", "race_date"]].copy()
    state["h_runs"] = g.cumcount() + 1.0
    state["h_dnf"] = g["dnf"].cumsum()
    state["h_prize"] = g["prize"].cumsum()
    for col, n in RUN_METRICS:
        state[f"{col}_last"] = h[col]
        state[f"{col}_mean{n}"] = g[col].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
    state["fig_best5"] = g["fig"].rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
    for col in ["race_date", "distance", "class_ord", "burden", "surface", "place"]:
        state[f"last_{col}"] = h[col]

    s = asof_before(state, ["horse_id"], df)
    out = s[[c for c in HISTORY_FEATURES if c in s.columns]].copy()
    out["h_runs"] = out["h_runs"].fillna(0.0)
    out["days_since"] = (df["race_date"] - s["last_race_date"]).dt.days.astype(float)
    out["dist_change"] = df["distance"] - s["last_distance"]
    out["class_change"] = df["class_ord"] - s["last_class_ord"]
    out["burden_change"] = df["burden"] - s["last_burden"]
    out["surface_change"] = _changed(df["surface"], s["last_surface"])
    out["place_change"] = _changed(df["place"], s["last_place"])
    return out


def entity_stats(df):
    ev = df[df["is_hist"]].assign(n=1.0)
    out = {}
    for name, keys, window, k in ENTITY_STATS:
        s = cum_before(ev, keys, df, ["n", "win", "top3"], window_days=window)
        out[f"{name}_n"] = s["n"]
        for col in ["win", "top3"]:
            out[f"{name}_{col}"] = (s[col] + BASE_RATE[col] * k) / (s["n"] + k)
    return pd.DataFrame(out, index=df.index)


def field_features(df):
    rg = df.groupby("race_id")
    out = pd.DataFrame(index=df.index)
    # 取消・除外があると馬番が頭数を超えるため、出走馬の中での馬番の順位で内外を表す
    out["gate_rel"] = (rg["horse_number"].rank(method="first") - 1) / (df["n_starters"] - 1).clip(lower=1)
    for c in FIELD_RELATIVE:
        out[f"{c}_vs_field"] = df[c] - rg[c].transform("mean")
        out[f"{c}_rank"] = rg[c].rank(ascending=False, pct=True)
    style = df["corner_rel_mean5"]
    front = pd.Series(np.where(style.isna(), np.nan, (style < 0.3).astype(float)), index=df.index)
    out["field_front_share"] = front.groupby(df["race_id"]).transform("mean")
    out["style_vs_field"] = style - rg["corner_rel_mean5"].transform("mean")
    return out


def build_features(t):
    """戻り値: 1行 = 1頭（取消・除外を除く）。ID・ラベル（status / finish_pos / win / top3）・特徴量（FEATURES）の列を持つ"""
    df = past_run_metrics(base_frame(t))
    df = pd.concat([df, horse_history(df), entity_stats(df)], axis=1)
    df = pd.concat([df, field_features(df)], axis=1)
    for c in CATEGORICAL:
        df[c] = df[c].astype(object).astype("category")
    return df[ID_COLS + LABEL_COLS + FEATURES]


def main():
    """全期間の特徴量を作って data/v2/features.parquet に保存する: python -m v2.features"""
    import time
    start = time.time()
    f = build_features(load_tables())
    f.to_parquet(table_path("features"), index=False)
    print(f"{len(f)}行 / {len(FEATURES)}特徴量 / {time.time() - start:.0f}秒 → {table_path('features')}")


if __name__ == "__main__":
    main()
