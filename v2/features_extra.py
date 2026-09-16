# v2/features_extra.py
# 追加の特徴量（2026-09-17）。既存の FEATURES / TRIP_FEATURES は変えず、別グループとして足す。
# build(df) は features.build_features の中から呼ばれ、馬の過去走・成績の集計・レース内比較が付いた後の df を受け取る。
#
# リークの原則は features.py と同じ「その日より前の結果だけ」。
# 例外は「当日バイアス」で、同じ日・同じ競馬場・同じ芝ダで**先に終わったレース**の結果だけを使う（買う時点では分かる）。
import numpy as np
import pandas as pd

from v2 import features as ft

# 条件別のスピード指数（名前, キー）
FIG_BY = [("h_fig_going", ["horse_id", "going"]), ("h_fig_surface", ["horse_id", "surface"]),
          ("h_fig_dist", ["horse_id", "dist_bucket"]), ("h_fig_course", ["horse_id", "place", "surface"])]
HORSE_EXTRA = [f"{n}_{s}" for n, _ in FIG_BY for s in ("mean", "n")] + ["fig_mean5", "fig_std3", "fig_trend", "prize_per_run"]
CONNECTION_EXTRA = ["jockey_dist_top3", "jockey_going_top3", "jockey_change_gain", "trainer_layoff_top3",
                    "owner_top3", "owner_n"]
PEDIGREE_EXTRA = ["sire_going_top3", "sire_going_n", "dam_fig_mean"]
SHAPE_EXTRA = ["field_fig_mean", "field_fig_std", "gap_to_best", "n_front", "weight_vs_field", "age_vs_field",
               "days_since_vs_field", "dist_change_x_style"]
SAMEDAY_EXTRA = ["bias_gate", "bias_style", "bias_n", "gate_x_bias", "style_x_bias"]
# 第2弾（2026-09-17）
CONNECTION2 = ["owner_trainer_top3", "is_club", "jockey_style_top3", "jockey_gate_top3"]
PREV_LEVEL = ["prev_field_fig", "prev_field_best", "prev_field_gap"]
HORSE2 = ["h_fig_extend_mean", "h_fig_shorten_mean", "h_fig_layoff_mean", "h_layoff_n"]
# 第3弾（2026-09-17）
OIKIRI2 = ["trainer_oikiri_mean", "oikiri_vs_trainer", "oikiri_prev_diff", "h_fig_when_sharp", "oikiri_vs_field"]
GATE_PACE2 = ["course_gate_bias", "gate_x_course_bias", "course_pace_mean", "style_x_course_pace"]
CONDITION2 = ["weight_vs_own_mean", "weight_vs_best", "h_fig_interval_mean", "is_first_distance", "is_class_up_first",
              "corner_rel_std5"]
# 第4弾（2026-09-17）
DAY_ADJ = ["fig_adj_last", "fig_adj_mean3", "fig_adj_best5", "h_fig_adj_mean"]
HORSE_PERSON = ["horse_jockey_n", "horse_jockey_top3", "horse_jockey_fig_mean", "trainer_class_top3"]
CAREER = ["runs_90d", "interval_sum3", "career_days", "h_fig_class_mean"]
GROUPS = {"馬の詳細実績": HORSE_EXTRA, "人": CONNECTION_EXTRA, "血統": PEDIGREE_EXTRA,
          "レースの形": SHAPE_EXTRA, "当日バイアス": SAMEDAY_EXTRA,
          "人2": CONNECTION2, "前走のレベル": PREV_LEVEL, "馬の条件別2": HORSE2,
          "調教2": OIKIRI2, "枠・展開2": GATE_PACE2, "馬の状態2": CONDITION2,
          "当日補正の指数": DAY_ADJ, "馬×人": HORSE_PERSON, "キャリア・疲労": CAREER}
EXTRA_FEATURES = [c for cols in GROUPS.values() for c in cols]
LAYOFF_DAYS = 60


def _shrunk(s, k, kind="top3"):
    return (s[kind] + ft.BASE_RATE[kind] * k) / (s["n"] + k)


def _horse_extra(df, ev, out):
    for name, keys in FIG_BY:
        s = ft.cum_before(ev, keys, df, ["fig_sum", "fig_n"])
        out[f"{name}_mean"] = s["fig_sum"] / s["fig_n"].replace(0, np.nan)
        out[f"{name}_n"] = s["fig_n"]

    h = ev.sort_values(["horse_id", "race_date"])
    g = h.groupby("horse_id", sort=False)
    state = h[["horse_id", "race_date"]].copy()
    state["fig_mean5"] = g["fig"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    state["fig_std3"] = g["fig"].rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    state["last_jockey"] = h["jockey_id"]
    s = ft.asof_before(state, ["horse_id"], df)
    out["fig_mean5"], out["fig_std3"] = s["fig_mean5"], s["fig_std3"]
    out["fig_trend"] = df["fig_last"] - s["fig_mean5"]          # 直近走が自分の平均よりどれだけ上か
    out["prize_per_run"] = df["h_prize"] / df["h_runs"].replace(0, np.nan)
    return s["last_jockey"]


def _connection_extra(df, ev, out, last_jockey):
    for name, keys in [("jockey_dist", ["jockey_id", "dist_bucket"]), ("jockey_going", ["jockey_id", "going"])]:
        out[f"{name}_top3"] = _shrunk(ft.cum_before(ev, keys, df, ["n", "top3"], window_days=365), 20)

    # 乗り替わりで騎手の複勝率がどれだけ変わるか（前走の騎手との差）
    q = df[["race_date"]].assign(_key=last_jockey.values)
    s = ft.cum_before(ev.rename(columns={"jockey_id": "_key"}), ["_key"], q, ["n", "top3"], window_days=365)
    out["jockey_change_gain"] = (df["jockey_top3"] - _shrunk(s, 30)).where(last_jockey.notna())

    layoff = (df["days_since"] >= LAYOFF_DAYS).astype(float)    # 休み明けかどうかは発走前に分かる
    s = ft.cum_before(ev.assign(layoff=(ev["days_since"] >= LAYOFF_DAYS).astype(float)), ["trainer_id", "layoff"],
                      df.assign(layoff=layoff), ["n", "top3"], window_days=1095)
    out["trainer_layoff_top3"] = _shrunk(s, 20)

    s = ft.cum_before(ev, ["owner"], df, ["n", "top3"])
    out["owner_top3"], out["owner_n"] = _shrunk(s, 20), s["n"]


def _pedigree_extra(df, ev, out):
    s = ft.cum_before(ev, ["sire_id", "going"], df, ["n", "top3"])
    out["sire_going_top3"], out["sire_going_n"] = _shrunk(s, 50), s["n"]
    s = ft.cum_before(ev, ["dam_id"], df, ["fig_sum", "fig_n"])
    out["dam_fig_mean"] = s["fig_sum"] / s["fig_n"].replace(0, np.nan)


def _shape_extra(df, out):
    rg = df.groupby("race_id")
    out["field_fig_mean"] = rg["fig_mean3"].transform("mean")
    out["field_fig_std"] = rg["fig_mean3"].transform("std")
    best = rg["fig_mean3"].transform("max")
    second = rg["fig_mean3"].transform(lambda x: x.nlargest(2).min() if x.notna().sum() >= 2 else np.nan)
    # 自分以外で一番強い馬との差（自分が一番なら2番手との差）
    out["gap_to_best"] = df["fig_mean3"] - np.where(df["fig_mean3"] >= best, second, best)
    front = (df["corner_rel_mean5"] < 0.3).astype(float).where(df["corner_rel_mean5"].notna())
    out["n_front"] = front.groupby(df["race_id"]).transform("sum")
    for c in ["weight", "age", "days_since"]:
        out[f"{c}_vs_field"] = df[c] - rg[c].transform("mean")
    out["dist_change_x_style"] = df["dist_change"] * (0.5 - df["corner_rel_mean5"])


def _sameday_bias(df, out):
    """同じ日・同じ競馬場・同じ芝ダで、先に終わったレースの勝ち馬の枠・1角位置の偏り（長期平均との差）"""
    races = df.drop_duplicates("race_id")[["race_id", "race_date", "place", "surface", "race_number"]].copy()
    win = df[(df["status"] == "finished") & (df["finish_pos"] == 1)]
    races = races.join(win.groupby("race_id")[["gate_rel", "corner_rel"]].mean(), on="race_id")
    races = races.sort_values(["race_date", "place", "surface", "race_number"])
    g = races.groupby(["race_date", "place", "surface"], sort=False)
    for c in ["gate_rel", "corner_rel"]:
        races[f"day_{c}"] = g[c].transform(lambda x: x.shift().expanding().mean())
    races["bias_n"] = g["gate_rel"].transform(lambda x: x.shift().expanding().count())

    ev = races[races["gate_rel"].notna()].assign(n=1.0)
    base = ft.cum_before(ev, ["place", "surface"], races, ["gate_rel", "corner_rel", "n"])
    denom = base["n"].replace(0, np.nan)
    races["bias_gate"] = races["day_gate_rel"] - base["gate_rel"] / denom      # 正なら外枠が勝っている日
    races["bias_style"] = races["day_corner_rel"] - base["corner_rel"] / denom  # 正なら差しが決まっている日

    r = races.set_index("race_id")
    for c in ["bias_gate", "bias_style", "bias_n"]:
        out[c] = df["race_id"].map(r[c])
    out["gate_x_bias"] = (df["gate_rel"] - 0.5) * out["bias_gate"]
    out["style_x_bias"] = (df["corner_rel_mean5"] - 0.5) * out["bias_style"]


CLUB_WORDS = ("レーシング", "クラブ", "ファーム", "ホースクラブ", "サラブレッド", "（株）", "(株)", "（有）", "(有)")


def _buckets(df):
    """脚質・枠の区分（発走前に分かる値だけから作る）"""
    style = pd.cut(df["corner_rel_mean5"], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=False).fillna(-1.0)
    gate = pd.cut(df["gate_rel"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=False).astype(float)
    return style, gate


def _connection2(df, ev, out):
    s = ft.cum_before(ev, ["owner", "trainer_id"], df, ["n", "top3"], window_days=1095)
    out["owner_trainer_top3"] = _shrunk(s, 10)
    owner = df["owner"].astype(object).fillna("")
    out["is_club"] = owner.map(lambda x: float(any(w in x for w in CLUB_WORDS)))

    style, gate = _buckets(df)
    ev_style, ev_gate = _buckets(ev)
    for name, key, q_col, e_col in [("jockey_style_top3", "_style", style, ev_style), ("jockey_gate_top3", "_gate", gate, ev_gate)]:
        s = ft.cum_before(ev.assign(**{key: e_col}), ["jockey_id", key], df.assign(**{key: q_col}),
                          ["n", "top3"], window_days=1095)
        out[name] = _shrunk(s, 20)


def _prev_level(df, out):
    """前走で相手にした馬の強さ（そのレースの発走前に分かっていた指数の、自分以外の平均・最高）"""
    rg = df.groupby("race_id")["fig_mean3"]
    n_known = rg.transform("count")
    others_mean = (rg.transform("sum") - df["fig_mean3"].fillna(0)) / (n_known - df["fig_mean3"].notna()).clip(lower=1)
    best = rg.transform("max")
    second = rg.transform(lambda x: x.nlargest(2).min() if x.notna().sum() >= 2 else np.nan)
    others_best = np.where(df["fig_mean3"] >= best, second, best)

    hist = df[df["is_hist"]]
    state = hist[["horse_id", "race_date"]].copy()
    state["prev_field_fig"] = others_mean[hist.index]
    state["prev_field_best"] = pd.Series(others_best, index=df.index)[hist.index]
    state["_own"] = hist["fig_mean3"]
    s = ft.asof_before(state, ["horse_id"], df)
    out["prev_field_fig"], out["prev_field_best"] = s["prev_field_fig"], s["prev_field_best"]
    out["prev_field_gap"] = s["_own"] - s["prev_field_best"]   # 前走で相手との力差がどれだけあったか


def _horse2(df, ev, out):
    change = np.sign(df["dist_change"]).fillna(0.0)
    ev_change = np.sign(ev["dist_change"]).fillna(0.0)
    s = ft.cum_before(ev.assign(_c=ev_change), ["horse_id", "_c"], df.assign(_c=change), ["fig_sum", "fig_n"])
    mean = s["fig_sum"] / s["fig_n"].replace(0, np.nan)
    out["h_fig_extend_mean"] = mean.where(change > 0)      # 距離を延ばしたときの自分の指数
    out["h_fig_shorten_mean"] = mean.where(change < 0)

    layoff = (df["days_since"] >= LAYOFF_DAYS).astype(float)
    s = ft.cum_before(ev.assign(_l=(ev["days_since"] >= LAYOFF_DAYS).astype(float)), ["horse_id", "_l"],
                      df.assign(_l=layoff), ["fig_sum", "fig_n"])
    out["h_fig_layoff_mean"] = (s["fig_sum"] / s["fig_n"].replace(0, np.nan)).where(layoff > 0)
    out["h_layoff_n"] = s["fig_n"].where(layoff > 0)


def _oikiri2(df, ev, out):
    """追い切り評価は厩舎ごとに基準が違うので、厩舎の平均との差を見る"""
    e = ev.assign(o_sum=ev["oikiri"].fillna(0.0), o_n=ev["oikiri"].notna().astype(float))
    s = ft.cum_before(e, ["trainer_id"], df, ["o_sum", "o_n"], window_days=1095)
    out["trainer_oikiri_mean"] = s["o_sum"] / s["o_n"].replace(0, np.nan)
    out["oikiri_vs_trainer"] = df["oikiri"] - out["trainer_oikiri_mean"]

    state = ev.sort_values(["horse_id", "race_date"])[["horse_id", "race_date", "oikiri"]].rename(columns={"oikiri": "prev_oikiri"})
    out["oikiri_prev_diff"] = df["oikiri"] - ft.asof_before(state, ["horse_id"], df)["prev_oikiri"]

    sharp = (df["oikiri"] >= 4).astype(float)     # A評価以上
    s = ft.cum_before(ev.assign(_s=(ev["oikiri"] >= 4).astype(float)), ["horse_id", "_s"], df.assign(_s=sharp),
                      ["fig_sum", "fig_n"])
    out["h_fig_when_sharp"] = (s["fig_sum"] / s["fig_n"].replace(0, np.nan)).where(sharp > 0)
    out["oikiri_vs_field"] = df["oikiri"] - df.groupby("race_id")["oikiri"].transform("mean")


def _gate_pace2(df, ev, out):
    """コース固有の枠の有利不利と平均ペース（その日より前の同条件のレースから）"""
    races = df.drop_duplicates("race_id")[["race_id", "race_date", "place", "surface", "distance", "dist_bucket"]].copy()
    win = df[(df["status"] == "finished") & (df["finish_pos"] == 1)]
    races = races.join(win.groupby("race_id")[["gate_rel"]].mean(), on="race_id")
    races["pace"] = races["race_id"].map(df.drop_duplicates("race_id").set_index("race_id")["pace_rel"].add(0))
    ev_r = races[races["gate_rel"].notna()].assign(n=1.0)
    s = ft.cum_before(ev_r, ["place", "surface", "dist_bucket"], races, ["gate_rel", "n"])
    races["course_gate_bias"] = s["gate_rel"] / s["n"].replace(0, np.nan) - 0.5   # 正なら外枠が勝ちやすいコース

    pace_ev = df.drop_duplicates("race_id")[["race_id", "race_date", "place", "surface", "distance"]].copy()
    pace_ev["pace"] = df.drop_duplicates("race_id").set_index("race_id")["pace_rel"].values
    pace_ev = pace_ev[pace_ev["pace"].notna()].assign(n=1.0)
    s = ft.cum_before(pace_ev, ["place", "surface", "distance"], races, ["pace", "n"])
    races["course_pace_mean"] = s["pace"] / s["n"].replace(0, np.nan)

    r = races.set_index("race_id")
    for c in ["course_gate_bias", "course_pace_mean"]:
        out[c] = df["race_id"].map(r[c])
    out["gate_x_course_bias"] = (df["gate_rel"] - 0.5) * out["course_gate_bias"]
    out["style_x_course_pace"] = (0.5 - df["corner_rel_mean5"]) * out["course_pace_mean"]


def _condition2(df, ev, out):
    """馬体重の水準、休養パターン、初距離・昇級初戦、1角位置のばらつき"""
    h = ev.sort_values(["horse_id", "race_date"])
    g = h.groupby("horse_id", sort=False)
    state = h[["horse_id", "race_date"]].copy()
    state["own_weight_mean"] = g["weight"].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    state["corner_rel_std5"] = g["corner_rel"].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    best = g["fig"].cummax()
    at_best = h["weight"].where(h["fig"] >= best)          # 自己ベストを更新した走の馬体重
    state["weight_at_best"] = at_best.groupby(h["horse_id"], sort=False).ffill()
    s = ft.asof_before(state, ["horse_id"], df)
    out["weight_vs_own_mean"] = df["weight"] - s["own_weight_mean"]
    out["weight_vs_best"] = df["weight"] - s["weight_at_best"]
    out["corner_rel_std5"] = s["corner_rel_std5"]

    bucket = pd.cut(df["days_since"], [-1, 8, 21, 42, 70, 180, 10000], labels=False).astype(float)
    ev_bucket = pd.cut(ev["days_since"], [-1, 8, 21, 42, 70, 180, 10000], labels=False).astype(float)
    s = ft.cum_before(ev.assign(_b=ev_bucket), ["horse_id", "_b"], df.assign(_b=bucket), ["fig_sum", "fig_n"])
    out["h_fig_interval_mean"] = s["fig_sum"] / s["fig_n"].replace(0, np.nan)

    s = ft.cum_before(ev.assign(n=1.0), ["horse_id", "distance"], df, ["n"])
    out["is_first_distance"] = (s["n"] == 0).astype(float)
    s = ft.cum_before(ev.assign(n=1.0), ["horse_id", "class_ord"], df, ["n"])
    out["is_class_up_first"] = ((s["n"] == 0) & (df["class_change"] > 0)).astype(float)


def _day_adjusted(df, out):
    """その日・その競馬場・芝ダの平均との差で指数を補正する（馬場が速い日の好時計を割り引く）。
    使うのは過去走の補正済み指数だけなので、今回のレース当日の結果は入らない"""
    day_mean = df.groupby([df["race_date"], df["place"], df["surface"]], observed=True)["fig"].transform("mean")
    df["fig_adj"] = df["fig"] - day_mean

    h = df[df["is_hist"]].sort_values(["horse_id", "race_date"])
    g = h.groupby("horse_id", sort=False)
    state = h[["horse_id", "race_date"]].copy()
    state["fig_adj_last"] = h["fig_adj"]
    state["fig_adj_mean3"] = g["fig_adj"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    state["fig_adj_best5"] = g["fig_adj"].rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
    s = ft.asof_before(state, ["horse_id"], df)
    for c in ["fig_adj_last", "fig_adj_mean3", "fig_adj_best5"]:
        out[c] = s[c]

    ev = df[df["is_hist"]].assign(adj_sum=lambda x: x["fig_adj"].fillna(0.0), adj_n=lambda x: x["fig_adj"].notna().astype(float))
    s = ft.cum_before(ev, ["horse_id"], df, ["adj_sum", "adj_n"])
    out["h_fig_adj_mean"] = s["adj_sum"] / s["adj_n"].replace(0, np.nan)


def _horse_person(df, ev, out):
    """その馬とその騎手の相性、厩舎のクラス別成績"""
    s = ft.cum_before(ev, ["horse_id", "jockey_id"], df, ["n", "top3", "fig_sum", "fig_n"])
    out["horse_jockey_n"] = s["n"]
    out["horse_jockey_top3"] = _shrunk(s, 3)
    out["horse_jockey_fig_mean"] = s["fig_sum"] / s["fig_n"].replace(0, np.nan)
    s = ft.cum_before(ev, ["trainer_id", "class_ord"], df, ["n", "top3"], window_days=1095)
    out["trainer_class_top3"] = _shrunk(s, 20)


def _career(df, ev, out):
    """使い詰めかどうか、キャリアの長さ、クラス別の自分の指数"""
    out["runs_90d"] = ft.cum_before(ev, ["horse_id"], df, ["n"], window_days=90)["n"]

    h = ev.sort_values(["horse_id", "race_date"])
    g = h.groupby("horse_id", sort=False)
    state = h[["horse_id", "race_date"]].copy()
    state["interval_sum3"] = g["days_since"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    state["first_date"] = g["race_date"].cummin()
    s = ft.asof_before(state, ["horse_id"], df)
    out["interval_sum3"] = s["interval_sum3"]
    out["career_days"] = (df["race_date"] - s["first_date"]).dt.days.astype(float)

    s = ft.cum_before(ev, ["horse_id", "class_ord"], df, ["fig_sum", "fig_n"])
    out["h_fig_class_mean"] = s["fig_sum"] / s["fig_n"].replace(0, np.nan)


def build(df):
    out = pd.DataFrame(index=df.index)
    hist = df[df["is_hist"]]
    ev = hist.assign(n=1.0, fig_sum=hist["fig"].fillna(0.0), fig_n=hist["fig"].notna().astype(float))
    last_jockey = _horse_extra(df, ev, out)
    _connection_extra(df, ev, out, last_jockey)
    _pedigree_extra(df, ev, out)
    _shape_extra(df, out)
    _sameday_bias(df, out)
    _connection2(df, ev, out)
    _prev_level(df, out)
    _horse2(df, ev, out)
    _oikiri2(df, ev, out)
    _gate_pace2(df, ev, out)
    _condition2(df, ev, out)
    _day_adjusted(df, out)
    _horse_person(df, ev, out)
    _career(df, ev, out)
    return out[EXTRA_FEATURES]
