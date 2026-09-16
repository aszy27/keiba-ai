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
GROUPS = {"馬の詳細実績": HORSE_EXTRA, "人": CONNECTION_EXTRA, "血統": PEDIGREE_EXTRA,
          "レースの形": SHAPE_EXTRA, "当日バイアス": SAMEDAY_EXTRA}
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


def build(df):
    out = pd.DataFrame(index=df.index)
    hist = df[df["is_hist"]]
    ev = hist.assign(n=1.0, fig_sum=hist["fig"].fillna(0.0), fig_n=hist["fig"].notna().astype(float))
    last_jockey = _horse_extra(df, ev, out)
    _connection_extra(df, ev, out, last_jockey)
    _pedigree_extra(df, ev, out)
    _shape_extra(df, out)
    _sameday_bias(df, out)
    return out[EXTRA_FEATURES]
