import numpy as np
import pandas as pd
import pytest

from v2 import features as ft
from v2 import features_extra as fx
from v2.paths import table_path


def test_cum_before_excludes_same_day_and_future():
    ev = pd.DataFrame({"k": ["a", "a", "a", "b"], "race_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-05", "2024-01-03"]),
                       "n": [1.0, 1.0, 1.0, 1.0]})
    q = pd.DataFrame({"k": ["a", "a", "a", "b", "c"],
                      "race_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-06", "2024-01-10", "2024-01-10"])})
    assert ft.cum_before(ev, ["k"], q, ["n"])["n"].tolist() == [0, 1, 3, 1, 0]
    # 5日の窓は [日付 - 5日, 日付)。01-06 からは 01-01〜01-05 の3件、b の 01-10 からは 01-05〜01-09 で0件
    assert ft.cum_before(ev, ["k"], q, ["n"], window_days=5)["n"].tolist() == [0, 1, 3, 0, 0]


def test_asof_before_with_missing_keys_and_unsorted_query():
    state = pd.DataFrame({"k": ["a", "a"], "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]), "v": [1.0, 2.0]})
    q = pd.DataFrame({"k": ["a", None, "a"], "race_date": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-01-15"])},
                     index=[10, 11, 12])
    out = ft.asof_before(state, ["k"], q)
    assert out.index.tolist() == [10, 11, 12]
    assert out["v"].iloc[0] == 2.0 and np.isnan(out["v"].iloc[1]) and out["v"].iloc[2] == 1.0


def test_race_pace_and_position_drop():
    # 200m ラップ: 前半3F 12.0 平均、後半3F 11.5 平均 → 0.5（前半が遅い）
    assert ft._race_pace([12.0, 12.0, 12.0, 11.5, 11.5, 11.5], 200) == pytest.approx(0.5)
    # 端数距離（最初が100m）は最初の区間を除く
    assert ft._race_pace([6.5, 12.0, 12.0, 12.0, 11.5, 11.5, 11.5], 100) == pytest.approx(0.5)
    assert np.isnan(ft._race_pace(None, 200)) and np.isnan(ft._race_pace([12.0] * 5, 200))
    assert ft._max_position_drop("3-3-8-6") == 5 and ft._max_position_drop("5-4-3-2") == 0
    assert np.isnan(ft._max_position_drop(None))


# ---- 実データでのリークテスト（data/v2 の表が無ければスキップ） ----
needs_data = pytest.mark.skipif(not table_path("runners").exists(), reason="data/v2 の表が無い")
WINDOW = ("2023-09-01", "2024-01-31")
CUTOFF = pd.Timestamp("2023-12-24")   # 開催日
# 当日バイアスは「同じ日の先に終わったレース」を使うため、当日を丸ごと消すテストの対象外にする
NOT_SAMEDAY = [c for c in ft.FEATURES_TRIP + fx.EXTRA_FEATURES if c not in fx.SAMEDAY_DEPENDENT]


@pytest.fixture(scope="module")
def tables():
    t = ft.load_tables()
    races = t["races"][t["races"]["race_date"].between(*WINDOW)]
    t["races"] = races
    for name in ["runners", "training", "laps"]:
        t[name] = t[name][t[name]["race_id"].isin(races["race_id"])]
    return t


@pytest.fixture(scope="module")
def full(tables):
    return ft.build_features(tables)


def _blank(tables, race_ids):
    """指定したレースを「結果が未確定」の状態にした表を返す"""
    r = tables["runners"].copy()
    target = r["race_id"].isin(race_ids) & (r["status"] != "scratched")
    r.loc[target, ft.RESULT_COLS] = np.nan
    r.loc[target, "status"] = "entry"
    laps = tables["laps"][~tables["laps"]["race_id"].isin(race_ids)]
    return dict(tables, runners=r, laps=laps)


def _same(a, b, cols=None):
    a = a.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    b = b.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    cols = cols or [c for c in a.columns if c not in ft.LABEL_COLS]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False, check_categorical=False, rtol=1e-9)


@needs_data
def test_future_data_does_not_change_features(tables, full):
    keep = tables["races"]["race_date"] <= CUTOFF
    cut = dict(tables, races=tables["races"][keep])
    for name in ["runners", "laps"]:
        cut[name] = tables[name][tables[name]["race_id"].isin(cut["races"]["race_id"])]
    before = ft.build_features(cut)
    _same(before, full[full["race_date"] <= CUTOFF])


@needs_data
def test_same_day_results_do_not_change_features(tables, full):
    day_ids = tables["races"].loc[tables["races"]["race_date"] == CUTOFF, "race_id"]
    assert len(day_ids) > 0
    blanked = ft.build_features(_blank(tables, day_ids))
    _same(blanked[blanked["race_date"] == CUTOFF], full[full["race_date"] == CUTOFF], NOT_SAMEDAY)


@needs_data
def test_later_races_on_the_same_day_do_not_change_features(tables, full):
    """当日バイアスを含めて、同じ日の「後のレース」の結果は使っていないこと"""
    day = tables["races"][tables["races"]["race_date"] == CUTOFF]
    last = day.loc[day.groupby(["place", "surface"])["race_number"].idxmax(), "race_id"]
    earlier = day.loc[~day["race_id"].isin(last), "race_id"]
    blanked = ft.build_features(_blank(tables, last))
    _same(blanked[blanked["race_id"].isin(earlier)], full[full["race_id"].isin(earlier)])


@needs_data
def test_features_are_not_constant(full):
    cols = ft.FEATURES_TRIP + fx.EXTRA_FEATURES
    later = full.loc[full["race_date"] >= "2023-12-01", cols]
    constant = [c for c in later.columns if later[c].nunique(dropna=True) <= 1]
    assert constant == [], constant
