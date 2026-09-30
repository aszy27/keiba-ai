import numpy as np
import pandas as pd

from v2.model.past_market import PAST_COLS, from_runs


def _runs():
    rng = np.random.default_rng(0)
    rows = []
    for i, day in enumerate(pd.date_range("2025-01-05", periods=12, freq="7D")):
        horses = rng.choice([f"h{k}" for k in range(10)], size=6, replace=False)
        for pos, h in enumerate(horses, start=1):
            rows.append(dict(race_id=f"r{i:02d}", horse_id=h, race_date=day, finish_pos=pos,
                             win_odds=float(rng.uniform(1.5, 50))))
    return pd.DataFrame(rows)


def test_future_races_do_not_change_values():
    runs = _runs()
    cut = runs["race_date"].sort_values().unique()[7]
    full = from_runs(runs).set_index(["race_id", "horse_id"])
    past_only = from_runs(runs[runs["race_date"] <= cut]).set_index(["race_id", "horse_id"])
    pd.testing.assert_frame_equal(full.loc[past_only.index, PAST_COLS], past_only[PAST_COLS])


def test_own_race_result_does_not_leak():
    runs = _runs()
    last = runs["race_id"].max()
    changed = runs.copy()
    m = changed["race_id"] == last
    changed.loc[m, "finish_pos"] = changed.loc[m, "finish_pos"].to_numpy()[::-1]
    changed.loc[m, "win_odds"] = 99.0
    a = from_runs(runs).set_index(["race_id", "horse_id"]).loc[last, PAST_COLS]
    b = from_runs(changed).set_index(["race_id", "horse_id"]).loc[last, PAST_COLS]
    pd.testing.assert_frame_equal(a, b)


def test_previous_values():
    runs = pd.DataFrame(dict(race_id=["a", "a", "b", "b"], horse_id=["h1", "h2", "h1", "h2"],
                             race_date=pd.to_datetime(["2025-01-01"] * 2 + ["2025-02-01"] * 2),
                             finish_pos=[2, 1, 1, 2], win_odds=[2.0, 4.0, 3.0, 3.0]))
    out = from_runs(runs).set_index(["race_id", "horse_id"])
    assert np.isnan(out.loc[("a", "h1"), "past_x1"])
    assert np.isclose(out.loc[("b", "h1"), "past_x1"], np.log((1 / 2) / (1 / 2 + 1 / 4)))
    assert out.loc[("b", "h1"), "past_gap1"] == 1 - 2       # 前走は1番人気で2着
    assert out.loc[("b", "h2"), "past_gap1"] == 2 - 1       # 前走は2番人気で1着
