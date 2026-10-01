import pandas as pd

from predict import load_snapshots


def test_snapshot_priority_and_filters(tmp_path):
    rows = []
    # レース r1: 3分前は発走後に取れてしまった（使わない）→ 10分前を使う
    for hn in (1, 2):
        rows.append(dict(race_id="r1", horse_number=hn, win_odds=3.0, minutes_before=3, seconds_to_post=-20, api_status="middle"))
        rows.append(dict(race_id="r1", horse_number=hn, win_odds=4.0, minutes_before=10, seconds_to_post=590, api_status="middle"))
    # レース r2: 3分前は予想オッズ（使わない）→ 30分前を使う
    for hn in (1, 2):
        rows.append(dict(race_id="r2", horse_number=hn, win_odds=5.0, minutes_before=3, seconds_to_post=170, api_status="yoso"))
        rows.append(dict(race_id="r2", horse_number=hn, win_odds=6.0, minutes_before=30, seconds_to_post=1790, api_status="middle"))
    # レース r3: 3分前が正常に取れている
    for hn in (1, 2):
        rows.append(dict(race_id="r3", horse_number=hn, win_odds=7.0, minutes_before=3, seconds_to_post=175, api_status="middle"))
    pd.DataFrame(rows).assign(place_odds_min=1.1, place_odds_max=1.5).to_csv(tmp_path / "20261003.csv", index=False)
    s = load_snapshots(tmp_path).groupby("race_id")["minutes_used"].first().to_dict()
    assert s == {"r1": 10, "r2": 30, "r3": 3}
