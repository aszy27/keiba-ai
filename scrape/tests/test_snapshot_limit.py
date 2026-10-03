from datetime import datetime, timedelta

import pandas as pd

from scrape.snapshot import LIMIT_PAUSE_MIN, LIMIT_STREAK, LimitBackoff, load_done


def test_backoff_after_streak_and_recovery(tmp_path):
    b = LimitBackoff(tmp_path / "limit_state.json")
    t = datetime(2026, 10, 3, 10, 0)
    for i in range(LIMIT_STREAK - 1):
        b.record("limit", t)
        assert b.allow(t)                                      # まだ控えない
    b.record("limit", t)
    assert not b.allow(t + timedelta(minutes=1))               # 控える
    probe = t + timedelta(minutes=LIMIT_PAUSE_MIN)
    assert b.allow(probe)                                      # 15分後に1回試す
    b.record("limit", probe)
    assert not b.allow(probe + timedelta(minutes=1))           # 試しも limit ならまた控える
    later = probe + timedelta(minutes=LIMIT_PAUSE_MIN)
    b.record("middle", later)
    assert b.allow(later) and not (tmp_path / "limit_state.json").exists()   # 解けたら元どおり


def test_state_survives_restart(tmp_path):
    path = tmp_path / "limit_state.json"
    t = datetime(2026, 10, 3, 10, 0)
    b = LimitBackoff(path)
    for _ in range(LIMIT_STREAK):
        b.record("limit", t)
    assert not LimitBackoff(path).allow(t + timedelta(minutes=5))   # 起動し直しても控える


def test_load_done_ignores_limit_rows(tmp_path):
    path = tmp_path / "20261003.csv"
    pd.DataFrame([dict(race_id="r1", minutes_before=3, api_status="middle"),
                  dict(race_id="r1", minutes_before=10, api_status="limit"),
                  dict(race_id="r2", minutes_before=3, api_status=" limit ")]).to_csv(path, index=False)
    assert load_done(path) == {("r1", 3)}
