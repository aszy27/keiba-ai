from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from scrape import jra_odds, snapshot


class Clock:
    t = datetime(2026, 10, 31, 9, 0)


class FakeDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return Clock.t


def _run(monkeypatch, tmp_path, start):
    """発走 10:00・10:05 の2レースを、偽の時計で最後まで回す。取った順番を返す"""
    Clock.t = start
    calls = []
    sched = [("r1", datetime(2026, 10, 31, 10, 0)), ("r2", datetime(2026, 10, 31, 10, 5))]

    def sleep(sec):
        Clock.t += timedelta(seconds=sec)

    def fetch_odds(rid, date=None):
        calls.append(("win", rid, Clock.t))
        return [{"race_id": rid, "horse_number": 1, "win_odds": 2.0}], "jra"

    def fetch_exotic(rid, date, bet, honor_cooldown=False):
        calls.append((bet, rid, Clock.t))
        return [{"race_id": rid, "combo": "010203", "odds": 10.0, "official_datetime": ""}], "jra"

    monkeypatch.setattr(snapshot, "datetime", FakeDatetime)
    monkeypatch.setattr(snapshot.time, "sleep", sleep)
    monkeypatch.setattr(snapshot, "race_schedule", lambda d: sched)
    monkeypatch.setattr(snapshot, "fetch_odds", fetch_odds)
    monkeypatch.setattr(jra_odds, "fetch_exotic", fetch_exotic)
    snapshot.run(SimpleNamespace(date="20261031", minutes=None, now=False, out_dir=tmp_path, no_exotic=False))
    return calls


def test_exotic_taken_at_the_same_points_after_win(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, datetime(2026, 10, 31, 9, 0))
    win = pd.read_csv(tmp_path / "20261031.csv", dtype={"race_id": str})
    want = {(r, m) for r in ("r1", "r2") for m in snapshot.LADDER}
    assert set(zip(win["race_id"], win["minutes_before"])) == want
    for bet in snapshot.EXOTIC_BETS:
        x = pd.read_csv(snapshot.exotic_path(tmp_path, "20261031", bet), dtype={"race_id": str, "combo": str})
        assert set(zip(x["race_id"], x["minutes_before"])) == want and set(x["combo"]) == {"010203"}
    assert list(win.columns) == snapshot.COLS                     # 単勝・複勝のファイルの形式は変えない
    for rid in ("r1", "r2"):                                       # 単勝・複勝が先、3連複・3連単、馬連・馬単の順
        kinds = [k for k, r, _ in calls if r == rid]
        assert kinds == ["win", "trio", "trifecta", "quinella", "exacta"] * 3


def test_restart_does_not_refetch_exotic(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, datetime(2026, 10, 31, 9, 0))
    calls = _run(monkeypatch, tmp_path, datetime(2026, 10, 31, 9, 58))   # 起動し直し（3分前の窓の中）
    assert calls == []


def test_restart_after_adding_bets_fetches_only_new_ones(monkeypatch, tmp_path):
    """3連複・3連単だけ取った日に（馬連・馬単を足したコードで）起動し直すと、足りない券種だけ取る"""
    _run(monkeypatch, tmp_path, datetime(2026, 10, 31, 9, 0))
    for bet in ("quinella", "exacta"):
        snapshot.exotic_path(tmp_path, "20261031", bet).unlink()
    calls = _run(monkeypatch, tmp_path, datetime(2026, 10, 31, 9, 58))
    # 9:58 に r1（3分前）と r2（30・10分前をまとめて1回）、10:02 に r2（3分前）。単勝・複勝は取り直さない
    assert [(k, r) for k, r, _ in calls] == [("quinella", "r1"), ("exacta", "r1"), ("quinella", "r2"), ("exacta", "r2"),
                                             ("quinella", "r2"), ("exacta", "r2")]
    x = pd.read_csv(snapshot.exotic_path(tmp_path, "20261031", "trio"), dtype={"race_id": str})
    assert len(x) == 6                                                    # 3連複は取り直していない（2レース × 3時点）
