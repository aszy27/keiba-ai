# odds-exotic の --job: 年の書き方と、リクエスト数の上限を共有して前から順に取ること。1日の上限は起動し直しても合計で守ること
from scrape import odds


def test_parse_job():
    assert odds.parse_job("2021-2025:7,8") == ([2021, 2022, 2023, 2024, 2025], [7, 8])
    assert odds.parse_job("2020:8") == ([2020], [8])
    assert odds.parse_job("2020,2022-2023:4,6") == ([2020, 2022, 2023], [4, 6])


def _fake(calls, results):
    def run(years, types, limit, sleep, max_requests, on_request=None):
        calls.append((years, types, max_requests))
        n, reason = results.pop(0)
        for _ in range(n if on_request else 0):
            on_request()
        return n, reason
    return run


def test_jobs_share_budget_and_stop(monkeypatch):
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(1200, None), (1800, "上限"), (0, None)]))
    odds.run_exotic_jobs([([2021], [7]), ([2020], [8]), ([2020], [4])], max_requests=3000)
    assert calls == [([2021], [7], 3000), ([2020], [8], 1800)]     # 2つ目が上限で止まったので3つ目は始めない


def test_jobs_no_budget_left(monkeypatch):
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(3000, None)]))
    odds.run_exotic_jobs([([2021], [7]), ([2020], [4])], max_requests=3000)
    assert len(calls) == 1


def test_daily_budget_survives_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(odds, "daily_path", lambda: tmp_path / "daily_state.json")
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(1000, "上限"), (2000, "上限")]))
    jobs = [([2021], [8])]
    odds.run_exotic_jobs(jobs, daily_max=3000)          # 1,000回でPCを切った
    odds.run_exotic_jobs(jobs, daily_max=3000)          # 起動し直すと残りの2,000回だけ
    odds.run_exotic_jobs(jobs, daily_max=3000)          # 使い切ったら取らない
    assert [c[2] for c in calls] == [3000, 2000]


def test_daily_stop_on_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(odds, "daily_path", lambda: tmp_path / "daily_state.json")
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(50, "制限"), (0, None)]))
    odds.run_exotic_jobs([([2021], [8])], daily_max=3000)
    odds.run_exotic_jobs([([2021], [8])], daily_max=3000)   # 制限で止めた日は起動し直しても取らない
    assert len(calls) == 1


def test_daily_resets_next_day(monkeypatch, tmp_path):
    monkeypatch.setattr(odds, "daily_path", lambda: tmp_path / "daily_state.json")
    odds.save_daily({"date": "20000101", "used": 3000, "stopped": "制限"})
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(10, None)]))
    odds.run_exotic_jobs([([2021], [8])], daily_max=3000)
    assert calls[0][2] == 3000
