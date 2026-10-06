# odds-exotic の --job: 年の書き方と、リクエスト数の上限を共有して前から順に取ること
from scrape import odds


def test_parse_job():
    assert odds.parse_job("2021-2025:7,8") == ([2021, 2022, 2023, 2024, 2025], [7, 8])
    assert odds.parse_job("2020:8") == ([2020], [8])
    assert odds.parse_job("2020,2022-2023:4,6") == ([2020, 2022, 2023], [4, 6])


def _fake(calls, results):
    def run(years, types, limit, sleep, max_requests):
        calls.append((years, types, max_requests))
        return results.pop(0)
    return run


def test_jobs_share_budget_and_stop(monkeypatch):
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(1200, True), (1800, False), (0, True)]))
    odds.run_exotic_jobs([([2021], [7]), ([2020], [8]), ([2020], [4])], max_requests=3000)
    assert calls == [([2021], [7], 3000), ([2020], [8], 1800)]     # 2つ目が上限で止まったので3つ目は始めない


def test_jobs_no_budget_left(monkeypatch):
    calls = []
    monkeypatch.setattr(odds, "run_exotic", _fake(calls, [(3000, True)]))
    odds.run_exotic_jobs([([2021], [7]), ([2020], [4])], max_requests=3000)
    assert len(calls) == 1
