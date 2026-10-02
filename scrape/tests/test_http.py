import pytest

from scrape import common


class FakeResponse:
    def __init__(self, code, headers=None):
        self.status_code = code
        self.headers = headers or {}
        self.content = b""


class FakeSession:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        code = self.codes.pop(0)
        return FakeResponse(*code) if isinstance(code, tuple) else FakeResponse(code)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """時計・待ち時間・セッション・冷却期間のファイルを偽物にする"""
    clock = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(common, "_now", lambda: clock["t"])

    def sleep(sec):
        slept.append(sec)
        clock["t"] += sec
    monkeypatch.setattr(common, "_sleep", sleep)
    monkeypatch.setattr(common, "COOLDOWN_FILE", tmp_path / "cooldown.json")
    monkeypatch.setattr(common, "_last_request", {})
    monkeypatch.setattr(common, "_block_times", common.deque())

    def use(codes):
        s = FakeSession(codes)
        monkeypatch.setattr(common, "_session", s)
        return s
    return clock, slept, use


def test_requests_to_same_host_are_spaced(fake):
    clock, slept, use = fake
    use([200, 200])
    common.http_get("https://race.netkeiba.com/a")
    common.http_get("https://race.netkeiba.com/b")
    assert len(slept) == 1 and common.MIN_INTERVAL <= slept[0] <= 2 * common.MIN_INTERVAL


def test_429_waits_retry_after_then_succeeds(fake):
    clock, slept, use = fake
    s = use([(429, {"Retry-After": "30"}), 200])
    r, status = common.http_get("https://race.netkeiba.com/a")
    assert status == "ok" and s.calls == 2 and 30 in slept


def test_repeated_blocks_start_cooldown(fake):
    clock, slept, use = fake
    s = use([403, 403, 403, 200, 200])
    for _ in range(3):
        assert common.http_get("https://db.netkeiba.com/x")[1] == "block"
    assert common.cooldown_until() is not None
    calls = s.calls
    assert common.http_get("https://db.netkeiba.com/y") == (None, "cooldown")
    assert s.calls == calls                                   # 冷却期間中は取りに行かない
    r, status = common.http_get("https://race.netkeiba.com/api", honor_cooldown=False)
    assert status == "ok" and s.calls == calls + 1           # スナップショット用は冷却期間でも取る


def test_404_is_not_retried(fake):
    clock, slept, use = fake
    s = use([404])
    assert common.http_get("https://db.netkeiba.com/none") == (None, "not_found") and s.calls == 1
