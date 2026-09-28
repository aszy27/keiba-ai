# scrape/tests/test_results.py
# scrape/results.py が keibascraper 3.1.4 と同じ値・型を返すことの回帰テスト。
# fixtures/ には db.netkeiba の結果ページ（2026-09-29 取得）と、同じページを keibascraper でパースした結果を保存してある。
#   201201010101: 取消・除外あり / 201204010111: 1着同着 / 202606040811: 最近のページ
#   202606040899: 存在しないレース番号（db.netkeiba はその日の1Rを返す）
# 実行: python -m pytest scrape/tests
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from scrape import results as R

FIX = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIX / "keibascraper_expected.json").read_text(encoding="utf-8"))


def page(race_id):
    with gzip.open(FIX / f"{race_id}.html.gz", "rt", encoding="utf-8") as f:
        return f.read()


def norm(records):
    return [[[k, type(v).__name__, repr(v)] for k, v in r.items()] for r in records]


def test_same_as_keibascraper():
    # 着差は「直前に読んだ1着のタイム」を引き継ぐので、保存したときと同じ順に読む
    R._winner_time = 0.0
    for race_id, exp in EXPECTED.items():
        info, rows = R.parse_page(page(race_id), race_id)
        assert norm(info) == exp["race"], race_id
        assert norm(rows) == exp["result"], race_id


def test_shown_race_id_detects_wrong_page():
    assert R.shown_race_id(page("202606040811")) == "202606040811"
    assert R.shown_race_id(page("202606040899")) == "202606040801"


def test_no_race_page():
    with pytest.raises(R.NoRace):
        R.parse_page("<html><body><div id='page'>該当するデータはありません</div></body></html>", "202699999999")


def test_phantom_of():
    df = pd.DataFrame({"race_id": ["202601010102"] * 2, "race_date": ["2026-01-01"] * 2, "horse_id": ["a", "b"]})
    day = {"2026-01-01": {"202601010101": frozenset({"a", "b"}), "202601010103": frozenset({"a", "c"})}}
    assert R.phantom_of(df, day) == "202601010101"
    assert R.phantom_of(df, {"2026-01-01": {"202601010103": frozenset({"a", "c"})}}) is None
