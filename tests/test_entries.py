from pathlib import Path

from bs4 import BeautifulSoup

from live.entries import parse_shutuba

FIX = Path(__file__).parent / "fixtures"


def _parse(race_id):
    soup = BeautifulSoup((FIX / f"shutuba_{race_id}.html").read_text(encoding="utf-8"), "html.parser")
    return parse_shutuba(soup, race_id)


def test_settled_page():
    """枠順・馬体重・天候・馬場が出たあとのページ（2026-09-06 中山11R、11頭）"""
    rows, warns = _parse("202606040211")
    assert len(rows) == 11 and not any(r["cancelled"] for r in rows)
    first = rows[0]
    assert (first["bracket"], first["horse_number"], first["horse_id"]) == ("1", "1", first["horse_id"])
    assert first["horse_id"] and first["jockey_id"] and first["trainer_id"]
    assert (first["gender"], first["age"], first["burden"]) == ("牝", "3", "55.0")
    assert (first["weight"], first["weight_diff"]) == ("442", "0")
    assert first["type"] == "芝" and first["race_date"] == "2026-09-06" and first["race_time"]
    assert first["condition"] in ("良", "稍", "重", "不良") and first["weather"]
    assert warns == []


def test_jump_race_by_name():
    """障害レースは距離の欄に「ダ3100m」としか書かれないので、レース名で判定する"""
    rows, _ = _parse("202605040104")
    assert rows[0]["type"] == "障害" and rows[0]["length"] == "3100"


def test_before_draw():
    """枠順が出る前のページ: 馬番が無いことを警告し、既定値で埋めない。レース名はタイトルから（格付け付き）"""
    rows, warns = _parse("202608040111")
    assert all(r["horse_number"] is None for r in rows)
    assert any("枠順" in w for w in warns)
    assert rows[0]["race_name"].endswith("(L)")
    assert rows[0]["condition"] is None and rows[0]["weather"] is None
