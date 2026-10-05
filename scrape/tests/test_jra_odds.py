from pathlib import Path

from scrape.jra_odds import meeting_links, parse_odds, race_links

FIX = Path(__file__).parent / "fixtures"


def test_list_page_to_meetings():
    html = (FIX / "jra_odds_list_20261005.html").read_text(encoding="utf-8")
    m = meeting_links(html, "20261004")
    assert len(m) == 2 and all(c.startswith("pw15orl10") and "20261004/" in c for c in m)
    assert meeting_links(html, "20260101") == []


def test_meeting_page_to_race_ids():
    html = (FIX / "jra_odds_meeting_20261004_tokyo.html").read_text(encoding="utf-8")
    links = race_links(html, "20261004")
    assert sorted(links) == [f"2026050402{r:02d}" for r in range(1, 13)]     # netkeiba のレース ID（年・場・回・日・R）
    assert links["202605040211"].startswith("pw151ou1005202604021120261004Z/")


def test_parse_final_odds():
    """毎日王冠 2026-10-04（17頭）。netkeiba の確定オッズと全頭一致することを確認したページ"""
    html = (FIX / "jra_odds_202605040211.html").read_bytes().decode("cp932", errors="replace")
    rows, status = parse_odds(html, "202605040211")
    assert status == "jra_final" and len(rows) == 17
    r1 = next(r for r in rows if r["horse_number"] == 1)
    assert (r1["win_odds"], r1["place_odds_min"], r1["place_odds_max"]) == (11.2, 2.7, 4.3)
    assert next(r for r in rows if r["horse_number"] == 2)["popularity"] == 1    # 2.9倍が1番人気
    assert sum(1 / r["win_odds"] for r in rows) > 1.18                            # 確定オッズの「1/オッズの合計」の範囲


def test_presale_page_without_numbers():
    html = '<table class="tanpuku"><tr><td class="num">1</td><td class="odds_tan">----</td><td class="odds_fuku">----</td></tr></table>'
    assert parse_odds(html, "x") == ([], "jra_presale")
