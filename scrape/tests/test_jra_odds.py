import gzip
from pathlib import Path

from scrape import jra_odds
from scrape.jra_odds import meeting_links, parse_exotic, parse_odds, race_links

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


def _gz(name):
    return gzip.decompress((FIX / name).read_bytes()).decode("utf-8")


def test_parse_trio_all_combinations():
    """東京11R 2026-10-25（16頭）の3連複の全通り。netkeiba の確定オッズと560通りすべて一致することを確認したページ"""
    rows, status = parse_exotic(_gz("jra_trio_202605040911.html.gz"), "202605040911", "trio")
    assert status == "jra_final" and len(rows) == 560 and len({r["combo"] for r in rows}) == 560
    odds = {r["combo"]: r["odds"] for r in rows}
    assert odds["010204"] == 28.1 and odds["010203"] == 169.6                  # 表の見出し「1-2」の行 4・3
    assert all(r["combo"][:2] < r["combo"][2:4] < r["combo"][4:] for r in rows)   # 馬番の昇順（API と同じ表記）


def test_parse_trifecta_all_combinations():
    """同じレースの3連単の全通り（3,360通り、netkeiba の確定オッズとすべて一致）"""
    rows, status = parse_exotic(_gz("jra_trifecta_202605040911.html.gz"), "202605040911", "trifecta")
    assert status == "jra_final" and len(rows) == 3360
    odds = {r["combo"]: r["odds"] for r in rows}
    assert odds["010203"] == 992.2 and odds["010204"] == 225.0 and "010201" not in odds   # 1着 1・2着 2 の表。同じ馬は空欄


def test_parse_quinella_all_combinations():
    """東京11R 2026-11-01（17頭）の馬連の全通り。netkeiba の確定オッズと136通りすべて一致することを確認したページ"""
    rows, status = parse_exotic(_gz("jra_quinella_202605041111.html.gz"), "202605041111", "quinella")
    assert status == "jra_final" and len(rows) == 136 and len({r["combo"] for r in rows}) == 136
    odds = {r["combo"]: r["odds"] for r in rows}
    assert odds["0102"] == 644.0 and odds["0103"] == 12.4 and odds["0304"] == 301.5   # 表の見出しが1頭目、行が2頭目
    assert all(r["combo"][:2] < r["combo"][2:] for r in rows)                         # 馬番の昇順（API と同じ表記）


def test_parse_exacta_all_combinations():
    """同じレースの馬単の全通り（272通り、netkeiba の確定オッズとすべて一致）"""
    rows, status = parse_exotic(_gz("jra_exacta_202605041111.html.gz"), "202605041111", "exacta")
    assert status == "jra_final" and len(rows) == 272
    odds = {r["combo"]: r["odds"] for r in rows}
    assert odds["0102"] == 700.1 and odds["0103"] == 17.5 and odds["0301"] == 34.4 and "0303" not in odds  # 見出しが1着


def test_harvest_quinella_exacta_links():
    jra_odds._exotic_cnames.clear()
    jra_odds.harvest_exotic(_gz("jra_quinella_202605041111.html.gz"))
    jra_odds.harvest_exotic(_gz("jra_exacta_202605041111.html.gz"))
    assert jra_odds._exotic_cnames[("202605041101", "quinella")].startswith("pw154ou1005202604110120261101Z/")
    assert jra_odds._exotic_cnames[("202605041101", "exacta")].startswith("pw156ou1005202604110120261101Z/")
    assert ("202605041111", "trio") in jra_odds._exotic_cnames                 # 馬連のページにも3連複のリンクがある


def test_harvest_exotic_links_for_the_whole_day():
    jra_odds._exotic_cnames.clear()
    jra_odds.harvest_exotic(_gz("jra_trio_202605040911.html.gz"))
    assert jra_odds._exotic_cnames[("202605040901", "trio")].startswith("pw157ou1005202604090120261025Z99/")
    assert ("202605040912", "trio") in jra_odds._exotic_cnames
