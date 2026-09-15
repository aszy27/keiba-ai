import pandas as pd

from v2.checks import check_odds, check_payouts, check_runners


def _races(ids, dates):
    return pd.DataFrame({"race_id": pd.array(ids, dtype="string"), "race_date": pd.to_datetime(dates),
                         "surface": "芝", "distance": 1600, "n_finishers": 3})


def _runners(race_id, horses, finish=(1, 2, 3)):
    return pd.DataFrame({
        "race_id": pd.array([race_id] * len(horses), dtype="string"),
        "horse_id": pd.array(horses, dtype="string"),
        "horse_number": pd.array(range(1, len(horses) + 1), dtype="Int8"),
        "bracket": pd.array(range(1, len(horses) + 1), dtype="Int8"),
        "jockey_id": pd.array(["01170"] * len(horses), dtype="string"),
        "trainer_id": pd.array(["00438"] * len(horses), dtype="string"),
        "finish_pos": pd.array(list(finish), dtype="Int8"),
        "status": pd.array(["finished"] * len(horses), dtype="string"),
        "weight": 470.0, "time_sec": 96.0, "last_3f": 34.0, "burden": 55.0,
    })


def _names(issues, level="ERROR"):
    return {i.name for i in issues if i.level == level}


def test_detects_phantom_race():
    races = _races(["202601010101", "202601010102"], ["2026-01-05"] * 2)
    horses = ["2020100001", "2020100002", "2020100003"]
    runners = pd.concat([_runners("202601010101", horses), _runners("202601010102", horses)], ignore_index=True)
    names = _names(check_runners(runners, races))
    assert "runners: 同じ馬が同じ日に2回出走" in names
    assert "runners: 同じ日に出走馬の組み合わせが同一のレース" in names


def test_clean_runners_have_no_errors():
    races = _races(["202601010101"], ["2026-01-05"])
    runners = _runners("202601010101", ["2020100001", "2020100002", "2020100003"])
    assert _names(check_runners(runners, races)) == set()


def test_payout_column_shift_is_error_when_frequent():
    races = _races([f"2026010101{i:02d}" for i in range(1, 11)], ["2026-01-05"] * 10)
    p = pd.DataFrame({"race_id": races["race_id"], "win": [[300]] * 10, "place": [[150, 200, 300]] * 10,
                      "quinella": [[1000]] * 10, "exacta": [[2000]] * 10,
                      "trio": [[5000]] * 5 + [[3000]] * 5, "trifecta": [[4000]] * 10})
    issues = check_payouts(p, races)
    assert "payouts: 3連複 > 3連単" in _names(issues)


def test_winner_odds_mismatch():
    races = _races(["202601010101"], ["2026-01-05"])
    runners = _runners("202601010101", ["2020100001", "2020100002", "2020100003"])
    odds = pd.DataFrame({"race_id": runners["race_id"], "horse_number": runners["horse_number"],
                         "win_odds": [2.5, 4.0, 9.0], "odds_status": "ok"})
    good = pd.DataFrame({"race_id": pd.array(["202601010101"], dtype="string"), "win": [[250]]})
    bad = good.assign(win=[[400]])
    assert _names(check_odds(odds, runners, good, races)) == set()
    assert "odds: 1着馬のオッズ×100 と単勝払戻が違う" in _names(check_odds(odds, runners, bad, races))
