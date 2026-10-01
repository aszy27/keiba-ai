import numpy as np
import pandas as pd
import pytest

from prep import normalize as nz


@pytest.mark.parametrize("broken, expected", [
    ("3КаЬЄОЁЭј", "3歳未勝利"),
    ("3КаАЪОх500ЫќВМ", "3歳以上500万下"),
    ("3ŗŠĢ¤¾”Ķų", "3歳未勝利"),
    ("4Κ–Α\xa0Ψε1ΨΓΞ·ΞιΞΙ", "4歳以上1勝クラス"),
    ("Тш67ВѓЅщЅИЅЊNIKKEIОо(GIII)", "第67回ラジオNIKKEI賞(GIII)"),
    ("ΑώΨκΩΖ\xa0Ι«’(2ΨΓ)", "茨城新聞杯(2勝)"),
])
def test_fix_mojibake(broken, expected):
    assert nz.fix_mojibake(broken) == (expected, "fixed")


def test_fix_mojibake_leaves_normal_names():
    assert nz.fix_mojibake("第67回ラジオNIKKEI賞(GIII)") == ("第67回ラジオNIKKEI賞(GIII)", "ok")
    assert nz.fix_mojibake(np.nan)[1] == "ok"


@pytest.mark.parametrize("name, col, expected", [
    ("第67回ラジオNIKKEI賞(GIII)", None, "G3"),
    ("第22回阪神スプリングJ(JGII)", None, "G2"),
    ("第70回有馬記念(GI)", None, "G1"),
    ("有馬記念(ＧⅠ)", None, "G1"),
    ("コーラルステークス(L)", None, "L"),
    ("第1回いちょうステークス(重賞)", None, "G3"),
    ("3歳未勝利", "一般", None),
    ("文字化け", "G2", "G2"),
])
def test_parse_grade(name, col, expected):
    assert nz.parse_grade(name, col) == expected


@pytest.mark.parametrize("name, expected", [
    ("2歳新馬", "新馬"),
    ("3歳未勝利", "未勝利"),
    ("障害3歳以上未勝利", "未勝利"),
    ("3歳以上500万下", "1勝"),
    ("4歳以上1000万下", "2勝"),
    ("3歳以上1600万下", "3勝"),
    ("磐梯山特別(1勝)", "1勝"),
    ("4歳以上2勝クラス", "2勝"),
    ("障害4歳以上オープン", "OP"),
    ("清秋ジャンプS(OP)", "OP"),
    ("春風S", None),
])
def test_parse_class(name, expected):
    assert nz.parse_class(name) == expected


def test_parse_class_prefers_grade():
    assert nz.parse_class("第70回有馬記念(GI)", "G1") == "G1"


def test_class_from_prize():
    ref = {"未勝利": 590, "1勝": 820, "2勝": 1604, "3勝": 1902, "OP": 2234}
    assert nz.class_from_prize(1903.6, ref) == "3勝"
    assert nz.class_from_prize(0, ref) is None
    assert nz.class_from_prize(800, {}) is None


def test_pad_id():
    s = pd.Series(["1170", "01170", "386", "5339.0", None])
    assert nz.pad_id(s).tolist()[:4] == ["01170", "01170", "00386", "05339"]
    assert pd.isna(nz.pad_id(s).iloc[4])


def test_corner_positions_and_status():
    passage = pd.Series(["5-5-3-3", "12", None, None], dtype="string")
    first, last = nz.corner_positions(passage)
    assert first.tolist()[:2] == [5, 12] and last.tolist()[:2] == [3, 12]
    finish = pd.Series([1, np.nan, np.nan, 2])
    assert nz.runner_status(finish, passage).tolist() == ["finished", "dnf", "scratched", "finished"]


def test_normalize_going_and_payouts():
    assert nz.normalize_going(pd.Series(["稍重", "稍", "良", "x"])).tolist()[:3] == ["稍", "稍", "良"]
    assert nz.parse_payouts(pd.Series(["170|260|800", np.nan, ""])).tolist() == [[170, 260, 800], None, None]
