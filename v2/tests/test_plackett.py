import numpy as np

from v2.plackett import PLGroups
from v2.softmax import RaceGroups

RACES = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3])
FINISH = np.array([2, 1, 3, 4, 3, 5, 1, np.nan, 2, 1, 2, 2])   # レース2に競走中止、レース3は2着同着（3着なし）


def test_k1_matches_softmax():
    u = np.random.default_rng(0).normal(size=len(RACES))
    win = (FINISH == 1).astype(float)
    ll_sm, p_sm = RaceGroups(RACES, win).ll(u)
    pl = PLGroups(RACES, FINISH, k=1)
    np.testing.assert_allclose(pl.ll(u), ll_sm)
    g, h = pl.grad_hess(u)
    np.testing.assert_allclose(g, p_sm - win)
    np.testing.assert_allclose(h, p_sm * (1 - p_sm))


def test_gradient_matches_numerical():
    u = np.random.default_rng(1).normal(size=len(RACES))
    pl = PLGroups(RACES, FINISH, k=3, lam=0.75, stage_w=[1.0, 0.5, 0.5])
    g, _ = pl.grad_hess(u)
    eps = 1e-6
    num = np.array([-(pl.ll(u + eps * np.eye(len(u))[i]).sum() - pl.ll(u - eps * np.eye(len(u))[i]).sum()) / (2 * eps)
                    for i in range(len(u))])
    np.testing.assert_allclose(g, num, atol=1e-6)


def test_tied_and_missing_stages_are_skipped():
    pl = PLGroups(RACES, FINISH, k=3)
    ok2, ok3 = pl.stages[1][4], pl.stages[2][4]
    assert list(ok2) == [True, True, False]    # レース3は2着同着 → 2着の段を使わない
    assert list(ok3) == [True, True, False]    # 3着がいない
    # 競走中止の馬は3着の段まで候補に残る
    assert pl.stages[2][0][7]
