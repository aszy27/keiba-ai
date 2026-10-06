import itertools

import numpy as np

from model.exotic import combos, order2_probs, order3_probs, winning_combo


def test_probabilities_sum_to_one():
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    for lam in (1.0, 0.75):
        assert np.isclose(order3_probs(p, lam).sum(), 1.0)
        for kind in ("trio", "trifecta"):
            names, pr = combos([3, 1, 7, 2, 5, 8], p, kind, lam)
            assert np.isclose(pr.sum(), 1.0)
            assert len(set(names)) == len(names)


def test_harville_matches_formula():
    p = np.array([0.5, 0.3, 0.2])
    prob = order3_probs(p, 1.0)
    assert np.isclose(prob[0, 1, 2], 0.5 * 0.3 / 0.5 * 0.2 / 0.2)
    names, pr = combos([1, 2, 3], p, "trio", 1.0)
    assert list(names) == ["010203"] and np.isclose(pr[0], 1.0)


def test_trio_is_sum_of_orders():
    p = np.array([0.35, 0.3, 0.2, 0.15])
    nums = [4, 2, 9, 1]
    t_names, t_pr = combos(nums, p, "trio")
    o_names, o_pr = combos(nums, p, "trifecta")
    o = dict(zip(o_names, o_pr))
    for name, pr in zip(t_names, t_pr):
        legs = [name[i:i + 2] for i in (0, 2, 4)]
        assert np.isclose(pr, sum(o["".join(x)] for x in itertools.permutations(legs)))


def test_winning_combo():
    finish = [3, 1, 2, 4]
    nums = [5, 8, 2, 11]
    assert winning_combo(finish, nums, "trifecta") == "080205"
    assert winning_combo(finish, nums, "trio") == "020508"
    assert winning_combo([1, 2, 2, 4], nums, "trio") is None      # 2着同着


def test_two_leg_probabilities():
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    nums = [3, 1, 7, 2, 5, 8]
    for lam in (1.0, 0.75):
        # 馬単の確率は3連単を3着について足したものと同じ
        assert np.allclose(order2_probs(p, lam), order3_probs(p, lam).sum(axis=2))
        e_names, e_pr = combos(nums, p, "exacta", lam)
        q_names, q_pr = combos(nums, p, "quinella", lam)
        assert np.isclose(e_pr.sum(), 1.0) and np.isclose(q_pr.sum(), 1.0)
        assert len(e_names) == 30 and len(q_names) == 15
        e = dict(zip(e_names, e_pr))
        for name, pr in zip(q_names, q_pr):
            a, b = name[:2], name[2:]
            assert int(a) < int(b)
            assert np.isclose(pr, e[a + b] + e[b + a])


def test_two_leg_winning_combo():
    nums = [5, 8, 2, 11]
    assert winning_combo([3, 1, 2, 4], nums, "exacta") == "0802"
    assert winning_combo([3, 1, 2, 4], nums, "quinella") == "0208"
    assert winning_combo([1, 2, 3, 3], nums, "exacta") == "0508"     # 3着同着は関係ない
    assert winning_combo([1, 1, 3, 4], nums, "quinella") is None     # 1着同着
