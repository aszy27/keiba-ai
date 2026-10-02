import itertools

import numpy as np

from model.exotic import combos, order3_probs, winning_combo


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
