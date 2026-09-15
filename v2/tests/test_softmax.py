import numpy as np
import pytest

from v2.softmax import RaceGroups, bootstrap_ci, fit_logit


def _simulate(n_races=3000, n_horses=10, beta=(1.5, -0.7), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_races * n_horses, len(beta)))
    u = X @ np.array(beta)
    race_ids = np.repeat(np.arange(n_races), n_horses)
    win = np.zeros(len(u))
    for r in range(n_races):
        s = slice(r * n_horses, (r + 1) * n_horses)
        p = np.exp(u[s] - u[s].max())
        win[r * n_horses + rng.choice(n_horses, p=p / p.sum())] = 1.0
    return X, RaceGroups(race_ids, win)


def test_fit_logit_recovers_coefficients():
    X, groups = _simulate()
    assert np.allclose(fit_logit(X, groups), [1.5, -0.7], atol=0.1)


def test_probabilities_sum_to_one_per_race():
    X, groups = _simulate(n_races=50)
    _, p = groups.ll(X[:, 0])
    assert np.allclose(np.add.reduceat(p, groups.starts), 1.0)


def test_uniform_scores_give_log_one_over_n():
    groups = RaceGroups(np.repeat([0, 1], [4, 8]), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ll, _ = groups.ll(np.zeros(12))
    assert np.allclose(ll, [-np.log(4), -np.log(8)])


def test_rejects_races_without_exactly_one_winner():
    with pytest.raises(ValueError):
        RaceGroups([0, 0, 1, 1], [1, 1, 0, 1])


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).normal(0.3, 1.0, 2000)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.2
