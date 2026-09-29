import numpy as np
import pytest

from sentport.optimize import black_litterman, ledoit_wolf, mean_variance, min_variance


def test_bl_hand_computed_single_view():
    # Two assets, one absolute view on asset 0.
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    delta, tau = 2.0, 0.1
    pi = delta * cov @ w  # [0.05, 0.10]
    assert np.allclose(pi, [0.05, 0.10])
    q = np.array([0.10, 0.0])
    conf = np.array([0.5, 0.0])  # view on asset 0 only; Omega = tau*cov00*(1-c)/c = 0.004
    # tau*Sigma = [[0.004, 0.001],[0.001, 0.009]]; P tau Sigma P' + Omega = 0.004 + 0.004 = 0.008
    # gain = tau*Sigma P' / 0.008 = [0.5, 0.125]; residual q - pi0 = 0.05
    expected = pi + np.array([0.5, 0.125]) * 0.05
    mu, prior = black_litterman(cov, w, q, conf, delta, tau)
    assert np.allclose(prior, pi)
    assert np.allclose(mu, expected)
    assert np.allclose(expected, [0.075, 0.10625])


def test_bl_zero_confidence_returns_prior():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 4))
    cov = np.cov(a.T)
    w = np.full(4, 0.25)
    mu, pi = black_litterman(cov, w, rng.normal(size=4), np.zeros(4), 2.5, 0.05)
    assert np.array_equal(mu, pi)


def test_bl_full_confidence_pins_view_when_single_asset():
    cov = np.array([[0.04]])
    mu, _ = black_litterman(cov, np.array([1.0]), np.array([0.2]), np.array([1.0]), 2.0, 0.05)
    assert mu[0] == pytest.approx(0.2)


def test_bl_moves_toward_view_monotonically_in_confidence():
    cov = np.diag([0.04, 0.04])
    w = np.array([0.5, 0.5])
    q = np.array([0.30, 0.0])
    m = [black_litterman(cov, w, q, np.array([c, 0.0]), 2.0, 0.05)[0][0] for c in (0.1, 0.5, 0.9)]
    assert m[0] < m[1] < m[2] <= 0.30


def _random_cov(n=8, t=120, seed=1):
    r = np.random.default_rng(seed).normal(0, 0.01, (t, n))
    return ledoit_wolf(r)[0], r


@pytest.mark.parametrize("cap", [0.15, 0.2, 0.5, 1.0])
def test_min_variance_respects_constraints(cap):
    cov, _ = _random_cov()
    w = min_variance(cov, cap)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= -1e-10).all() and (w <= cap + 1e-8).all()


def test_mean_variance_respects_constraints_and_prefers_high_mu():
    cov, _ = _random_cov()
    mu = np.linspace(0, 0.3, 8)
    w = mean_variance(mu, cov * 252, 2.5, 0.25)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= -1e-10).all() and (w <= 0.25 + 1e-8).all()
    assert w[-1] > w[0]


def test_infeasible_cap_raises():
    cov, _ = _random_cov(n=4)
    with pytest.raises(ValueError):
        min_variance(cov, 0.2)


def test_ledoit_wolf_is_psd_and_shrinks():
    cov, r = _random_cov(n=30, t=40)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.linalg.cond(cov) < np.linalg.cond(np.cov(r.T))
