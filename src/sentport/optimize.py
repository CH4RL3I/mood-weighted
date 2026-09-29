"""Covariance shrinkage, constrained long-only optimisers, Black-Litterman."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards mu*I.

    ``returns`` is (T, N). Returns (covariance, shrinkage intensity).
    """
    x = returns - returns.mean(axis=0, keepdims=True)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = np.sum((s - mu * np.eye(n)) ** 2) / n
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / (t**2 * n)
    b2 = min(b2, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    return shrink * mu * np.eye(n) + (1 - shrink) * s, shrink


def _solve(objective, n: int, cap: float, w0: np.ndarray | None = None) -> np.ndarray:
    if cap * n < 1 - 1e-12:
        raise ValueError("weight cap too small to be fully invested")
    x0 = np.full(n, 1.0 / n) if w0 is None else w0
    res = minimize(objective, x0, method="SLSQP", bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})  # fmt: skip
    w = np.clip(res.x, 0.0, cap)
    return w / w.sum() if cap >= 1.0 / n else w


def min_variance(cov: np.ndarray, cap: float) -> np.ndarray:
    """Long-only, fully invested, capped minimum variance."""
    n = cov.shape[0]
    scale = 1.0 / np.trace(cov) * n
    return _solve(lambda w: scale * w @ cov @ w, n, cap)


def mean_variance(mu: np.ndarray, cov: np.ndarray, delta: float, cap: float) -> np.ndarray:
    """Long-only, capped maximiser of w'mu - delta/2 w'cov w."""
    n = len(mu)
    sc = 1.0 / np.trace(cov) * n
    return _solve(lambda w: sc * (-(w @ mu) + 0.5 * delta * w @ cov @ w), n, cap)


def black_litterman(
    cov: np.ndarray,
    prior_w: np.ndarray,
    q: np.ndarray,
    conf: np.ndarray,
    delta: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean (with absolute views on every asset).

    Prior (equilibrium) returns: pi = delta * cov @ prior_w.
    Views: one absolute view per asset, P = I, expected return q_i, confidence c_i in [0, 1].
    View uncertainty follows Idzorek's mapping: Omega_ii = tau * cov_ii * (1 - c_i) / c_i, so
    c = 1 pins the view (Omega -> 0) and c = 0 removes it (Omega -> infinity).
    Posterior: mu = pi + tau*cov P' (P tau cov P' + Omega)^-1 (q - P pi).

    Returns (posterior mean, prior pi). Views with c <= 0 are dropped, so an all-zero
    confidence vector returns pi exactly.
    """
    pi = delta * cov @ prior_w
    active = conf > 0
    if not active.any():
        return pi.copy(), pi
    c = np.clip(conf[active], 1e-9, 1.0)
    idx = np.where(active)[0]
    p = np.zeros((len(idx), len(pi)))
    p[np.arange(len(idx)), idx] = 1.0
    omega = np.diag(tau * np.diag(cov)[idx] * (1 - c) / c)
    tc = tau * cov
    mid = p @ tc @ p.T + omega
    mu = pi + tc @ p.T @ np.linalg.solve(mid, q[idx] - p @ pi)
    return mu, pi
