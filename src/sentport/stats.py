"""Performance statistics, information coefficient, bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def spearman_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """Cross-sectional Spearman rank correlation, NaN if fewer than 5 valid pairs."""
    d = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(d) < 5 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def stationary_bootstrap_idx(n: int, reps: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (reps, n)."""
    p = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    restart = rng.random((reps, n)) < p
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def ic_stats(ic: pd.Series, reps: int = 5000, seed: int = 0) -> dict:
    """Mean IC, t-stat (mean / standard error), hit rate and a stationary-bootstrap 95% CI."""
    x = ic.dropna().to_numpy()
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, reps, 3.0, rng)
    boot = x[idx].mean(axis=1)
    return {"n": n, "mean_ic": m, "t_stat": m / se, "hit_rate": float((x > 0).mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))}  # fmt: skip


def bucket_returns(panel: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.Series]:
    """Sort stocks into signal buckets within each date.

    ``panel`` has columns date, signal, fwd. Bucket returns are relative to the date's
    cross-sectional mean. Returns (per-bucket mean/se table, per-date top-minus-bottom spread).
    """
    p = panel.dropna().copy()
    p["rel"] = p["fwd"] - p.groupby("date")["fwd"].transform("mean")
    p["bucket"] = p.groupby("date")["signal"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False) if len(s) >= n_bins else np.nan
    )
    p = p.dropna(subset=["bucket"])
    if p.empty:  # fewer stocks per date than buckets
        return pd.DataFrame(columns=["mean", "std", "count", "se"]), pd.Series(dtype=float)
    tab = p.groupby("bucket")["rel"].agg(["mean", "std", "count"])
    tab["se"] = tab["std"] / np.sqrt(tab["count"])
    per = p.groupby(["date", "bucket"])["rel"].mean().unstack()
    spread = per[n_bins - 1] - per[0]
    return tab, spread


def max_drawdown(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float((eq / eq.cummax() - 1).min())


def perf(ret: pd.Series) -> dict:
    """Annualised statistics from daily returns (risk-free rate taken as zero)."""
    n = len(ret)
    cagr = (1 + ret).prod() ** (TRADING_DAYS / n) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "vol": vol, "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS),
            "max_dd": max_drawdown(ret)}  # fmt: skip


def sharpe_ci(rets: pd.DataFrame, ref: str | None, reps: int = 5000, mean_block: float = 21.0,
              seed: int = 1) -> pd.DataFrame:  # fmt: skip
    """Stationary-bootstrap 95% CIs for each column's Sharpe and for Sharpe minus ``ref``'s.

    The same resampled dates are used for every column, so differences are paired.
    """
    x = rets.to_numpy()
    idx = stationary_bootstrap_idx(len(x), reps, mean_block, np.random.default_rng(seed))
    samp = x[idx]  # (reps, n, k)
    sh = samp.mean(axis=1) / samp.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    out = pd.DataFrame(index=rets.columns)
    out["sharpe_lo"] = np.percentile(sh, 2.5, axis=0)
    out["sharpe_hi"] = np.percentile(sh, 97.5, axis=0)
    if ref is not None:
        j = list(rets.columns).index(ref)
        diff = sh - sh[:, [j]]
        out["dsharpe_lo"] = np.percentile(diff, 2.5, axis=0)
        out["dsharpe_hi"] = np.percentile(diff, 97.5, axis=0)
        out["p_better"] = (diff > 0).mean(axis=0)
        out.loc[ref, ["dsharpe_lo", "dsharpe_hi", "p_better"]] = np.nan
    return out
