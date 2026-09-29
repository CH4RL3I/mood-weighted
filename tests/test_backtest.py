import numpy as np
import pandas as pd
import pytest

from sentport.backtest import run_all, schedule, simulate
from sentport.config import BacktestConfig
from sentport.stats import spearman_ic


def _prices(n_days=700, n=6, seed=0):
    idx = pd.bdate_range("2017-01-02", periods=n_days)
    r = np.random.default_rng(seed).normal(0.0004, 0.012, (n_days, n))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=list("ABCDEF"))


def _daily(px, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in px.index[::3]:
        for t in px.columns:
            rows.append((t, d, 2, rng.normal(0, 0.3)))
    df = pd.DataFrame(rows, columns=["ticker", "date", "n", "score_sum"])
    df["avail_date"] = df["date"] + pd.Timedelta(days=1)
    return df


CFG = BacktestConfig(first_rebalance="2018-03-01", last_rebalance="2018-12-31", cov_lookback=60,
                     max_weight=0.4, top_k=3, min_headlines=1.0)  # fmt: skip


def test_costs_flat_prices_turnover_accounting():
    idx = pd.bdate_range("2020-01-01", periods=80)
    px = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    sched = schedule(idx, "M", "2020-01-01", "2020-03-31", 1)
    targets = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([0.5, 0.5])}
    order = {t: i for i, t in enumerate(sched["decision"])}
    gross, net, turn, _ = simulate(px, sched, {"s": None}, lambda nm, t: targets[order[t]], 10.0)
    # flat prices: gross is zero; cost = 10bp * traded notional on each execution day
    assert (gross["s"] == 0).all()
    traded = [1.0, 2.0, 1.0]  # cash->A, A->B, B->half/half
    exp = [-tr * 10 / 1e4 for tr in traded]
    got = [net["s"].loc[e] for e in sched["exec"]]
    assert got == pytest.approx(exp)
    assert turn["s"].to_numpy() == pytest.approx([0.5 * t for t in traded])
    assert net["s"].sum() == pytest.approx(sum(exp))


def test_costs_reduce_returns_versus_gross():
    px = _prices()
    res = run_all(_daily(px), px, CFG, boot_reps=50)
    assert (res["net"].sum() <= res["gross"].sum() + 1e-12).all()
    zero = run_all(_daily(px), px, BacktestConfig(**{**CFG.__dict__, "cost_bps": 0.0}), boot_reps=50)
    pd.testing.assert_frame_equal(zero["net"], zero["gross"])


def test_weights_valid_and_no_lookahead_in_prices():
    px = _prices()
    daily = _daily(px)
    base = run_all(daily, px, CFG, boot_reps=50)
    for name, ws in base["weights"].items():
        for w in ws:
            assert w.sum() == pytest.approx(1.0, abs=1e-8), name
            assert (w >= -1e-10).all() and (w <= max(CFG.max_weight, 1 / 6 + 1e-9) + 1e-8).all(), name
    # perturb prices and headlines after a cut-off: earlier weights must not change
    cut = base["sched"]["decision"].iloc[3]
    px2 = px.copy()
    px2.loc[px2.index > cut] *= 1.7 + np.random.default_rng(5).random(px2.loc[px2.index > cut].shape)
    d2 = daily.copy()
    d2.loc[d2["date"] > cut, "score_sum"] = 99.0
    alt = run_all(d2, px2, CFG, boot_reps=50)
    for name in base["weights"]:
        for k in range(4):
            np.testing.assert_allclose(base["weights"][name][k], alt["weights"][name][k],
                                       atol=1e-8, err_msg=name)  # fmt: skip


def test_spearman_ic_known_values():
    s = pd.Series([1, 2, 3, 4, 5, 6.0])
    assert spearman_ic(s, s * 2 + 1) == pytest.approx(1.0)
    assert spearman_ic(s, -s) == pytest.approx(-1.0)
    r = pd.Series([2, 1, 4, 3, 6, 5.0])  # d = 1 everywhere: rho = 1 - 6*6/(6*35) = 0.828571
    assert spearman_ic(s, r) == pytest.approx(1 - 6 * 6 / (6 * 35))
    assert np.isnan(spearman_ic(s.iloc[:3], s.iloc[:3]))  # too few observations
    assert np.isnan(spearman_ic(pd.Series([1.0] * 6), s))  # constant signal


def test_ic_uses_only_forward_returns_after_execution():
    # signal perfectly predicts returns *after* execution -> IC of 1 in the panel
    px = _prices()
    daily = _daily(px)
    res = run_all(daily, px, CFG, boot_reps=50)
    p = res["panels"]["monthly"]
    assert set(p.columns) == {"date", "ticker", "signal", "fwd"}
    first = res["sched"].iloc[0]
    row = p[(p["date"] == first["decision"]) & (p["ticker"] == "A")].iloc[0]
    nxt = res["sched"]["exec"].iloc[1]
    assert row["fwd"] == pytest.approx(px.loc[nxt, "A"] / px.loc[first["exec"], "A"] - 1)
