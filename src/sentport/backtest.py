"""Strict walk-forward backtest.

Timeline for a rebalance with decision date t (last trading day of the month):

1. At the close of t we know prices up to t and headlines whose ``avail_date <= t``.
2. Target weights are computed from that information only.
3. Orders execute at the close of the ``execution_lag_days``-th trading day after t
   (default 1), and the new weights earn returns from that close onwards.
4. Between rebalances weights drift with returns. At each rebalance the traded notional is
   sum |w_target - w_drifted| and costs are ``cost_bps`` per unit of traded notional.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import stats
from .config import BacktestConfig
from .optimize import black_litterman, ledoit_wolf, mean_variance, min_variance
from .signal import decayed_signal

WeightFn = Callable[[dict], np.ndarray]


def schedule(index: pd.DatetimeIndex, freq: str, start: str, end: str, lag: int) -> pd.DataFrame:
    """Decision dates (last trading day of each month/week) with execution dates.

    freq: 'M' or 'W'. Only decision dates in [start, end] with an execution date and a
    following execution date available are kept (the last one is dropped for lack of a horizon).
    """
    s = pd.Series(index, index=index)
    key = s.dt.to_period(freq)
    dec = s.groupby(key).last()
    dec = dec[(dec >= pd.Timestamp(start)) & (dec <= pd.Timestamp(end))]
    pos = index.get_indexer(dec.to_numpy())
    exe_pos = pos + lag
    ok = exe_pos < len(index)
    return pd.DataFrame({"decision": dec.to_numpy()[ok], "exec": index[exe_pos[ok]]})


def _returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change()


def strategy_weights(cfg: BacktestConfig) -> dict[str, WeightFn]:
    """Weight functions. ctx keys: hist (returns up to t), sig (signal frame), n."""

    def ew(ctx):
        return np.full(ctx["n"], 1.0 / ctx["n"])

    def minvar(ctx):
        cov, _ = ledoit_wolf(ctx["hist"].to_numpy())
        return min_variance(cov, cfg.max_weight)

    def make_bl(ic: float):
        def bl(ctx):
            cov, _ = ledoit_wolf(ctx["hist"].to_numpy())
            cov = cov * stats.TRADING_DAYS
            n = ctx["n"]
            sig = ctx["sig"]
            sigma = np.sqrt(np.diag(cov))
            pi = cfg.delta * cov @ np.full(n, 1.0 / n)
            q = pi + ic * sigma * sig["z"].to_numpy()  # Grinold: alpha = IC * vol * score
            cnt = sig["count"].to_numpy()
            conf = np.where(cnt > 0, cnt / (cnt + cfg.conf_ref_count), 0.0)
            mu, _ = black_litterman(cov, np.full(n, 1.0 / n), q, conf, cfg.delta, cfg.tau)
            return mean_variance(mu, cov, cfg.delta, cfg.max_weight)

        return bl

    def topk(ctx):
        sig = ctx["sig"]
        elig = sig[sig["count"] >= cfg.min_headlines]
        w = np.zeros(ctx["n"])
        if len(elig) < cfg.top_k:  # not enough news: fall back to equal weight
            return np.full(ctx["n"], 1.0 / ctx["n"])
        pick = elig["sent"].nlargest(cfg.top_k).index
        w[[sig.index.get_loc(t) for t in pick]] = 1.0 / cfg.top_k
        return w

    return {"Equal weight": ew, "Min variance (LW)": minvar,
            "BL + sentiment": make_bl(cfg.assumed_ic),
            "BL + sentiment (IC=0.20)": make_bl(0.20), "Sentiment top-k": topk}  # fmt: skip


def simulate(px: pd.DataFrame, sched: pd.DataFrame, wfn_by_name: dict[str, WeightFn],
             weights_at: Callable[[str, pd.Timestamp], np.ndarray], cost_bps: float):  # fmt: skip
    """Run all strategies through the same schedule. Returns gross, net, turnover, weights."""
    rets = _returns(px)
    names = list(wfn_by_name)
    idx = px.index
    first_exec = sched["exec"].iloc[0]
    days = idx[idx >= first_exec]
    net = pd.DataFrame(0.0, index=days, columns=names)
    gross = net.copy()
    turnover = pd.DataFrame(index=sched["decision"], columns=names, dtype=float)
    all_w: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for nm in names:
        w_drift = np.zeros(px.shape[1])
        for k in range(len(sched)):
            t, e = sched["decision"].iloc[k], sched["exec"].iloc[k]
            e_next = sched["exec"].iloc[k + 1] if k + 1 < len(sched) else days[-1]
            w = weights_at(nm, t)
            traded = np.abs(w - w_drift).sum()
            turnover.loc[t, nm] = 0.5 * traded
            all_w[nm].append(w)
            hold = rets.loc[(idx > e) & (idx <= e_next)].fillna(0.0)
            hr = hold.to_numpy()
            wt = w.copy()  # weights drift with returns between rebalances
            port = np.empty(len(hold))
            for j in range(len(hold)):
                r = hr[j]
                pr = wt @ r
                port[j] = pr
                wt = wt * (1 + r) / (1 + pr)
            w_drift = wt
            gross.loc[hold.index, nm] = port
            net.loc[hold.index, nm] = port
            net.loc[e, nm] -= traded * cost_bps / 1e4
    keep = (net.index >= first_exec)
    return gross[keep], net[keep], turnover, all_w


def run_all(daily: pd.DataFrame, px: pd.DataFrame, cfg: BacktestConfig | None = None,
            boot_reps: int = 5000) -> dict:  # fmt: skip
    cfg = cfg or BacktestConfig()
    tickers = list(px.columns)
    rets = _returns(px)
    sched = schedule(px.index, "M", cfg.first_rebalance, cfg.last_rebalance, cfg.execution_lag_days)
    fns = strategy_weights(cfg)
    cache: dict[pd.Timestamp, dict] = {}

    def ctx_at(t):
        if t not in cache:
            hist = rets.loc[:t].dropna(how="all").iloc[-cfg.cov_lookback:]
            if len(hist) < cfg.cov_lookback:
                raise ValueError(f"only {len(hist)} return days before {t.date()}")
            sig = decayed_signal(daily, t, cfg.news_halflife_days, cfg.news_window_days, tickers)
            cache[t] = {"hist": hist.fillna(0.0), "sig": sig, "n": len(tickers)}
        return cache[t]

    wcache: dict[tuple[str, pd.Timestamp], np.ndarray] = {}

    def weights_at(name, t):
        key = (name, t)
        if key not in wcache:
            wcache[key] = fns[name](ctx_at(t))
        return wcache[key]

    gross, net, turnover, all_w = simulate(px, sched, fns, weights_at, cfg.cost_bps)

    # information coefficient: monthly (the tradable frequency) and weekly (more power)
    ic = {}
    panels = {}
    for label, freq in (("monthly", "M"), ("weekly", "W")):
        sc = schedule(px.index, freq, cfg.first_rebalance, cfg.last_rebalance, cfg.execution_lag_days)
        rows, vals = [], []
        for k in range(len(sc) - 1):
            t, e, e2 = sc["decision"].iloc[k], sc["exec"].iloc[k], sc["exec"].iloc[k + 1]
            fwd = px.loc[e2] / px.loc[e] - 1
            sig = decayed_signal(daily, t, cfg.news_halflife_days, cfg.news_window_days, tickers)
            ok = sig["count"] >= 1.0
            s = sig["sent"].where(ok)
            vals.append((t, stats.spearman_ic(s, fwd)))
            rows.append(pd.DataFrame({"date": t, "ticker": tickers, "signal": s.to_numpy(),
                                      "fwd": fwd.to_numpy()}))  # fmt: skip
        ic[label] = pd.Series(dict(vals))
        panels[label] = pd.concat(rows, ignore_index=True)

    ic_summary = pd.DataFrame({k: stats.ic_stats(v) for k, v in ic.items()}).T
    dec, spread = stats.bucket_returns(panels["monthly"], 10)
    quint, qspread = stats.bucket_returns(panels["monthly"], 5)
    wdec, wspread = stats.bucket_returns(panels["weekly"], 5)

    ref = "Equal weight"
    ci = stats.sharpe_ci(net, ref, reps=boot_reps)
    rows = {}
    for nm in net.columns:
        g, n_ = stats.perf(gross[nm]), stats.perf(net[nm])
        rows[nm] = {"cagr_net": n_["cagr"], "vol": n_["vol"], "sharpe_net": n_["sharpe"],
                    "sharpe_gross": g["sharpe"], "max_dd": n_["max_dd"],
                    "turnover_ann": turnover[nm].sum() / (len(net) / stats.TRADING_DAYS)}  # fmt: skip
    summary = pd.DataFrame(rows).T.join(ci)
    return {"cfg": cfg, "gross": gross, "net": net, "turnover": turnover, "weights": all_w,
            "sched": sched, "ic": ic, "ic_summary": ic_summary, "deciles": dec,
            "decile_spread": spread, "quintiles": quint, "quintile_spread": qspread,
            "weekly_quintiles": wdec, "weekly_spread": wspread, "summary": summary,
            "panels": panels, "tickers": tickers}  # fmt: skip
