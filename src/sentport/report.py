"""Figures and docs/results.md from a backtest result."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .backtest import run_all  # noqa: E402

COLORS = {
    "Equal weight": "#8c8c8c",
    "Min variance (LW)": "#4c78a8",
    "BL + sentiment": "#e45756",
    "BL + sentiment (IC=0.20)": "#f2a3a2",
    "Sentiment top-k": "#54a24b",
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def fig_equity(res, path):
    net = res["net"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for nm in net.columns:
        eq = (1 + net[nm]).cumprod()
        ax.plot(eq.index, eq, label=nm, color=COLORS[nm], lw=2.2 if nm == "BL + sentiment" else 1.5)
    ax.set_ylabel("Growth of 1 (net of costs)")
    ax.set_title(f"Walk-forward equity curves, {C.BacktestConfig.cost_bps:.0f} bp per unit traded",
                 loc="left", fontsize=11)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ic(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (label, win) in zip(axes, (("monthly", 12), ("weekly", 26)), strict=True):
        ic = res["ic"][label].dropna()
        w = 0.8 * (28 if label == "monthly" else 6)
        ax.bar(ic.index, ic, width=w, color=np.where(ic > 0, "#4c78a8", "#e45756"), alpha=0.55)
        ax.plot(ic.index, ic.rolling(win).mean(), color="black", lw=1.6, label=f"{win}-period mean")
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(ic.mean(), color="black", lw=0.8, ls="--", label=f"full mean {ic.mean():.3f}")
        st = res["ic_summary"].loc[label]
        ax.set_title(f"{label.capitalize()} rank IC (t = {st['t_stat']:.2f}, n = {int(st['n'])})",
                     loc="left", fontsize=10)  # fmt: skip
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_deciles(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (key, title) in zip(
        axes, (("deciles", "Monthly, deciles"), ("weekly_quintiles", "Weekly, quintiles")),
        strict=True,
    ):
        t = res[key]
        ax.bar(t.index + 1, t["mean"] * 100, yerr=1.96 * t["se"] * 100, color="#4c78a8",
               capsize=3, alpha=0.85)  # fmt: skip
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Sentiment bucket (1 = most negative)")
        ax.set_ylabel("Next-period return vs cross-section mean (%)")
        ax.set_title(f"{title}, error bars = 95% CI", loc="left", fontsize=10)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _t(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def summary_table(res) -> str:
    s = res["summary"]
    lines = ["| Strategy | CAGR | Vol | Sharpe (net) | 95% CI | Sharpe (gross) | Max DD | Turnover/yr |"
             " Sharpe diff vs EW (95% CI) |",
             "|---|---:|---:|---:|---|---:|---:|---:|---|"]  # fmt: skip
    for nm, r in s.iterrows():
        d = "-" if nm == "Equal weight" else f"{r.sharpe_net - s.loc['Equal weight', 'sharpe_net']:+.2f} " \
            f"[{r.dsharpe_lo:+.2f}, {r.dsharpe_hi:+.2f}]"  # fmt: skip
        lines.append(f"| {nm} | {r.cagr_net:.1%} | {r.vol:.1%} | {r.sharpe_net:.2f} | "
                     f"[{r.sharpe_lo:.2f}, {r.sharpe_hi:.2f}] | {r.sharpe_gross:.2f} | "
                     f"{r.max_dd:.1%} | {r.turnover_ann:.0%} | {d} |")  # fmt: skip
    return "\n".join(lines)


def make_report(res, daily, px, cfg, out_dir=None) -> None:
    out = out_dir or C.DOCS_DIR
    out.mkdir(exist_ok=True)
    fig_equity(res, out / "equity_curves.png")
    fig_ic(res, out / "ic_over_time.png")
    fig_deciles(res, out / "sentiment_deciles.png")

    ics = res["ic_summary"]
    sens = {}
    for bps in (0, 10, 25, 50):
        r = res if bps == cfg.cost_bps else run_all(daily, px, C.BacktestConfig(cost_bps=bps), 200)
        sens[bps] = r["summary"]["sharpe_net"]
    sens = pd.DataFrame(sens)
    sens.columns = [f"{b} bp" for b in sens.columns]

    _, qt = _t(res["quintile_spread"])
    qm = res["quintile_spread"].mean()
    _, wt = _t(res["weekly_spread"])
    wm = res["weekly_spread"].mean()
    n_days = len(res["net"])
    s = res["summary"]
    sent_rows = ["BL + sentiment", "Sentiment top-k"]
    beat = [n for n in sent_rows if s.loc[n, "dsharpe_lo"] > 0]

    ic_rows = ["| Horizon | n | Mean IC | t-stat | Bootstrap 95% CI | Hit rate |", "|---|---:|---:|---:|---|---:|"]
    for k, r in ics.iterrows():
        ic_rows.append(f"| {k} | {int(r.n)} | {r.mean_ic:.3f} | {r.t_stat:.2f} | "
                       f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {r.hit_rate:.0%} |")  # fmt: skip

    n_head = int(daily["n"].sum())
    md = f"""# Results

Out-of-sample window: {res['net'].index[0].date()} to {res['net'].index[-1].date()}
({n_days} trading days, {len(res['sched'])} monthly rebalances), {len(res['tickers'])} US stocks,
{n_head:,} scored headlines. Costs: {cfg.cost_bps:.0f} bp per unit of traded notional. Risk-free rate is
taken as zero for every strategy. Sharpe confidence intervals are stationary-bootstrap
(5,000 resamples, mean block 21 days); differences versus equal weight use paired resamples.

## Portfolios

{summary_table(res)}

Sentiment strategies that beat equal weight with the paired Sharpe-difference interval strictly
above zero: {', '.join(beat) if beat else 'none'}.

![Equity curves](equity_curves.png)

### Sensitivity of net Sharpe to the cost assumption

{_md(sens)}

## Does sentiment predict returns?

Rank information coefficient of the decayed FinBERT signal against the next-period return
(execution one trading day after the decision date).

{chr(10).join(ic_rows)}

Top-minus-bottom quintile spread of next-month return (relative to the cross-section mean):
{qm:.2%} per month, t = {qt:.2f}. Weekly quintile spread: {wm:.2%} per week, t = {wt:.2f}.

![IC over time](ic_over_time.png)
![Sentiment vs next return](sentiment_deciles.png)

## Parameters

```
{cfg}
```
"""
    (out / "results.md").write_text(md)


def _md(df: pd.DataFrame) -> str:
    head = "| Strategy | " + " | ".join(df.columns) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = [f"| {i} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep, *rows])
