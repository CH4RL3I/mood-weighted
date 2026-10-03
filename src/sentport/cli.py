"""Command line interface: sentport fetch | score | backtest | report."""

from __future__ import annotations

import argparse

import pandas as pd

from . import config as C


def cmd_fetch(args) -> None:
    from .data import fetch_news, fetch_prices

    news = fetch_news(force=args.force)
    px = fetch_prices(force=args.force)
    print(f"headlines: {len(news)} | tickers: {news['ticker'].nunique()} | "
          f"{news['date'].min().date()} to {news['date'].max().date()}")  # fmt: skip
    print(f"prices: {px.shape[1]} tickers, {px.index[0].date()} to {px.index[-1].date()}")


def cmd_score(args) -> None:
    from .data import fetch_news
    from .sentiment import aggregate_daily, finbert_scorer, score_headlines

    news = fetch_news()
    scorer = finbert_scorer(batch_size=args.batch_size, threads=4)
    scores = score_headlines(news, scorer, C.DATA_DIR / "finbert_scores.parquet")
    daily = aggregate_daily(news, scores)
    daily.to_parquet(C.DATA_DIR / "sentiment_daily.parquet")
    print(f"daily aggregates: {len(daily)} rows; mean score {scores['score'].mean():.3f}")


def _load(sample: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample:
        from .data import load_sample

        return load_sample()
    return (pd.read_parquet(C.DATA_DIR / "sentiment_daily.parquet"),
            pd.read_parquet(C.DATA_DIR / "prices.parquet"))  # fmt: skip


def cmd_backtest(args) -> None:
    from .backtest import run_all

    daily, px = _load(args.sample)
    res = run_all(daily, px, C.BacktestConfig(cost_bps=args.cost_bps))
    C.DATA_DIR.mkdir(exist_ok=True)
    out = C.DATA_DIR / "backtest.pkl"
    pd.to_pickle(res, out)
    print(res["summary"].round(3).to_string())


def cmd_report(args) -> None:
    from .backtest import run_all
    from .report import make_report

    daily, px = _load(args.sample)
    res = run_all(daily, px, C.BacktestConfig())
    make_report(res, daily, px, C.BacktestConfig())
    print("wrote docs/results.md and figures")


def cmd_sample(args) -> None:
    """Write the committed sample: aggregated sentiment only (no headline text, no prices)."""
    daily, _ = _load(False)
    C.SAMPLE_DIR.mkdir(exist_ok=True)
    daily.to_parquet(C.SAMPLE_DIR / "sentiment_daily.parquet", compression="zstd")
    print("sample written")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sentport", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch", help="download headlines and prices into data/")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("score", help="score headlines with FinBERT (cached)")
    s.add_argument("--batch-size", type=int, default=64)
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("backtest", help="run the walk-forward backtest")
    s.add_argument("--sample", action="store_true", help="use the committed sample")
    s.add_argument("--cost-bps", type=float, default=C.BacktestConfig.cost_bps)
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("report", help="write figures and docs/results.md")
    s.add_argument("--sample", action="store_true", help="use the committed sample")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("sample", help="write the committed processed sample")
    s.set_defaults(fn=cmd_sample)
    args = p.parse_args(argv)
    args.fn(args)
