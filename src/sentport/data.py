"""Fetching and caching of headlines and prices."""

from __future__ import annotations

import pandas as pd

from . import config as C


def fetch_news(force: bool = False) -> pd.DataFrame:
    """Download the Benzinga-derived headline dataset and reduce it to the universe.

    Keeps ticker, date (date-only, no intra-day time is available), headline, publisher.
    Drops exact duplicates of (ticker, date, headline). Cached at data/headlines.parquet.
    """
    out = C.DATA_DIR / "headlines.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    from huggingface_hub import hf_hub_download

    C.DATA_DIR.mkdir(exist_ok=True)
    raw = hf_hub_download(C.HF_REPO, C.HF_FILE, repo_type="dataset", local_dir=C.DATA_DIR / "hf")
    df = pd.read_parquet(raw, columns=["headline", "publisher", "date", "stock"])
    df = df[df["stock"].isin(C.UNIVERSE)].rename(columns={"stock": "ticker"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df["date"] >= C.NEWS_START) & (df["date"] < C.NEWS_END)]
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]
    df = df.drop_duplicates(["ticker", "date", "headline"]).sort_values(["ticker", "date"])
    df = df.reset_index(drop=True)
    df.to_parquet(out)
    return df


def fetch_prices(force: bool = False) -> pd.DataFrame:
    """Daily split/dividend-adjusted closes from Yahoo Finance via yfinance, cached."""
    out = C.DATA_DIR / "prices.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    import yfinance as yf

    C.DATA_DIR.mkdir(exist_ok=True)
    ymap = {t: C.YAHOO_MAP.get(t, t) for t in C.UNIVERSE}
    px = yf.download(list(ymap.values()), start=C.PRICE_START, end=C.PRICE_END,
                     auto_adjust=True, progress=False)["Close"]  # fmt: skip
    inv = {v: k for k, v in ymap.items()}
    px = px.rename(columns=inv)[C.UNIVERSE]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    missing = px.columns[px.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"no price data for {missing}")
    px = px.sort_index().ffill(limit=3)
    px.to_parquet(out)
    return px


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Committed processed sample: (daily sentiment aggregates, prices)."""
    return (pd.read_parquet(C.SAMPLE_DIR / "sentiment_daily.parquet"),
            pd.read_parquet(C.SAMPLE_DIR / "prices.parquet"))  # fmt: skip
