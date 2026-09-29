"""Central configuration: universe, dates, paths, and backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = ROOT / "sample"
DOCS_DIR = ROOT / "docs"

HF_REPO = "ashraq/financial-news"
HF_FILE = "data/train-00000-of-00001-8ec327f23bbe0948.parquet"

# 29 US large/mid caps. Chosen by (i) headline coverage in the dataset (at least 3 headlines
# in >= 97% of months 2015-01..2020-05), (ii) being a US-listed operating company rather than
# an ETF/ADR, and (iii) still having Yahoo price history in 2026. Criterion (iii) is a
# survivorship filter and is documented as a bias. Selection never looked at returns.
UNIVERSE = [
    "MDT", "OXY", "FCX", "TXN", "ANTM", "BLK", "WDC", "STZ", "LEN", "CSX",
    "CI", "ULTA", "DUK", "TJX", "HUM", "TSN", "KMB", "MDLZ", "EW", "TRV",
    "MAR", "NOW", "MET", "TMO", "PRU", "CTSH", "KSS", "PANW", "DPZ",
]  # fmt: skip
# Dataset ticker -> Yahoo ticker (Anthem renamed itself Elevance Health, ticker ELV).
YAHOO_MAP = {"ANTM": "ELV"}

NEWS_START = "2015-01-01"
NEWS_END = "2020-06-01"
PRICE_START = "2014-01-01"
PRICE_END = "2020-07-01"


@dataclass(frozen=True)
class BacktestConfig:
    first_rebalance: str = "2015-03-01"
    last_rebalance: str = "2020-05-01"
    cov_lookback: int = 252  # trading days for the covariance estimate
    news_halflife_days: float = 15.0  # calendar days, exponential decay of headline weights
    news_window_days: int = 90  # headlines older than this get (near) zero weight anyway
    max_weight: float = 0.20  # applies to every strategy; top-k needs 1/k <= cap
    cost_bps: float = 10.0  # per unit of traded notional, both buys and sells
    top_k: int = 6
    min_headlines: float = 2.0  # decayed headline count needed for a stock to be tradable on news
    # Black-Litterman
    delta: float = 2.5  # risk aversion
    tau: float = 0.05
    assumed_ic: float = 0.05  # Grinold: alpha = IC * sigma * z-score
    conf_ref_count: float = 5.0  # decayed headline count at which view confidence reaches 0.5
    execution_lag_days: int = 1  # trade at the close of the day after the decision
    news_lag_days: int = 0  # extra embargo; date-only stamps are already delayed to next day

    def __post_init__(self):
        if self.top_k * self.max_weight < 1.0:
            raise ValueError("top_k * max_weight must be >= 1 for a fully invested top-k rule")
