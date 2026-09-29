"""Timestamp discipline: when is a headline allowed to influence a decision?

Decisions are taken at the US equity close (16:00 America/New_York) of a decision date ``D``.
A headline may be used only if it was published no later than that cutoff.

Two kinds of timestamps occur:

* Timestamped headlines (tz-aware, or naive and assumed UTC) are converted to New York time.
  Published before 16:00 ET on day X: usable from decision date X. At or after 16:00: usable
  from the next calendar day X+1.
* Date-only headlines (midnight, as in the dataset used here) carry no intra-day information,
  so they may have appeared after the close. The conservative rule is that they become usable
  on the *next* calendar day, X+1.
"""

from __future__ import annotations

import pandas as pd

NY = "America/New_York"
CLOSE_HOUR = 16


def available_date(ts: pd.Series, date_only: bool = False, lag_days: int = 0) -> pd.Series:
    """First calendar date (naive, normalised) whose 16:00 ET decision may use the headline."""
    s = pd.to_datetime(ts)
    if date_only:
        base = s.dt.tz_localize(None).dt.normalize() if s.dt.tz is not None else s.dt.normalize()
        out = base + pd.Timedelta(days=1)
    else:
        s = s.dt.tz_localize("UTC") if s.dt.tz is None else s
        ny = s.dt.tz_convert(NY)
        day = ny.dt.tz_localize(None).dt.normalize()
        after_close = ny.dt.hour >= CLOSE_HOUR
        out = day + pd.to_timedelta(after_close.astype(int), unit="D")
    return out + pd.Timedelta(days=lag_days)


def filter_available(df: pd.DataFrame, decision_date, col: str = "avail_date") -> pd.DataFrame:
    """Rows usable for a decision at the close of ``decision_date``."""
    return df[df[col] <= pd.Timestamp(decision_date).normalize()]
