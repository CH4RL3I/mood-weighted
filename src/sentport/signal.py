"""Per-stock sentiment signal from daily aggregates, strictly using data available at t."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decayed_signal(
    daily: pd.DataFrame,
    decision_date,
    halflife_days: float = 15.0,
    window_days: int = 90,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Exponentially decayed sentiment at the close of ``decision_date``.

    Uses only rows with ``avail_date <= decision_date``. Weights are 0.5 ** (age / halflife)
    with age measured in calendar days from the availability date.

    Returns a frame indexed by ticker with columns: sent (decayed mean score),
    count (decayed headline count), z (cross-sectional z-score of sent among stocks with
    count > 0; 0 for stocks without news).
    """
    t = pd.Timestamp(decision_date).normalize()
    d = daily[(daily["avail_date"] <= t) & (daily["avail_date"] > t - pd.Timedelta(days=window_days))]
    age = (t - d["avail_date"]).dt.days.to_numpy()
    w = 0.5 ** (age / halflife_days)
    d = d.assign(w_n=w * d["n"].to_numpy(), w_s=w * d["score_sum"].to_numpy())
    g = d.groupby("ticker")[["w_n", "w_s"]].sum()
    if tickers is not None:
        g = g.reindex(tickers).fillna(0.0)
    out = pd.DataFrame({"count": g["w_n"]})
    out["sent"] = np.where(g["w_n"] > 0, g["w_s"] / g["w_n"].where(g["w_n"] > 0, 1.0), np.nan)
    has = out["count"] > 0
    z = pd.Series(0.0, index=out.index)
    if has.sum() >= 3 and out.loc[has, "sent"].std(ddof=0) > 0:
        s = out.loc[has, "sent"]
        z[has] = (s - s.mean()) / s.std(ddof=0)
    out["z"] = z
    return out
