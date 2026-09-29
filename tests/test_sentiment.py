import numpy as np
import pandas as pd

from sentport.sentiment import aggregate_daily, score_headlines
from sentport.stats import bucket_returns, ic_stats


def fake_scorer(calls):
    def score(texts):
        calls.append(list(texts))
        pos = np.array([1.0 if "up" in t else 0.0 for t in texts])
        neg = np.array([1.0 if "down" in t else 0.0 for t in texts])
        return np.c_[pos, neg, 1 - pos - neg]

    return score


def test_scores_cached_and_only_new_texts_rescored(tmp_path):
    heads = pd.DataFrame({"headline": ["stock up", "stock down", "stock up"]})
    calls: list = []
    cache = tmp_path / "s.parquet"
    r1 = score_headlines(heads, fake_scorer(calls), cache, log=lambda *_: None)
    assert len(calls) == 1 and len(calls[0]) == 2  # duplicates scored once
    assert set(r1["score"]) == {1.0, -1.0}
    heads2 = pd.DataFrame({"headline": ["stock up", "flat news"]})
    r2 = score_headlines(heads2, fake_scorer(calls), cache, log=lambda *_: None)
    assert len(calls) == 2 and calls[1] == ["flat news"]
    assert len(r2) == 3


def test_aggregate_daily_has_no_text_and_counts():
    heads = pd.DataFrame({"ticker": ["A", "A", "B"],
                          "date": pd.to_datetime(["2020-01-02"] * 3),
                          "headline": ["stock up", "stock down", "stock up"]})  # fmt: skip
    sc = pd.DataFrame({"text": ["stock up", "stock down"], "score": [1.0, -1.0]})
    d = aggregate_daily(heads, sc)
    assert "headline" not in d.columns
    a = d[d.ticker == "A"].iloc[0]
    assert (a["n"], a["score_sum"]) == (2, 0.0)
    assert a["avail_date"] == pd.Timestamp("2020-01-03")


def test_bucket_spread_positive_when_signal_predicts():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2020-01-31", periods=24, freq="ME"):
        s = rng.normal(size=30)
        rows.append(pd.DataFrame({"date": d, "signal": s, "fwd": 0.02 * s + rng.normal(0, 0.01, 30)}))
    tab, spread = bucket_returns(pd.concat(rows), 5)
    assert tab["mean"].is_monotonic_increasing
    assert spread.mean() > 0


def test_ic_stats_tstat_matches_formula():
    x = pd.Series([0.1, 0.0, 0.2, -0.05, 0.1, 0.15, 0.05, 0.0])
    st = ic_stats(x, reps=200)
    assert st["t_stat"] == np.float64(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    assert st["ci_lo"] < st["mean_ic"] < st["ci_hi"]
    assert st["n"] == 8
