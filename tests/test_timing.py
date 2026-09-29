import pandas as pd

from sentport.sentiment import aggregate_daily
from sentport.signal import decayed_signal
from sentport.timing import available_date, filter_available


def av(ts, date_only=False, lag=0):
    return available_date(pd.Series(pd.to_datetime([ts])), date_only, lag).iloc[0]


def test_date_only_headline_usable_next_day_only():
    assert av("2020-03-10 00:00:00", date_only=True) == pd.Timestamp("2020-03-11")


def test_timed_headline_before_close_usable_same_day():
    # 15:59 New York on 2020-03-10 (EDT, UTC-4) = 19:59 UTC
    assert av("2020-03-10 19:59:00") == pd.Timestamp("2020-03-10")


def test_timed_headline_after_close_usable_next_day():
    assert av("2020-03-10 20:01:00") == pd.Timestamp("2020-03-11")


def test_timezone_conversion_respects_dst():
    # 2020-01-10 is EST (UTC-5): 20:59 UTC = 15:59 ET, 21:00 UTC = 16:00 ET
    assert av("2020-01-10 20:59:00") == pd.Timestamp("2020-01-10")
    assert av("2020-01-10 21:00:00") == pd.Timestamp("2020-01-11")


def test_tz_aware_input_is_converted():
    ts = pd.Series(pd.to_datetime(["2020-03-10 15:30:00"]).tz_localize("America/New_York"))
    assert available_date(ts).iloc[0] == pd.Timestamp("2020-03-10")
    ts = pd.Series(pd.to_datetime(["2020-03-10 16:30:00"]).tz_localize("America/New_York"))
    assert available_date(ts).iloc[0] == pd.Timestamp("2020-03-11")


def test_embargo_lag():
    assert av("2020-03-10", date_only=True, lag=2) == pd.Timestamp("2020-03-13")


def test_filter_never_returns_future_rows():
    df = pd.DataFrame({"avail_date": pd.to_datetime(["2020-03-09", "2020-03-10", "2020-03-11"])})
    out = filter_available(df, "2020-03-10")
    assert out["avail_date"].max() <= pd.Timestamp("2020-03-10")
    assert len(out) == 2


def _daily(same_day_score):
    heads = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
        "date": pd.to_datetime(["2020-03-05", "2020-03-10"] * 3),
        "headline": list("abcdef"),
    })
    scores = pd.DataFrame({"text": list("abcdef"),
                           "score": [0.1, same_day_score, -0.2, 0.0, 0.3, 0.0]})  # fmt: skip
    return aggregate_daily(heads, scores)


def test_headline_dated_on_decision_day_cannot_move_signal():
    """A headline dated D must not affect the signal computed at the close of D."""
    a = decayed_signal(_daily(0.9), "2020-03-10")
    b = decayed_signal(_daily(-0.9), "2020-03-10")
    pd.testing.assert_frame_equal(a, b)


def test_headline_dated_previous_day_does_move_signal():
    a = decayed_signal(_daily(0.9), "2020-03-11")
    b = decayed_signal(_daily(-0.9), "2020-03-11")
    assert a.loc["AAA", "sent"] > b.loc["AAA", "sent"]
