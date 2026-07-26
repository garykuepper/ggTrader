"""Research-bench daily bars must resolve to one row per calendar date.

Regression test for the SPY duplicate-timestamp bug (audit 2026-07-25 §2.0):
`SPY` accumulated 826 rows at 04:00/05:00 (yfinance's native midnight-ET
convention) alongside the canonical 16:00/17:00 rows every other symbol
uses. Because `lab/cli.py:152` and `lab/blend.py:86` join SPY into the
same frame as the universe, the union index gained a second row per
trading day from 2023 on -- every non-SPY symbol NaN on the duplicate --
deflating every equity Sharpe covering 2023+ by ~1/sqrt(2) (measured:
AAPL buy-and-hold 0.848 contaminated vs 1.187 clean; CAGR/MaxDD
unaffected).

The fix lives at the research boundary (`lab.data.load_ohlcv`) rather
than in the shared yfinance loader, which is live-trader runtime code.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.lab.data import collapse_daily_duplicates


def _contaminated_wide() -> pd.DataFrame:
    """Two symbols over two bar dates, where SPY carries an extra row per
    date at the yfinance-native midnight-ET time -- the real bug shape.
    AAPL is NaN on those rows, exactly as the pivot produces."""
    idx = pd.DatetimeIndex(
        [
            "2023-06-01 04:00:00",
            "2023-06-01 16:00:00",
            "2023-06-02 04:00:00",
            "2023-06-02 16:00:00",
        ],
        tz="UTC",
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "close"),
            ("AAPL", "open"),
            ("MSFT", "close"),
            ("MSFT", "open"),
            ("SPY", "close"),
            ("SPY", "open"),
        ]
    )
    data = [
        # contaminating rows carry the stray symbol only -- every other
        # column is NaN, which is what makes them identifiable.
        [None, None, None, None, 20.0, 20.0],
        [10.0, 10.0, 30.0, 30.0, 20.0, 20.0],  # canonical
        [None, None, None, None, 21.0, 21.0],
        [11.0, 11.0, 31.0, 31.0, 21.0, 21.0],  # canonical
    ]
    return pd.DataFrame(data, index=idx, columns=cols, dtype="float64")


class TestCollapseDailyDuplicates:
    def test_duplicate_timestamps_do_not_split_the_index(self):
        out = collapse_daily_duplicates(_contaminated_wide())
        assert len(out) == 2, (
            f"expected one row per bar date, got {len(out)} -- "
            "the contaminating 04:00 rows split the index"
        )

    def test_other_symbols_are_not_nan_holed(self):
        out = collapse_daily_duplicates(_contaminated_wide())
        assert out[("AAPL", "close")].isna().sum() == 0
        assert list(out[("AAPL", "close")]) == [10.0, 11.0]
        assert list(out[("SPY", "close")]) == [20.0, 21.0]

    def test_minority_convention_is_dropped_not_merged(self):
        """The two conventions are offset by one session, so a merge would
        align SPY's Friday close against AAPL's Thursday close. Rows on the
        minority time-of-day must be dropped outright."""
        out = collapse_daily_duplicates(_contaminated_wide())
        tods = {t.strftime("%H:%M") for t in out.index}
        assert tods == {"16:00"}, f"minority 04:00 rows survived: {tods}"

    def test_index_has_no_duplicate_dates(self):
        out = collapse_daily_duplicates(_contaminated_wide())
        dates = pd.Series(out.index.date)
        assert not dates.duplicated().any()

    def test_index_stays_tz_aware_and_sorted(self):
        out = collapse_daily_duplicates(_contaminated_wide())
        assert out.index.tz is not None
        assert out.index.is_monotonic_increasing

    def test_already_clean_frame_is_unchanged(self):
        clean = _contaminated_wide().iloc[[1, 3]]
        out = collapse_daily_duplicates(clean)
        assert len(out) == 2
        assert list(out[("AAPL", "close")]) == [10.0, 11.0]
        assert list(out[("SPY", "close")]) == [20.0, 21.0]

    def test_empty_frame_is_passed_through(self):
        out = collapse_daily_duplicates(pd.DataFrame())
        assert out.empty
