"""Tests for CachedYFinanceLoader's inception tracking and vectorized cache write.

Phase C item 13: a symbol whose true first bar postdates ``start_date`` by
more than 7 days (recent IPO, 2020+ index addition, leveraged ETF vs. a 2010
data_start) must not trigger a full network refetch on every single run once
its inception has been confirmed once. Phase C item 14: ``_cache_to_db``'s
row construction is vectorized (numpy, not iterrows) but must produce
identical rows -- including None for any NaN OHLCV field, and skipping any
row with a NaN close or NaT timestamp -- to the original implementation.

No real DB or network connection is made: DB-touching methods are patched
per test.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader


def _loader() -> CachedYFinanceLoader:
    """A loader instance with no real DB connection (never calls __init__)."""
    loader = object.__new__(CachedYFinanceLoader)
    loader.connection_string = "postgresql://unused/unused"
    import logging

    loader.logger = logging.getLogger("test_cached_yfinance_loader")
    return loader


def _wide(symbol: str, dates, closes) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, tz="UTC")
    cols = pd.MultiIndex.from_tuples(
        [(symbol, m) for m in ("open", "high", "low", "close", "volume")]
    )
    data = [[c, c, c, c, 100.0] for c in closes]
    return pd.DataFrame(data, index=idx, columns=cols, dtype="float64")


class TestCacheToDbVectorized:
    """Item 14: vectorized row construction, output-identical to iterrows."""

    def _captured_records(self, df: pd.DataFrame, interval: str = "1d") -> list[tuple]:
        loader = _loader()
        captured: dict = {}

        def fake_execute_values(cur, query, records, page_size=5000):
            captured["records"] = list(records)

        with patch.object(loader, "_connect", return_value=MagicMock()):
            with patch(
                "ggTrader.data.live.cached_yfinance_loader.execute_values",
                side_effect=fake_execute_values,
            ):
                loader._cache_to_db(df, interval)
        return captured.get("records", [])

    def test_basic_row_shape_and_values(self):
        df = _wide("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 101.0])
        records = self._captured_records(df)
        assert len(records) == 2
        ts0, sym0, interval0, venue0, o, h, low, c, v, trades = records[0]
        assert sym0 == "AAPL"
        assert interval0 == "1d"
        assert venue0 == "yfinance"
        assert o == pytest.approx(100.0)
        assert c == pytest.approx(100.0)
        assert v == pytest.approx(100.0)
        assert trades == 0

    def test_nan_close_row_is_dropped(self):
        df = _wide("AAPL", ["2024-01-02", "2024-01-03"], [100.0, np.nan])
        records = self._captured_records(df)
        assert len(records) == 1
        assert records[0][7] == pytest.approx(100.0)

    def test_nan_optional_field_becomes_none(self):
        df = _wide("AAPL", ["2024-01-02"], [100.0])
        df[("AAPL", "open")] = np.nan
        df[("AAPL", "volume")] = np.nan
        records = self._captured_records(df)
        assert len(records) == 1
        _, _, _, _, o, h, low, c, v, _ = records[0]
        assert o is None
        assert v is None
        assert c == pytest.approx(100.0)

    def test_multi_symbol_frame(self):
        a = _wide("AAPL", ["2024-01-02"], [100.0])
        b = _wide("MSFT", ["2024-01-02"], [200.0])
        df = pd.concat([a, b], axis=1)
        records = self._captured_records(df)
        symbols = {r[1] for r in records}
        assert symbols == {"AAPL", "MSFT"}
        assert len(records) == 2

    def test_empty_frame_produces_no_records(self):
        loader = _loader()
        with patch.object(loader, "_connect") as mock_connect:
            loader._cache_to_db(pd.DataFrame(), "1d")
            mock_connect.assert_not_called()

    def test_all_nan_close_produces_no_db_call(self):
        df = _wide("AAPL", ["2024-01-02"], [np.nan])
        loader = _loader()
        with patch.object(loader, "_connect") as mock_connect:
            loader._cache_to_db(df, "1d")
            mock_connect.assert_not_called()


class TestInceptionTracking:
    """Item 13: known-inception symbols skip the redundant full refetch."""

    def test_unknown_gap_triggers_full_fetch(self):
        """No recorded inception yet -> full fetch as before (first run)."""
        loader = _loader()
        start = pd.Timestamp("2010-01-01", tz="UTC")
        first = pd.Timestamp("2022-06-01", tz="UTC")
        last = pd.Timestamp("2024-01-01", tz="UTC")

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(loader, "_db_coverage", return_value={"NEWCO": (first, last)})
            )
            mock_known = stack.enter_context(
                patch.object(loader, "_get_known_inceptions", return_value={})
            )
            mock_live = stack.enter_context(
                patch(
                    "ggTrader.data.live.cached_yfinance_loader.YFinanceDataLoader.fetch_ohlcv",
                    return_value=_wide("NEWCO", [first], [10.0]),
                )
            )
            stack.enter_context(patch.object(loader, "_cache_to_db"))
            mock_record = stack.enter_context(patch.object(loader, "_record_inceptions"))
            stack.enter_context(
                patch.object(loader, "_db_fetch", return_value=_wide("NEWCO", [first], [10.0]))
            )
            loader.fetch_ohlcv(["NEWCO"], "1d", start_date=start, end_date=last)

        mock_known.assert_called_once()
        assert mock_live.call_count == 1
        full_fetch_symbols = mock_live.call_args[0][0]
        assert full_fetch_symbols == ["NEWCO"]
        # A confirmed inception is recorded after the full fetch.
        mock_record.assert_called_once()
        recorded = mock_record.call_args[0][0]
        assert recorded == {"NEWCO": first}

    def test_confirmed_inception_skips_full_refetch(self):
        """Second run: recorded inception matches the cached first bar ->
        no network full-fetch call for that symbol."""
        loader = _loader()
        start = pd.Timestamp("2010-01-01", tz="UTC")
        first = pd.Timestamp("2022-06-01", tz="UTC")
        last = pd.Timestamp("2024-01-01", tz="UTC")
        now = last  # end freshness satisfied, no incremental fetch either

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(loader, "_db_coverage", return_value={"NEWCO": (first, last)})
            )
            stack.enter_context(
                patch.object(loader, "_get_known_inceptions", return_value={"NEWCO": first})
            )
            mock_live = stack.enter_context(
                patch("ggTrader.data.live.cached_yfinance_loader.YFinanceDataLoader.fetch_ohlcv")
            )
            stack.enter_context(patch.object(loader, "_cache_to_db"))
            mock_record = stack.enter_context(patch.object(loader, "_record_inceptions"))
            stack.enter_context(
                patch.object(loader, "_db_fetch", return_value=_wide("NEWCO", [first], [10.0]))
            )
            stack.enter_context(patch("pandas.Timestamp.now", return_value=now))
            loader.fetch_ohlcv(["NEWCO"], "1d", start_date=start, end_date=now)

        mock_live.assert_not_called()
        mock_record.assert_not_called()

    def test_mismatched_recorded_inception_forces_refetch(self):
        """If the cached first bar regressed away from the recorded
        inception (e.g. table repopulated), a full refetch happens again --
        genuinely missing history is never silently accepted."""
        loader = _loader()
        start = pd.Timestamp("2010-01-01", tz="UTC")
        recorded_first = pd.Timestamp("2022-06-01", tz="UTC")
        cached_first = pd.Timestamp("2022-09-01", tz="UTC")  # later than recorded -> gap widened
        last = pd.Timestamp("2024-01-01", tz="UTC")

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(loader, "_db_coverage", return_value={"NEWCO": (cached_first, last)})
            )
            stack.enter_context(
                patch.object(
                    loader, "_get_known_inceptions", return_value={"NEWCO": recorded_first}
                )
            )
            mock_live = stack.enter_context(
                patch(
                    "ggTrader.data.live.cached_yfinance_loader.YFinanceDataLoader.fetch_ohlcv",
                    return_value=_wide("NEWCO", [cached_first], [10.0]),
                )
            )
            stack.enter_context(patch.object(loader, "_cache_to_db"))
            stack.enter_context(patch.object(loader, "_record_inceptions"))
            stack.enter_context(
                patch.object(
                    loader, "_db_fetch", return_value=_wide("NEWCO", [cached_first], [10.0])
                )
            )
            loader.fetch_ohlcv(["NEWCO"], "1d", start_date=start, end_date=last)

        assert mock_live.call_count == 1
        assert mock_live.call_args[0][0] == ["NEWCO"]

    def test_genuine_gappy_cache_with_no_confirmation_still_refetches(self):
        """A symbol never confirmed (known-inceptions lookup empty because
        the table has no row for it, e.g. a prior fetch failure) always
        gets a full refetch -- we never suppress a refetch for data that
        might genuinely be fetchable."""
        loader = _loader()
        start = pd.Timestamp("2010-01-01", tz="UTC")
        first = pd.Timestamp("2015-01-01", tz="UTC")
        last = pd.Timestamp("2024-01-01", tz="UTC")

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(loader, "_db_coverage", return_value={"GAPPY": (first, last)})
            )
            stack.enter_context(patch.object(loader, "_get_known_inceptions", return_value={}))
            mock_live = stack.enter_context(
                patch(
                    "ggTrader.data.live.cached_yfinance_loader.YFinanceDataLoader.fetch_ohlcv",
                    return_value=_wide("GAPPY", [first], [10.0]),
                )
            )
            stack.enter_context(patch.object(loader, "_cache_to_db"))
            stack.enter_context(patch.object(loader, "_record_inceptions"))
            stack.enter_context(
                patch.object(loader, "_db_fetch", return_value=_wide("GAPPY", [first], [10.0]))
            )
            loader.fetch_ohlcv(["GAPPY"], "1d", start_date=start, end_date=last)

        assert mock_live.call_count == 1

    def test_no_start_date_skips_inception_lookup_entirely(self):
        """start_date=None means the start-gap check never fires, so the
        inception-lookup query is never issued (matches prior behavior)."""
        loader = _loader()
        first = pd.Timestamp("2022-06-01", tz="UTC")
        last = pd.Timestamp("2024-01-01", tz="UTC")

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(loader, "_db_coverage", return_value={"NEWCO": (first, last)})
            )
            mock_known = stack.enter_context(patch.object(loader, "_get_known_inceptions"))
            mock_live = stack.enter_context(
                patch("ggTrader.data.live.cached_yfinance_loader.YFinanceDataLoader.fetch_ohlcv")
            )
            stack.enter_context(patch.object(loader, "_cache_to_db"))
            stack.enter_context(patch.object(loader, "_record_inceptions"))
            stack.enter_context(
                patch.object(loader, "_db_fetch", return_value=_wide("NEWCO", [first], [10.0]))
            )
            stack.enter_context(patch("pandas.Timestamp.now", return_value=last))
            loader.fetch_ohlcv(["NEWCO"], "1d", start_date=None, end_date=last)

        mock_known.assert_not_called()
        mock_live.assert_not_called()


class TestCacheToDbTimestampNaiveUtc:
    """Regression: bars must reach the DB as naive UTC on their true date.

    ``ohlcv.timestamp`` is `timestamp WITHOUT time zone`. Handing psycopg2 a
    tz-aware datetime let Postgres rebase the offset into the session
    timezone (``America/Los_Angeles`` on this host), storing
    ``2026-08-03 00:00+00:00`` as ``2026-08-02 17:00`` -- every equity bar
    one calendar day EARLY. The whole 5.68M-row store ran Sun-Thu (1.07M
    Sunday bars vs 168 Friday ones) before this was caught on 2026-08-20.
    """

    def _timestamps(self, df: pd.DataFrame) -> list:
        return [r[0] for r in TestCacheToDbVectorized()._captured_records(df)]

    def test_tz_aware_index_is_stripped_to_naive(self):
        ts = self._timestamps(_wide("AAPL", ["2026-08-03"], [303.16]))
        assert ts[0].tzinfo is None, "tz-aware datetime lets Postgres shift the date"

    def test_calendar_date_is_preserved(self):
        """The bar for Mon 2026-08-03 must not land on Sun 2026-08-02."""
        ts = self._timestamps(_wide("AAPL", ["2026-08-03"], [303.16]))
        assert ts[0].date() == date(2026, 8, 3)
        assert ts[0].hour == 0

    def test_no_weekend_stamps_for_a_trading_week(self):
        """A Mon-Fri week must stay Mon-Fri, not slide back to Sun-Thu."""
        week = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
        ts = self._timestamps(_wide("AAPL", week, [1.0] * 5))
        assert [t.date().isoformat() for t in ts] == week
        assert not [t for t in ts if t.weekday() >= 5]

    def test_naive_index_passes_through_unchanged(self):
        """A tz-naive index is already naive UTC -- don't touch it."""
        df = _wide("AAPL", ["2026-08-03"], [303.16])
        df.index = df.index.tz_localize(None)
        ts = self._timestamps(df)
        assert ts[0].tzinfo is None
        assert ts[0].date() == date(2026, 8, 3)


@pytest.mark.integration
class TestCacheToDbRoundTripDate:
    """The test that would have caught the day-shift: a real DB round-trip.

    The unit tests above can only assert what Python hands psycopg2. The
    corruption happened *inside* Postgres (tz-aware literal rebased into the
    session timezone on insert into a naive column), so only a real
    write-then-read proves the calendar date survives. Uses a throwaway
    symbol and cleans up after itself.
    """

    SYMBOL = "__TZ_ROUNDTRIP__"

    def test_written_bar_reads_back_on_the_same_calendar_date(self):
        import psycopg2

        from ggTrader.data.live.cached_yfinance_loader import STOCK_VENUE
        from ggTrader.utils.config import get_db_connection_string

        dsn = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
        week = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
        df = _wide(self.SYMBOL, week, [100.0] * 5)

        loader = object.__new__(CachedYFinanceLoader)
        loader.connection_string = dsn
        import logging

        loader.logger = logging.getLogger("tz_roundtrip")

        try:
            loader._cache_to_db(df, "1d")
            with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT timestamp FROM ohlcv WHERE symbol=%s AND interval='1d' "
                    "AND venue=%s ORDER BY timestamp",
                    (self.SYMBOL, STOCK_VENUE),
                )
                got = [r[0].date().isoformat() for r in cur.fetchall()]
            assert got == week, f"calendar dates shifted on the DB round-trip: {got}"
        finally:
            with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM ohlcv WHERE symbol=%s", (self.SYMBOL,))
                conn.commit()
