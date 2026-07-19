"""Tests for the Google Trends (retail search-attention) data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.google_trends_data import PUBLISH_LAG_DAYS, available_as_of, parse_interest_series


class _FakeInterestFrame:
    """Mimics pytrends' interest_over_time() DataFrame shape: DatetimeIndex
    named 'date', one column per queried term, plus an 'isPartial' column."""

    def __init__(self, dates, values, column):
        self._df = pd.DataFrame({column: values, "isPartial": [False] * len(values)}, index=dates)
        self._df.index.name = "date"

    def __call__(self):
        return self._df


class TestParseInterestSeries:
    def test_extracts_values_for_the_query_column(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        raw = pd.DataFrame(
            {"AAPL stock": [10, 25, 15], "isPartial": [False, False, True]}, index=dates
        )
        raw.index.name = "date"
        series = parse_interest_series(raw, "AAPL stock")
        assert list(series.values) == [10, 25, 15]
        assert list(series.index) == list(dates)

    def test_empty_frame_returns_empty_series(self):
        raw = pd.DataFrame(columns=["AAPL stock", "isPartial"])
        series = parse_interest_series(raw, "AAPL stock")
        assert series.empty

    def test_missing_column_returns_empty_series(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        raw = pd.DataFrame({"other": [1, 2], "isPartial": [False, False]}, index=dates)
        series = parse_interest_series(raw, "AAPL stock")
        assert series.empty


class TestAvailableAsOf:
    def test_excludes_dates_within_the_publish_lag(self):
        df = pd.DataFrame(
            {"symbol": ["AAPL", "AAPL"], "date": pd.to_datetime(["2026-06-01", "2026-06-15"])}
        )
        asof = pd.Timestamp("2026-06-10")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-06-01"]

    def test_handles_tz_aware_asof(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "date": pd.to_datetime(["2026-06-01"])})
        asof = pd.Timestamp("2026-06-15", tz="UTC")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert len(out) == 1
        assert out["date"].dt.tz is None


@pytest.mark.integration
def test_cache_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.google_trends_data import (
        cache_symbol_interest,
        ensure_schema,
        load_search_interest,
    )
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_GT"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM google_trends_interest WHERE symbol = :s"), {"s": marker})

    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    raw = pd.DataFrame(
        {f"{marker} stock": [10, 25, 15], "isPartial": [False, False, True]}, index=dates
    )
    raw.index.name = "date"
    n = cache_symbol_interest(
        marker, "2020-01-01", "2020-03-01", query_fn=lambda term, timeframe: raw
    )
    assert n == 3

    df = load_search_interest([marker], "2020-01-01", "2020-12-31")
    assert len(df) == 3
    assert df.iloc[0]["symbol"] == marker

    # Re-caching upserts, not duplicates.
    cache_symbol_interest(marker, "2020-01-01", "2020-03-01", query_fn=lambda term, timeframe: raw)
    df2 = load_search_interest([marker], "2020-01-01", "2020-12-31")
    assert len(df2) == 3

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM google_trends_interest WHERE symbol = :s"), {"s": marker})
