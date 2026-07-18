"""Tests for the yfinance earnings-surprise (PEAD) data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.earnings_surprise_data import (
    REPORT_LAG_DAYS,
    available_as_of,
    fetch_symbol_surprises,
)


def _yf_earnings_dates_frame(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """Mimics yfinance's Ticker.get_earnings_dates() return shape: a
    DatetimeIndex named 'Earnings Date' with EPS Estimate/Reported EPS/
    Surprise(%) columns, most-recent first (yfinance's natural order)."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") for d, *_ in rows], name="Earnings Date"
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [r[1] for r in rows],
            "Reported EPS": [r[2] for r in rows],
            "Surprise(%)": [r[3] for r in rows],
        },
        index=idx,
    )


class TestFetchSymbolSurprises:
    def test_parses_into_expected_columns(self):
        raw = _yf_earnings_dates_frame(
            [("2026-01-29", 2.67, 2.84, 6.34), ("2025-10-30", 1.77, 1.85, 4.52)]
        )
        df = fetch_symbol_surprises("AAPL", earnings_dates_fn=lambda symbol, limit: raw)
        assert set(df.columns) == {
            "symbol",
            "earnings_date",
            "eps_estimate",
            "eps_reported",
            "surprise_pct",
        }
        assert set(df["symbol"]) == {"AAPL"}
        assert df["surprise_pct"].tolist() == pytest.approx([6.34, 4.52])

    def test_drops_rows_with_no_reported_eps_yet(self):
        """A future/unreported earnings date (Reported EPS is NaN) must be
        dropped -- it isn't a realized surprise and can't drive a signal."""
        raw = _yf_earnings_dates_frame([("2026-07-30", 1.89, float("nan"), float("nan"))])
        df = fetch_symbol_surprises("AAPL", earnings_dates_fn=lambda symbol, limit: raw)
        assert df.empty

    def test_none_response_returns_empty_frame_with_expected_columns(self):
        df = fetch_symbol_surprises("AAPL", earnings_dates_fn=lambda symbol, limit: None)
        assert df.empty
        assert "surprise_pct" in df.columns

    def test_empty_response_returns_empty_frame(self):
        df = fetch_symbol_surprises("AAPL", earnings_dates_fn=lambda symbol, limit: pd.DataFrame())
        assert df.empty


class TestAvailableAsOf:
    def test_excludes_earnings_within_the_report_lag(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "earnings_date": pd.to_datetime(["2026-06-01", "2026-06-15"]),
            }
        )
        asof = pd.Timestamp("2026-06-10")  # 9 days after 06-01, before 06-15 even happens
        out = available_as_of(df, asof, lag_days=REPORT_LAG_DAYS)
        assert list(out["earnings_date"].dt.strftime("%Y-%m-%d")) == ["2026-06-01"]

    def test_handles_tz_aware_asof_against_naive_earnings_dates(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "earnings_date": pd.to_datetime(["2026-06-01"])})
        asof = pd.Timestamp("2026-06-05", tz="UTC")
        out = available_as_of(df, asof, lag_days=REPORT_LAG_DAYS)
        assert len(out) == 1

    def test_returned_frame_has_tz_naive_earnings_date_even_if_input_was_tz_aware(self):
        """Regression: the filter mask was tz-normalized internally but the
        RETURNED dataframe's column kept the original tz-aware values --
        any downstream datetime arithmetic on the result (e.g. computing an
        age-in-days) would crash with a tz-naive/tz-aware mismatch."""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "earnings_date": [pd.Timestamp("2026-06-01", tz="UTC")],
            }
        )
        asof = pd.Timestamp("2026-06-05", tz="UTC")
        out = available_as_of(df, asof, lag_days=REPORT_LAG_DAYS)
        assert out["earnings_date"].dt.tz is None
        # Must not raise: tz-naive minus tz-naive.
        age = pd.Timestamp(asof).tz_localize(None) - out["earnings_date"].iloc[0]
        assert age.days == 4


@pytest.mark.integration
def test_cache_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.earnings_surprise_data import (
        cache_symbol_surprises,
        ensure_schema,
        load_earnings_surprises,
    )
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_EARN"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM earnings_surprise WHERE symbol = :s"), {"s": marker})

    raw = _yf_earnings_dates_frame([("2026-01-29", 2.67, 2.84, 6.34)])
    n = cache_symbol_surprises(marker, earnings_dates_fn=lambda symbol, limit: raw)
    assert n == 1

    df = load_earnings_surprises([marker], "2025-01-01", "2026-12-31")
    assert len(df) == 1
    assert df.iloc[0]["surprise_pct"] == pytest.approx(6.34)

    # Re-caching upserts, not duplicates.
    cache_symbol_surprises(marker, earnings_dates_fn=lambda symbol, limit: raw)
    df2 = load_earnings_surprises([marker], "2025-01-01", "2026-12-31")
    assert len(df2) == 1

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM earnings_surprise WHERE symbol = :s"), {"s": marker})
