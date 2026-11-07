"""Tests for the FINRA consolidated short-interest data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.short_interest_data import (
    PUBLISH_LAG_DAYS,
    available_as_of,
    discover_settlement_dates,
    fetch_settlement_date,
)


def _raw_row(symbol, settlement_date, current=1000, previous=900, dtc=2.5, adv=500, pct=11.1):
    return {
        "symbolCode": symbol,
        "settlementDate": settlement_date,
        "currentShortPositionQuantity": current,
        "previousShortPositionQuantity": previous,
        "daysToCoverQuantity": dtc,
        "averageDailyVolumeQuantity": adv,
        "changePercent": pct,
    }


class TestFetchSettlementDate:
    def test_parses_single_page_into_expected_columns(self):
        page = [_raw_row("AAPL", "2026-06-15"), _raw_row("MSFT", "2026-06-15")]

        def fake_http(payload):
            return page if payload["offset"] == 0 else []

        df = fetch_settlement_date("2026-06-15", http_fetch=fake_http)
        assert set(df.columns) == {
            "symbol",
            "settlement_date",
            "current_short_position",
            "previous_short_position",
            "days_to_cover",
            "avg_daily_volume",
            "change_percent",
        }
        assert set(df["symbol"]) == {"AAPL", "MSFT"}
        assert df.loc[df["symbol"] == "AAPL", "days_to_cover"].iloc[0] == pytest.approx(2.5)

    def test_paginates_until_a_short_page(self):
        """A full PAGE_SIZE page must trigger another fetch; a short page ends it."""
        from ggTrader.lab.short_interest_data import PAGE_SIZE

        full_page = [_raw_row(f"SYM{i}", "2026-06-15") for i in range(PAGE_SIZE)]
        short_page = [_raw_row("LAST", "2026-06-15")]
        calls = []

        def fake_http(payload):
            calls.append(payload["offset"])
            if payload["offset"] == 0:
                return full_page
            if payload["offset"] == PAGE_SIZE:
                return short_page
            return []

        df = fetch_settlement_date("2026-06-15", http_fetch=fake_http)
        assert calls == [0, PAGE_SIZE]
        assert len(df) == PAGE_SIZE + 1
        assert "LAST" in set(df["symbol"])

    def test_empty_response_returns_empty_frame_with_expected_columns(self):
        df = fetch_settlement_date("2026-06-15", http_fetch=lambda payload: [])
        assert df.empty
        assert "symbol" in df.columns
        assert "days_to_cover" in df.columns


class TestDiscoverSettlementDates:
    def test_returns_sorted_unique_dates(self):
        """Settlement dates nominally fall on the 15th/month-end but shift to
        the nearest business day around weekends/holidays -- this discovers
        the real ones from data rather than assuming a fixed calendar rule."""
        rows = [
            _raw_row("AAPL", "2020-05-29"),  # 05-31 was a Sunday, shifted
            _raw_row("AAPL", "2020-05-15"),
            _raw_row("AAPL", "2020-04-30"),
        ]

        def fake_http(payload):
            return rows if payload["offset"] == 0 else []

        dates = discover_settlement_dates("2020-04-01", "2020-05-31", http_fetch=fake_http)
        assert dates == ["2020-04-30", "2020-05-15", "2020-05-29"]

    def test_uses_anchor_symbol_filter(self):
        seen_payloads = []

        def fake_http(payload):
            seen_payloads.append(payload)
            return [] if payload["offset"] > 0 else [_raw_row("MSFT", "2020-04-30")]

        discover_settlement_dates(
            "2020-04-01", "2020-04-30", anchor_symbol="MSFT", http_fetch=fake_http
        )
        assert seen_payloads[0]["compareFilters"][0]["fieldValue"] == "MSFT"

    def test_paginates_like_fetch_settlement_date(self):
        from ggTrader.lab.short_interest_data import PAGE_SIZE

        full_page = [_raw_row("AAPL", f"2020-0{(i % 9) + 1}-15") for i in range(PAGE_SIZE)]
        short_page = [_raw_row("AAPL", "2020-12-15")]
        calls = []

        def fake_http(payload):
            calls.append(payload["offset"])
            if payload["offset"] == 0:
                return full_page
            if payload["offset"] == PAGE_SIZE:
                return short_page
            return []

        dates = discover_settlement_dates("2020-01-01", "2020-12-31", http_fetch=fake_http)
        assert calls == [0, PAGE_SIZE]
        assert "2020-12-15" in dates

    def test_empty_response_returns_empty_list(self):
        assert discover_settlement_dates("2020-01-01", "2020-12-31", http_fetch=lambda p: []) == []


class TestAvailableAsOf:
    def test_excludes_rows_within_the_publish_lag(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "settlement_date": pd.to_datetime(["2026-06-01", "2026-06-15"]),
            }
        )
        asof = pd.Timestamp("2026-06-20")  # 19 days after 06-01, only 5 after 06-15
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        # 06-01 settlement (+15 lag = 06-16) has been published by 06-20;
        # 06-15 (+15 = 06-30) has not.
        assert list(out["settlement_date"].dt.strftime("%Y-%m-%d")) == ["2026-06-01"]

    def test_includes_row_exactly_at_the_lag_boundary(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "settlement_date": pd.to_datetime(["2026-06-01"])})
        asof = pd.Timestamp("2026-06-01") + pd.Timedelta(days=PUBLISH_LAG_DAYS)
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert len(out) == 1

    def test_no_rows_available_before_any_lag_has_elapsed(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "settlement_date": pd.to_datetime(["2026-06-15"])})
        asof = pd.Timestamp("2026-06-16")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert out.empty

    def test_handles_tz_aware_asof_against_naive_settlement_dates(self):
        """FINRA settlement dates parse as tz-naive; every OHLCV-derived asof
        in this lab is tz-aware UTC -- must not raise/misbehave on the mix."""
        df = pd.DataFrame({"symbol": ["AAPL"], "settlement_date": pd.to_datetime(["2026-06-01"])})
        asof = pd.Timestamp("2026-06-20", tz="UTC")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert len(out) == 1

    def test_returned_frame_has_tz_naive_settlement_date_even_if_input_was_tz_aware(self):
        """Regression: the filter mask was tz-normalized internally but the
        RETURNED dataframe's column kept the original tz-aware values --
        any downstream datetime arithmetic on the result (e.g. a trend
        calc's age-in-days) would crash with a tz-naive/tz-aware mismatch."""
        df = pd.DataFrame(
            {"symbol": ["AAPL"], "settlement_date": [pd.Timestamp("2026-06-01", tz="UTC")]}
        )
        asof = pd.Timestamp("2026-06-20", tz="UTC")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert out["settlement_date"].dt.tz is None
        age = pd.Timestamp(asof).tz_localize(None) - out["settlement_date"].iloc[0]
        assert age.days == 19


@pytest.mark.integration
def test_cache_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.persist import get_engine
    from ggTrader.lab.short_interest_data import (
        cache_settlement_date,
        ensure_schema,
        load_short_interest,
    )

    ensure_schema()
    marker = "ZZTEST_SI"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM short_interest WHERE symbol = :s"), {"s": marker})

    page = [_raw_row(marker, "2026-06-15", dtc=3.2, pct=25.0)]
    n = cache_settlement_date(
        "2026-06-15", http_fetch=lambda payload: page if payload["offset"] == 0 else []
    )
    assert n == 1

    df = load_short_interest([marker], "2026-01-01", "2026-12-31")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == marker
    assert df.iloc[0]["days_to_cover"] == pytest.approx(3.2)

    # Re-caching the same settlement date upserts, not duplicates.
    cache_settlement_date(
        "2026-06-15", http_fetch=lambda payload: page if payload["offset"] == 0 else []
    )
    df2 = load_short_interest([marker], "2026-01-01", "2026-12-31")
    assert len(df2) == 1

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM short_interest WHERE symbol = :s"), {"s": marker})


@pytest.mark.integration
def test_cache_settlement_date_filters_to_symbol_universe():
    from sqlalchemy import text

    from ggTrader.lab.persist import get_engine
    from ggTrader.lab.short_interest_data import (
        cache_settlement_date,
        ensure_schema,
        load_short_interest,
    )

    ensure_schema()
    keep, drop = "ZZTEST_KEEP", "ZZTEST_DROP"
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM short_interest WHERE symbol IN (:a, :b)"), {"a": keep, "b": drop}
        )

    page = [_raw_row(keep, "2026-06-15"), _raw_row(drop, "2026-06-15")]
    n = cache_settlement_date(
        "2026-06-15",
        symbols=[keep],
        http_fetch=lambda payload: page if payload["offset"] == 0 else [],
    )
    assert n == 1

    df = load_short_interest([keep, drop], "2026-01-01", "2026-12-31")
    assert list(df["symbol"]) == [keep]

    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM short_interest WHERE symbol IN (:a, :b)"), {"a": keep, "b": drop}
        )
