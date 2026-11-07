"""Tests for the FINRA daily short-sale volume data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.short_volume_data import (
    EARLIEST_DATE,
    available_as_of,
    parse_daily_file,
)


def _raw_file(rows: list[tuple[str, str, int, int, int]]) -> str:
    header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    body = "\n".join(
        f"{date}|{symbol}|{short_vol}|0|{total_vol}|B,Q,N"
        for date, symbol, short_vol, _, total_vol in rows
    )
    return header + body


class TestParseDailyFile:
    def test_extracts_short_volume_ratio(self):
        raw = _raw_file([("20250715", "AAPL", 500, 0, 1000), ("20250715", "MSFT", 250, 0, 1000)])
        rows = parse_daily_file(raw)
        assert len(rows) == 2
        aapl = next(r for r in rows if r["symbol"] == "AAPL")
        assert aapl["date"] == pd.Timestamp("2025-07-15")
        assert aapl["short_volume"] == 500
        assert aapl["total_volume"] == 1000
        assert aapl["short_volume_ratio"] == pytest.approx(0.5)

    def test_zero_total_volume_gives_nan_ratio(self):
        raw = _raw_file([("20250715", "DEAD", 0, 0, 0)])
        rows = parse_daily_file(raw)
        assert pd.isna(rows[0]["short_volume_ratio"])

    def test_empty_file_returns_empty_list(self):
        raw = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        rows = parse_daily_file(raw)
        assert rows == []


class TestAvailableAsOf:
    def test_excludes_dates_within_the_publish_lag(self):
        df = pd.DataFrame(
            {"symbol": ["AAPL", "AAPL"], "date": pd.to_datetime(["2026-06-01", "2026-06-05"])}
        )
        asof = pd.Timestamp("2026-06-03")
        out = available_as_of(df, asof, lag_days=1)
        assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-06-01"]

    def test_handles_tz_aware_asof(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "date": pd.to_datetime(["2026-06-01"])})
        asof = pd.Timestamp("2026-06-05", tz="UTC")
        out = available_as_of(df, asof, lag_days=1)
        assert len(out) == 1
        assert out["date"].dt.tz is None


def test_earliest_date_matches_verified_finra_cdn_retention_boundary():
    assert EARLIEST_DATE == "2018-08-01"


@pytest.mark.integration
def test_cache_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.persist import get_engine
    from ggTrader.lab.short_volume_data import cache_daily_file, ensure_schema, load_short_volume

    ensure_schema()
    marker = "ZZTEST_SV"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM short_volume WHERE symbol = :s"), {"s": marker})

    raw = _raw_file([("20250715", marker, 500, 0, 1000)])
    n = cache_daily_file("2025-07-15", http_fetch=lambda url: raw)
    assert n == 1

    df = load_short_volume([marker], "2025-01-01", "2025-12-31")
    assert len(df) == 1
    assert df.iloc[0]["short_volume_ratio"] == pytest.approx(0.5)

    # Re-caching upserts, not duplicates.
    cache_daily_file("2025-07-15", http_fetch=lambda url: raw)
    df2 = load_short_volume([marker], "2025-01-01", "2025-12-31")
    assert len(df2) == 1

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM short_volume WHERE symbol = :s"), {"s": marker})


@pytest.mark.integration
def test_cache_daily_file_filters_to_symbol_universe():
    from sqlalchemy import text

    from ggTrader.lab.persist import get_engine
    from ggTrader.lab.short_volume_data import cache_daily_file, ensure_schema, load_short_volume

    ensure_schema()
    keep, drop = "ZZTEST_KEEP", "ZZTEST_DROP"
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM short_volume WHERE symbol IN (:a, :b)"), {"a": keep, "b": drop}
        )

    raw = _raw_file([("20250715", keep, 500, 0, 1000), ("20250715", drop, 200, 0, 1000)])
    n = cache_daily_file("2025-07-15", symbols=[keep], http_fetch=lambda url: raw)
    assert n == 1

    df = load_short_volume([keep, drop], "2025-01-01", "2025-12-31")
    assert list(df["symbol"]) == [keep]

    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM short_volume WHERE symbol IN (:a, :b)"), {"a": keep, "b": drop}
        )
