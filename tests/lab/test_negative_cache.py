"""Tests for the OHLCV negative cache (skip permanently-delisted symbols)."""

from __future__ import annotations

import pytest
from sqlalchemy import text


@pytest.mark.integration
def test_record_and_load_roundtrip():
    from ggTrader.lab.negative_cache import ensure_schema, load_skip_symbols, record_no_data
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_DEAD"
    # Clean any prior run
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM ohlcv_no_data WHERE symbol = :s"), {"s": marker})

    record_no_data([marker], "1d")
    skip = load_skip_symbols("1d")
    assert marker in skip

    # Different interval must not pick it up
    assert marker not in load_skip_symbols("1h")

    # TTL=0 days excludes the just-written row (checked_at is not >= now())
    assert marker not in load_skip_symbols("1d", ttl_days=0)

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM ohlcv_no_data WHERE symbol = :s"), {"s": marker})


@pytest.mark.integration
def test_record_no_data_is_idempotent():
    from ggTrader.lab.negative_cache import ensure_schema, load_skip_symbols, record_no_data
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_DUP"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM ohlcv_no_data WHERE symbol = :s"), {"s": marker})

    record_no_data([marker, marker], "1d")
    record_no_data([marker], "1d")  # upsert, no duplicate-key error

    with get_engine().connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM ohlcv_no_data WHERE symbol = :s"), {"s": marker}
        ).scalar()
    assert n == 1
    assert marker in load_skip_symbols("1d")

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM ohlcv_no_data WHERE symbol = :s"), {"s": marker})


def test_record_no_data_empty_is_noop(monkeypatch):
    """Empty input must not touch the DB at all."""
    import ggTrader.lab.negative_cache as nc

    called = False

    def _boom():
        nonlocal called
        called = True
        raise AssertionError("get_engine should not be called for empty input")

    monkeypatch.setattr(nc, "get_engine", _boom)
    nc.record_no_data([], "1d")
    assert called is False
