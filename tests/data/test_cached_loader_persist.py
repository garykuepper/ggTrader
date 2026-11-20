"""Tests for the public CachedYFinanceLoader.persist() writer entry point.

persist() is a one-line delegation to the private ``_cache_to_db``, which keeps every line of
its tz-stripping and upsert logic. No real DB or network connection is made.
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader

OHLCV = ["open", "high", "low", "close", "volume"]


def _loader() -> CachedYFinanceLoader:
    """A loader with no real connection (``__init__`` is never run)."""
    loader = object.__new__(CachedYFinanceLoader)
    loader.logger = logging.getLogger("test.persist")
    return loader


def _utc_frame() -> pd.DataFrame:
    """A tz-aware UTC-midnight frame, the shape the loader hands the writer."""
    return pd.DataFrame(
        [[1.0, 2.0, 0.5, 1.5, 100.0]],
        index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
        columns=pd.MultiIndex.from_product([["SPY"], OHLCV]),
    )


def test_persist_exists_on_the_class() -> None:
    assert callable(getattr(CachedYFinanceLoader, "persist", None))
    params = list(inspect.signature(CachedYFinanceLoader.persist).parameters)
    assert params == ["self", "df", "interval"]


def test_persist_forwards_the_frame_and_interval_untouched() -> None:
    loader = _loader()
    frame = _utc_frame()
    before = frame.copy(deep=True)
    loader._cache_to_db = MagicMock()

    assert loader.persist(frame, "1d") is None

    loader._cache_to_db.assert_called_once()
    (got_frame, got_interval), kwargs = loader._cache_to_db.call_args
    assert got_frame is frame
    assert got_interval == "1d"
    assert kwargs == {}
    assert str(got_frame.index.tz) == "UTC"
    pd.testing.assert_frame_equal(frame, before)


def test_persist_swallows_nothing_new() -> None:
    loader = _loader()
    loader._cache_to_db = MagicMock(side_effect=RuntimeError("write"))
    with pytest.raises(RuntimeError, match="write"):
        loader.persist(_utc_frame(), "1d")


def test_persist_through_the_real_writer_still_swallows_driver_errors(monkeypatch, caplog) -> None:
    loader = _loader()
    conn = MagicMock()
    monkeypatch.setattr(loader, "_connect", lambda: conn)

    def boom(*args, **kwargs):
        raise RuntimeError("driver down")

    monkeypatch.setattr("ggTrader.data.live.cached_yfinance_loader.execute_values", boom)

    with caplog.at_level(logging.ERROR, logger="test.persist"):
        loader.persist(_utc_frame(), "1d")  # must not raise

    assert conn.close.called
    assert "Failed to cache stock data to DB: driver down" in caplog.text
