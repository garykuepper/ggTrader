"""Tests for CAGR and benchmark helpers in benchmarking.py."""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ggTrader.core.benchmarking import (
    _buy_hold_portfolio_stats,
    _btc_buy_hold_portfolio_stats,
    _cagr_percent,
    _sp500_buy_hold_portfolio_stats,
    _years_from_price_index,
)


# ---------------------------------------------------------------------------
# _cagr_percent
# ---------------------------------------------------------------------------

def test_cagr_percent_matches_geometric_mean() -> None:
    """13.72% over 3 calendar years -> ~4.4% CAGR."""
    cagr = _cagr_percent(13.72, 3.0)
    expected = (1.1372 ** (1.0 / 3.0) - 1.0) * 100.0
    assert cagr == pytest.approx(expected, rel=1e-6)


def test_cagr_percent_invalid_years() -> None:
    assert math.isnan(_cagr_percent(10.0, 0.0))
    assert math.isnan(_cagr_percent(10.0, float("nan")))


def test_cagr_percent_negative_return() -> None:
    # -50% over 2 years → (0.5^0.5 - 1)*100 ≈ -29.3%
    cagr = _cagr_percent(-50.0, 2.0)
    expected = (0.5 ** 0.5 - 1.0) * 100.0
    assert cagr == pytest.approx(expected, rel=1e-6)


def test_cagr_percent_total_loss() -> None:
    # -100% → mult=0, which is not > 0 → nan
    assert math.isnan(_cagr_percent(-100.0, 2.0))


# ---------------------------------------------------------------------------
# _years_from_price_index
# ---------------------------------------------------------------------------

def test_years_from_price_index() -> None:
    idx = pd.date_range("2023-01-01", periods=10, freq="4h")
    y = _years_from_price_index(idx)
    assert y > 0
    assert math.isfinite(y)


def test_years_from_price_index_single_bar() -> None:
    idx = pd.date_range("2023-01-01", periods=1, freq="4h")
    assert math.isnan(_years_from_price_index(idx))


# ---------------------------------------------------------------------------
# _buy_hold_portfolio_stats (shared helper)
# ---------------------------------------------------------------------------

def _rising_close(n=80, start=100.0, end=110.0, col="ASSET"):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({col: np.linspace(start, end, n)}, index=idx)


def test_buy_hold_portfolio_stats_positive_return() -> None:
    close_df = _rising_close()
    config = {"START_CASH": 1000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    out = _buy_hold_portfolio_stats(close_df, "ASSET", config)
    assert out["total_trades"] >= 1
    assert out["profit_pct"] is not None
    assert out["profit_pct"] > 0
    assert out["cagr_pct"] is not None


def test_buy_hold_portfolio_stats_empty_returns_empty() -> None:
    close_df = pd.DataFrame({"ASSET": []})
    config = {"START_CASH": 1000.0, "FREQ": "1D"}
    out = _buy_hold_portfolio_stats(close_df, "ASSET", config)
    assert out["profit_pct"] is None
    assert out["total_trades"] == 0


# ---------------------------------------------------------------------------
# _btc_buy_hold_portfolio_stats
# ---------------------------------------------------------------------------

def _make_ohlcv_with_btc(n=200):
    """Build minimal MultiIndex OHLCV with BTC-USD already present."""
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = np.linspace(20000, 30000, n)
    data = {}
    for field in ["open", "high", "low", "close", "volume"]:
        data[("BTC-USD", field)] = close
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def test_btc_bnh_uses_existing_ohlcv_without_db_fetch() -> None:
    """When BTC-USD is already in the close columns, no DB fetch should occur."""
    ohlcv = _make_ohlcv_with_btc()
    close = ohlcv.xs("close", axis=1, level=1)
    config = {"BENCHMARK_SYMBOL": "BTC-USD", "START_CASH": 1000.0, "FEES": 0.001,
              "SLIPPAGE": 0.0005, "FREQ": "4h"}
    with patch("ggTrader.data.historical.timescaledb_loader.TimescaleDBLoader") as mock_db:
        out = _btc_buy_hold_portfolio_stats(close, config)
        mock_db.assert_not_called()
    assert out.get("profit_pct") is not None
    assert out.get("benchmark_symbol") == "BTC-USD"


def test_btc_bnh_empty_close_returns_empty() -> None:
    close = pd.DataFrame()
    config = {"BENCHMARK_SYMBOL": "BTC-USD", "START_CASH": 1000.0, "FREQ": "4h"}
    out = _btc_buy_hold_portfolio_stats(close, config)
    assert out["profit_pct"] is None
    assert out["total_trades"] == 0


# ---------------------------------------------------------------------------
# _sp500_buy_hold_portfolio_stats
# ---------------------------------------------------------------------------

def test_sp500_bnh_returns_empty_on_yfinance_failure() -> None:
    """If yfinance raises, the function should return an empty-stats dict, not raise."""
    idx = pd.date_range("2023-01-01", periods=100, freq="4h", tz="UTC")
    config = {"START_CASH": 1000.0, "FREQ": "4h"}
    with patch("ggTrader.core.benchmarking._load_spy_close", side_effect=RuntimeError("network error")):
        out = _sp500_buy_hold_portfolio_stats(idx, config)
    assert out["profit_pct"] is None
    assert out["total_trades"] == 0


def test_sp500_bnh_empty_index_returns_empty() -> None:
    idx = pd.DatetimeIndex([])
    config = {"START_CASH": 1000.0, "FREQ": "4h"}
    out = _sp500_buy_hold_portfolio_stats(idx, config)
    assert out["profit_pct"] is None
