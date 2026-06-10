"""Unit tests for the FastBacktest engine."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.core.fast_backtest import FastBacktest


def create_trending_data(n: int = 100):
    """Simple trending data for backtest verification with proper MultiIndex."""
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    # Linear uptrend: 100 -> 200
    close = np.linspace(100, 200, n)
    data = {
        ("BTC", "open"): close * 0.999,
        ("BTC", "high"): close * 1.001,
        ("BTC", "low"): close * 0.998,
        ("BTC", "close"): close,
    }
    df = pd.DataFrame(data, index=idx)
    # Ensure MultiIndex with levels named 'symbol' and 'field'
    df.columns.names = ["symbol", "field"]
    return df


def test_fast_backtest_basic_run():
    """Test a basic backtest run with trending data."""
    ohlcv = create_trending_data()
    params = {
        "adx_threshold": 10,  # Low threshold to ensure some signals
        "adx_length": 5,
        "atr_length": 5,
        "atr_multiplier": 3.0,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
    }
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "PORTFOLIO_SHARE": 1.0,
        "FREQ": "4h",
    }

    engine = FastBacktest(ohlcv, params, config=config)
    pf = engine.run()
    stats = engine.get_stats()

    assert pf is not None
    assert stats["total_value"] >= 10000.0
    assert "profit_pct" in stats
    assert stats["total_trades"] >= 0


def test_fast_backtest_empty_data():
    """Ensure engine raises ValueError on empty input."""
    with pytest.raises(ValueError, match="OHLCV data is empty"):
        engine = FastBacktest(pd.DataFrame(), {})
        engine.run()


def test_fast_backtest_stats_safety():
    """Stats must handle a zero-trade portfolio (NaN metrics) safely."""
    ohlcv = create_trending_data(100)
    params = {
        "adx_threshold": 150,  # beyond ADX's [0, 100] range -> no entries, no trades
        "adx_length": 14,
        "atr_length": 14,
        "atr_multiplier": 3.0,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
    }
    engine = FastBacktest(ohlcv, params)
    engine.run()
    stats = engine.get_stats()

    # Should not crash on division by zero or NaN
    assert stats["total_trades"] == 0
    assert isinstance(stats["win_rate"], float)
    assert isinstance(stats["sharpe"], float)


def test_fast_backtest_insufficient_warmup_raises():
    """Too few bars for the configured indicators raises instead of silently
    returning empty signals (WFO skips such folds; callers must size windows)."""
    ohlcv = create_trending_data(10)
    params = {
        "adx_threshold": 25,
        "adx_length": 14,
        "atr_length": 14,
        "atr_multiplier": 3.0,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
    }
    engine = FastBacktest(ohlcv, params)
    with pytest.raises(Exception):
        engine.run()
