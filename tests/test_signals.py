"""Unit tests for signal generation logic."""

import numpy as np
import pandas as pd

from ggTrader.indicators.signals import SignalFactory, Signals


def create_dummy_ohlcv(n_rows: int = 100, trend: str = "flat") -> pd.DataFrame:
    """Create a dummy OHLCV MultiIndex DataFrame with proper names."""
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="4h")
    if trend == "up":
        close = np.linspace(100, 200, n_rows)
    elif trend == "down":
        close = np.linspace(200, 100, n_rows)
    else:
        close = np.full(n_rows, 100.0)

    data = {
        ("BTC", "open"): close * 0.99,
        ("BTC", "high"): close * 1.01,
        ("BTC", "low"): close * 0.98,
        ("BTC", "close"): close,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def test_entry_signals_logic():
    """Test that entry signals are correctly generated."""
    ohlcv = create_dummy_ohlcv(n_rows=50, trend="up")
    c = ohlcv.xs("close", axis=1, level="field")
    h = ohlcv.xs("high", axis=1, level="field")
    l = ohlcv.xs("low", axis=1, level="field")

    entries = Signals.entry_signals(
        close=c,
        high=h,
        low=l,
        adx_length=14,
        adx_threshold=20,
        sar_acceleration=0.02,
        sar_maximum=0.2,
        use_dmp_cross=False,
    )

    assert isinstance(entries, np.ndarray)
    assert entries.dtype == bool
    assert entries.shape == c.shape


def test_signals_insufficient_data():
    """Test that signals handle short data frames gracefully."""
    ohlcv = create_dummy_ohlcv(n_rows=5)  # Too short for default ADX(14)
    res = Signals.calculate_ohlcv_signals(ohlcv, adx_length=14)

    for symbol, df in res.items():
        assert (df["signal"] == 0).all()
        assert df["stop_loss"].isna().all()


def test_stop_fill_price_gap_handling():
    """Test that exit fill price accounts for gap-downs."""
    exits = np.array([True, False, True])
    stops = np.array([100.0, 100.0, 100.0])
    opens = np.array([90.0, 110.0, 105.0])
    base_price = np.array([102.0, 102.0, 102.0])

    fills = Signals.stop_fill_price(exits, stops, opens, None, None, base_price)

    assert fills[0] == 90.0
    assert fills[1] == 102.0
    assert fills[2] == 100.0


def test_signal_factory_broadcasting():
    """Test that SignalFactory handles multiple parameters."""
    ohlcv = create_dummy_ohlcv(n_rows=50)
    # Unpack for factory
    close = ohlcv.xs("close", axis=1, level="field")
    high = ohlcv.xs("high", axis=1, level="field")
    low = ohlcv.xs("low", axis=1, level="field")
    open_ = ohlcv.xs("open", axis=1, level="field")

    sf = SignalFactory.run(close, high, low, open_, adx_threshold=[20, 30], param_product=True)

    assert isinstance(sf.entries.columns, pd.MultiIndex)
