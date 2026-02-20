"""Unit tests for setup and data loading utilities."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from ggTrader.utils.setup import load_data_and_setup, build_mover_mask


@pytest.fixture
def mock_ohlcv():
    idx = pd.date_range("2023-01-01", periods=100, freq="4h").tz_localize("UTC")
    data = {
        ("BTC", "open"): np.random.rand(100),
        ("BTC", "high"): np.random.rand(100),
        ("BTC", "low"): np.random.rand(100),
        ("BTC", "close"): np.random.rand(100),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


@patch("ggTrader.utils.setup.TimescaleDBLoader")
def test_load_data_and_setup(mock_hdata, mock_ohlcv):
    mock_hdata.return_value.fetch_ohlcv.return_value = mock_ohlcv

    config = {
        "SYMBOLS": ["BTC"],
        "INTERVAL": "4h",
        "START_DATE": "2023-01-01",
        "END_DATE": "2023-01-05",
    }

    df = load_data_and_setup(config)
    assert not df.empty
    assert "BTC" in df.columns.levels[0]


@patch("ggTrader.utils.setup.TimescaleDBLoader")
def test_build_mover_mask(mock_hdata, mock_ohlcv):
    # Mock daily mover mask (only dates, no intraday)
    daily_idx = pd.date_range("2023-01-01", periods=2, freq="1d").tz_localize("UTC")
    daily_mask = pd.DataFrame({"BTC": [True, False]}, index=daily_idx)
    mock_hdata.return_value.get_daily_mover_mask.return_value = daily_mask

    config = {"START_DATE": "2023-01-01", "END_DATE": "2023-01-02"}

    mask = build_mover_mask(mock_ohlcv, config, top_n=5)

    assert mask.shape[0] == mock_ohlcv.shape[0]
    # Check that it's a boolean mask
    assert mask.values.dtype == bool
    # BTC should be True for the first day, False for the second
    assert mask["BTC"].iloc[0] == True
