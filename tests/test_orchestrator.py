"""Unit tests for orchestrators using mocks."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import vectorbt as vbt
from ggTrader.core.orchestrator import (
    run_backtest_orchestrator,
    run_sensitivity_orchestrator,
    run_wfo_orchestrator,
)


@pytest.fixture
def mock_ohlcv():
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    data = {
        ("BTC", "open"): np.random.rand(100),
        ("BTC", "high"): np.random.rand(100),
        ("BTC", "low"): np.random.rand(100),
        ("BTC", "close"): np.random.rand(100),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


@patch("ggTrader.core.orchestrator.load_data_with_movers")
@patch("ggTrader.core.orchestrator.ResultsManager")
def test_run_backtest_orchestrator(mock_rm, mock_load, mock_ohlcv):
    mock_load.return_value = (mock_ohlcv, None)

    config = {"USE_MOVERS": 0}
    params = {
        "adx_threshold": 25,
        "adx_length": 14,
        "atr_length": 14,
        "atr_multiplier": 3.0,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
    }

    res = run_backtest_orchestrator(config, params, save_results=True)

    assert "portfolio" in res
    assert "stats" in res
    mock_rm.return_value.save_run_results.assert_called_once()


@patch("ggTrader.core.orchestrator.load_data_with_movers")
@patch("ggTrader.core.orchestrator.ResultsManager")
def test_run_sensitivity_orchestrator(mock_rm, mock_load, mock_ohlcv):
    mock_load.return_value = (mock_ohlcv, None)

    config = {
        "CHUNK_SIZE": 100,  # Large chunk to avoid loop issues
        "USE_CASH_SHARING": True,
        "START_CASH": 10000,
        "FEES": 0.001,
        "SLIPPAGE": 0.001,
        "FREQ": "4h",
    }
    # Provide multiple thresholds
    param_grid = {"adx_threshold": [20, 25, 30]}

    res = run_sensitivity_orchestrator(config, param_grid, save_results=False)

    assert "best_params" in res
    assert len(res["results_df"]) == 3


@patch("ggTrader.core.orchestrator.load_data_with_movers")
@patch("ggTrader.core.orchestrator.ResultsManager")
@patch("ggTrader.core.orchestrator.make_end_anchored_tscv")
def test_run_wfo_orchestrator(mock_tscv, mock_rm, mock_load, mock_ohlcv):
    mock_load.return_value = (mock_ohlcv, None)

    mock_splitter = MagicMock()
    mock_splitter.split.return_value = [(np.arange(50), np.arange(50, 100))]
    mock_tscv.return_value = (mock_splitter, None, None)

    config = {
        "N_SPLITS": 1,
        "TEST_RATIO": 0.5,
        "USE_MOVERS": 0,
        "START_CASH": 10000,
        "FEES": 0.001,
        "SLIPPAGE": 0.001,
        "FREQ": "4h",
    }
    # Provide multiple thresholds
    param_grid = {"adx_threshold": [20, 25]}

    res = run_wfo_orchestrator(config, param_grid, save_results=False)

    assert len(res["wfo_stats"]) == 1
    assert "best_robust_params" in res
