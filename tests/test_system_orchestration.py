import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Standard imports
import ggTrader.core.orchestrator as orch


@pytest.fixture
def mock_ohlcv():
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    symbols = ["BTC-USD"]
    data = {}
    for sym in symbols:
        data[(sym, "open")] = np.linspace(30000, 31000, 100)
        data[(sym, "high")] = data[(sym, "open")] + 10
        data[(sym, "low")] = data[(sym, "open")] - 10
        data[(sym, "close")] = data[(sym, "open")] + 5
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


@pytest.fixture
def temp_results_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@patch("ggTrader.core.orchestrator.load_data_with_movers")
def test_system_backtest_flow(mock_load, mock_ohlcv, temp_results_dir):
    mock_load.return_value = (mock_ohlcv, None)

    config = {"USE_MOVERS": 0, "START_CASH": 10000, "FEES": 0, "SLIPPAGE": 0, "FREQ": "4h"}
    params = {
        "adx_threshold": 25,
        "adx_length": 14,
        "atr_length": 14,
        "atr_multiplier": 3.0,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
    }

    # Manually configure ResultsManager to use temp dir
    with patch("ggTrader.utils.results_manager.find_project_root") as mock_root:
        mock_root.return_value = temp_results_dir
        res = orch.run_backtest_orchestrator(config, params, save_results=True)

    assert "portfolio" in res
    # Check if run results JSON was created in the temp results folder
    # ResultsManager creates a "results" subfolder under project_root
    results_base = temp_results_dir / "results"
    run_dirs = [d for d in os.listdir(results_base) if d.startswith("run_backtest_")]
    assert len(run_dirs) == 1
    assert (results_base / run_dirs[0] / "run_results.json").exists()


@patch("ggTrader.core.orchestrator.load_data_with_movers")
def test_system_wfo_auto_window(mock_load, mock_ohlcv, temp_results_dir):
    mock_load.return_value = (mock_ohlcv, None)

    config = {
        "N_SPLITS": 2,
        "TEST_RATIO": 0.5,
        "USE_MOVERS": 0,
        "START_CASH": 1000,
        "FEES": 0,
        "SLIPPAGE": 0,
        "FREQ": "4h",
    }
    param_grid = {"adx_threshold": [25]}

    with patch("ggTrader.utils.results_manager.find_project_root") as mock_root:
        mock_root.return_value = temp_results_dir
        with patch("ggTrader.utils.plotting.plot_wfo_splits") as mock_plot:
            res = orch.run_wfo_orchestrator(config, param_grid, save_results=True)
            assert len(res["wfo_stats"]) == 2
            assert mock_plot.called
