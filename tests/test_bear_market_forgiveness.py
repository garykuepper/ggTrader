"""Unit tests verifying the Bear Market 'zero-trade' forgiveness in WFO."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ggTrader.core.orchestrator import _process_wfo_fold


@pytest.fixture
def bear_market_ohlcv():
    """Returns a MultiIndex DataFrame mimicking a dropping symbol (bear market)."""
    # Create 3 days of dropping prices
    dates = pd.date_range("2025-01-01", periods=3, freq="D")

    # We need MultiIndex columns for (symbol, price_type) as expected by the engine
    cols = pd.MultiIndex.from_product([["BTC"], ["open", "high", "low", "close", "volume"]])

    # Prices drop from 100 -> 90 -> 80 (Return = -20%)
    data = [
        [100, 105, 95, 100, 10],
        [90, 95, 85, 90, 10],
        [80, 85, 75, 80, 10],
    ]
    df = pd.DataFrame(data, index=dates, columns=cols)
    return df


@pytest.fixture
def bull_market_ohlcv():
    """Returns a MultiIndex DataFrame mimicking a rising symbol (bull market)."""
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([["BTC"], ["open", "high", "low", "close", "volume"]])

    # Prices rise from 100 -> 110 -> 120 (Return = +20%)
    data = [
        [100, 105, 95, 100, 10],
        [110, 115, 105, 110, 10],
        [120, 125, 115, 120, 10],
    ]
    df = pd.DataFrame(data, index=dates, columns=cols)
    return df


@patch("ggTrader.core.wfo.FastBacktest")
@patch("ggTrader.core.wfo._vectorized_grid_metrics")
def test_bear_market_forgives_zero_trades(mock_metrics, mock_engine_cls, bear_market_ohlcv):
    """
    Tests that if the Buy&Hold return over the train fold is negative,
    parameter combinations with exactly 0 trades are spared from the MIN_CLOSED_TRADES_TRAIN
    rejection gate, and are given a neutral 0.0 score.
    """
    # Setup our mocked metrics:
    # Combo A: took 0 trades (should be forgiven and scored 0.0)
    # Combo B: took 5 trades, lost money (should be kept and scored -1.0)
    # Combo C: took 0 trades (should be forgiven and scored 0.0)

    mock_train_metrics = pd.Series([np.nan, -1.0, np.nan], index=["combo_A", "combo_B", "combo_C"])
    mock_trade_counts = pd.Series([0, 5, 0], index=["combo_A", "combo_B", "combo_C"])

    mock_metrics.return_value = (mock_train_metrics, mock_trade_counts)

    # Mock the test-phase backtest to prevent errors at the end of the fold processing
    mock_test_engine = MagicMock()
    mock_test_pf = MagicMock()
    mock_test_pf.sharpe_ratio.return_value.mean.return_value = 0.5
    mock_test_pf.sortino_ratio.return_value.mean.return_value = 0.5
    mock_test_pf.total_profit.return_value.sum.return_value = 100
    mock_test_pf.init_cash.sum.return_value = 1000
    mock_test_pf.value.return_value.iloc.__getitem__.return_value.sum.return_value = 1100
    mock_test_pf.total_return.return_value.mean.return_value = 0.1
    mock_test_pf.returns.return_value = pd.Series(dtype=float)
    mock_test_engine.run.return_value = mock_test_pf

    # FastBacktest is instantiated twice per fold (train, then test)
    mock_engine_cls.return_value = mock_test_engine

    config = {"MIN_CLOSED_TRADES_TRAIN": 1, "MIN_FOLD_BARS": 1}

    train_idx = bear_market_ohlcv.index
    test_idx = bear_market_ohlcv.index

    result = _process_wfo_fold(
        fold_idx=1,
        train_idx=train_idx,
        test_idx=test_idx,
        ohlcv=bear_market_ohlcv,
        mover_mask=None,
        param_grid={"dummy": [1]},
        config=config,
        show_progress=False,
        param_names=["dummy"],
    )

    final_train_metrics = result["train_metrics"]

    # Let's assert that the zero-trade combos were given exactly 0.0!
    assert final_train_metrics.loc["combo_A"] == 0.0, (
        "Combo A traded 0 times in a bear market; Sharpe should be coerced to 0.0"
    )
    assert final_train_metrics.loc["combo_C"] == 0.0, (
        "Combo C traded 0 times in a bear market; Sharpe should be coerced to 0.0"
    )

    # Assert Combo B (which traded but lost) kept its real score
    assert final_train_metrics.loc["combo_B"] == -1.0, (
        "Combo B traded > min_trades, should retain its -1.0 score"
    )


@patch("ggTrader.core.wfo.FastBacktest")
@patch("ggTrader.core.wfo._vectorized_grid_metrics")
def test_bull_market_rejects_zero_trades(mock_metrics, mock_engine_cls, bull_market_ohlcv):
    """
    Tests that if the Buy&Hold return is positive (bull market),
    parameter combinations with 0 trades are strictly rejected by the MIN_CLOSED_TRADES_TRAIN gate.
    """
    # Combo A: took 0 trades
    # Combo B: took 5 trades (gained money)
    mock_train_metrics = pd.Series([np.nan, 2.0], index=["combo_A", "combo_B"])
    mock_trade_counts = pd.Series([0, 5], index=["combo_A", "combo_B"])

    mock_metrics.return_value = (mock_train_metrics, mock_trade_counts)

    mock_test_engine = MagicMock()
    mock_test_pf = MagicMock()
    mock_test_pf.sharpe_ratio.return_value.mean.return_value = 0.5
    mock_test_pf.sortino_ratio.return_value.mean.return_value = 0.5
    mock_test_pf.total_profit.return_value.sum.return_value = 100
    mock_test_pf.init_cash.sum.return_value = 1000
    mock_test_pf.value.return_value.iloc.__getitem__.return_value.sum.return_value = 1100
    mock_test_pf.total_return.return_value.mean.return_value = 0.1
    mock_test_pf.returns.return_value = pd.Series(dtype=float)
    mock_test_engine.run.return_value = mock_test_pf
    mock_engine_cls.return_value = mock_test_engine

    config = {"MIN_CLOSED_TRADES_TRAIN": 1, "MIN_FOLD_BARS": 1}

    train_idx = bull_market_ohlcv.index
    test_idx = bull_market_ohlcv.index

    result = _process_wfo_fold(
        fold_idx=1,
        train_idx=train_idx,
        test_idx=test_idx,
        ohlcv=bull_market_ohlcv,
        mover_mask=None,
        param_grid={"dummy": [1]},
        config=config,
        show_progress=False,
        param_names=["dummy"],
    )

    final_train_metrics = result["train_metrics"]

    # Assert Combo A was rejected (remains NaN) because taking 0 trades in a
    # bull market is a failure!
    assert pd.isna(final_train_metrics.loc["combo_A"]), (
        "Combo A traded 0 times in a bull market; should be rejected (NaN)"
    )

    # Assert Combo B retained its real success score
    assert final_train_metrics.loc["combo_B"] == 2.0, (
        "Combo B traded > min_trades, should retain its 2.0 score"
    )
