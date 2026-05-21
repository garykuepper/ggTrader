"""Unit tests for the Live Execution Engine."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ggTrader.core.crypto_execution_engine import CryptoExecutionEngine as ExecutionEngine


@pytest.fixture
def dummy_results_file(tmp_path: str) -> str:
    """Create a mock run_results.json."""
    results = {
        "per_coin_results": {
            "BTC-USD": {
                "best_strategy": "rsi_reversal",
                "best_exit": "fixed_sl_tp",
                "best_params": {
                    "rsi_length": 14,
                    "rsi_oversold": 30,
                    "stop_pct": 2.0,
                    "take_profit_pct": 5.0,
                },
            }
        }
    }
    path = os.path.join(tmp_path, "run_results.json")
    with open(path, "w") as f:
        json.dump(results, f)
    return path


@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    """Create dummy OHLCV data for testing signals."""
    idx = pd.date_range("2025-01-01", periods=100, freq="4h")
    # To trigger an RSI Cross Above 30 exactly on the last bar:
    # 80 bars of 50.0, 18 bars of 10.0 (RSI -> 0), then jump to 35.0 on the last bar.
    close = [50.0] * 80 + [10.0] * 19 + [35.0]
    data = {
        ("BTC-USD", "open"): [c * 0.99 for c in close],
        ("BTC-USD", "high"): [c * 1.01 for c in close],
        ("BTC-USD", "low"): [c * 0.98 for c in close],
        ("BTC-USD", "close"): close,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def test_engine_load_params(dummy_results_file: str):
    """Test that parameters are correctly loaded from JSON."""
    config = {"SYMBOLS": ["BTC-USD"], "DRY_RUN": True}
    engine = ExecutionEngine(config, results_path=dummy_results_file)

    assert "BTC-USD" in engine.per_coin_params
    assert engine.per_coin_params["BTC-USD"]["strategy_name"] == "rsi_reversal"
    assert engine.per_coin_params["BTC-USD"]["params"]["rsi_length"] == 14


def test_compute_signals(dummy_results_file: str, dummy_ohlcv: pd.DataFrame):
    """Test signal computation using real precomputer on dummy data."""
    config = {"SYMBOLS": ["BTC-USD"], "DRY_RUN": True}
    with patch("ggTrader.data.live.exchange_loader.LiveExchangeLoader") as mock_loader:
        mock_loader.return_value.exchange = MagicMock()
        engine = ExecutionEngine(config, results_path=dummy_results_file)

        signals = engine._compute_latest_signals(dummy_ohlcv)
        assert "BTC-USD" in signals
        assert isinstance(signals["BTC-USD"]["entry"], bool)
        # Entry should be True on the last bar with our sharp reversal
        assert signals["BTC-USD"]["entry"] is True


@patch("ccxt.kraken")
def test_execute_orders_mocked(mock_kraken: MagicMock, dummy_results_file: str):
    """Test that CCXT order methods are called with correct params."""
    config = {"SYMBOLS": ["BTC-USD"], "DRY_RUN": False}
    # Mock exchange instance
    mock_instance = mock_kraken.return_value
    mock_instance.fetch_ticker.return_value = {"last": 50000.0}
    mock_instance.market.return_value = {"symbol": "BTC/USD", "precision": {"price": 1}}

    mock_instance.amount_to_precision.return_value = "0.002"
    mock_instance.create_market_buy_order.return_value = {"id": "123"}
    mock_instance.create_order.return_value = {"id": "456"}

    with patch("ggTrader.data.live.exchange_loader.LiveExchangeLoader") as mock_loader:
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.exchange = mock_instance
        # Mock reconcile to prevent it from dropping positions
        with patch.object(ExecutionEngine, "_reconcile_positions"):
            engine = ExecutionEngine(config, results_path=dummy_results_file)

            # Test Market Buy
            order_id = engine._execute_market_buy_order("BTC-USD", 100.0)
            assert order_id == "123"
            mock_instance.create_market_buy_order.assert_called_once()

            # Test Trailing Stop
            tsl_id = engine._execute_trailing_stop_order("BTC-USD", 0.002, 3.0)
            assert tsl_id == "456"
            # Check params for create_order(symbol, type, side, amount, price, params)
            args, kwargs = mock_instance.create_order.call_args
            assert args[1] == "trailing-stop"
            params = args[5]
            assert "trailingAmount" in params


def test_persistence_logic(tmp_path: str):
    """Test that state is correctly saved and loaded."""
    persist_path = os.path.join(tmp_path, "active_positions.json")
    config = {"PERSISTENCE_PATH": persist_path}
    with patch("ggTrader.data.live.exchange_loader.LiveExchangeLoader") as mock_loader:
        mock_loader.return_value.exchange = MagicMock()
        # Mock reconcile to prevent dropping positions
        with patch.object(ExecutionEngine, "_reconcile_positions"):
            engine = ExecutionEngine(config)

            engine.active_positions = {"BTC-USD": {"amount": 0.1}}
            engine.save_state()

            assert os.path.exists(persist_path)

            new_engine = ExecutionEngine(config)
            assert new_engine.active_positions["BTC-USD"]["amount"] == 0.1
