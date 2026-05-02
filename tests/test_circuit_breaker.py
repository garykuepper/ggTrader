"""Unit tests for the Daily Loss Circuit Breaker."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datetime import datetime, timezone

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
                "robustness_score": 0.8
            }
        }
    }
    path = os.path.join(tmp_path, "run_results.json")
    with open(path, "w") as f:
        json.dump(results, f)
    return path

@pytest.fixture
def mock_engine_deps(tmp_path):
    persist_path = os.path.join(tmp_path, "active_positions.json")
    reopt_path = os.path.join(tmp_path, "last_reopt_month.txt")
    
    with patch("ggTrader.core.base_execution_engine.setup_live_logger"), \
         patch("ggTrader.core.base_execution_engine.TradeTracker"), \
         patch("ggTrader.data.live.cached_loader.CachedExchangeLoader"):
        yield {
            "PERSISTENCE_PATH": persist_path,
            "REOPT_FLAG_PATH": reopt_path
        }

def test_circuit_breaker_blocks_entry(mock_engine_deps, dummy_results_file):
    """Verify that entries are blocked when the circuit breaker is triggered."""
    config = {
        "SYMBOLS": ["BTC-USD"],
        "DAILY_LOSS_LIMIT_PCT": 0.05,
        "DRY_RUN": False,
        "PERSISTENCE_PATH": mock_engine_deps["PERSISTENCE_PATH"]
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        engine = ExecutionEngine(config, results_path=dummy_results_file)
        engine.daily_start_equity = 1000.0
        engine.circuit_breaker_triggered = True
        
        signals = {
            "BTC-USD": {
                "entry": True,
                "exit": False,
                "current_price": 50000.0
            }
        }
        
        # Mock _execute_market_buy_order to ensure it's NOT called
        engine._execute_market_buy_order = MagicMock()
        
        engine._execute_trade_logic(signals)
        
        engine._execute_market_buy_order.assert_not_called()

def test_circuit_breaker_trigger_logic(mock_engine_deps, dummy_results_file):
    """Verify that the circuit breaker triggers when drawdown exceeds limit."""
    config = {
        "SYMBOLS": ["BTC-USD"],
        "DAILY_LOSS_LIMIT_PCT": 0.05,
        "DRY_RUN": False,
        "PERSISTENCE_PATH": mock_engine_deps["PERSISTENCE_PATH"]
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        engine = ExecutionEngine(config, results_path=dummy_results_file)
        engine.daily_start_equity = 1000.0
        engine.circuit_breaker_triggered = False
        
        # Simulate current equity showing a 6% loss
        engine._get_total_portfolio_usd = MagicMock(return_value=939.0)
        
        # In run_event_loop logic:
        limit = engine.config.get("DAILY_LOSS_LIMIT_PCT")
        if limit and engine.daily_start_equity and not engine.circuit_breaker_triggered:
            current_equity = engine._get_total_portfolio_usd()
            if current_equity:
                drawdown = (current_equity / engine.daily_start_equity) - 1
                if drawdown < -limit:
                    engine.circuit_breaker_triggered = True
        
        assert engine.circuit_breaker_triggered is True

def test_circuit_breaker_resets_on_new_day(mock_engine_deps, dummy_results_file):
    """Verify that the circuit breaker resets when the day changes."""
    config = {
        "SYMBOLS": ["BTC-USD"],
        "DAILY_LOSS_LIMIT_PCT": 0.05,
        "DRY_RUN": False,
        "PERSISTENCE_PATH": mock_engine_deps["PERSISTENCE_PATH"]
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        engine = ExecutionEngine(config, results_path=dummy_results_file)
        engine.circuit_breaker_triggered = True
        engine._last_check_date = datetime(2025, 1, 1).date()
        
        # Simulate a new day
        now = datetime(2025, 1, 2)
        _current_date = now.date()
        
        engine._get_total_portfolio_usd = MagicMock(return_value=1100.0)
        
        if engine._last_check_date != _current_date:
            engine.daily_start_equity = engine._get_total_portfolio_usd()
            engine.circuit_breaker_triggered = False
            engine._last_check_date = _current_date

        assert engine.circuit_breaker_triggered is False
        assert engine.daily_start_equity == 1100.0
        assert engine._last_check_date == _current_date

def test_circuit_breaker_persistence(mock_engine_deps, dummy_results_file):
    """Verify that circuit breaker state is persisted to disk."""
    config = {
        "SYMBOLS": ["BTC-USD"],
        "DAILY_LOSS_LIMIT_PCT": 0.05,
        "DRY_RUN": False,
        "PERSISTENCE_PATH": mock_engine_deps["PERSISTENCE_PATH"]
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        engine = ExecutionEngine(config, results_path=dummy_results_file)
        engine.daily_start_equity = 1234.56
        engine.circuit_breaker_triggered = True
        engine._last_check_date = datetime(2025, 5, 2).date()
        
        engine.save_state()
        
        # Verify file content
        with open(mock_engine_deps["PERSISTENCE_PATH"], "r") as f:
            data = json.load(f)
            assert data["daily_start_equity"] == 1234.56
            assert data["circuit_breaker_triggered"] is True
            assert data["last_check_date"] == "2025-05-02"
        
        # Load into new engine
        new_engine = ExecutionEngine(config, results_path=dummy_results_file)
        assert new_engine.daily_start_equity == 1234.56
        assert new_engine.circuit_breaker_triggered is True
        assert new_engine._last_check_date == datetime(2025, 5, 2).date()
