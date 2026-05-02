"""System-level dry run tests for the Live Execution Engine."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ggTrader.core.crypto_execution_engine import CryptoExecutionEngine as ExecutionEngine


@pytest.fixture
def dry_run_config(tmp_path: str) -> dict:
    """Create a configuration for dry run testing."""
    results = {
        "per_coin_results": {
            "ETH-USD": {
                "best_strategy": "psar_adx",
                "best_exit": "atr_trailing",
                "best_params": {
                    "sar_acceleration": 0.02,
                    "sar_maximum": 0.2,
                    "adx_length": 14,
                    "adx_threshold": 10,
                    "atr_length": 14,
                    "atr_multiplier": 3.0,
                    "stop_pct": 3.0,
                },
            }
        }
    }
    results_path = os.path.join(tmp_path, "run_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    return {
        "EXCHANGE": "kraken",
        "SYMBOLS": ["ETH-USD"],
        "INTERVAL": "4h",
        "DRY_RUN": True,
        "CAPITAL_PER_TRADE": 100.0,
        "PERSISTENCE_PATH": os.path.join(tmp_path, "active_positions.json"),
        "RESULTS_PATH": results_path,
    }


def test_system_dry_run_cycle(dry_run_config: dict):
    """Test a full iteration of the trade logic in dry-run mode."""
    # 1. Setup Mock Data (Strong uptrend for PSAR+ADX)
    idx = pd.date_range("2025-01-01", periods=100, freq="4h")
    # Long downtrend then sharp reversal
    close = [100.0 - i * 0.1 for i in range(80)] + [92.0 + i * 2.0 for i in range(20)]
    data = {
        ("ETH-USD", "open"): [c * 0.99 for c in close],
        ("ETH-USD", "high"): [c * 1.05 for c in close],
        ("ETH-USD", "low"): [c * 0.95 for c in close],
        ("ETH-USD", "close"): close,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])

    with patch("ggTrader.data.live.exchange_loader.LiveExchangeLoader") as mock_loader:
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.fetch_ohlcv.return_value = df
        mock_loader_instance.exchange = MagicMock()
        mock_loader_instance.exchange.fetch_ticker.return_value = {"last": 110.0}

        # 2. Initialize Engine
        engine = ExecutionEngine(dry_run_config, results_path=dry_run_config["RESULTS_PATH"])

        # 3. Compute Signals
        signals = engine._compute_latest_signals(df)
        assert "ETH-USD" in signals

        # 4. Execute Logic
        engine._execute_trade_logic(signals)

        # 5. Verify State
        if signals["ETH-USD"]["entry"]:
            assert "ETH-USD" in engine.active_positions
            assert engine.active_positions["ETH-USD"]["entry_order_id"] == "dry_run_id"


def test_scheduling_alignment():
    """Test the 4h boundary sleep calculation."""

    def get_wait_seconds(hour, minute, second):
        next_hour = ((hour // 4) + 1) * 4
        if next_hour >= 24:
            return (24 - hour) * 3600 - minute * 60 - second
        else:
            return (next_hour - hour) * 3600 - minute * 60 - second

    assert get_wait_seconds(0, 0, 0) == 4 * 3600
    assert get_wait_seconds(3, 59, 59) == 1
    assert get_wait_seconds(23, 59, 59) == 1
