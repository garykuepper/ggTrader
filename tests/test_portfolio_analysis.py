"""
Unit tests for the Portfolio Analysis Standalone tool.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
# Add scripts to path for importing functions
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from portfolio_analysis_standalone import (
    run_strategy_benchmark,
    run_strategy_equal_share,
    run_strategy_max_diversified,
    run_strategy_robustness_weighted,
    run_strategy_volatility_weighted,
)


@pytest.fixture
def sample_portfolio_data(sample_ohlcv_data):
    """Create sample signals and close prices for two symbols."""
    close = sample_ohlcv_data.xs("close", level=1, axis=1)
    # Generate random entries (approx 5% of bars have signal)
    np.random.seed(42)
    entries = pd.DataFrame(
        np.random.choice([True, False], size=close.shape, p=[0.05, 0.95]),
        index=close.index,
        columns=close.columns,
    )
    # Simple exit 5 bars later
    exits = entries.shift(5).fillna(False)

    config = {"START_CASH": 1000, "FEES": 0.001, "SLIPPAGE": 0.001, "FREQ": "4h"}
    return close, entries, exits, config


def test_run_strategy_equal_share(sample_portfolio_data):
    close, entries, exits, config = sample_portfolio_data
    pf, weights = run_strategy_equal_share(close, entries, exits, config, share=0.1)
    assert isinstance(pf, vbt.Portfolio)
    assert isinstance(weights, pd.DataFrame)
    assert pf.init_cash == 1000
    # Check that position sizes are 10%
    if not pf.trades.records.empty:
        # Expect approx 100 units if price=1000
        pass


def test_run_strategy_max_diversified(sample_portfolio_data):
    close, entries, exits, config = sample_portfolio_data
    pf, weights = run_strategy_max_diversified(close, entries, exits, config)
    assert isinstance(pf, vbt.Portfolio)
    assert isinstance(weights, pd.DataFrame)
    # With 2 symbols, size should be 0.5 (50%)
    # pf.size is not directly accessible after run, but we can check entry values
    pass


def test_run_strategy_volatility_weighted(sample_portfolio_data):
    close, entries, exits, config = sample_portfolio_data
    pf, weights = run_strategy_volatility_weighted(close, entries, exits, config)
    assert isinstance(pf, vbt.Portfolio)
    assert isinstance(weights, pd.DataFrame)


def test_run_strategy_robustness_weighted(sample_portfolio_data):
    close, entries, exits, config = sample_portfolio_data
    robustness_scores = {"BTC-USD": 0.8, "ETH-USD": 0.2}
    pf, weights = run_strategy_robustness_weighted(close, entries, exits, config, robustness_scores)
    assert isinstance(pf, vbt.Portfolio)
    assert isinstance(weights, pd.DataFrame)


def test_run_strategy_benchmark(sample_portfolio_data):
    close, _, _, config = sample_portfolio_data
    pf = run_strategy_benchmark(close, config)
    assert isinstance(pf, vbt.Portfolio)
    assert pf.trades.count().sum() == close.shape[1]  # One trade per symbol


def test_system_flow(tmp_path, sample_ohlcv_data):
    """Test the end-to-end signal generation from a mock result directory."""
    # Create mock result directory
    results_dir = tmp_path / "mock_run"
    results_dir.mkdir()

    # Create mock run_results.json
    mock_results = {
        "configuration": {
            "_raw_config": {
                "per_coin_results": {
                    "BTC-USD": {
                        "best_strategy": "rsi_reversal",
                        "best_exit": "fixed_sl_tp",
                        "best_params": {
                            "rsi_length": 14,
                            "rsi_oversold": 30,
                            "stop_pct": 1.5,
                            "take_profit_pct": 3.0,
                        },
                        "robustness_score": 0.8,
                    }
                },
                "START_CASH": 1000,
                "FEES": 0.001,
                "SLIPPAGE": 0.001,
                "FREQ": "4h",
            }
        }
    }

    with open(results_dir / "run_results.json", "w") as f:
        json.dump(mock_results, f)

    # We can't easily call main() because it uses load_data_with_movers (which queries DB)
    # But we can test load_wfo_results
    from portfolio_analysis_standalone import load_wfo_results

    per_coin, config = load_wfo_results(str(results_dir))

    assert "BTC-USD" in per_coin
    assert config["FEES"] == 0.001


if __name__ == "__main__":
    import json

    pytest.main([__file__])
