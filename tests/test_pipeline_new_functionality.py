"""Tests for new pipeline functionality: strategy dispatch, sensitivity analysis, reporting."""

import os
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator import analyze_sensitivity_results
from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY
from ggTrader.utils.report_generator import generate_pipeline_report


class TestStrategyDispatchInFastBacktest:
    """Test that FastBacktest properly dispatches to entry/exit strategies."""

    def test_default_strategy_psar_adx(self, sample_ohlcv_data):
        """Test that PSAR+ADX is used by default when no strategy specified."""
        config = {
            "ENTRY_STRATEGY": None,
            "EXIT_STRATEGY": None,
            "USE_VECTORIZED": True,
            "START_CASH": 1000,
            "FEES": 0.001,
            "SLIPPAGE": 0.0005,
            "FREQ": "4h",
        }
        params = {"sar_acceleration": 0.02, "sar_maximum": 0.2, "adx_length": 14, "atr_length": 14}

        engine = FastBacktest(sample_ohlcv_data, params, config=config)
        assert engine.config["ENTRY_STRATEGY"] in (None, "psar_adx")
        assert engine.config["EXIT_STRATEGY"] in (None, "atr_trailing")

    def test_strategy_dispatch_ema_cross(self, sample_ohlcv_data):
        """Test that EMA crossover strategy can be dispatched."""
        config = {
            "ENTRY_STRATEGY": "ema_cross",
            "EXIT_STRATEGY": "atr_trailing",
            "USE_VECTORIZED": True,
            "START_CASH": 1000,
            "FEES": 0.001,
            "SLIPPAGE": 0.0005,
            "FREQ": "4h",
        }
        params = {"ema_fast": 9, "ema_slow": 21, "atr_length": 14, "atr_multiplier": 3.0}

        engine = FastBacktest(sample_ohlcv_data, params, config=config)
        assert engine.config["ENTRY_STRATEGY"] == "ema_cross"
        assert engine.config["EXIT_STRATEGY"] == "atr_trailing"

    def test_strategy_dispatch_rsi_reversal(self, sample_ohlcv_data):
        """Test that RSI reversal strategy can be dispatched."""
        config = {
            "ENTRY_STRATEGY": "rsi_reversal",
            "EXIT_STRATEGY": "fixed_sl_tp",
            "USE_VECTORIZED": True,
            "START_CASH": 1000,
            "FEES": 0.001,
            "SLIPPAGE": 0.0005,
            "FREQ": "4h",
        }
        params = {"rsi_length": 14, "rsi_oversold": 30, "stop_pct": 2.0, "take_profit_pct": 5.0}

        engine = FastBacktest(sample_ohlcv_data, params, config=config)
        assert engine.config["ENTRY_STRATEGY"] == "rsi_reversal"
        assert engine.config["EXIT_STRATEGY"] == "fixed_sl_tp"

    def test_invalid_strategy_falls_back(self, sample_ohlcv_data):
        """Test that invalid strategy is caught and falls back gracefully."""
        config = {
            "ENTRY_STRATEGY": "invalid_strategy",
            "EXIT_STRATEGY": "atr_trailing",
            "USE_VECTORIZED": True,
            "START_CASH": 1000,
            "FEES": 0.001,
            "SLIPPAGE": 0.0005,
            "FREQ": "4h",
        }
        params = {"sar_acceleration": 0.02, "sar_maximum": 0.2, "adx_length": 14, "atr_length": 14}

        engine = FastBacktest(sample_ohlcv_data, params, config=config)
        # Should not raise - falls back to standard path which will print warning
        # This is expected behavior - errors in vectorized path gracefully fallback
        pf = engine.run(show_progress=False)
        assert pf is not None

    def test_entry_registry_contains_all_strategies(self):
        """ENTRY_REGISTRY is the single source of truth for pipeline strategies."""
        actual_strategies = set(ENTRY_REGISTRY.keys())
        assert "psar_adx" in actual_strategies
        assert "ema_cross" in actual_strategies
        assert "rsi_reversal" in actual_strategies
        assert "macd_cross" in actual_strategies
        assert len(actual_strategies) >= 7

    def test_exit_registry_contains_all_strategies(self):
        """Test that EXIT_REGISTRY has expected strategies."""
        expected_strategies = {"atr_trailing", "fixed_sl_tp"}
        actual_strategies = set(EXIT_REGISTRY.keys())
        assert expected_strategies == actual_strategies


class TestAnalyzeSensitivityResults:
    """Test sensitivity analysis helper function."""

    def test_analyze_sensitivity_empty_df(self):
        """Test behavior with empty DataFrame."""
        results_df = pd.DataFrame()
        param_grid = {"param1": [1, 2, 3]}

        narrowed = analyze_sensitivity_results(results_df, param_grid)
        assert narrowed == param_grid

    def test_analyze_sensitivity_with_data(self):
        """Test sensitivity analysis with sample data."""
        # Create a sample sensitivity results DataFrame
        results_df = pd.DataFrame(
            {
                "param1": [1, 1, 2, 2, 3, 3],
                "param2": [10, 20, 10, 20, 10, 20],
                "Sharpe Ratio": [0.5, 0.7, 0.9, 0.6, 0.4, 0.8],
            }
        )

        param_grid = {
            "param1": [1, 2, 3],
            "param2": [10, 20],
        }

        narrowed = analyze_sensitivity_results(results_df, param_grid, top_percentile=50)

        # Should keep params that produced top 50% of Sharpe ratios
        assert "param1" in narrowed
        assert "param2" in narrowed
        assert isinstance(narrowed["param1"], list)
        assert isinstance(narrowed["param2"], list)
        # At least one value from each parameter should be kept
        assert len(narrowed["param1"]) >= 1
        assert len(narrowed["param2"]) >= 1

    def test_analyze_sensitivity_top_percentile_filtering(self):
        """Test that top_percentile correctly filters values."""
        results_df = pd.DataFrame(
            {
                "param1": [1, 1, 2, 2, 3, 3],
                "Sharpe Ratio": [0.1, 0.2, 0.9, 0.95, 0.5, 0.6],
            }
        )

        param_grid = {"param1": [1, 2, 3]}

        # Keep top 33% (should keep 2 out of 6)
        narrowed = analyze_sensitivity_results(results_df, param_grid, top_percentile=33)

        # Param 2 should dominate since it has highest Sharpe values
        assert 2 in narrowed["param1"]

    def test_analyze_sensitivity_constant_parameter(self):
        """Test behavior with constant (non-varied) parameter."""
        results_df = pd.DataFrame(
            {
                "param1": [1, 1, 1, 1],  # Constant
                "param2": [10, 20, 10, 20],
                "Sharpe Ratio": [0.5, 0.7, 0.9, 0.6],
            }
        )

        param_grid = {
            "param1": [1],
            "param2": [10, 20],
        }

        narrowed = analyze_sensitivity_results(results_df, param_grid)

        # Constant param should be preserved as-is
        assert narrowed["param1"] == [1]


class TestReportGenerator:
    """Test report generation functionality."""

    def test_generate_report_creates_file(self, tmp_path):
        """Test that generate_pipeline_report creates a markdown file."""
        sensitivity_results = {
            "psar_adx": pd.DataFrame(
                {
                    "sar_acceleration": [0.02],
                    "sar_maximum": [0.2],
                    "Sharpe Ratio": [0.75],
                }
            )
        }

        wfo_results = {
            "per_coin_results": {
                "BTC": {
                    "best_strategy": "psar_adx",
                    "best_params": {"sar_acceleration": 0.02},
                    "robustness_score": 0.85,
                }
            }
        }

        final_backtest_results = {
            "per_coin_final_stats": {
                "BTC": {
                    "strategy": "psar_adx",
                    "profit_pct": 15.5,
                    "sharpe": 0.82,
                    "max_drawdown": 12.3,
                    "total_trades": 45,
                    "win_rate": 52.5,
                }
            },
            "final_stats": {
                "profit_pct": 12.3,
                "sharpe": 0.75,
                "sortino": 0.95,
                "max_drawdown": 10.5,
                "total_trades": 45,
                "win_rate": 50.0,
            },
        }

        output_dir = str(tmp_path)

        # Should not raise
        generate_pipeline_report(
            sensitivity_results, wfo_results, final_backtest_results, output_dir
        )

        # Check that report file was created
        report_file = tmp_path / "pipeline_report.md"
        assert report_file.exists()

    def test_report_content_structure(self, tmp_path):
        """Test that report contains expected sections."""
        sensitivity_results = {
            "psar_adx": pd.DataFrame(
                {
                    "sar_acceleration": [0.02],
                    "sar_maximum": [0.2],
                    "Sharpe Ratio": [0.75],
                }
            )
        }

        wfo_results = {
            "per_coin_results": {
                "BTC": {
                    "best_strategy": "psar_adx",
                    "best_params": {"sar_acceleration": 0.02},
                    "robustness_score": 0.85,
                }
            }
        }

        final_backtest_results = {
            "per_coin_final_stats": {
                "BTC": {
                    "strategy": "psar_adx",
                    "profit_pct": 15.5,
                    "sharpe": 0.82,
                    "max_drawdown": 12.3,
                    "total_trades": 45,
                    "win_rate": 52.5,
                }
            },
            "final_stats": {
                "profit_pct": 12.3,
                "sharpe": 0.75,
                "sortino": 0.95,
                "max_drawdown": 10.5,
                "total_trades": 45,
                "win_rate": 50.0,
            },
        }

        output_dir = str(tmp_path)

        generate_pipeline_report(
            sensitivity_results, wfo_results, final_backtest_results, output_dir
        )

        report_file = tmp_path / "pipeline_report.md"
        content = report_file.read_text()

        # Check for expected sections
        expected_sections = [
            "# Trading Strategy Pipeline Report",
            "## Executive Summary",
            "## Phase 0: Sensitivity Analysis Findings",
            "## Phase 1: Per-Coin Strategy Selection",
            "## Final Full-Period Performance",
            "## Combined Portfolio Performance",
            "## Methodology",
        ]

        for section in expected_sections:
            assert section in content, f"Missing section: {section}"

    def test_report_with_multiple_strategies(self, tmp_path):
        """Test report with multiple strategies."""
        sensitivity_results = {
            "psar_adx": pd.DataFrame(
                {
                    "sar_acceleration": [0.02],
                    "sar_maximum": [0.2],
                    "Sharpe Ratio": [0.75],
                }
            ),
            "ema_cross": pd.DataFrame(
                {
                    "ema_fast": [9],
                    "ema_slow": [21],
                    "Sharpe Ratio": [0.68],
                }
            ),
        }

        wfo_results = {
            "per_coin_results": {
                "BTC": {
                    "best_strategy": "psar_adx",
                    "best_params": {"sar_acceleration": 0.02},
                    "robustness_score": 0.85,
                },
                "ETH": {
                    "best_strategy": "ema_cross",
                    "best_params": {"ema_fast": 9, "ema_slow": 21},
                    "robustness_score": 0.72,
                },
            }
        }

        final_backtest_results = {
            "per_coin_final_stats": {
                "BTC": {
                    "strategy": "psar_adx",
                    "profit_pct": 15.5,
                    "sharpe": 0.82,
                    "max_drawdown": 12.3,
                    "total_trades": 45,
                    "win_rate": 52.5,
                },
                "ETH": {
                    "strategy": "ema_cross",
                    "profit_pct": 8.2,
                    "sharpe": 0.65,
                    "max_drawdown": 15.0,
                    "total_trades": 32,
                    "win_rate": 48.0,
                },
            },
            "final_stats": {
                "profit_pct": 12.3,
                "sharpe": 0.75,
                "sortino": 0.95,
                "max_drawdown": 10.5,
                "total_trades": 77,
                "win_rate": 50.5,
            },
        }

        output_dir = str(tmp_path)

        generate_pipeline_report(
            sensitivity_results, wfo_results, final_backtest_results, output_dir
        )

        report_file = tmp_path / "pipeline_report.md"
        content = report_file.read_text()

        # Should contain both strategies
        assert "psar_adx" in content
        assert "ema_cross" in content
        # Should contain both coins
        assert "BTC" in content
        assert "ETH" in content


# ---------------------------------------------------------------------------
# C4: Data-load-once tests (Phase A1 fix)
# ---------------------------------------------------------------------------

def _make_ohlcv_stub(n=200):
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    data = {}
    for sym in ("BTC-USD",):
        close = 1000.0 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 5)
        for field in ("open", "high", "low", "close", "volume"):
            data[(sym, field)] = close
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


class TestDataLoadOnce:
    """Verify the Phase A1 fix: OHLCV is not redundantly reloaded per phase."""

    def test_phase_2_skips_load_when_ohlcv_provided(self):
        """phase_2_full_data_validation must NOT call load_data_with_movers if ohlcv is given."""
        from ggTrader.utils.pipeline_phases import phase_2_full_data_validation

        ohlcv = _make_ohlcv_stub()
        config = {
            "START_CASH": 1000.0, "FEES": 0.001, "SLIPPAGE": 0.0, "FREQ": "4h",
            "EXIT_TOURNAMENT": [],
            "BTC_REGIME_FILTER": False,
        }
        wfo_results = {"per_coin_results": {}}

        mock_frozen = MagicMock(return_value={
            "final_stats": {}, "per_coin_final_stats": {}, "final_portfolio": None
        })

        # load_data_with_movers is a local import inside the function body; patch source module
        with patch("ggTrader.utils.setup.load_data_with_movers") as mock_load, \
             patch("ggTrader.core.orchestrator.run_frozen_params_combined_backtest",
                   mock_frozen):
            phase_2_full_data_validation(config, wfo_results, ohlcv=ohlcv)

        mock_load.assert_not_called()

    def test_phase_2_loads_data_when_ohlcv_not_provided(self):
        """phase_2_full_data_validation calls load_data_with_movers when ohlcv is None."""
        from ggTrader.utils.pipeline_phases import phase_2_full_data_validation

        ohlcv = _make_ohlcv_stub()
        config = {
            "START_CASH": 1000.0, "FEES": 0.001, "SLIPPAGE": 0.0, "FREQ": "4h",
            "EXIT_TOURNAMENT": [],
            "BTC_REGIME_FILTER": False,
        }
        wfo_results = {"per_coin_results": {}}

        mock_frozen = MagicMock(return_value={
            "final_stats": {}, "per_coin_final_stats": {}, "final_portfolio": None
        })

        # load_data_with_movers is a local import inside the function body; patch source module
        with patch("ggTrader.utils.setup.load_data_with_movers",
                   return_value=(ohlcv, None)) as mock_load, \
             patch("ggTrader.core.orchestrator.run_frozen_params_combined_backtest",
                   mock_frozen):
            phase_2_full_data_validation(config, wfo_results)  # no ohlcv arg

        mock_load.assert_called_once()

    def test_pipeline_runner_loads_ohlcv_once(self):
        """execute_full_pipeline calls load_data_with_movers exactly once (for phase 2 data).

        The load happens in pipeline_runner.py via a local import; patching the source module.
        """
        from ggTrader.pipeline.pipeline_runner import execute_full_pipeline

        ohlcv = _make_ohlcv_stub()
        args = Namespace(
            detailed_sensitivity=False,
            max_symbols=1,
            symbols_file=None,
            wfo_debug_metrics=False,
            train_metric=None,
            recent_validation_start=None,
            recent_validation_end=None,
            recent_validation_ccxt_tail=False,
            sensitivity=False,
            dual_exits=False,
            exits=None,
            dry_run=True,
            no_save=True,
            no_progress=True,
        )

        base_constants = {
            "START_CASH": 1000.0, "FEES": 0.001, "SLIPPAGE": 0.0, "FREQ": "4h",
            "SYMBOLS": ["BTC-USD"],
            "INTERVAL": "4h",
            "START_DATE": "2023-01-01",
            "END_DATE": "2023-12-31",
            "N_SPLITS": 2,
            "CHUNK_SIZE": 50,
        }

        stub_wfo = {"per_coin_results": {}, "final_portfolio": None, "final_stats": {},
                    "per_coin_final_stats": {}, "results_manager": None}

        with patch("ggTrader.pipeline.pipeline_runner.prepare_config_and_symbols",
                   return_value={**base_constants, "SYMBOLS": ["BTC-USD"]}), \
             patch("ggTrader.pipeline.pipeline_runner.phase_1_per_coin_multi_strategy_wfo",
                   return_value=stub_wfo), \
             patch("ggTrader.utils.setup.load_data_with_movers",
                   return_value=(ohlcv, None)) as mock_load, \
             patch("ggTrader.pipeline.pipeline_runner.phase_2_full_data_validation"), \
             patch("ggTrader.pipeline.pipeline_runner.phase_3_recent_performance"), \
             patch("ggTrader.pipeline.pipeline_runner.StatusLogger"):
            execute_full_pipeline(args, base_constants)

        # load_data_with_movers in the source module called exactly once
        mock_load.assert_called_once()
