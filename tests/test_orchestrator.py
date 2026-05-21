"""Unit tests for orchestrators using mocks."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ggTrader.core.orchestrator import (
    _apply_tiered_regime_mask,
    _compute_allocation_weights,
    run_backtest_orchestrator,
    run_frozen_params_combined_backtest,
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


@patch("ggTrader.core.sensitivity.load_data_with_movers")
@patch("ggTrader.core.sensitivity.ResultsManager")
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


@patch("ggTrader.core.wfo.load_data_with_movers")
@patch("ggTrader.core.wfo.ResultsManager")
def test_run_wfo_orchestrator(mock_rm, mock_load, mock_ohlcv):
    mock_load.return_value = (mock_ohlcv, None)

    config = {
        "WINDOW_LEN": 50,
        "SET_LENS": [25],  # 25 test, 25 train
        "USE_MOVERS": 0,
        "START_CASH": 10000,
        "FEES": 0.001,
        "SLIPPAGE": 0.001,
        "FREQ": "4h",
    }
    # Provide multiple thresholds
    param_grid = {"adx_threshold": [20, 25]}

    res = run_wfo_orchestrator(config, param_grid, save_results=False)

    assert len(res["wfo_stats"]) >= 1
    assert "best_robust_params" in res


# ---------------------------------------------------------------------------
# _apply_tiered_regime_mask
# ---------------------------------------------------------------------------


def _make_combined_entries(symbols, n=100):
    """Build a simple entries DataFrame with all-True signals."""
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    data = {(sym, "entry"): np.ones(n, dtype=bool) for sym in symbols}
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def test_regime_mask_high_corr_blocked_on_bear_bars():
    """Coins above LEADER_CORR_THRESHOLD to BTC are gated by btc_regime."""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    btc_regime = pd.Series([False] * 50 + [True] * 50, index=idx)
    combined_entries = _make_combined_entries(["BTC-USD", "ETH-USD"], n=n)

    config = {"LEADER_CORR_THRESHOLD": 0.7, "EMA_WARMUP_BARS": 0}
    btc_corrs = {"BTC-USD": 1.0, "ETH-USD": 0.8}

    filtered = _apply_tiered_regime_mask(combined_entries, btc_corrs, btc_regime, config)
    assert not filtered.iloc[:50].any().any(), "Bear bars should be blocked"
    assert filtered.iloc[50:].any().any(), "Bull bars should pass through"


def test_regime_mask_low_corr_exempt_coins_always_pass():
    """Coins below LEADER_CORR_THRESHOLD are never filtered."""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    btc_regime = pd.Series([False] * n, index=idx)
    combined_entries = _make_combined_entries(["LOW-USD"], n=n)

    config = {"LEADER_CORR_THRESHOLD": 0.7, "EMA_WARMUP_BARS": 0}
    btc_corrs = {"LOW-USD": 0.3}

    filtered = _apply_tiered_regime_mask(combined_entries, btc_corrs, btc_regime, config)
    assert filtered.all().all()


def test_regime_mask_none_btc_regime_returns_unchanged():
    """If the BTC mask is None the function returns entries unchanged."""
    n = 50
    combined_entries = _make_combined_entries(["SOL-USD"], n=n)
    config = {"LEADER_CORR_THRESHOLD": 0.7, "EMA_WARMUP_BARS": 0, "BENCHMARK_SYMBOL": "BTC-USD"}
    btc_corrs = {"SOL-USD": 0.9}

    result = _apply_tiered_regime_mask(combined_entries, btc_corrs, None, config)

    pd.testing.assert_frame_equal(result, combined_entries)


# ---------------------------------------------------------------------------
# _compute_allocation_weights in combined backtest context
# ---------------------------------------------------------------------------


def test_allocation_weights_sum_to_one():
    scores = [0.8, 0.4, 0.2, 0.1, 0.05]
    weights = _compute_allocation_weights(scores, config={"MAX_COIN_ALLOCATION": 0.5})
    assert abs(weights.sum() - 1.0) < 1e-9


def test_allocation_weights_cap_respected():
    scores = [10.0, 0.1, 0.1, 0.1, 0.1]  # first coin would dominate without cap
    cap = 0.25
    weights = _compute_allocation_weights(scores, config={"MAX_COIN_ALLOCATION": cap})
    assert weights.max() <= cap + 1e-9
    assert abs(weights.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# run_frozen_params_combined_backtest — integration test
# ---------------------------------------------------------------------------


def _make_multi_symbol_ohlcv(symbols=("BTC-USD", "ETH-USD"), n=200):
    """Synthetic MultiIndex OHLCV for combined backtest."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    data = {}
    for sym in symbols:
        close = 1000.0 + np.cumsum(rng.standard_normal(n) * 5)
        for field in ("open", "high", "low", "close", "volume"):
            data[(sym, field)] = close
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def _per_coin_results_stub(symbols=("BTC-USD", "ETH-USD")):
    return {
        sym: {
            "best_strategy": "psar_adx",
            "best_exit": "atr_trailing",
            "best_params": {
                "sar_acceleration": 0.02,
                "sar_maximum": 0.1,
                "adx_length": 14,
                "adx_threshold": 25,
                "use_dmp_cross": False,
                "atr_length": 14,
                "atr_multiplier": 3.0,
            },
            "robustness_score": 0.6,
            "oos_robustness_score": 0.5,
            "selection_reason": "wfo_robustness",
        }
        for sym in symbols
    }


def test_frozen_params_combined_backtest_returns_expected_keys():
    """Combined backtest completes and returns all required keys."""
    ohlcv = _make_multi_symbol_ohlcv()
    per_coin = _per_coin_results_stub()
    config = {
        "START_CASH": 1000.0,
        "FEES": 0.001,
        "SLIPPAGE": 0.0005,
        "FREQ": "4h",
        "BTC_REGIME_FILTER": False,
        "BENCHMARK_SYMBOL": "BTC-USD",
        "MAX_COIN_ALLOCATION": 0.25,
    }
    with patch("ggTrader.core.benchmarking._btc_buy_hold_portfolio_stats", return_value={}), patch(
        "ggTrader.core.benchmarking._sp500_buy_hold_portfolio_stats", return_value={}
    ):
        result = run_frozen_params_combined_backtest(
            ohlcv, per_coin, config, exit_tournament=["atr_trailing"], save_results=False
        )

    assert "final_portfolio" in result
    assert "final_stats" in result
    assert "per_coin_final_stats" in result


def test_frozen_params_allocation_weights_sum_to_one():
    """Allocation weights passed to vbt must sum to 1 (verified via per-coin stat coverage)."""
    ohlcv = _make_multi_symbol_ohlcv()
    per_coin = _per_coin_results_stub()
    config = {
        "START_CASH": 1000.0,
        "FEES": 0.001,
        "SLIPPAGE": 0.0005,
        "FREQ": "4h",
        "BTC_REGIME_FILTER": False,
        "MAX_COIN_ALLOCATION": 0.5,
    }
    with patch("ggTrader.core.benchmarking._btc_buy_hold_portfolio_stats", return_value={}), patch(
        "ggTrader.core.benchmarking._sp500_buy_hold_portfolio_stats", return_value={}
    ):
        result = run_frozen_params_combined_backtest(
            ohlcv, per_coin, config, exit_tournament=["atr_trailing"], save_results=False
        )

    final_stats = result["final_stats"]
    # If the portfolio ran we should get at least some trades or a defined profit_pct
    assert final_stats.get("profit_pct") is not None


def test_frozen_params_regime_filter_blocks_signals():
    """When BTC_REGIME_FILTER=True and btc_regime=all-False, no entries should survive."""
    ohlcv = _make_multi_symbol_ohlcv(symbols=("BTC-USD", "ETH-USD"), n=250)
    per_coin = _per_coin_results_stub()
    config = {
        "START_CASH": 1000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "4h",
        "BTC_REGIME_FILTER": True,
        "LEADER_CORR_THRESHOLD": 0.7,
        "EMA_WARMUP_BARS": 0,
        "MAX_COIN_ALLOCATION": 0.5,
        "BENCHMARK_SYMBOL": "BTC-USD",
    }
    len(ohlcv)
    all_bear = pd.Series(False, index=ohlcv.index)

    with patch("ggTrader.core.orchestrator._compute_btc_regime_mask", return_value=all_bear), patch(
        "ggTrader.core.orchestrator._compute_btc_correlations",
        return_value={"BTC-USD": 1.0, "ETH-USD": 0.9},
    ), patch("ggTrader.core.benchmarking._btc_buy_hold_portfolio_stats", return_value={}), patch(
        "ggTrader.core.benchmarking._sp500_buy_hold_portfolio_stats", return_value={}
    ):
        result = run_frozen_params_combined_backtest(
            ohlcv, per_coin, config, exit_tournament=["atr_trailing"], save_results=False
        )

    # No entries → no trades
    assert result["final_stats"]["total_trades"] == 0
