"""Tests for exit-strategy WFO tournament: fixed_sl_tp product, WFO smoke, best_exit outputs."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.fast_backtest import (
    FastBacktest,
    _expand_entries_for_exit_product,
    _get_exit_axis_keys,
    _merge_entry_exit_param_combos,
)
from ggTrader.indicators.strategies import (
    EXIT_PARAM_AXIS_KEYS,
    EXIT_REGISTRY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.1,
    "FEES": 0.001,
    "SLIPPAGE": 0.0005,
    "FREQ": "4h",
}


def _make_ohlcv(n_time: int = 300, symbol: str = "BTC-USD", seed: int = 42) -> pd.DataFrame:
    """Small synthetic OHLCV with a deterministic random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_time, freq="4h")
    close = 1000 + np.cumsum(rng.standard_normal(n_time) * 10)
    high = close + np.abs(rng.standard_normal(n_time) * 5)
    low = close - np.abs(rng.standard_normal(n_time) * 5)
    open_ = close.copy()
    volume = rng.integers(100, 1000, n_time).astype(float)
    cols = pd.MultiIndex.from_tuples(
        [(symbol, f) for f in ("open", "high", "low", "close", "volume")],
        names=["symbol", "field"],
    )
    return pd.DataFrame(
        np.column_stack([open_, high, low, close, volume]),
        index=dates,
        columns=cols,
    )


# ---------------------------------------------------------------------------
# 1) EXIT_PARAM_AXIS_KEYS and _get_exit_axis_keys
# ---------------------------------------------------------------------------


class TestExitParamAxisKeys:
    def test_atr_trailing_keys(self):
        keys = EXIT_PARAM_AXIS_KEYS["atr_trailing"]
        assert "atr_length" in keys
        assert "atr_multiplier" in keys

    def test_fixed_sl_tp_keys(self):
        keys = EXIT_PARAM_AXIS_KEYS["fixed_sl_tp"]
        assert "stop_pct" in keys
        assert "take_profit_pct" in keys

    def test_unknown_exit_returns_empty(self):
        assert _get_exit_axis_keys("nonexistent") == ()

    def test_helper_matches_registry(self):
        for name in EXIT_REGISTRY:
            assert name in EXIT_PARAM_AXIS_KEYS, f"{name} missing from EXIT_PARAM_AXIS_KEYS"


# ---------------------------------------------------------------------------
# 2) _expand_entries_for_exit_product / backward-compat alias
# ---------------------------------------------------------------------------


class TestExpandEntriesForExitProduct:
    def _make_entries(self, n_time: int, n_entry_combos: int, n_symbols: int) -> np.ndarray:
        """Random sparse bool array shaped (n_time, n_entry_combos * n_symbols)."""
        rng = np.random.default_rng(0)
        return rng.random((n_time, n_entry_combos * n_symbols)) > 0.97

    def test_n_exit_one_is_no_op(self):
        entries = self._make_entries(100, 3, 2)
        result = _expand_entries_for_exit_product(entries, n_symbols=2, n_exit=1)
        np.testing.assert_array_equal(entries, result)

    def test_width_doubles_with_n_exit_2(self):
        entries = self._make_entries(100, 3, 2)
        result = _expand_entries_for_exit_product(entries, n_symbols=2, n_exit=2)
        assert result.shape == (100, 3 * 2 * 2)  # 3 combos × 2 exits × 2 symbols

    def test_each_block_repeated_correctly(self):
        """Block 0 and block 1 of expanded array should equal original block 0."""
        entries = self._make_entries(50, 2, 1)
        expanded = _expand_entries_for_exit_product(entries, n_symbols=1, n_exit=2)
        # Original block 0 → expanded blocks 0 and 1
        np.testing.assert_array_equal(expanded[:, 0:1], entries[:, 0:1])
        np.testing.assert_array_equal(expanded[:, 1:2], entries[:, 0:1])



# ---------------------------------------------------------------------------
# 3) _merge_entry_exit_param_combos
# ---------------------------------------------------------------------------


class TestMergeEntryExitParamCombos:
    def test_atr_axes_merged_correctly(self):
        entry_combos = [{"ema_fast": 5, "ema_slow": 21}, {"ema_fast": 9, "ema_slow": 50}]
        strat_params = {
            "ema_fast": [5, 9],
            "ema_slow": [21, 50],
            "atr_length": [10, 20],
            "atr_multiplier": [2.0, 3.0],
        }
        result = _merge_entry_exit_param_combos(
            entry_combos, strat_params, ("atr_length", "atr_multiplier")
        )
        # 2 entry combos × 2 lengths × 2 multipliers = 8
        assert len(result) == 8
        # All atr_length values should be present
        atr_lengths_seen = {r["atr_length"] for r in result}
        assert atr_lengths_seen == {10, 20}

    def test_fixed_sl_tp_axes_merged_correctly(self):
        entry_combos = [{"rsi_length": 14}]
        strat_params = {"rsi_length": [14], "stop_pct": [2.0, 4.0], "take_profit_pct": [5.0, 10.0]}
        result = _merge_entry_exit_param_combos(
            entry_combos, strat_params, ("stop_pct", "take_profit_pct")
        )
        # 1 entry combo × 2 stops × 2 tps = 4
        assert len(result) == 4
        tp_vals = {r["take_profit_pct"] for r in result}
        assert tp_vals == {5.0, 10.0}

    def test_no_exit_axes_returns_entry_combos(self):
        entry_combos = [{"ema_fast": 5}]
        strat_params = {"ema_fast": [5]}
        result = _merge_entry_exit_param_combos(entry_combos, strat_params, ())
        assert len(result) == 1
        assert result[0] == {"ema_fast": 5}


# ---------------------------------------------------------------------------
# 4) FixedStopTakeProfit multi-combo correctness
# ---------------------------------------------------------------------------


class TestFixedStopTakeProfitMultiCombo:
    """Verify that FixedStopTakeProfit uses the correct (stop, tp) per block."""

    def _make_indicator_precomputer(
        self, close_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray
    ):
        """Minimal object matching IndicatorPrecomputer's attribute interface."""
        import pandas as pd

        from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer

        idx = pd.date_range("2023-01-01", periods=len(close_arr), freq="4h")
        c = pd.DataFrame({"SYM": close_arr}, index=idx)
        h = pd.DataFrame({"SYM": high_arr}, index=idx)
        lo = pd.DataFrame({"SYM": low_arr}, index=idx)
        return IndicatorPrecomputer(c, h, lo)

    def test_last_param_combos_length_matches_columns(self, sample_ohlcv_data):
        """_last_param_combos must equal columns // n_symbols for fixed_sl_tp grid."""
        ohlcv = sample_ohlcv_data[["BTC-USD"]]
        config = {
            **_BASE_CONFIG,
            "ENTRY_STRATEGY": "ema_cross",
            "EXIT_STRATEGY": "fixed_sl_tp",
            "USE_VECTORIZED": True,
        }
        params = {
            "ema_fast": [5, 9],
            "ema_slow": [21, 50],
            "stop_pct": [2.0, 4.0],
            "take_profit_pct": [5.0, 10.0],
        }
        engine = FastBacktest(ohlcv, params, config=config)
        pf = engine.run(show_progress=False)
        combos = engine._last_param_combos
        assert combos is not None and len(combos) > 0
        n_symbols = 1
        expected_columns = len(combos) * n_symbols
        assert pf.wrapper.shape[1] == expected_columns, (
            f"Portfolio columns {pf.wrapper.shape[1]} != combos × symbols {expected_columns}"
        )

    def test_different_stops_produce_different_exits(self, sample_ohlcv_data):
        """Tight stop (1%) should trigger more exits than loose stop (10%)."""
        ohlcv = sample_ohlcv_data[["BTC-USD"]]
        base_cfg = {
            **_BASE_CONFIG,
            "ENTRY_STRATEGY": "ema_cross",
            "EXIT_STRATEGY": "fixed_sl_tp",
            "USE_VECTORIZED": True,
        }

        tight_engine = FastBacktest(
            ohlcv,
            {"ema_fast": 9, "ema_slow": 21, "stop_pct": 1.0, "take_profit_pct": 2.0},
            config=base_cfg,
        )
        tight_pf = tight_engine.run(show_progress=False)

        loose_engine = FastBacktest(
            ohlcv,
            {"ema_fast": 9, "ema_slow": 21, "stop_pct": 10.0, "take_profit_pct": 20.0},
            config=base_cfg,
        )
        loose_pf = loose_engine.run(show_progress=False)

        # Tight stop should produce >= trades (higher turnover) than very loose stop.
        tight_trades = int(tight_pf.trades.count().sum())
        loose_trades = int(loose_pf.trades.count().sum())
        assert tight_trades >= loose_trades, (
            f"Expected tight stop to have >= trades; got tight={tight_trades} loose={loose_trades}"
        )


# ---------------------------------------------------------------------------
# 5) Vectorized path: _last_param_combos alignment for atr_trailing (regression)
# ---------------------------------------------------------------------------


class TestVectorizedLastParamCombosAtrRegression:
    def test_last_param_combos_aligns_with_sharpe_series(self, sample_ohlcv_data):
        """Sharpe series length must equal _last_param_combos count (regression guard)."""
        ohlcv = sample_ohlcv_data[["BTC-USD"]]
        config = {
            **_BASE_CONFIG,
            "ENTRY_STRATEGY": "ema_cross",
            "EXIT_STRATEGY": "atr_trailing",
            "USE_VECTORIZED": True,
        }
        params = {
            "ema_fast": [5, 9],
            "ema_slow": [21, 50],
            "atr_length": [10, 20],
            "atr_multiplier": [2.0, 3.0],
        }
        engine = FastBacktest(ohlcv, params, config=config)
        pf = engine.run(show_progress=False)
        combos = engine._last_param_combos
        sharpe = pf.sharpe_ratio()
        sharpe_arr = np.asarray(sharpe).ravel()
        assert len(combos) == sharpe_arr.size, (
            f"Sharpe size {sharpe_arr.size} != combo count {len(combos)}"
        )


# ---------------------------------------------------------------------------
# 6) EXIT_TOURNAMENT key stored in per_coin_results
# ---------------------------------------------------------------------------


class TestExitTournamentInOrchestratorOutput:
    """Smoke-test that per_coin_results contains best_exit after WFO tournament."""

    def test_per_coin_results_has_best_exit_key(self, sample_ohlcv_data):
        """run_multi_strategy_per_coin_wfo must record best_exit in per_coin_results."""
        from ggTrader.core.orchestrator import run_multi_strategy_per_coin_wfo

        ohlcv = sample_ohlcv_data[["BTC-USD"]]
        # Use only one symbol (BTC) and a tiny grid to keep runtime minimal.
        config = {
            **_BASE_CONFIG,
            "SYMBOLS": ["BTC-USD"],
            "START_DATE": "2023-01-01",
            "END_DATE": "2023-12-31",
            "INTERVAL": "4h",
            "N_SPLITS": 2,
            "TEST_RATIO": 2.0,
            "MIN_CLOSED_TRADES_TRAIN": 0,
            "USE_VECTORIZED": True,
            "USE_VECTORIZED_SENSITIVITY": True,
            "USE_MOVERS": 0,
            "EXIT_TOURNAMENT": ["atr_trailing", "fixed_sl_tp"],
            "_OHLCV_OVERRIDE": ohlcv,  # Injected below via monkeypatch
        }
        strategy_grids = {
            "ema_cross": {
                "ema_fast": [5],
                "ema_slow": [21],
                "atr_length": [10],
                "atr_multiplier": [2.0],
                "stop_pct": [2.0],
                "take_profit_pct": [5.0],
            }
        }

        # Monkeypatch load_data_with_movers to return our synthetic data.
        import ggTrader.core.orchestrator as orch_mod

        original_loader = orch_mod.load_data_with_movers

        def _mock_loader(cfg):
            return ohlcv, None

        orch_mod.load_data_with_movers = _mock_loader
        try:
            result = run_multi_strategy_per_coin_wfo(
                config=config,
                strategy_param_grids=strategy_grids,
                save_results=False,
                show_progress=False,
            )
        finally:
            orch_mod.load_data_with_movers = original_loader

        per_coin = result["per_coin_results"]
        assert "BTC-USD" in per_coin
        coin_result = per_coin["BTC-USD"]
        assert "best_exit" in coin_result, "per_coin_results must contain 'best_exit'"
        assert coin_result["best_exit"] in EXIT_REGISTRY, (
            f"best_exit '{coin_result['best_exit']}' not in EXIT_REGISTRY"
        )

    def test_per_coin_final_stats_has_exit_key(self, sample_ohlcv_data):
        """per_coin_final_stats dict must contain 'exit' key after Phase 3 replay."""
        from ggTrader.core.orchestrator import run_multi_strategy_per_coin_wfo

        ohlcv = sample_ohlcv_data[["BTC-USD"]]
        config = {
            **_BASE_CONFIG,
            "SYMBOLS": ["BTC-USD"],
            "START_DATE": "2023-01-01",
            "END_DATE": "2023-12-31",
            "INTERVAL": "4h",
            "N_SPLITS": 2,
            "TEST_RATIO": 2.0,
            "MIN_CLOSED_TRADES_TRAIN": 0,
            "USE_VECTORIZED": True,
            "USE_VECTORIZED_SENSITIVITY": True,
            "USE_MOVERS": 0,
            "EXIT_TOURNAMENT": ["atr_trailing"],
        }
        strategy_grids = {
            "ema_cross": {
                "ema_fast": [5],
                "ema_slow": [21],
                "atr_length": [10],
                "atr_multiplier": [2.0],
            }
        }

        import ggTrader.core.orchestrator as orch_mod

        original_loader = orch_mod.load_data_with_movers

        def _mock_loader(cfg):
            return ohlcv, None

        orch_mod.load_data_with_movers = _mock_loader
        try:
            result = run_multi_strategy_per_coin_wfo(
                config=config,
                strategy_param_grids=strategy_grids,
                save_results=False,
                show_progress=False,
            )
        finally:
            orch_mod.load_data_with_movers = original_loader

        final_stats = result["per_coin_final_stats"]
        assert "BTC-USD" in final_stats
        assert "exit" in final_stats["BTC-USD"], "per_coin_final_stats must contain 'exit'"


# ---------------------------------------------------------------------------
# 7) Pipeline-level: build_param_grid and EXIT_AXIS_GRIDS
# ---------------------------------------------------------------------------


class TestBuildParamGrid:
    def test_atr_trailing_grid_has_no_fixed_sl_tp_keys(self):
        from ggTrader.pipeline.param_grids import (
            COARSE_ENTRY_PARAM_GRIDS,
            EXIT_AXIS_GRIDS,
            build_param_grid,
        )

        grid = build_param_grid(
            "ema_cross", "atr_trailing", COARSE_ENTRY_PARAM_GRIDS, EXIT_AXIS_GRIDS
        )
        assert "atr_length" in grid
        assert "stop_pct" not in grid
        assert "take_profit_pct" not in grid

    def test_fixed_sl_tp_grid_has_no_atr_keys(self):
        from ggTrader.pipeline.param_grids import (
            COARSE_ENTRY_PARAM_GRIDS,
            EXIT_AXIS_GRIDS,
            build_param_grid,
        )

        grid = build_param_grid(
            "ema_cross", "fixed_sl_tp", COARSE_ENTRY_PARAM_GRIDS, EXIT_AXIS_GRIDS
        )
        assert "stop_pct" in grid
        assert "take_profit_pct" in grid
        assert "atr_length" not in grid
        assert "atr_multiplier" not in grid

    def test_entry_params_preserved(self):
        from ggTrader.pipeline.param_grids import (
            COARSE_ENTRY_PARAM_GRIDS,
            EXIT_AXIS_GRIDS,
            build_param_grid,
        )

        grid = build_param_grid(
            "ema_cross", "atr_trailing", COARSE_ENTRY_PARAM_GRIDS, EXIT_AXIS_GRIDS
        )
        assert "ema_fast" in grid
        assert "ema_slow" in grid

    def test_all_entries_can_be_merged_with_all_exits(self):
        from ggTrader.pipeline.param_grids import (
            COARSE_ENTRY_PARAM_GRIDS,
            EXIT_AXIS_GRIDS,
            build_param_grid,
        )

        for entry_name in COARSE_ENTRY_PARAM_GRIDS:
            for exit_name in EXIT_AXIS_GRIDS:
                grid = build_param_grid(
                    entry_name, exit_name, COARSE_ENTRY_PARAM_GRIDS, EXIT_AXIS_GRIDS
                )
                assert isinstance(grid, dict)
                assert len(grid) > 0


class TestParseExitTournament:
    def test_filters_unknown_and_keeps_order(self):
        from ggTrader.indicators.strategies import EXIT_REGISTRY
        from ggTrader.pipeline.exit_tournament import parse_exit_tournament

        out = parse_exit_tournament(["atr_trailing", "bogus", "fixed_sl_tp"], EXIT_REGISTRY)
        assert out == ["atr_trailing", "fixed_sl_tp"]

    def test_default_all_registered(self):
        from ggTrader.indicators.strategies import EXIT_REGISTRY
        from ggTrader.pipeline.exit_tournament import parse_exit_tournament

        out = parse_exit_tournament(None, EXIT_REGISTRY)
        assert set(out) == set(EXIT_REGISTRY.keys())


class TestFilterStratParamsForExit:
    def test_atr_trailing_drops_sl_tp_keys(self):
        from ggTrader.indicators.strategies import filter_strat_params_for_exit

        raw = {
            "ema_fast": [5],
            "atr_length": [10],
            "atr_multiplier": [2.0],
            "stop_pct": [2.0],
            "take_profit_pct": [5.0],
        }
        out = filter_strat_params_for_exit(raw, "atr_trailing")
        assert "stop_pct" not in out
        assert "take_profit_pct" not in out
        assert "atr_length" in out

    def test_fixed_sl_tp_drops_atr_keys(self):
        from ggTrader.indicators.strategies import filter_strat_params_for_exit

        raw = {
            "ema_fast": [5],
            "atr_length": [10],
            "stop_pct": [2.0],
            "take_profit_pct": [5.0],
        }
        out = filter_strat_params_for_exit(raw, "fixed_sl_tp")
        assert "atr_length" not in out
        assert "stop_pct" in out
