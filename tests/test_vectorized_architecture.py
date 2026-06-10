"""Tests for vectorized signal generation and strategy architecture."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.strategies import (
    AtrTrailingExit,
    BollingerMeanReversionEntry,
    DonchianBreakoutEntry,
    EmaCrossEntry,
    KeltnerBreakoutEntry,
    MacdCrossEntry,
    PsarAdxEntry,
    RsiReversalEntry,
    StochRsiReversalEntry,
    SupertrendFlipEntry,
    get_entry_strategy,
    get_exit_strategy,
)


# Fixtures
@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_time = 500
    n_symbols = 3

    # Generate realistic OHLCV data
    dates = pd.date_range("2023-01-01", periods=n_time, freq="4h")
    price = np.random.uniform(100, 200, (n_time, n_symbols))

    close = pd.DataFrame(price, index=dates, columns=[f"SYM{i}" for i in range(n_symbols)])
    high = close * (1 + np.random.uniform(0, 0.01, close.shape))
    low = close * (1 - np.random.uniform(0, 0.01, close.shape))
    open_ = close * (1 + np.random.uniform(-0.005, 0.005, close.shape))

    return close, high, low, open_


class TestIndicatorPrecomputer:
    """Test IndicatorPrecomputer caching and computation."""

    def test_initialization(self, sample_ohlcv):
        """Test precomputer initialization."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        assert pc.close.shape == (500, 3)
        assert pc.high.shape == (500, 3)
        assert pc.low.shape == (500, 3)
        assert len(pc._cache) == 0

    def test_psar_computation(self, sample_ohlcv):
        """Test PSAR pre-computation."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        psar_ind = pc.compute_psar([0.02, 0.03], [0.2])
        assert psar_ind is not None
        assert hasattr(psar_ind, "psarl")

    def test_adx_computation(self, sample_ohlcv):
        """Test ADX pre-computation."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        adx_ind = pc.compute_adx([14, 21])
        assert adx_ind is not None
        assert hasattr(adx_ind, "adx")

    def test_atr_computation(self, sample_ohlcv):
        """Test ATR pre-computation."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        atr_ind = pc.compute_atr([14, 21])
        assert atr_ind is not None
        assert hasattr(atr_ind, "atrr")

    def test_ema_computation(self, sample_ohlcv):
        """Test EMA pre-computation."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        ema_ind = pc.compute_ema([9, 21])
        assert ema_ind is not None
        assert hasattr(ema_ind, "ema")

    def test_rsi_computation(self, sample_ohlcv):
        """Test RSI pre-computation."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        rsi_ind = pc.compute_rsi([14])
        assert rsi_ind is not None
        assert hasattr(rsi_ind, "rsi")

    def test_caching(self, sample_ohlcv):
        """Test that indicators are cached."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        # First call
        atr_1 = pc.compute_atr([14])
        assert len(pc._cache) == 1

        # Second call with same params should return cached
        atr_2 = pc.compute_atr([14])
        assert len(pc._cache) == 1
        assert atr_1 is atr_2

    def test_clear_cache(self, sample_ohlcv):
        """Test cache clearing."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        pc.compute_atr([14])
        pc.compute_adx([14])
        assert len(pc._cache) == 2

        pc.clear_cache()
        assert len(pc._cache) == 0


class TestVectorizedSignals:
    """Test vectorized multi-combo signal generation via the strategy registry."""

    def test_psar_adx_entries(self, sample_ohlcv):
        """Test vectorized PSAR+ADX entry generation across a param grid."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        param_grid = {
            "sar_acceleration": [0.02, 0.03],
            "sar_maximum": [0.2],
            "adx_length": [14],
            "adx_threshold": [25, 30],
            "use_dmp_cross": [False],
        }

        entries, param_combos = PsarAdxEntry(use_dmp_cross=False).compute_entries(pc, param_grid)

        assert entries.shape[0] == 500  # n_time
        assert entries.shape[1] == 4 * 3  # n_combos * n_symbols (2*2*1 combos, 3 symbols)
        assert len(param_combos) == 4
        assert entries.dtype == bool


class TestStrategies:
    """Test strategy implementations."""

    def test_psar_adx_entry_strategy(self, sample_ohlcv):
        """Test PsarAdxEntry strategy."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        strategy = PsarAdxEntry(use_dmp_cross=False)
        param_grid = {
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "adx_length": [14],
            "adx_threshold": [25],
        }

        entries, param_combos = strategy.compute_entries(pc, param_grid)
        assert entries.shape[0] == 500
        assert entries.dtype == bool

    def test_ema_cross_entry_strategy(self, sample_ohlcv):
        """Test EmaCrossEntry strategy."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        strategy = EmaCrossEntry()
        param_grid = {"ema_fast": [9, 12], "ema_slow": [21, 26]}

        entries, param_combos = strategy.compute_entries(pc, param_grid)
        assert entries.shape[0] == 500
        assert entries.dtype == bool
        assert len(param_combos) > 0

    def test_rsi_reversal_entry_strategy(self, sample_ohlcv):
        """Test RsiReversalEntry strategy."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        strategy = RsiReversalEntry()
        param_grid = {"rsi_length": [14], "rsi_oversold": [30, 35]}

        entries, param_combos = strategy.compute_entries(pc, param_grid)
        assert entries.shape[0] == 500
        assert entries.dtype == bool

    def test_macd_cross_entry_strategy(self, sample_ohlcv):
        """Test MacdCrossEntry shape (fast/slow/signal product × symbols)."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = MacdCrossEntry()
        param_grid = {
            "macd_fast": [10, 12],
            "macd_slow": [22, 26],
            "macd_signal": [9],
        }
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        n_sym = 3
        n_combo = 2 * 2 * 1
        assert entries.shape == (500, n_combo * n_sym)
        assert entries.dtype == bool
        assert len(param_combos) == n_combo

    def test_bbands_mean_reversion_entry_strategy(self, sample_ohlcv):
        """Test BollingerMeanReversionEntry."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = BollingerMeanReversionEntry()
        param_grid = {"bb_length": [15, 20], "bb_std": [2.0]}
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        assert entries.shape == (500, 2 * 3)
        assert entries.dtype == bool
        assert len(param_combos) == 2

    def test_donchian_breakout_entry_strategy(self, sample_ohlcv):
        """Test DonchianBreakoutEntry."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = DonchianBreakoutEntry()
        param_grid = {"donchian_length": [10, 20]}
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        assert entries.shape == (500, 2 * 3)
        assert entries.dtype == bool
        assert len(param_combos) == 2

    def test_supertrend_flip_entry_strategy(self, sample_ohlcv):
        """Test SupertrendFlipEntry (stacked length × multiplier × symbols)."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = SupertrendFlipEntry()
        param_grid = {"st_length": [7, 10], "st_multiplier": [2.0, 3.0]}
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        n_combo = 2 * 2
        assert entries.shape == (500, n_combo * 3)
        assert entries.dtype == bool
        assert len(param_combos) == n_combo

    def test_stoch_rsi_reversal_entry_strategy(self, sample_ohlcv):
        """Test StochRsiReversalEntry shape and dtype."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = StochRsiReversalEntry()
        param_grid = {
            "stochrsi_rsi_length": [10, 14],
            "stochrsi_stoch_length": [14],
            "stochrsi_oversold": [20],
        }
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        n_combo = 2 * 1 * 1  # 2 rsi_lengths × 1 stoch_length × 1 oversold
        assert entries.shape == (500, n_combo * 3)
        assert entries.dtype == bool
        assert len(param_combos) == n_combo

    def test_keltner_breakout_entry_strategy(self, sample_ohlcv):
        """Test KeltnerBreakoutEntry shape and dtype."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)
        strategy = KeltnerBreakoutEntry()
        param_grid = {"kc_length": [14, 20], "kc_multiplier": [1.5]}
        entries, param_combos = strategy.compute_entries(pc, param_grid)
        n_combo = 2 * 1  # 2 lengths × 1 multiplier
        assert entries.shape == (500, n_combo * 3)
        assert entries.dtype == bool
        assert len(param_combos) == n_combo

    def test_atr_trailing_exit_strategy(self, sample_ohlcv):
        """Test AtrTrailingExit strategy."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        # First generate entries
        entry_strategy = PsarAdxEntry()
        entry_grid = {
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "adx_length": [14],
            "adx_threshold": [25],
        }
        entries, _ = entry_strategy.compute_entries(pc, entry_grid)

        # Then generate exits
        exit_strategy = AtrTrailingExit()
        exit_grid = {"atr_length": [14], "atr_multiplier": [3.0]}

        exits, stops, prices = exit_strategy.compute_exits(entries, pc, exit_grid, n_symbols=3)
        assert exits.shape == entries.shape
        assert stops.shape == entries.shape
        assert prices.shape == entries.shape

    def test_fixed_stop_tp_exit_strategy(self, sample_ohlcv):
        """Test FixedStopTakeProfit strategy."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        entry_strategy = PsarAdxEntry()
        entry_grid = {
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "adx_length": [14],
            "adx_threshold": [25],
        }
        entries, _ = entry_strategy.compute_entries(pc, entry_grid)

        from ggTrader.indicators.strategies import FixedStopTakeProfit

        exit_strategy = FixedStopTakeProfit()
        exit_grid = {"stop_pct": [2.0], "take_profit_pct": [5.0]}

        exits, stops, prices = exit_strategy.compute_exits(entries, pc, exit_grid, n_symbols=3)
        assert exits.shape == entries.shape

    def test_strategy_registry(self):
        """Test strategy registry lookups."""
        entry_strat = get_entry_strategy("psar_adx")
        assert entry_strat.name == "psar_adx"

        exit_strat = get_exit_strategy("atr_trailing")
        assert exit_strat.name == "atr_trailing"

    def test_unknown_strategy(self):
        """Test error handling for unknown strategies."""
        with pytest.raises(ValueError):
            get_entry_strategy("unknown_strategy")

        with pytest.raises(ValueError):
            get_exit_strategy("unknown_strategy")


class TestStrategyCompatibility:
    """Test that strategies work together."""

    def test_entry_exit_compatibility(self, sample_ohlcv):
        """Test that entry and exit strategies work together."""
        close, high, low, _ = sample_ohlcv
        pc = IndicatorPrecomputer(close, high, low)

        # Use EMA entry
        entry_strategy = EmaCrossEntry()
        entry_grid = {"ema_fast": [9, 12], "ema_slow": [21, 26]}
        entries, _ = entry_strategy.compute_entries(pc, entry_grid)

        # Use ATR exit
        exit_strategy = AtrTrailingExit()
        exit_grid = {"atr_length": [14], "atr_multiplier": [3.0]}
        exits, stops, prices = exit_strategy.compute_exits(entries, pc, exit_grid, n_symbols=3)

        # Verify compatibility - entries may be reshaped, just check time dimension
        assert entries.shape[0] == exits.shape[0]
        assert entries.dtype == bool
        assert exits.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
