"""Unit tests for registry-path signal generation (PsarAdxEntry + AtrTrailingExit).

The legacy Signals/SignalFactory classes were deleted with USE_VECTORIZED;
these tests cover the same behaviors through the strategy registry and the
kept numba kernels in indicators/signals.py.
"""

import numpy as np
import pandas as pd

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.signals import _atr_trailing_stop_long_ohlc_touch_2d_numba
from ggTrader.indicators.strategies import AtrTrailingExit, PsarAdxEntry


def create_dummy_ohlcv(n_rows: int = 100, trend: str = "flat") -> pd.DataFrame:
    """Create a dummy OHLCV MultiIndex DataFrame with proper names."""
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="4h")
    if trend == "up":
        close = np.linspace(100, 200, n_rows)
    elif trend == "down":
        close = np.linspace(200, 100, n_rows)
    else:
        close = np.full(n_rows, 100.0)

    data = {
        ("BTC", "open"): close * 0.99,
        ("BTC", "high"): close * 1.01,
        ("BTC", "low"): close * 0.98,
        ("BTC", "close"): close,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def _precomputer(ohlcv: pd.DataFrame) -> IndicatorPrecomputer:
    close = ohlcv.xs("close", axis=1, level="field")
    high = ohlcv.xs("high", axis=1, level="field")
    low = ohlcv.xs("low", axis=1, level="field")
    return IndicatorPrecomputer(close, high, low)


def test_entry_signals_logic():
    """Entry signals are boolean arrays of the right shape on trending data."""
    ohlcv = create_dummy_ohlcv(n_rows=50, trend="up")
    pc = _precomputer(ohlcv)

    entries, combos = PsarAdxEntry(use_dmp_cross=False).compute_entries(
        pc,
        {
            "adx_length": [14],
            "adx_threshold": [20],
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "use_dmp_cross": [False],
        },
    )

    assert isinstance(entries, np.ndarray)
    assert entries.dtype == bool
    assert entries.shape == (50, 1)  # 1 combo x 1 symbol
    assert len(combos) == 1


def test_trailing_stop_kernel_insufficient_data():
    """All-NaN ATR (insufficient warmup bars) must yield no exits and NaN stops."""
    n = 5
    high = np.full((n, 1), 101.0)
    low = np.full((n, 1), 99.0)
    atr = np.full((n, 1), np.nan)
    entries = np.zeros((n, 1), dtype=np.bool_)
    entries[1, 0] = True

    stops, exits = _atr_trailing_stop_long_ohlc_touch_2d_numba(high, low, atr, entries, 3.0)
    assert not exits.any()
    # entered with NaN ATR -> stop is NaN, never touched
    assert np.isnan(stops[1, 0])


def test_exit_fill_price_gap_handling():
    """On exit bars the fill price must be min(open-proxy, stop), elsewhere close."""
    ohlcv = create_dummy_ohlcv(n_rows=60, trend="up")
    pc = _precomputer(ohlcv)
    params = {"atr_length": [14], "atr_multiplier": [0.5]}  # tight stop -> exits fire

    entries, _ = PsarAdxEntry(use_dmp_cross=False).compute_entries(
        pc,
        {
            "adx_length": [14],
            "adx_threshold": [5],
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "use_dmp_cross": [False],
        },
    )
    exits, stops, price = AtrTrailingExit().compute_exits(entries, pc, params, n_symbols=1)

    close = pc.close if pc.close.ndim == 2 else pc.close[:, None]
    on_exit = exits.astype(bool)
    if on_exit.any():
        expected = np.minimum(close, stops)[on_exit]
        assert np.allclose(price[on_exit], expected, equal_nan=True)
    off_exit = ~on_exit
    assert np.allclose(price[off_exit], np.broadcast_to(close, price.shape)[off_exit])


def test_grid_broadcasting_column_shape():
    """A param grid produces (param_combo, symbol) MultiIndex columns."""
    ohlcv = create_dummy_ohlcv(n_rows=50, trend="up")
    engine = FastBacktest(
        ohlcv,
        {
            "adx_length": [14],
            "adx_threshold": [20, 30],
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "use_dmp_cross": [False],
            "atr_length": [14],
            "atr_multiplier": [3.0],
        },
        config={"FREQ": "4h"},
    )
    engine.run(show_progress=False)

    assert isinstance(engine.entries.columns, pd.MultiIndex)
    assert engine.entries.shape[1] == 2  # 2 combos x 1 symbol
    assert engine.entries.columns.names == ["param_combo", "symbol"]


def test_scalar_params_equal_one_cell_grid():
    """Scalar param dicts must behave exactly like 1-cell grids (WFO test-fold path)."""
    ohlcv = create_dummy_ohlcv(n_rows=120, trend="up")
    scalar_params = {
        "adx_length": 14,
        "adx_threshold": 20,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": False,
        "atr_length": 14,
        "atr_multiplier": 3.0,
    }
    grid_params = {k: [v] for k, v in scalar_params.items()}
    config = {"FREQ": "4h", "ENTRY_STRATEGY": "psar_adx", "EXIT_STRATEGY": "atr_trailing"}

    scalar_engine = FastBacktest(ohlcv, scalar_params, config=config)
    scalar_pf = scalar_engine.run(show_progress=False)
    grid_engine = FastBacktest(ohlcv, grid_params, config=config)
    grid_engine.run(show_progress=False)

    assert np.array_equal(scalar_engine.entries.values, grid_engine.entries.values)
    assert np.array_equal(scalar_engine.exits.values, grid_engine.exits.values)
    assert scalar_engine.get_stats() == grid_engine.get_stats()

    # single combo -> one portfolio group, and trades happen on trending data
    sharpe = scalar_pf.sharpe_ratio()
    n_groups = len(sharpe) if isinstance(sharpe, pd.Series) else 1
    assert n_groups == 1
    assert scalar_engine.get_stats()["total_trades"] > 0
