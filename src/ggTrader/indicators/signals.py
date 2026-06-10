"""Numba trailing-stop kernels shared by the exit strategies in strategies.py.

The legacy ``Signals`` class and ``SignalFactory`` (vbt IndicatorFactory) that used
to live here were deleted with the ``USE_VECTORIZED`` flag: signal generation now
always goes through the strategy registry (``indicators/strategies.py``), which
dispatches on ``ENTRY_STRATEGY``/``EXIT_STRATEGY``. The old factory silently ran
psar_adx regardless of the configured entry strategy and filled stop exits at the
close instead of the gap-adjusted stop price.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(parallel=False)
def _atr_trailing_stop_long_ohlc_touch_2d_numba(
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    atr_vals: np.ndarray,
    entry_vals: np.ndarray,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, m = high_vals.shape
    stop = np.empty((n, m), dtype=np.float64)
    stop[:] = np.nan
    exits = np.zeros((n, m), dtype=np.bool_)

    for j in range(m):
        in_pos = False
        peak = 0.0
        current_stop = 0.0

        for i in range(n):
            if entry_vals[i, j] and not in_pos:
                in_pos = True
                peak = high_vals[i, j]
                current_stop = peak - mult * atr_vals[i, j]
                stop[i, j] = current_stop
                exits[i, j] = False
                continue

            if in_pos:
                if high_vals[i, j] > peak:
                    peak = high_vals[i, j]

                new_trail = peak - mult * atr_vals[i, j]
                if new_trail > current_stop:
                    current_stop = new_trail

                stop[i, j] = current_stop

                if low_vals[i, j] <= stop[i, j]:
                    exits[i, j] = True
                    in_pos = False
                    current_stop = 0.0
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits


@njit(parallel=False)
def _trailing_stop_long_ohlc_touch_2d_numba(
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    entry_vals: np.ndarray,
    stop_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate percentage-based trailing stop exits (long only) with low-touch logic."""
    n, m = high_vals.shape
    stop = np.empty((n, m), dtype=np.float64)
    stop[:] = np.nan
    exits = np.zeros((n, m), dtype=np.bool_)

    for j in range(m):
        in_pos = False
        peak = 0.0
        current_stop = 0.0

        for i in range(n):
            if entry_vals[i, j] and not in_pos:
                in_pos = True
                peak = high_vals[i, j]
                current_stop = peak * (1.0 - stop_pct / 100.0)
                stop[i, j] = current_stop
                exits[i, j] = False
                continue

            if in_pos:
                if high_vals[i, j] > peak:
                    peak = high_vals[i, j]
                    new_trail = peak * (1.0 - stop_pct / 100.0)
                    if new_trail > current_stop:
                        current_stop = new_trail

                stop[i, j] = current_stop

                if low_vals[i, j] <= stop[i, j]:
                    exits[i, j] = True
                    in_pos = False
                    current_stop = 0.0
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits
