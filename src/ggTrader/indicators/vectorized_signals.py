"""Vectorized signal generation using numpy broadcasting and pre-computed indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.signals import _atr_trailing_stop_long_ohlc_touch_2d_numba


def generate_psar_adx_entries_vectorized(
    precomputer: IndicatorPrecomputer,
    param_grid: dict[str, list],
    use_dmp_cross: bool = True,
) -> tuple[np.ndarray, list]:
    """Generate PSAR + ADX entry signals via vectorized broadcasting.

    Args:
        precomputer: IndicatorPrecomputer instance.
        param_grid: Parameter grid with keys like 'sar_acceleration', 'adx_threshold', etc.
        use_dmp_cross: Whether to include DMP > DMN condition.

    Returns:
        (entries_array, param_combo_list) where entries_array is (n_time, n_symbol_combos).
        param_combo_list contains dicts with the parameter values for each combo.
    """
    close = precomputer.close
    if close.ndim == 1:
        close = close[:, np.newaxis]
    n_time, n_symbols = close.shape

    sar_accel = param_grid.get("sar_acceleration", [0.02])
    sar_max = param_grid.get("sar_maximum", [0.2])
    adx_lengths = param_grid.get("adx_length", [14])
    adx_thresholds = param_grid.get("adx_threshold", [25])

    sar_accel = sar_accel if isinstance(sar_accel, list) else [sar_accel]
    sar_max = sar_max if isinstance(sar_max, list) else [sar_max]
    adx_lengths = adx_lengths if isinstance(adx_lengths, list) else [adx_lengths]
    adx_thresholds = adx_thresholds if isinstance(adx_thresholds, list) else [adx_thresholds]

    # Pre-compute indicators
    psar_ind = precomputer.compute_psar(sar_accel, sar_max)
    adx_ind = precomputer.compute_adx(adx_lengths)

    psarl = psar_ind.psarl.values if hasattr(psar_ind.psarl, "values") else psar_ind.psarl
    adx_vals = adx_ind.adx.values if hasattr(adx_ind.adx, "values") else adx_ind.adx

    if adx_vals.ndim == 1:
        adx_vals = adx_vals[:, np.newaxis]

    # Build entries via broadcasting
    param_combos = []
    entries_list = []

    for sar_accel_val in sar_accel:
        for sar_max_val in sar_max:
            for adx_len_idx, adx_len in enumerate(adx_lengths):
                for adx_thresh in adx_thresholds:
                    # Extract PSAR and ADX for this combo
                    # PSAR result shape: (n_time, n_sar_combos, n_symbols) after param_product
                    sar_idx = sar_accel.index(sar_accel_val) * len(sar_max) + sar_max.index(sar_max_val)
                    if psarl.ndim == 3:
                        psar_close_compare = psarl[:, sar_idx, :] < close
                    else:
                        psar_close_compare = psarl < close

                    # ADX condition: (n_time, n_adx_lengths, n_symbols)
                    if adx_vals.ndim == 3:
                        adx_ok = adx_vals[:, adx_len_idx, :] >= float(adx_thresh)
                    else:
                        adx_ok = adx_vals >= float(adx_thresh)

                    entries_combo = psar_close_compare & adx_ok

                    if use_dmp_cross:
                        dmp = adx_ind.dmp.values if hasattr(adx_ind.dmp, "values") else adx_ind.dmp
                        dmn = adx_ind.dmn.values if hasattr(adx_ind.dmn, "values") else adx_ind.dmn
                        if dmp.ndim == 3:
                            dmp_ok = dmp[:, adx_len_idx, :] > dmn[:, adx_len_idx, :]
                        else:
                            dmp_ok = dmp > dmn
                        entries_combo = entries_combo & dmp_ok

                    entries_list.append(entries_combo)
                    param_combos.append(
                        {
                            "sar_acceleration": sar_accel_val,
                            "sar_maximum": sar_max_val,
                            "adx_length": adx_len,
                            "adx_threshold": adx_thresh,
                            "use_dmp_cross": use_dmp_cross,
                        }
                    )

    # Stack all entries: (n_time, n_combos * n_symbols)
    entries_stacked = []
    for entries_combo in entries_list:
        if entries_combo.ndim == 2:
            entries_stacked.append(entries_combo.reshape(n_time, -1))
        else:
            entries_stacked.append(entries_combo)

    entries_array = np.hstack(entries_stacked) if entries_stacked else np.zeros((n_time, n_symbols), dtype=bool)

    return entries_array.astype(bool), param_combos


def generate_atr_trailing_exits_vectorized(
    entries: np.ndarray,
    precomputer: IndicatorPrecomputer,
    param_grid: dict[str, list],
    n_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ATR trailing stop exits via vectorized Numba processing.

    Args:
        entries: Entry signals (n_time, n_entry_combos * n_symbols).
        precomputer: IndicatorPrecomputer instance.
        param_grid: Parameter grid with 'atr_length', 'atr_multiplier'.
        n_symbols: Number of symbols in the original data.

    Returns:
        (exits_array, stop_array, price_for_orders_array).
    """
    n_time, n_cols = entries.shape
    atr_lengths = param_grid.get("atr_length", [14])
    atr_multipliers = param_grid.get("atr_multiplier", [3.0])

    atr_lengths = atr_lengths if isinstance(atr_lengths, list) else [atr_lengths]
    atr_multipliers = atr_multipliers if isinstance(atr_multipliers, list) else [atr_multipliers]

    close = precomputer.close
    high = precomputer.high
    low = precomputer.low
    open_ = close.copy()  # Fallback if open not available

    if close.ndim == 1:
        close = close[:, np.newaxis]
        high = high[:, np.newaxis]
        low = low[:, np.newaxis]
        open_ = open_[:, np.newaxis]

    # Pre-compute ATR for all lengths
    atr_ind = precomputer.compute_atr(atr_lengths)
    atr_vals = atr_ind.atrr.values if hasattr(atr_ind.atrr, "values") else atr_ind.atrr

    if atr_vals.ndim == 2:
        atr_vals = atr_vals[:, :, np.newaxis]

    exits_list = []
    stops_list = []

    for atr_len_idx, atr_len in enumerate(atr_lengths):
        for atr_mult in atr_multipliers:
            # Broadcast ATR values to match entry columns
            if atr_vals.ndim == 3:
                atr_combo = np.tile(atr_vals[:, atr_len_idx, :], (1, len(atr_lengths)))
            else:
                atr_combo = atr_vals

            high_broad = np.tile(high, (1, n_cols // n_symbols))
            low_broad = np.tile(low, (1, n_cols // n_symbols))

            # Run Numba-accelerated trailing stop
            stops, exits = _atr_trailing_stop_long_ohlc_touch_2d_numba(
                np.asarray(high_broad, dtype=np.float64),
                np.asarray(low_broad, dtype=np.float64),
                np.asarray(atr_combo, dtype=np.float64),
                np.asarray(entries, dtype=np.bool_),
                float(atr_mult),
            )

            exits_list.append(exits)
            stops_list.append(stops)

    # Average or combine exits/stops (take any exit across ATR multipliers)
    if exits_list:
        exits_array = np.any(np.stack(exits_list, axis=2), axis=2).astype(bool)
        stops_array = np.nanmean(np.stack(stops_list, axis=2), axis=2)
    else:
        exits_array = np.zeros_like(entries, dtype=bool)
        stops_array = np.full_like(entries, np.nan, dtype=np.float64)

    price_for_orders = stop_fill_price_vectorized(exits_array, stops_array, close, high, low, open_)

    return exits_array, stops_array, price_for_orders


def stop_fill_price_vectorized(
    exits: np.ndarray, stop_arr: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray
) -> np.ndarray:
    """Compute fill prices with gap adjustment for stop exits (vectorized).

    Args:
        exits: Exit signal array.
        stop_arr: Stop loss array.
        close: Close prices.
        high: High prices.
        low: Low prices.
        open_: Open prices.

    Returns:
        Fill price array adjusted for gaps.
    """
    gap_adjusted_stops = np.minimum(open_, stop_arr)
    price_for_orders = close.copy()

    if close.ndim == 1:
        close = close[:, np.newaxis]
    if close.shape[1] == 1 and exits.shape[1] > 1:
        close = np.tile(close, (1, exits.shape[1]))
    if open_.shape[1] == 1 and exits.shape[1] > 1:
        gap_adjusted_stops = np.tile(gap_adjusted_stops, (1, exits.shape[1]))

    price_for_orders = np.where(exits, gap_adjusted_stops, close)
    return price_for_orders
