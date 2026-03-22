"""Pluggable entry and exit strategy framework with concrete implementations."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.signals import _atr_trailing_stop_long_ohlc_touch_2d_numba


def _nanmean_axis2_no_empty_warn(stacked: np.ndarray) -> np.ndarray:
    """Column-wise nanmean along axis=2 without numpy 'Mean of empty slice' noise."""
    summed = np.nansum(stacked, axis=2)
    counts = np.sum(np.isfinite(stacked), axis=2)
    out = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, counts, out=out, where=counts > 0)
    return out


class EntryStrategy(Protocol):
    """Protocol for entry signal strategies."""

    name: str
    param_schema: dict[str, list]

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Compute entry signals.

        Args:
            precomputer: IndicatorPrecomputer with OHLCV data.
            param_grid: Parameter ranges for this strategy.

        Returns:
            (entries_array, param_combos) where entries_array is (n_time, n_combos * n_symbols)
            and param_combos is a list of dicts describing each combo.
        """
        ...


class ExitStrategy(Protocol):
    """Protocol for exit signal strategies."""

    name: str
    param_schema: dict[str, list]

    def compute_exits(
        self, entries: np.ndarray, precomputer: IndicatorPrecomputer, param_grid: dict, n_symbols: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute exit signals and fill prices.

        Args:
            entries: Entry signals (n_time, n_entry_combos * n_symbols).
            precomputer: IndicatorPrecomputer with OHLCV data.
            param_grid: Parameter ranges for this strategy.
            n_symbols: Number of symbols in the portfolio.

        Returns:
            (exits_array, stops_array, price_for_orders_array).
        """
        ...


class PsarAdxEntry:
    """Parabolic SAR + ADX entry strategy (current default)."""

    name = "psar_adx"
    param_schema = {
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "adx_length": [14],
        "adx_threshold": [25],
        "use_dmp_cross": [True],
    }

    def __init__(self, use_dmp_cross: bool = True):
        """Initialize strategy.

        Args:
            use_dmp_cross: Include DMP > DMN condition in entry logic.
        """
        self.use_dmp_cross = use_dmp_cross

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate PSAR + ADX entries."""
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

        psar_ind = precomputer.compute_psar(sar_accel, sar_max)
        adx_ind = precomputer.compute_adx(adx_lengths)

        psarl = psar_ind.psarl.values if hasattr(psar_ind.psarl, "values") else psar_ind.psarl
        adx_vals = adx_ind.adx.values if hasattr(adx_ind.adx, "values") else adx_ind.adx

        if adx_vals.ndim == 1:
            adx_vals = adx_vals[:, np.newaxis]

        param_combos = []
        entries_list = []

        for sar_accel_val in sar_accel:
            for sar_max_val in sar_max:
                for adx_len_idx, adx_len in enumerate(adx_lengths):
                    for adx_thresh in adx_thresholds:
                        sar_idx = sar_accel.index(sar_accel_val) * len(sar_max) + sar_max.index(sar_max_val)
                        if psarl.ndim == 3:
                            psar_close_compare = psarl[:, sar_idx, :] < close
                        else:
                            psar_close_compare = psarl < close

                        if adx_vals.ndim == 3:
                            adx_ok = adx_vals[:, adx_len_idx, :] >= float(adx_thresh)
                        else:
                            adx_ok = adx_vals >= float(adx_thresh)

                        entries_combo = psar_close_compare & adx_ok

                        if self.use_dmp_cross:
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
                                "use_dmp_cross": self.use_dmp_cross,
                            }
                        )

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked) if entries_stacked else np.zeros((n_time, n_symbols), dtype=bool)
        )

        return entries_array.astype(bool), param_combos


class EmaCrossEntry:
    """EMA fast/slow crossover entry strategy."""

    name = "ema_cross"
    param_schema = {
        "ema_fast": [9, 12],
        "ema_slow": [21, 26],
    }

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate EMA crossover entries (fast > slow)."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        ema_fast_vals = param_grid.get("ema_fast", [9, 12])
        ema_slow_vals = param_grid.get("ema_slow", [21, 26])

        ema_fast_vals = ema_fast_vals if isinstance(ema_fast_vals, list) else [ema_fast_vals]
        ema_slow_vals = ema_slow_vals if isinstance(ema_slow_vals, list) else [ema_slow_vals]

        all_ema_lengths = sorted(set(ema_fast_vals + ema_slow_vals))
        ema_ind = precomputer.compute_ema(all_ema_lengths)
        ema_vals = ema_ind.ema.values if hasattr(ema_ind.ema, "values") else ema_ind.ema

        if ema_vals.ndim == 2:
            ema_vals = ema_vals[:, :, np.newaxis]

        param_combos = []
        entries_list = []

        for fast_len in ema_fast_vals:
            for slow_len in ema_slow_vals:
                if fast_len >= slow_len:
                    continue
                fast_idx = all_ema_lengths.index(fast_len)
                slow_idx = all_ema_lengths.index(slow_len)

                if ema_vals.ndim == 3:
                    ema_fast = ema_vals[:, fast_idx, :]
                    ema_slow = ema_vals[:, slow_idx, :]
                else:
                    ema_fast = ema_vals[:, fast_idx] if ema_vals.shape[1] > fast_idx else ema_vals
                    ema_slow = ema_vals[:, slow_idx] if ema_vals.shape[1] > slow_idx else ema_vals

                # Ensure 2D shape for crossover logic
                if ema_fast.ndim == 1:
                    ema_fast = ema_fast[:, np.newaxis]
                if ema_slow.ndim == 1:
                    ema_slow = ema_slow[:, np.newaxis]

                # Crossover signal: fast crosses above slow (golden cross)
                ema_fast_prev = np.roll(ema_fast.copy(), 1, axis=0)
                ema_slow_prev = np.roll(ema_slow.copy(), 1, axis=0)
                ema_fast_prev[0] = ema_fast[0]
                ema_slow_prev[0] = ema_slow[0]
                entries_combo = (ema_fast > ema_slow) & (ema_fast_prev <= ema_slow_prev)
                entries_combo[0] = False

                entries_list.append(entries_combo.astype(bool))
                param_combos.append({"ema_fast": fast_len, "ema_slow": slow_len})

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked) if entries_stacked else np.zeros((n_time, n_symbols), dtype=bool)
        )

        return entries_array.astype(bool), param_combos


class RsiReversalEntry:
    """RSI oversold/overbought reversal entry strategy."""

    name = "rsi_reversal"
    param_schema = {
        "rsi_length": [14],
        "rsi_oversold": [30],
    }

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate RSI reversal entries (RSI < oversold threshold)."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        rsi_lengths = param_grid.get("rsi_length", [14])
        rsi_oversold_vals = param_grid.get("rsi_oversold", [30])

        rsi_lengths = rsi_lengths if isinstance(rsi_lengths, list) else [rsi_lengths]
        rsi_oversold_vals = rsi_oversold_vals if isinstance(rsi_oversold_vals, list) else [rsi_oversold_vals]

        rsi_ind = precomputer.compute_rsi(rsi_lengths)
        rsi_vals = rsi_ind.rsi.values if hasattr(rsi_ind.rsi, "values") else rsi_ind.rsi

        if rsi_vals.ndim == 2:
            rsi_vals = rsi_vals[:, :, np.newaxis]

        param_combos = []
        entries_list = []

        for rsi_len_idx, rsi_len in enumerate(rsi_lengths):
            for rsi_thresh in rsi_oversold_vals:
                if rsi_vals.ndim == 3:
                    rsi_col = rsi_vals[:, rsi_len_idx, :]
                else:
                    rsi_col = rsi_vals

                # Ensure 2D shape for crossover logic
                if rsi_col.ndim == 1:
                    rsi_col = rsi_col[:, np.newaxis]

                # Reversal signal: RSI crosses above oversold threshold (was below, now above)
                rsi_col_prev = np.roll(rsi_col.copy(), 1, axis=0)
                rsi_col_prev[0] = rsi_col[0]
                entries_combo = (rsi_col > float(rsi_thresh)) & (rsi_col_prev <= float(rsi_thresh))
                entries_combo[0] = False

                entries_list.append(entries_combo.astype(bool))
                param_combos.append({"rsi_length": rsi_len, "rsi_oversold": rsi_thresh})

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked) if entries_stacked else np.zeros((n_time, n_symbols), dtype=bool)
        )

        return entries_array.astype(bool), param_combos


class AtrTrailingExit:
    """ATR-based trailing stop exit strategy."""

    name = "atr_trailing"
    param_schema = {
        "atr_length": [14],
        "atr_multiplier": [3.0],
    }

    def compute_exits(
        self, entries: np.ndarray, precomputer: IndicatorPrecomputer, param_grid: dict, n_symbols: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ATR trailing stop exits."""
        n_time, n_cols = entries.shape
        atr_lengths = param_grid.get("atr_length", [14])
        atr_multipliers = param_grid.get("atr_multiplier", [3.0])

        atr_lengths = atr_lengths if isinstance(atr_lengths, list) else [atr_lengths]
        atr_multipliers = atr_multipliers if isinstance(atr_multipliers, list) else [atr_multipliers]

        close = precomputer.close
        high = precomputer.high
        low = precomputer.low
        open_ = close.copy()

        if close.ndim == 1:
            close = close[:, np.newaxis]
            high = high[:, np.newaxis]
            low = low[:, np.newaxis]
            open_ = open_[:, np.newaxis]

        atr_ind = precomputer.compute_atr(atr_lengths)
        atr_vals = atr_ind.atrr.values if hasattr(atr_ind.atrr, "values") else atr_ind.atrr

        if atr_vals.ndim == 2:
            atr_vals = atr_vals[:, :, np.newaxis]

        exits_list = []
        stops_list = []

        # Handle case where n_cols might not match expected
        n_entry_combos = n_cols // n_symbols if n_cols >= n_symbols else 1

        for atr_len_idx, _ in enumerate(atr_lengths):
            for atr_mult in atr_multipliers:
                if atr_vals.ndim == 3:
                    atr_combo = np.tile(atr_vals[:, atr_len_idx, :], (1, n_entry_combos))
                else:
                    atr_combo = atr_vals

                high_broad = np.tile(high, (1, n_entry_combos))
                low_broad = np.tile(low, (1, n_entry_combos))

                stops, exits = _atr_trailing_stop_long_ohlc_touch_2d_numba(
                    np.asarray(high_broad, dtype=np.float64),
                    np.asarray(low_broad, dtype=np.float64),
                    np.asarray(atr_combo, dtype=np.float64),
                    np.asarray(entries, dtype=np.bool_),
                    float(atr_mult),
                )

                exits_list.append(exits)
                stops_list.append(stops)

        if exits_list:
            exits_array = np.any(np.stack(exits_list, axis=2), axis=2).astype(bool)
            stops_array = _nanmean_axis2_no_empty_warn(np.stack(stops_list, axis=2))
        else:
            exits_array = np.zeros_like(entries, dtype=bool)
            stops_array = np.full_like(entries, np.nan, dtype=np.float64)

        open_broad = np.tile(open_, (1, n_entry_combos))
        gap_adjusted_stops = np.minimum(open_broad, stops_array)

        # Ensure close is broadcasted to match exits shape
        if close.shape[1] != exits_array.shape[1]:
            close_broad = np.tile(close, (1, n_entry_combos))
        else:
            close_broad = close

        price_for_orders = np.where(exits_array, gap_adjusted_stops, close_broad)

        return exits_array, stops_array, price_for_orders


class FixedStopTakeProfit:
    """Fixed percentage stop loss and take profit exit strategy."""

    name = "fixed_sl_tp"
    param_schema = {
        "stop_pct": [2.0],
        "take_profit_pct": [5.0],
    }

    def compute_exits(
        self, entries: np.ndarray, precomputer: IndicatorPrecomputer, param_grid: dict, n_symbols: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fixed stop/TP exits."""
        n_time, n_cols = entries.shape
        stop_pcts = param_grid.get("stop_pct", [2.0])
        tp_pcts = param_grid.get("take_profit_pct", [5.0])

        stop_pcts = stop_pcts if isinstance(stop_pcts, list) else [stop_pcts]
        tp_pcts = tp_pcts if isinstance(tp_pcts, list) else [tp_pcts]

        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]

        open_ = close.copy()

        exits_array = np.zeros_like(entries, dtype=bool)
        stops_array = np.full_like(entries, np.nan, dtype=np.float64)
        price_for_orders = close.copy()

        if close.shape[1] == 1 and n_cols > 1:
            close = np.tile(close, (1, n_cols // n_symbols))
            open_ = np.tile(open_, (1, n_cols // n_symbols))
            price_for_orders = close.copy()

        for col in range(n_cols):
            in_position = False
            entry_price = 0.0

            for t in range(n_time):
                if entries[t, col] and not in_position:
                    in_position = True
                    entry_price = close[t, col]

                if in_position:
                    stop_level = entry_price * (1 - stop_pcts[0] / 100)
                    tp_level = entry_price * (1 + tp_pcts[0] / 100)
                    stops_array[t, col] = stop_level

                    if close[t, col] <= stop_level or close[t, col] >= tp_level:
                        exits_array[t, col] = True
                        price_for_orders[t, col] = stop_level if close[t, col] <= stop_level else tp_level
                        in_position = False

        return exits_array, stops_array, price_for_orders


# Strategy registries
ENTRY_REGISTRY: dict[str, type] = {
    "psar_adx": PsarAdxEntry,
    "ema_cross": EmaCrossEntry,
    "rsi_reversal": RsiReversalEntry,
}

EXIT_REGISTRY: dict[str, type] = {
    "atr_trailing": AtrTrailingExit,
    "fixed_sl_tp": FixedStopTakeProfit,
}


def get_entry_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """Instantiate an entry strategy by name."""
    if strategy_name not in ENTRY_REGISTRY:
        raise ValueError(f"Unknown entry strategy: {strategy_name}. Available: {list(ENTRY_REGISTRY.keys())}")
    return ENTRY_REGISTRY[strategy_name](**kwargs)


def get_exit_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """Instantiate an exit strategy by name."""
    if strategy_name not in EXIT_REGISTRY:
        raise ValueError(f"Unknown exit strategy: {strategy_name}. Available: {list(EXIT_REGISTRY.keys())}")
    return EXIT_REGISTRY[strategy_name](**kwargs)
