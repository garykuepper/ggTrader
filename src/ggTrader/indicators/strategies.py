"""Pluggable entry and exit strategy framework with concrete implementations."""

from __future__ import annotations

import itertools
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


def _vbt_multi_output_to_tps(
    arr: Any,
    n_time: int,
    n_params: int,
    n_symbols: int,
) -> np.ndarray:
    """Reshape VectorBT / pandas-ta outputs to ``(time, n_params, n_symbols)``.

    With multiple OHLC columns and ``param_product``, VBT often returns a 2D array
    shaped ``(T, n_params * n_symbols)`` (param blocks contiguous). Single-param
    multi-symbol runs may be ``(T, n_symbols)``.
    """
    a = np.asarray(arr, dtype=np.float64)
    if n_params < 1 or n_symbols < 1:
        return a
    if a.ndim == 3:
        if a.shape == (n_time, n_params, n_symbols):
            return a
        if a.shape == (n_time, n_symbols, n_params):
            return np.swapaxes(a, 1, 2)
        return a
    if a.ndim == 1 and a.size == n_time * n_params * n_symbols:
        return a.reshape(n_time, n_params, n_symbols)
    if a.ndim != 2 or a.shape[0] != n_time:
        return a

    ncol = a.shape[1]
    if n_params == 1 and ncol == n_symbols:
        return a.reshape(n_time, 1, n_symbols)
    if n_symbols == 1 and ncol == n_params:
        return a.reshape(n_time, n_params, 1)
    if ncol == n_params * n_symbols:
        return a.reshape(n_time, n_params, n_symbols)
    if ncol == n_symbols * n_params:
        return np.swapaxes(a.reshape(n_time, n_symbols, n_params), 1, 2)
    return a


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
        self,
        entries: np.ndarray,
        precomputer: IndicatorPrecomputer,
        param_grid: dict,
        n_symbols: int,
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

        # Entry-only axes (ATR lives in param_grid for exits; expanded in FastBacktest).
        val_by_key = {
            "sar_acceleration": param_grid.get("sar_acceleration", [0.02]),
            "sar_maximum": param_grid.get("sar_maximum", [0.2]),
            "adx_length": param_grid.get("adx_length", [14]),
            "adx_threshold": param_grid.get("adx_threshold", [25]),
            "use_dmp_cross": param_grid.get("use_dmp_cross", [self.use_dmp_cross]),
        }
        for _k, _v in list(val_by_key.items()):
            val_by_key[_k] = _v if isinstance(_v, list) else [_v]

        _psar_canon = (
            "sar_acceleration",
            "sar_maximum",
            "adx_length",
            "adx_threshold",
            "use_dmp_cross",
        )
        keys_psar = [k for k in param_grid.keys() if k in val_by_key]
        for k in _psar_canon:
            if k in val_by_key and k not in keys_psar:
                keys_psar.append(k)

        sar_accel = val_by_key["sar_acceleration"]
        sar_max = val_by_key["sar_maximum"]
        adx_lengths = val_by_key["adx_length"]
        val_by_key["adx_threshold"]

        psar_ind = precomputer.compute_psar(sar_accel, sar_max)
        adx_ind = precomputer.compute_adx(adx_lengths)

        psarl = psar_ind.psarl.values if hasattr(psar_ind.psarl, "values") else psar_ind.psarl
        adx_vals = adx_ind.adx.values if hasattr(adx_ind.adx, "values") else adx_ind.adx

        n_psar = len(sar_accel) * len(sar_max)
        n_adx = len(adx_lengths)
        psarl = _vbt_multi_output_to_tps(psarl, n_time, n_psar, n_symbols)
        adx_vals = _vbt_multi_output_to_tps(adx_vals, n_time, n_adx, n_symbols)

        dmp = adx_ind.dmp.values if hasattr(adx_ind.dmp, "values") else adx_ind.dmp
        dmn = adx_ind.dmn.values if hasattr(adx_ind.dmn, "values") else adx_ind.dmn
        dmp = _vbt_multi_output_to_tps(dmp, n_time, n_adx, n_symbols)
        dmn = _vbt_multi_output_to_tps(dmn, n_time, n_adx, n_symbols)

        axis_val_lists = [val_by_key[k] for k in keys_psar]

        param_combos = []
        entries_list = []

        for tup in itertools.product(*axis_val_lists):
            combo = dict(zip(keys_psar, tup))
            sar_accel_val = combo["sar_acceleration"]
            sar_max_val = combo["sar_maximum"]
            adx_len = combo["adx_length"]
            adx_thresh = combo["adx_threshold"]
            use_dmp = bool(combo["use_dmp_cross"])

            adx_len_idx = adx_lengths.index(adx_len)
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

            if use_dmp:
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
                    "use_dmp_cross": use_dmp,
                }
            )

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
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
        n_ema = len(all_ema_lengths)
        ema_vals = _vbt_multi_output_to_tps(ema_vals, n_time, n_ema, n_symbols)

        ema_val_by_key = {"ema_fast": ema_fast_vals, "ema_slow": ema_slow_vals}
        ema_keys = [k for k in param_grid.keys() if k in ema_val_by_key]
        for k in ("ema_fast", "ema_slow"):
            if k in ema_val_by_key and k not in ema_keys:
                ema_keys.append(k)

        param_combos = []
        entries_list = []

        for tup in itertools.product(*[ema_val_by_key[k] for k in ema_keys]):
            lens = dict(zip(ema_keys, tup))
            fast_len = lens["ema_fast"]
            slow_len = lens["ema_slow"]
            fast_idx = all_ema_lengths.index(fast_len)
            slow_idx = all_ema_lengths.index(slow_len)

            if fast_len >= slow_len:
                entries_combo = np.zeros((n_time, n_symbols), dtype=bool)
            else:
                if ema_vals.ndim == 3:
                    ema_fast = ema_vals[:, fast_idx, :]
                    ema_slow = ema_vals[:, slow_idx, :]
                else:
                    ema_fast = ema_vals[:, fast_idx] if ema_vals.shape[1] > fast_idx else ema_vals
                    ema_slow = ema_vals[:, slow_idx] if ema_vals.shape[1] > slow_idx else ema_vals

                if ema_fast.ndim == 1:
                    ema_fast = ema_fast[:, np.newaxis]
                if ema_slow.ndim == 1:
                    ema_slow = ema_slow[:, np.newaxis]

                ema_fast_prev = np.roll(ema_fast.copy(), 1, axis=0)
                ema_slow_prev = np.roll(ema_slow.copy(), 1, axis=0)
                ema_fast_prev[0] = ema_fast[0]
                ema_slow_prev[0] = ema_slow[0]
                entries_combo = (ema_fast > ema_slow) & (ema_fast_prev <= ema_slow_prev)
                entries_combo[0] = False
                entries_combo = entries_combo.astype(bool)

            entries_list.append(entries_combo)
            param_combos.append({"ema_fast": fast_len, "ema_slow": slow_len})

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
        )

        return entries_array.astype(bool), param_combos


class RsiReversalEntry:
    """RSI oversold/overbought reversal entry strategy."""

    name = "rsi_reversal"
    param_schema = {
        "rsi_length": [14],
        "rsi_oversold": [30],
        "rsi_trend_filter": [False],
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
        rsi_filters = param_grid.get("rsi_trend_filter", [False])

        rsi_lengths = rsi_lengths if isinstance(rsi_lengths, list) else [rsi_lengths]
        rsi_oversold_vals = (
            rsi_oversold_vals if isinstance(rsi_oversold_vals, list) else [rsi_oversold_vals]
        )
        rsi_filters = rsi_filters if isinstance(rsi_filters, list) else [rsi_filters]

        ema200 = None
        if any(rsi_filters):
            ema_ind = precomputer.compute_ema([200])
            ema_vals = ema_ind.ema.values if hasattr(ema_ind.ema, "values") else ema_ind.ema
            ema200 = _vbt_multi_output_to_tps(ema_vals, n_time, 1, n_symbols)

        rsi_ind = precomputer.compute_rsi(rsi_lengths)
        rsi_vals = rsi_ind.rsi.values if hasattr(rsi_ind.rsi, "values") else rsi_ind.rsi
        n_rsi = len(rsi_lengths)
        rsi_vals = _vbt_multi_output_to_tps(rsi_vals, n_time, n_rsi, n_symbols)

        rsi_val_by_key = {
            "rsi_length": rsi_lengths,
            "rsi_oversold": rsi_oversold_vals,
            "rsi_trend_filter": rsi_filters,
        }
        rsi_keys = [k for k in param_grid.keys() if k in rsi_val_by_key]
        for k in ("rsi_length", "rsi_oversold", "rsi_trend_filter"):
            if k in rsi_val_by_key and k not in rsi_keys:
                rsi_keys.append(k)

        param_combos = []
        entries_list = []

        for tup in itertools.product(*[rsi_val_by_key[k] for k in rsi_keys]):
            lens = dict(zip(rsi_keys, tup))
            rsi_len = lens["rsi_length"]
            rsi_thresh = lens["rsi_oversold"]
            use_filter = bool(lens.get("rsi_trend_filter", False))
            rsi_len_idx = rsi_lengths.index(rsi_len)

            if rsi_vals.ndim == 3:
                rsi_col = rsi_vals[:, rsi_len_idx, :]
            else:
                rsi_col = rsi_vals

            if rsi_col.ndim == 1:
                rsi_col = rsi_col[:, np.newaxis]

            rsi_col_prev = np.roll(rsi_col.copy(), 1, axis=0)
            rsi_col_prev[0] = rsi_col[0]
            entries_combo = (rsi_col > float(rsi_thresh)) & (rsi_col_prev <= float(rsi_thresh))
            entries_combo[0] = False

            if use_filter and ema200 is not None:
                if ema200.ndim == 3:
                    ema_col = ema200[:, 0, :]
                else:
                    ema_col = ema200
                if ema_col.ndim == 1:
                    ema_col = ema_col[:, np.newaxis]
                entries_combo = entries_combo & (close > ema_col)

            entries_list.append(entries_combo.astype(bool))
            combo_dict = {"rsi_length": rsi_len, "rsi_oversold": rsi_thresh}
            if "rsi_trend_filter" in rsi_keys:
                combo_dict["rsi_trend_filter"] = use_filter
            param_combos.append(combo_dict)

        entries_stacked = []
        for entries_combo in entries_list:
            if entries_combo.ndim == 2:
                entries_stacked.append(entries_combo.reshape(n_time, -1))
            else:
                entries_stacked.append(entries_combo)

        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
        )

        return entries_array.astype(bool), param_combos


class MacdCrossEntry:
    """MACD line crosses above signal line (bullish)."""

    name = "macd_cross"
    param_schema = {
        "macd_fast": [12],
        "macd_slow": [26],
        "macd_signal": [9],
    }

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate entries on MACD bullish crossover."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        fasts = param_grid.get("macd_fast", [12])
        slows = param_grid.get("macd_slow", [26])
        signals = param_grid.get("macd_signal", [9])
        fasts = fasts if isinstance(fasts, list) else [fasts]
        slows = slows if isinstance(slows, list) else [slows]
        signals = signals if isinstance(signals, list) else [signals]

        macd_ind = precomputer.compute_macd(fasts, slows, signals)
        macd_line = macd_ind.macd.values if hasattr(macd_ind.macd, "values") else macd_ind.macd
        macd_sig = macd_ind.macds.values if hasattr(macd_ind.macds, "values") else macd_ind.macds

        n_p = len(fasts) * len(slows) * len(signals)
        macd_line = _vbt_multi_output_to_tps(macd_line, n_time, n_p, n_symbols)
        macd_sig = _vbt_multi_output_to_tps(macd_sig, n_time, n_p, n_symbols)

        val_by_key = {
            "macd_fast": fasts,
            "macd_slow": slows,
            "macd_signal": signals,
        }
        # Column order matches VBT ``macd`` param_product: fast × slow × signal.
        macd_axis = ("macd_fast", "macd_slow", "macd_signal")

        combos = list(itertools.product(*[val_by_key[k] for k in macd_axis]))
        idx_map = {tup: i for i, tup in enumerate(combos)}

        param_combos: list[dict] = []
        entries_list: list[np.ndarray] = []

        for tup in combos:
            f_v = int(tup[0])
            s_v = int(tup[1])
            g_v = int(tup[2])
            ci = idx_map[tup]

            if f_v >= s_v:
                entries_combo = np.zeros((n_time, n_symbols), dtype=bool)
            elif macd_line.ndim == 3:
                m = macd_line[:, ci, :]
                ms = macd_sig[:, ci, :]
                m_prev = np.roll(m.copy(), 1, axis=0)
                ms_prev = np.roll(ms.copy(), 1, axis=0)
                m_prev[0] = m[0]
                ms_prev[0] = ms[0]
                entries_combo = (m > ms) & (m_prev <= ms_prev)
                entries_combo[0] = False
                entries_combo = entries_combo.astype(bool)
            else:
                entries_combo = np.zeros((n_time, n_symbols), dtype=bool)

            param_combos.append({"macd_fast": f_v, "macd_slow": s_v, "macd_signal": g_v})
            entries_list.append(entries_combo)

        entries_stacked = [e.reshape(n_time, -1) if e.ndim == 2 else e for e in entries_list]
        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
        )
        return entries_array.astype(bool), param_combos


class BollingerMeanReversionEntry:
    """Long when price crosses back above the lower Bollinger band."""

    name = "bbands_mean_reversion"
    param_schema = {
        "bb_length": [20],
        "bb_std": [2.0],
    }

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Entries on close crossing up through the lower band (oversold bounce)."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        lengths = param_grid.get("bb_length", [20])
        stds = param_grid.get("bb_std", [2.0])
        lengths = lengths if isinstance(lengths, list) else [lengths]
        stds = stds if isinstance(stds, list) else [stds]

        bbl_by_std: dict[float, np.ndarray] = {}
        for st in stds:
            stf = float(st)
            bb_ind = precomputer.compute_bbands(lengths, stf)
            bbl = bb_ind.bbl.values if hasattr(bb_ind.bbl, "values") else bb_ind.bbl
            n_l = len(lengths)
            bbl_by_std[stf] = _vbt_multi_output_to_tps(bbl, n_time, n_l, n_symbols)

        val_by_key = {"bb_length": lengths, "bb_std": stds}
        bb_axis = ("bb_length", "bb_std")

        param_combos = []
        entries_list = []

        for tup in itertools.product(*[val_by_key[k] for k in bb_axis]):
            lens = dict(zip(bb_axis, tup))
            bb_len = int(lens["bb_length"])
            bb_st = float(lens["bb_std"])
            li = lengths.index(bb_len)
            bbl_plane = bbl_by_std[bb_st][:, li, :]
            if bbl_plane.ndim == 1:
                bbl_plane = bbl_plane[:, np.newaxis]

            bbl_prev = np.roll(bbl_plane.copy(), 1, axis=0)
            bbl_prev[0] = bbl_plane[0]
            close_prev = np.roll(close.copy(), 1, axis=0)
            close_prev[0] = close[0]

            entries_combo = (close > bbl_plane) & (close_prev <= bbl_prev)
            entries_combo[0] = False
            entries_list.append(entries_combo.astype(bool))
            param_combos.append({"bb_length": bb_len, "bb_std": bb_st})

        entries_stacked = [e.reshape(n_time, -1) for e in entries_list]
        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
        )
        return entries_array.astype(bool), param_combos


class DonchianBreakoutEntry:
    """Long on close breaking above the prior bar's Donchian upper band."""

    name = "donchian_breakout"
    param_schema = {"donchian_length": [20]}

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Breakout entry without same-bar lookahead (vs prior upper channel)."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        lengths = param_grid.get("donchian_length", [20])
        lengths = lengths if isinstance(lengths, list) else [lengths]

        dc_ind = precomputer.compute_donchian(lengths)
        dcu = dc_ind.dcu.values if hasattr(dc_ind.dcu, "values") else dc_ind.dcu
        dcu = np.asarray(dcu, dtype=np.float64)
        n_dc = len(lengths)
        if dcu.ndim == 2:
            dcu = _vbt_multi_output_to_tps(dcu, n_time, n_dc, n_symbols)

        param_combos = []
        entries_list = []

        for d_len in lengths:
            d_len = int(d_len)
            li = lengths.index(d_len)
            if dcu.ndim == 3:
                u = dcu[:, li, :]
            else:
                u = dcu

            if u.ndim == 1:
                u = u[:, np.newaxis]

            u_prev = np.roll(u.copy(), 1, axis=0)
            u_prev[0, :] = u[0, :]
            close_prev = np.roll(close.copy(), 1, axis=0)
            close_prev[0, :] = close[0, :]

            entries_combo = (close > u_prev) & (close_prev <= u_prev)
            entries_combo[0] = False
            entries_list.append(entries_combo.astype(bool))
            param_combos.append({"donchian_length": d_len})

        entries_stacked = [e.reshape(n_time, -1) for e in entries_list]
        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
        )
        return entries_array.astype(bool), param_combos


class SupertrendFlipEntry:
    """Long on Supertrend bullish flip (direction -1 to 1)."""

    name = "supertrend_flip"
    param_schema = {
        "st_length": [10],
        "st_multiplier": [3.0],
    }

    def compute_entries(
        self, precomputer: IndicatorPrecomputer, param_grid: dict
    ) -> tuple[np.ndarray, list[dict]]:
        """Entry when ``supertd`` transitions from bearish to bullish."""
        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]
        n_time, n_symbols = close.shape

        lengths = param_grid.get("st_length", [10])
        mults = param_grid.get("st_multiplier", [3.0])
        lengths = lengths if isinstance(lengths, list) else [lengths]
        mults = mults if isinstance(mults, list) else [mults]

        st_ind = precomputer.compute_supertrend(lengths, mults)
        td = st_ind.supertd.values if hasattr(st_ind.supertd, "values") else st_ind.supertd
        td = np.asarray(td, dtype=np.float64)
        n_p = len(lengths) * len(mults)

        if td.ndim == 2:
            td = _vbt_multi_output_to_tps(td, n_time, n_p, n_symbols)

        val_by_key = {"st_length": lengths, "st_multiplier": mults}
        st_axis = ("st_length", "st_multiplier")
        combo_list = list(itertools.product(*[val_by_key[k] for k in st_axis]))

        param_combos = []
        entries_list = []

        for ci, tup in enumerate(combo_list):
            L = int(tup[0])
            m = float(tup[1])

            if td.ndim == 3:
                d = td[:, ci, :]
            else:
                d = td

            if d.ndim == 1:
                d = d[:, np.newaxis]

            d_prev = np.roll(d.copy(), 1, axis=0)
            d_prev[0, :] = d[0, :]
            entries_combo = np.isfinite(d) & np.isfinite(d_prev)
            entries_combo &= (d >= 1.0) & (d_prev <= -1.0)
            entries_combo[0] = False
            entries_list.append(entries_combo.astype(bool))
            param_combos.append({"st_length": L, "st_multiplier": m})

        entries_stacked = [e.reshape(n_time, -1) for e in entries_list]
        entries_array = (
            np.hstack(entries_stacked)
            if entries_stacked
            else np.zeros((n_time, n_symbols), dtype=bool)
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
        self,
        entries: np.ndarray,
        precomputer: IndicatorPrecomputer,
        param_grid: dict,
        n_symbols: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ATR trailing stop exits (paired per entry block when ATR grid > 1)."""
        n_time, n_cols = entries.shape
        atr_lengths = param_grid.get("atr_length", [14])
        atr_multipliers = param_grid.get("atr_multiplier", [3.0])

        atr_lengths = atr_lengths if isinstance(atr_lengths, list) else [atr_lengths]
        atr_multipliers = (
            atr_multipliers if isinstance(atr_multipliers, list) else [atr_multipliers]
        )

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

        n_sym = n_symbols
        n_atr_len = len(atr_lengths)
        atr_vals = _vbt_multi_output_to_tps(atr_vals, n_time, n_atr_len, n_sym)
        n_blocks = n_cols // n_sym if n_cols >= n_sym else 1
        n_atr = len(atr_lengths) * len(atr_multipliers)

        # FastBacktest expands entries so each (entry_combo, atr_pair) gets its own columns
        # when n_atr > 1; block i corresponds to e = i // n_atr, k = i % n_atr.
        if n_atr > 1:
            if n_blocks % n_atr != 0:
                raise ValueError(
                    f"ATR trailing: entries width {n_cols} not divisible by n_sym * n_atr "
                    f"({n_sym} * {n_atr})."
                )
            atr_axis_keys = [k for k in param_grid.keys() if k in ("atr_length", "atr_multiplier")]
            atr_lists = [
                (param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]])
                for k in atr_axis_keys
            ]
            pair_tuples = list(itertools.product(*atr_lists))
            if len(pair_tuples) != n_atr:
                raise ValueError("ATR param product size mismatch in vectorized exit.")

            exits_parts: list[np.ndarray] = []
            stops_parts: list[np.ndarray] = []
            price_parts: list[np.ndarray] = []

            for i in range(n_blocks):
                ent_sl = entries[:, i * n_sym : (i + 1) * n_sym]
                j = i % n_atr
                pmap = dict(zip(atr_axis_keys, pair_tuples[j]))
                alen = pmap["atr_length"]
                atr_mult = float(pmap["atr_multiplier"])
                atr_len_idx = atr_lengths.index(alen)
                if atr_vals.ndim == 3:
                    atr_sl = atr_vals[:, atr_len_idx, :]
                else:
                    atr_sl = atr_vals

                stops, ex = _atr_trailing_stop_long_ohlc_touch_2d_numba(
                    np.asarray(high, dtype=np.float64),
                    np.asarray(low, dtype=np.float64),
                    np.asarray(atr_sl, dtype=np.float64),
                    np.asarray(ent_sl, dtype=np.bool_),
                    atr_mult,
                )
                gap_adj = np.minimum(open_, stops)
                price_b = np.where(ex, gap_adj, close).astype(np.float64)
                exits_parts.append(ex)
                stops_parts.append(stops)
                price_parts.append(price_b)

            exits_array = np.hstack(exits_parts).astype(bool)
            stops_array = np.hstack(stops_parts)
            price_for_orders = np.hstack(price_parts)
            return exits_array, stops_array, price_for_orders

        # Single (atr_length, atr_multiplier): one ATR setting applied to all entry combos.
        n_entry_combos = n_blocks
        atr_len_idx = 0
        atr_mult = float(atr_multipliers[0])
        if atr_vals.ndim == 3:
            atr_combo = np.tile(atr_vals[:, atr_len_idx, :], (1, n_entry_combos))
        else:
            atr_combo = atr_vals

        high_broad = np.tile(high, (1, n_entry_combos))
        low_broad = np.tile(low, (1, n_entry_combos))

        stops, exits_array = _atr_trailing_stop_long_ohlc_touch_2d_numba(
            np.asarray(high_broad, dtype=np.float64),
            np.asarray(low_broad, dtype=np.float64),
            np.asarray(atr_combo, dtype=np.float64),
            np.asarray(entries, dtype=np.bool_),
            atr_mult,
        )

        open_broad = np.tile(open_, (1, n_entry_combos))
        gap_adjusted_stops = np.minimum(open_broad, stops)
        close_broad = (
            np.tile(close, (1, n_entry_combos)) if close.shape[1] != n_entry_combos else close
        )
        price_for_orders = np.where(exits_array, gap_adjusted_stops, close_broad)

        return exits_array, stops, price_for_orders


class FixedStopTakeProfit:
    """Fixed percentage stop loss and take profit exit strategy."""

    name = "fixed_sl_tp"
    param_schema = {
        "stop_pct": [2.0],
        "take_profit_pct": [5.0],
    }

    def compute_exits(
        self,
        entries: np.ndarray,
        precomputer: IndicatorPrecomputer,
        param_grid: dict,
        n_symbols: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fixed stop/TP exits with correct per-block (stop, tp) pairing.

        When multiple stop_pct × take_profit_pct values are given, FastBacktest
        expands entry columns so that each (entry_combo, exit_combo) pair occupies
        its own block of n_symbols columns.  Block index i maps to exit pair
        ``i % n_exit`` (same modulo convention as AtrTrailingExit).
        """
        n_time, n_cols = entries.shape
        stop_pcts = param_grid.get("stop_pct", [2.0])
        tp_pcts = param_grid.get("take_profit_pct", [5.0])

        stop_pcts = stop_pcts if isinstance(stop_pcts, list) else [stop_pcts]
        tp_pcts = tp_pcts if isinstance(tp_pcts, list) else [tp_pcts]

        close = precomputer.close
        if close.ndim == 1:
            close = close[:, np.newaxis]

        open_ = close.copy()

        # Broadcast close/open to full column width.
        if close.shape[1] == 1 and n_cols > 1:
            close = np.tile(close, (1, n_cols // n_symbols))
            open_ = np.tile(open_, (1, n_cols // n_symbols))

        exits_array = np.zeros((n_time, n_cols), dtype=bool)
        stops_array = np.full((n_time, n_cols), np.nan, dtype=np.float64)
        price_for_orders = close.copy()

        # Build ordered list of (stop_pct, take_profit_pct) pairs so block index
        # matches the Cartesian product order produced by _expand_entries_for_exit_product.
        pair_tuples = list(itertools.product(stop_pcts, tp_pcts))
        n_exit = len(pair_tuples)
        n_blocks = n_cols // n_symbols

        for i in range(n_blocks):
            j = i % n_exit
            stop_pct_val, tp_pct_val = pair_tuples[j]
            col_start = i * n_symbols
            col_end = col_start + n_symbols

            for col in range(col_start, col_end):
                in_position = False
                entry_price = 0.0

                for t in range(n_time):
                    if entries[t, col] and not in_position:
                        in_position = True
                        entry_price = close[t, col]

                    if in_position:
                        stop_level = entry_price * (1 - stop_pct_val / 100)
                        tp_level = entry_price * (1 + tp_pct_val / 100)
                        stops_array[t, col] = stop_level

                        if close[t, col] <= stop_level or close[t, col] >= tp_level:
                            exits_array[t, col] = True
                            price_for_orders[t, col] = (
                                stop_level if close[t, col] <= stop_level else tp_level
                            )
                            in_position = False

        return exits_array, stops_array, price_for_orders


class TrailingStopExit:
    """Percentage-based trailing stop loss exit strategy."""

    name = "trailing_stop"
    param_schema = {
        "trailing_stop_pct": [3.0, 5.0, 10.0],
    }

    def compute_exits(
        self,
        entries: np.ndarray,
        precomputer: IndicatorPrecomputer,
        param_grid: dict,
        n_symbols: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate trailing stop exits using the Numba-accelerated percentage logic."""
        from ggTrader.indicators.signals import _trailing_stop_long_ohlc_touch_2d_numba

        n_time, n_cols = entries.shape
        stop_pcts = param_grid.get("trailing_stop_pct", [3.0, 5.0, 10.0])
        stop_pcts = stop_pcts if isinstance(stop_pcts, list) else [stop_pcts]

        high = precomputer.high
        low = precomputer.low
        close = precomputer.close
        if high.ndim == 1:
            high = high[:, np.newaxis]
            low = low[:, np.newaxis]
            close = close[:, np.newaxis]

        n_exit = len(stop_pcts)
        n_blocks = n_cols // n_symbols if n_cols >= n_symbols else 1

        if n_exit > 1:
            if n_blocks % n_exit != 0:
                raise ValueError(
                    f"Trailing stop: entries width {n_cols} not divisible by n_sym * n_exit "
                    f"({n_symbols} * {n_exit})."
                )

            exits_parts: list[np.ndarray] = []
            stops_parts: list[np.ndarray] = []
            price_parts: list[np.ndarray] = []

            for i in range(n_blocks):
                ent_sl = entries[:, i * n_symbols : (i + 1) * n_symbols]
                j = i % n_exit
                spct = float(stop_pcts[j])

                stops, ex = _trailing_stop_long_ohlc_touch_2d_numba(
                    np.asarray(high, dtype=np.float64),
                    np.asarray(low, dtype=np.float64),
                    np.asarray(ent_sl, dtype=np.bool_),
                    spct,
                )
                # Fill price logic (gap adjustment)
                open_ = (
                    close.copy()
                )  # Approximating open with close if not available, consistent with other exits
                gap_adj = np.minimum(open_, stops)
                price_b = np.where(ex, gap_adj, close).astype(np.float64)

                exits_parts.append(ex)
                stops_parts.append(stops)
                price_parts.append(price_b)

            exits_array = np.hstack(exits_parts).astype(bool)
            stops_array = np.hstack(stops_parts)
            price_for_orders = np.hstack(price_parts)
            return exits_array, stops_array, price_for_orders

        # Single stop value path
        spct = float(stop_pcts[0])
        stops, exits_array = _trailing_stop_long_ohlc_touch_2d_numba(
            np.tile(high, (1, n_blocks)),
            np.tile(low, (1, n_blocks)),
            entries,
            spct,
        )
        open_broad = np.tile(close, (1, n_blocks))
        gap_adjusted_stops = np.minimum(open_broad, stops)
        price_for_orders = np.where(exits_array, gap_adjusted_stops, open_broad)

        return exits_array, stops, price_for_orders


# Strategy registries
ENTRY_REGISTRY: dict[str, type] = {
    "psar_adx": PsarAdxEntry,
    "ema_cross": EmaCrossEntry,
    "rsi_reversal": RsiReversalEntry,
    "macd_cross": MacdCrossEntry,
    "bbands_mean_reversion": BollingerMeanReversionEntry,
    "donchian_breakout": DonchianBreakoutEntry,
    "supertrend_flip": SupertrendFlipEntry,
}

EXIT_REGISTRY: dict[str, type] = {
    "atr_trailing": AtrTrailingExit,
    "fixed_sl_tp": FixedStopTakeProfit,
    "trailing_stop": TrailingStopExit,
}


# Parameter axes that drive the exit dimension of the vectorized grid.
# These keys appear in the param_grid and determine how many exit-combo columns
# get appended per entry-combo block (Cartesian product of all listed keys).
EXIT_PARAM_AXIS_KEYS: dict[str, tuple[str, ...]] = {
    "atr_trailing": ("atr_length", "atr_multiplier"),
    "fixed_sl_tp": ("stop_pct", "take_profit_pct"),
    "trailing_stop": ("trailing_stop_pct",),
}


def all_exit_axis_param_keys() -> frozenset[str]:
    """Union of every param key used by any registered exit strategy."""
    keys: set[str] = set()
    for tup in EXIT_PARAM_AXIS_KEYS.values():
        keys.update(tup)
    return frozenset(keys)


def foreign_exit_axis_keys(exit_strategy_name: str) -> frozenset[str]:
    """Exit-axis keys that belong to exits other than ``exit_strategy_name``."""
    active = set(EXIT_PARAM_AXIS_KEYS.get(exit_strategy_name, ()))
    return frozenset(k for k in all_exit_axis_param_keys() if k not in active)


def filter_strat_params_for_exit(
    strat_params: dict[str, Any], exit_strategy_name: str
) -> dict[str, Any]:
    """Drop exit-axis keys not used by the active exit (for superset WFO grids).

    When the parameter grid merges axes from every exit in EXIT_TOURNAMENT, the
    vectorized path must not pass e.g. ``stop_pct`` to ``atr_trailing`` or
    ``atr_length`` to ``fixed_sl_tp``.
    """
    drop = foreign_exit_axis_keys(exit_strategy_name)
    return {k: v for k, v in strat_params.items() if k not in drop}


def get_entry_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """Instantiate an entry strategy by name."""
    if strategy_name not in ENTRY_REGISTRY:
        raise ValueError(
            f"Unknown entry strategy: {strategy_name}. Available: {list(ENTRY_REGISTRY.keys())}"
        )
    return ENTRY_REGISTRY[strategy_name](**kwargs)


def get_exit_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """Instantiate an exit strategy by name."""
    if strategy_name not in EXIT_REGISTRY:
        raise ValueError(
            f"Unknown exit strategy: {strategy_name}. Available: {list(EXIT_REGISTRY.keys())}"
        )
    return EXIT_REGISTRY[strategy_name](**kwargs)
