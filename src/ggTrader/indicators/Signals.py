import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit, prange


class Signals:
    """
    Signal generation and modification logic for ggTrader.
    Includes logic for PSAR + ADX entries and ATR trailing stops.
    """

    @staticmethod
    def entry_signals(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        adx_length: int = 14,
        adx_threshold: int = 25,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        use_dmp_cross: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculate entry signals based on PSAR and ADX.
        Long entry: PSAR below close AND (ADX >= threshold) AND (optionally DMP > DMN).
        """
        # Ensure input is numpy for speed
        c = close.values if hasattr(close, "values") else close
        h = high.values if hasattr(high, "values") else high
        l = low.values if hasattr(low, "values") else low

        # Vectorized PSAR - Using VBT bridge but ensuring numpy output
        psar_ind = vbt.IndicatorFactory.from_pandas_ta("psar").run(
            h,
            l,
            close=c,
            acceleration=sar_acceleration,
            maximum=sar_maximum,
        )
        psarl = psar_ind.psarl.values if hasattr(psar_ind.psarl, "values") else psar_ind.psarl
        sar_buy = psarl < c

        # Safety check: ADX needs enough data
        # pandas_ta.adx often returns None if data is too short
        if len(c) < int(adx_length) + 1:
            return np.zeros(c.shape, dtype=bool)

        # Vectorized ADX
        try:
            adx_ind = vbt.IndicatorFactory.from_pandas_ta("adx").run(
                h, l, c, length=int(adx_length)
            )
            adx_val = adx_ind.adx.values if hasattr(adx_ind.adx, "values") else adx_ind.adx
            dmp = adx_ind.dmp.values if hasattr(adx_ind.dmp, "values") else adx_ind.dmp
            dmn = adx_ind.dmn.values if hasattr(adx_ind.dmn, "values") else adx_ind.dmn
        except Exception:
            # Fallback for insufficient data or other pandas_ta errors
            adx_val = np.full(c.shape, np.nan)
            dmp = np.full(c.shape, np.nan)
            dmn = np.full(c.shape, np.nan)

        adx_ok = adx_val >= float(adx_threshold)

        if use_dmp_cross:
            dmp_ok = dmp > dmn
            entries = sar_buy & adx_ok & dmp_ok
        else:
            entries = sar_buy & adx_ok

        return entries.astype(bool)

    @staticmethod
    def calculate_ohlcv_signals(
        ohlcv_df: pd.DataFrame,
        adx_length: int = 14,
        adx_threshold: int = 25,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        use_dmp_cross: bool = True,
        atr_length: int = 14,
        atr_multiplier: float = 3.0,
    ) -> dict[str, pd.DataFrame]:
        """
        Calculates trailing stop signals for a wide OHLCV DataFrame (MultiIndex).
        Returns a dictionary mapping symbol to a signals DataFrame.
        Used by the Trading engine.
        """
        symbols = ohlcv_df.columns.levels[0].tolist()

        # Use drop_level=True to get DataFrames with symbols as columns
        close = ohlcv_df.xs("close", axis=1, level=1, drop_level=True)
        high = ohlcv_df.xs("high", axis=1, level=1, drop_level=True)
        low = ohlcv_df.xs("low", axis=1, level=1, drop_level=True)
        open_ = ohlcv_df.xs("open", axis=1, level=1, drop_level=True)

        entries, exits, stop_arr, price_arr = Signals.calc_signals(
            close=close,
            high=high,
            low=low,
            open_=open_,
            adx_length=adx_length,
            adx_threshold=adx_threshold,
            sar_acceleration=sar_acceleration,
            sar_maximum=sar_maximum,
            use_dmp_cross=use_dmp_cross,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
        )

        res_dict = {}
        for symbol in symbols:
            if symbol not in close.columns:
                continue
            idx = close.columns.get_loc(symbol)
            sig_df = pd.DataFrame(index=ohlcv_df.index)
            sig_df["close"] = close[symbol].copy()
            sig_df["signal"] = 0

            # Extract column from numpy results
            s_entries = entries[:, idx]
            s_exits = exits[:, idx]
            s_stops = stop_arr[:, idx]

            sig_df.loc[s_entries, "signal"] = 1
            sig_df.loc[s_exits, "signal"] = -1
            sig_df["stop_loss"] = s_stops
            res_dict[symbol] = sig_df

        return res_dict

    @staticmethod
    def trailing_stop_and_exits(
        entries: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr_length: int = 14,
        atr_multiplier: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a 3xATR trailing stop and intrabar-touch exits (low <= stop).
        """
        # Ensure inputs are numpy
        c = close.values if hasattr(close, "values") else close
        h = high.values if hasattr(high, "values") else high
        l = low.values if hasattr(low, "values") else low
        e = entries.values if hasattr(entries, "values") else entries

        # Safety check: ATR needs enough data
        if len(c) < int(atr_length) + 1:
            n_rows = c.shape[0]
            n_cols = c.shape[1] if len(c.shape) > 1 else 1
            stop_vals = np.full(c.shape, np.nan)
            exits_vals = np.zeros(c.shape, dtype=bool)
            return stop_vals, exits_vals

        # ATR calculation via VBT bridge (fast)
        try:
            atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
                h, l, c, length=int(atr_length)
            )
            atr_vals = atr_ind.atrr.values if hasattr(atr_ind.atrr, "values") else atr_ind.atrr
        except Exception:
            # Fallback for insufficient data
            atr_vals = np.full(c.shape, np.nan)

        # Prepare arrays for numba
        high_vals = np.asarray(h, dtype=np.float64)
        low_vals = np.asarray(l, dtype=np.float64)
        atr_vals_np = np.asarray(atr_vals, dtype=np.float64)
        entry_vals = np.asarray(e, dtype=np.bool_)

        stop_vals, exits_vals = _atr_trailing_stop_long_ohlc_touch_2d_numba(
            high_vals, low_vals, atr_vals_np, entry_vals, float(atr_multiplier)
        )

        return stop_vals, exits_vals

    @staticmethod
    def stop_fill_price(
        exits: np.ndarray,
        stop_arr: np.ndarray,
        open_arr: np.ndarray,
        low_arr: np.ndarray,
        high_arr: np.ndarray,
        base_price: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the actual fill price for orders, accounting for gap downs.
        Inputs are expected to be numpy arrays or have a .values attribute.
        """

        def _to_np(x):
            return x.values if hasattr(x, "values") else x

        e = _to_np(exits).astype(bool)
        s = _to_np(stop_arr)
        o = _to_np(open_arr)
        bp = _to_np(base_price)

        # Logic: Fill = min(Open, Stop) where exit occurs
        gap_adjusted_stops = np.minimum(o, s)

        out = bp.copy()
        out[e] = gap_adjusted_stops[e]
        return out

    @staticmethod
    def calc_signals(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        adx_length: int = 14,
        adx_threshold: int = 25,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        use_dmp_cross: bool = True,
        atr_length: int = 14,
        atr_multiplier: float = 3.0,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Orchestrates everything and returns entries, exits, stop_arr, price_for_orders.
        Expects numpy arrays or objects with .values.
        """

        def _to_np(x):
            return x.values if hasattr(x, "values") else x

        c = _to_np(close)
        h = _to_np(high)
        l = _to_np(low)
        o = _to_np(open_)

        # Cast params
        adx_len = int(adx_length)
        adx_th = float(adx_threshold)
        sar_acc = float(sar_acceleration)
        sar_max = float(sar_maximum)
        atr_len = int(atr_length)
        atr_mult = float(atr_multiplier)

        entries = Signals.entry_signals(
            c,
            h,
            l,
            adx_length=adx_len,
            adx_threshold=adx_th,
            sar_acceleration=sar_acc,
            sar_maximum=sar_max,
            use_dmp_cross=use_dmp_cross,
            **kwargs,
        )

        stop_arr, exits = Signals.trailing_stop_and_exits(
            entries=entries,
            close=c,
            high=h,
            low=l,
            atr_length=atr_len,
            atr_multiplier=atr_mult,
        )

        price_for_orders = Signals.stop_fill_price(
            exits=exits,
            stop_arr=stop_arr,
            open_arr=o,
            low_arr=l,
            high_arr=h,
            base_price=c,
        )

        return entries, exits, stop_arr, price_for_orders


# Create a VectorBT IndicatorFactory to enable easy broadcasting/parameterizing
SignalFactory = vbt.IndicatorFactory(
    class_name="SignalFactory",
    short_name="sf",
    input_names=["close", "high", "low", "open_"],
    param_names=[
        "adx_length",
        "adx_threshold",
        "sar_acceleration",
        "sar_maximum",
        "use_dmp_cross",
        "atr_length",
        "atr_multiplier",
    ],
    output_names=["entries", "exits", "stop_df", "price_for_orders"],
).from_apply_func(
    Signals.calc_signals,
    adx_length=14,
    adx_threshold=25,
    sar_acceleration=0.02,
    sar_maximum=0.2,
    use_dmp_cross=True,
    atr_length=14,
    atr_multiplier=3.0,
    keep_pd=False,  # Optimization: Use numpy arrays internally for massive speedup
)


@njit(parallel=True)
def _atr_trailing_stop_long_ohlc_touch_2d_numba(
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    atr_vals: np.ndarray,
    entry_vals: np.ndarray,
    mult: float,
):
    """
    Computes a trailing stop and exact exit signals for a 2D array of OHLC/ATR data.
    The outer loop iterates over columns (assets/parameters) in parallel via prange.
    The inner loop iterates sequentially over time to maintain causality.
    """
    n, m = high_vals.shape
    stop = np.empty((n, m), dtype=np.float64)
    stop[:] = np.nan
    exits = np.zeros((n, m), dtype=np.bool_)

    # Use prange for the outer loop over columns (parameters/assets)
    # This allows Numba to execute multi-core processing at the group level
    for j in prange(m):
        in_pos = False
        peak = 0.0
        current_stop = 0.0

        # Inner loop is strictly sequential over time for this specific column
        for i in range(n):
            # 1. Check for new entry only if we are flat
            if entry_vals[i, j] and not in_pos:
                in_pos = True
                peak = high_vals[i, j]
                current_stop = peak - mult * atr_vals[i, j]
                stop[i, j] = current_stop
                exits[i, j] = False
                continue

            if in_pos:
                # 2. Update Peak High while in position
                if high_vals[i, j] > peak:
                    peak = high_vals[i, j]

                # 3. Calculate theoretical new stop
                new_trail = peak - mult * atr_vals[i, j]

                # 4. Enforce Monotonicity: Stop can ONLY go UP
                if new_trail > current_stop:
                    current_stop = new_trail

                stop[i, j] = current_stop

                # 5. Check for Exit (Low touches Stop)
                if low_vals[i, j] <= stop[i, j]:
                    exits[i, j] = True
                    in_pos = False
                    current_stop = 0.0  # reset for next trade
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits
