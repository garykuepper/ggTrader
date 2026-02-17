import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit


class Signals:
    """
    Signal generation and modification logic for ggTrader.
    Includes logic for PSAR + ADX entries and ATR trailing stops.
    """

    @staticmethod
    def entry_signals(
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        adx_length: int = 14,
        adx_threshold: int = 25,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        use_dmp_cross: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate entry signals based on PSAR and ADX.

        Long entry: PSAR below close AND (ADX >= threshold) AND (optionally DMP > DMN).
        Ensures flat column names before calculation to avoid MultiIndex level mismatch errors.
        """
        # Help vbt by providing flat column names if MultiIndex
        close_f = (
            close.columns.get_level_values(-1)
            if isinstance(close.columns, pd.MultiIndex)
            else close.columns
        )
        high_f = (
            high.columns.get_level_values(-1)
            if isinstance(high.columns, pd.MultiIndex)
            else high.columns
        )
        low_f = (
            low.columns.get_level_values(-1)
            if isinstance(low.columns, pd.MultiIndex)
            else low.columns
        )

        # We create temporary flat copies for vbt
        c_flat = close.copy()
        c_flat.columns = close_f
        h_flat = high.copy()
        h_flat.columns = high_f
        l_flat = low.copy()
        l_flat.columns = low_f

        # PSAR buy signal
        psar = vbt.pandas_ta("psar").run(
            h_flat,
            l_flat,
            close=c_flat,
            acceleration=sar_acceleration,
            maximum=sar_maximum,
        )
        sar_buy = psar.psarl_below(c_flat)

        # ADX block
        adx = vbt.pandas_ta("adx").run(h_flat, l_flat, c_flat, length=adx_length)
        adx_ok = adx.adx_above(adx_threshold)
        dmp_ok = adx.dmp_above(adx.dmn) if use_dmp_cross else adx_ok

        # Combine using numpy values to avoid MultiIndex naming/alignment issues
        sar_buy_v = sar_buy.values
        adx_ok_v = adx_ok.values
        dmp_ok_v = (
            dmp_ok.values if isinstance(dmp_ok, (pd.Series, pd.DataFrame)) else dmp_ok
        )

        if use_dmp_cross:
            entries_v = sar_buy_v & adx_ok_v & dmp_ok_v
        else:
            entries_v = sar_buy_v & adx_ok_v

        entries = pd.DataFrame(entries_v, index=close.index, columns=close.columns)
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

        entries, exits, stop_df, _ = Signals.calc_signals(
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
            if symbol not in entries.columns:
                continue
            sig_df = pd.DataFrame(index=ohlcv_df.index)
            sig_df["close"] = close[symbol].copy()
            sig_df["signal"] = 0
            sig_df.loc[entries[symbol], "signal"] = 1
            sig_df.loc[exits[symbol], "signal"] = -1
            sig_df["stop_loss"] = stop_df[symbol]
            res_dict[symbol] = sig_df

        return res_dict

    @staticmethod
    def trailing_stop_and_exits(
        entries: pd.DataFrame,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        atr_length: int = 14,
        atr_multiplier: float = 3.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a 3xATR trailing stop and intrabar-touch exits (low <= stop).
        """
        # Help vbt by providing flat column names if MultiIndex
        close_f = (
            close.columns.get_level_values(-1)
            if isinstance(close.columns, pd.MultiIndex)
            else close.columns
        )
        high_f = (
            high.columns.get_level_values(-1)
            if isinstance(high.columns, pd.MultiIndex)
            else high.columns
        )
        low_f = (
            low.columns.get_level_values(-1)
            if isinstance(low.columns, pd.MultiIndex)
            else low.columns
        )

        # We create temporary flat copies for vbt
        c_flat = close.copy()
        c_flat.columns = close_f
        h_flat = high.copy()
        h_flat.columns = high_f
        l_flat = low.copy()
        l_flat.columns = low_f

        # ATR calculation on flat DataFrames
        atr = vbt.pandas_ta("atr").run(h_flat, l_flat, c_flat, length=atr_length)
        atr_vals_df = atr.atrr

        # Ensure entries also has matching flat columns for logical operations
        entries_flat = entries.copy()
        if isinstance(entries_flat.columns, pd.MultiIndex):
            entries_flat.columns = entries_flat.columns.get_level_values(-1)

        # Prepare arrays for numba
        high_vals = np.asarray(h_flat.values, dtype=np.float64)
        low_vals = np.asarray(l_flat.values, dtype=np.float64)
        atr_vals = np.asarray(atr_vals_df.values, dtype=np.float64)
        entry_vals = np.asarray(entries_flat.values, dtype=np.bool_)

        stop_vals, exits_vals = _atr_trailing_stop_long_ohlc_touch_2d_numba(
            high_vals, low_vals, atr_vals, entry_vals, float(atr_multiplier)
        )

        # Return DataFrames with matching labels
        stop_df = pd.DataFrame(stop_vals, index=close.index, columns=c_flat.columns)
        exits_df = pd.DataFrame(
            exits_vals, index=close.index, columns=c_flat.columns, dtype=bool
        )
        return stop_df, exits_df

    @staticmethod
    def stop_fill_price(
        exits: pd.DataFrame,
        stop_df: pd.DataFrame,
        open_df: pd.DataFrame,
        low_df: pd.DataFrame,
        high_df: pd.DataFrame,
        base_price: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate the actual fill price for orders, accounting for gap downs.
        """
        # Start with base_price as the default
        out = base_price.copy()

        # Create boolean mask for exits
        exit_mask = (
            exits.to_numpy(dtype=bool)
            if hasattr(exits, "to_numpy")
            else exits.values.astype(bool)
        )

        # Get numpy arrays for all inputs
        stop_values = (
            stop_df.to_numpy(copy=True)
            if hasattr(stop_df, "to_numpy")
            else stop_df.values.copy()
        )
        open_values = (
            open_df.to_numpy(copy=True)
            if hasattr(open_df, "to_numpy")
            else open_df.values.copy()
        )

        # Logic: Fill = min(Open, Stop)
        gap_adjusted_stops = np.minimum(open_values, stop_values)

        # Create a writable copy of the output values
        out_values = (
            out.to_numpy(copy=True) if hasattr(out, "to_numpy") else out.values.copy()
        )

        # Apply the gap-adjusted prices where exits occur
        out_values[exit_mask] = gap_adjusted_stops[exit_mask]

        # Create new DataFrame with the modified values
        result = pd.DataFrame(out_values, index=out.index, columns=out.columns)

        return result

    @staticmethod
    def calc_signals(
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        open_: pd.DataFrame,
        adx_length: int = 14,
        adx_threshold: int = 25,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        use_dmp_cross: bool = True,
        atr_length: int = 14,
        atr_multiplier: float = 3.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Orchestrates everything and returns entries, exits, stop_df, price_for_orders.
        """
        # Cast params to native Python types to avoid issues with numpy scalars in pandas-ta
        adx_len = int(adx_length)
        adx_th = float(adx_threshold)
        sar_acc = float(sar_acceleration)
        sar_max = float(sar_maximum)
        atr_len = int(atr_length)
        atr_mult = float(atr_multiplier)

        entries = Signals.entry_signals(
            close,
            high,
            low,
            adx_length=adx_len,
            adx_threshold=adx_th,
            sar_acceleration=sar_acc,
            sar_maximum=sar_max,
            use_dmp_cross=use_dmp_cross,
        )

        stop_df, exits = Signals.trailing_stop_and_exits(
            entries=entries,
            close=close,
            high=high,
            low=low,
            atr_length=atr_len,
            atr_multiplier=atr_mult,
        )

        price_for_orders = Signals.stop_fill_price(
            exits=exits,
            stop_df=stop_df,
            open_df=open_,
            low_df=low,
            high_df=high,
            base_price=close,
        )

        return entries, exits, stop_df, price_for_orders


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
    keep_pd=True,  # Ensure inputs/outputs remain pandas objects inside the function
)


@njit
def _atr_trailing_stop_long_ohlc_touch_2d_numba(
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    atr_vals: np.ndarray,
    entry_vals: np.ndarray,
    mult: float,
):
    n, m = high_vals.shape
    stop = np.empty((n, m), dtype=np.float64)
    stop[:] = np.nan
    exits = np.zeros((n, m), dtype=np.bool_)
    in_pos = np.zeros(m, dtype=np.bool_)
    peak = np.zeros(m, dtype=np.float64)
    current_stop = np.zeros(m, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            # Check for new entry only if we are flat
            if entry_vals[i, j] and not in_pos[j]:
                in_pos[j] = True
                peak[j] = high_vals[i, j]
                current_stop[j] = peak[j] - mult * atr_vals[i, j]
                stop[i, j] = current_stop[j]
                exits[i, j] = False
                continue

            if in_pos[j]:
                # Update Peak High while in position
                if high_vals[i, j] > peak[j]:
                    peak[j] = high_vals[i, j]

                # Calculate theoretical new stop
                new_trail = peak[j] - mult * atr_vals[i, j]

                # Enforce Monotonicity: Stop can ONLY go UP
                if new_trail > current_stop[j]:
                    current_stop[j] = new_trail

                stop[i, j] = current_stop[j]

                # Check for Exit (Low touches Stop)
                if low_vals[i, j] <= stop[i, j]:
                    exits[i, j] = True
                    in_pos[j] = False
                    current_stop[j] = 0.0  # reset
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits
