import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit


# # Given wide OHLC DataFrames with same index/columns:
# # open_df, high_df, low_df, close_df
#
# entries, exits, stop_df, price_for_orders = calc_signals(
#     close=close_df, high=high_df, low=low_df, open_=open_df,
#     adx_length=14, adx_threshold=25,
#     sar_acceleration=0.02, sar_maximum=0.2,
#     use_dmp_cross=True,
#     atr_length=14, atr_multiplier=3.0,
# )
#
# pf = vbt.Portfolio.from_signals(
#     close=price_for_orders,
#     entries=entries,
#     exits=exits,
#     init_cash=100_000,
#     fees=0.0005,
#     slippage=0.0002
# )
# print(pf.stats())
#


class Signals:
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
        Long entry: PSAR below close AND (ADX >= threshold) AND (optionally DMP > DMN).
        Ensures flat column names before calculation to avoid MultiIndex level mismatch errors.
        """
        # Help vbt by providing flat column names if MultiIndex
        close_f = close.columns.get_level_values(-1) if isinstance(close.columns, pd.MultiIndex) else close.columns
        high_f = high.columns.get_level_values(-1) if isinstance(high.columns, pd.MultiIndex) else high.columns
        low_f = low.columns.get_level_values(-1) if isinstance(low.columns, pd.MultiIndex) else low.columns

        # We create temporary flat copies for vbt
        c_flat = close.copy()
        c_flat.columns = close_f
        h_flat = high.copy()
        h_flat.columns = high_f
        l_flat = low.copy()
        l_flat.columns = low_f

        # PSAR buy signal
        psar = vbt.pandas_ta('psar').run(
            h_flat, l_flat, close=c_flat,
            acceleration=sar_acceleration,
            maximum=sar_maximum
        )
        sar_buy = psar.psarl_below(c_flat)

        # ADX block
        adx = vbt.pandas_ta('adx').run(h_flat, l_flat, c_flat, length=adx_length)
        adx_ok = adx.adx_above(adx_threshold)
        dmp_ok = adx.dmp_above(adx.dmn) if use_dmp_cross else adx_ok

        # Combine using numpy values to avoid MultiIndex naming/alignment issues
        sar_buy_v = sar_buy.values
        adx_ok_v = adx_ok.values
        dmp_ok_v = dmp_ok.values if isinstance(dmp_ok, (pd.Series, pd.DataFrame)) else dmp_ok

        if use_dmp_cross:
            entries_v = sar_buy_v & adx_ok_v & dmp_ok_v
        else:
            entries_v = sar_buy_v & adx_ok_v

        entries = pd.DataFrame(entries_v, index=close.index, columns=close.columns)
        return entries.astype(bool)


    @staticmethod
    def _atr_trailing_stop_long_ohlc_touch_multi(
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
        """
        symbols = ohlcv_df.columns.levels[0].tolist()
        
        # Use drop_level=True to get DataFrames with symbols as columns
        close = ohlcv_df.xs('close', axis=1, level=1, drop_level=True)
        high = ohlcv_df.xs('high', axis=1, level=1, drop_level=True)
        low = ohlcv_df.xs('low', axis=1, level=1, drop_level=True)
        open_ = ohlcv_df.xs('open', axis=1, level=1, drop_level=True)
        
        entries, exits, stop_df, price_for_orders = Signals.calc_signals(
            close=close, high=high, low=low, open_=open_,
            adx_length=adx_length, adx_threshold=adx_threshold,
            sar_acceleration=sar_acceleration, sar_maximum=sar_maximum,
            use_dmp_cross=use_dmp_cross,
            atr_length=atr_length, atr_multiplier=atr_multiplier
        )
        
        res_dict = {}
        for symbol in symbols:
            if symbol not in entries.columns:
                continue
            sig_df = pd.DataFrame(index=ohlcv_df.index)
            sig_df['close'] = close[symbol]
            sig_df['signal'] = 0
            sig_df.loc[entries[symbol], 'signal'] = 1
            sig_df.loc[exits[symbol], 'signal'] = -1
            sig_df['stop_loss'] = stop_df[symbol]
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
        Build a 3×ATR trailing stop and intrabar-touch exits (low <= stop).
        """
        # Help vbt by providing flat column names if MultiIndex
        close_f = close.columns.get_level_values(-1) if isinstance(close.columns, pd.MultiIndex) else close.columns
        high_f = high.columns.get_level_values(-1) if isinstance(high.columns, pd.MultiIndex) else high.columns
        low_f = low.columns.get_level_values(-1) if isinstance(low.columns, pd.MultiIndex) else low.columns

        # We create temporary flat copies for vbt
        c_flat = close.copy()
        c_flat.columns = close_f
        h_flat = high.copy()
        h_flat.columns = high_f
        l_flat = low.copy()
        l_flat.columns = low_f

        # ATR calculation on flat DataFrames
        atr = vbt.pandas_ta('atr').run(h_flat, l_flat, c_flat, length=atr_length)
        atr_vals_df = atr.atrr

        # Ensure entries also has matching flat columns for logical operations
        entries_flat = entries.copy()
        if isinstance(entries_flat.columns, pd.MultiIndex):
            entries_flat.columns = entries_flat.columns.get_level_values(-1)

        # Optional: avoid entries before ATR is available
        # Use values to avoid MultiIndex alignment issues
        entries_flat_v = entries_flat.values
        atr_notna_v = atr_vals_df.notna().values
        entries_flat_v = entries_flat_v & atr_notna_v
        
        # We need to recreate entries_flat for array conversion below
        entries_flat = pd.DataFrame(entries_flat_v, index=entries_flat.index, columns=entries_flat.columns)

        # Prepare arrays for numba
        high_vals = np.asarray(h_flat.values, dtype=np.float64)
        low_vals = np.asarray(l_flat.values, dtype=np.float64)
        atr_vals = np.asarray(atr_vals_df.values, dtype=np.float64)
        entry_vals = np.asarray(entries_flat.values, dtype=np.bool_)

        stop_vals, exits_vals = _atr_trailing_stop_long_ohlc_touch_2d_numba(
            high_vals, low_vals, atr_vals, entry_vals, float(atr_multiplier)
        )

        # Return DataFrames with matching labels (using original close to preserve index/cols if needed)
        # But we use c_flat labels here which are flat
        stop_df = pd.DataFrame(stop_vals, index=close.index, columns=c_flat.columns)
        exits_df = pd.DataFrame(exits_vals, index=close.index, columns=c_flat.columns, dtype=bool)
        return stop_df, exits_df

    @staticmethod
    def stop_fill_price(exits, stop_df, open_df, low_df, high_df, base_price):
        """
        Calculate the actual fill price for orders.
        """
        # Start with base_price as the default
        out = base_price.copy()

        # Create boolean mask for exits
        exit_mask = exits.to_numpy(dtype=bool) if hasattr(exits, 'to_numpy') else exits.values.astype(bool)

        # Get numpy arrays for all inputs
        stop_values = stop_df.to_numpy(copy=True) if hasattr(stop_df, 'to_numpy') else stop_df.values.copy()
        
        # Gap Handling for Long Exits:
        # If the market opens below our stop (Gap Down), we execute at the Open.
        # If the market opens above our stop but trades through it (Intrabar), we execute at the Stop.
        # Logic: Fill = min(Open, Stop)
        # Note: We assume 'open_df' aligns with 'stop_df'
        open_values = open_df.to_numpy(copy=True) if hasattr(open_df, 'to_numpy') else open_df.values.copy()
        
        # Calculate the realistic fill price taking gaps into account
        # We use np.minimum because for a Long exit (Sell), a lower price is the "worse" case (gap down).
        # This replaces the naive np.clip(stop, low, high) which could optimistically fill at High during a gap down.
        gap_adjusted_stops = np.minimum(open_values, stop_values)

        # Create a writable copy of the output values
        out_values = out.to_numpy(copy=True) if hasattr(out, 'to_numpy') else out.values.copy()

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
        entries = Signals.entry_signals(
            close, high, low,
            adx_length=adx_length,
            adx_threshold=adx_threshold,
            sar_acceleration=sar_acceleration,
            sar_maximum=sar_maximum,
            use_dmp_cross=use_dmp_cross,
        )

        stop_df, exits = Signals.trailing_stop_and_exits(
            entries=entries,
            close=close,
            high=high,
            low=low,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
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


@njit
def _atr_trailing_stop_long_ohlc_touch_2d_numba(
        high_vals: np.ndarray,
        low_vals: np.ndarray,
        atr_vals: np.ndarray,
        entry_vals: np.ndarray,
        mult: float
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
                    current_stop[j] = 0.0 # reset
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits
