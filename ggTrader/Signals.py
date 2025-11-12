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
    Works per-asset on wide DataFrames sharing index/columns.
    """
    # PSAR buy signal
    psar = vbt.pandas_ta('psar').run(
        high, low, close=close,
        acceleration=sar_acceleration,
        maximum=sar_maximum
    )
    sar_buy = psar.psarl_below(close)

    # ADX block (we'll also reuse its ATR later in the stop function, but here we only need ADX/DMP/DMN)
    adx = vbt.pandas_ta('adx').run(high, low, close, length=adx_length)
    adx_ok = adx.adx_above(adx_threshold)
    dmp_ok = adx.dmp_above(adx.dmn) if use_dmp_cross else adx_ok  # if disabled, dmp_ok gets ignored below

    entries = sar_buy & adx_ok & (dmp_ok if use_dmp_cross else adx_ok)
    return entries.astype(bool)


@njit
def _atr_trailing_stop_long_ohlc_touch_2d(
        high_vals: np.ndarray,
        low_vals: np.ndarray,
        atr_vals: np.ndarray,
        entry_vals: np.ndarray,
        mult: float
):
    """
    Numba kernel: multi-asset long-only trailing stop with intrabar touch.
    stop_t = peak_since_entry - mult * ATR_t
    exit when low_t <= stop_t
    """
    n, m = high_vals.shape
    stop = np.empty((n, m), dtype=np.float64);
    stop[:] = np.nan
    exits = np.zeros((n, m), dtype=np.bool_)
    in_pos = np.zeros(m, dtype=np.bool_)
    peak = np.zeros(m, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if entry_vals[i, j]:
                in_pos[j] = True
                peak[j] = high_vals[i, j]
                stop[i, j] = peak[j] - mult * atr_vals[i, j]
                exits[i, j] = False
                continue

            if in_pos[j]:
                if high_vals[i, j] > peak[j]:
                    peak[j] = high_vals[i, j]
                stop[i, j] = peak[j] - mult * atr_vals[i, j]

                if low_vals[i, j] <= stop[i, j]:
                    exits[i, j] = True
                    in_pos[j] = False
                else:
                    exits[i, j] = False
            else:
                stop[i, j] = np.nan
                exits[i, j] = False

    return stop, exits


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
    ATR is obtained from pandas_ta('adx').run(...).atr to honor your constraint.
    Returns:
        stop_df  : trailing stop level (price) per bar/asset (NaN when flat)
        exits_df : boolean exit signals per bar/asset
    """
    # ATR from ADX output
    atr = vbt.pandas_ta('atr').run(high, low, close, length=atr_length)
    atr = atr.atrr

    # Optional: avoid entries before ATR is available
    entries = entries & atr.notna()

    # Prepare arrays for numba
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    atr_vals = np.asarray(atr.values, dtype=np.float64)
    entry_vals = np.asarray(entries.values, dtype=np.bool_)

    stop_vals, exits_vals = _atr_trailing_stop_long_ohlc_touch_2d(
        high_vals, low_vals, atr_vals, entry_vals, float(atr_multiplier)
    )

    stop_df = pd.DataFrame(stop_vals, index=close.index, columns=close.columns)
    exits_df = pd.DataFrame(exits_vals, index=close.index, columns=close.columns, dtype=bool)
    return stop_df, exits_df


def stop_fill_price(exits, stop_df, open_df, low_df, high_df, base_price):
    """
    Calculate the actual fill price for orders, considering stop levels and price bounds.

    For exit bars:
    - If stop was hit, use the stop price (clipped to the bar's low/high range)
    - Otherwise use the base price
    """
    # Start with base_price as the default
    out = base_price.copy()

    # Create boolean mask for exits
    exit_mask = exits.to_numpy(dtype=bool) if hasattr(exits, 'to_numpy') else exits.values.astype(bool)

    # Get numpy arrays for all inputs
    stop_values = stop_df.to_numpy(copy=True) if hasattr(stop_df, 'to_numpy') else stop_df.values.copy()
    low_values = low_df.to_numpy(copy=True) if hasattr(low_df, 'to_numpy') else low_df.values.copy()
    high_values = high_df.to_numpy(copy=True) if hasattr(high_df, 'to_numpy') else high_df.values.copy()

    # Clip stop prices to the bar's low/high range
    clipped_stops = np.clip(stop_values, low_values, high_values)

    # Create a writable copy of the output values
    out_values = out.to_numpy(copy=True) if hasattr(out, 'to_numpy') else out.values.copy()

    # Apply the clipped stop prices where exits occur
    out_values[exit_mask] = clipped_stops[exit_mask]

    # Create new DataFrame with the modified values
    result = pd.DataFrame(out_values, index=out.index, columns=out.columns)

    return result


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
    Orchestrates everything and returns:
        entries, exits, stop_df, price_for_orders

    Pass these into:
        vbt.Portfolio.from_signals(
            close=price_for_orders, entries=entries, exits=exits, ...
        )
    """
    entries = entry_signals(
        close, high, low,
        adx_length=adx_length,
        adx_threshold=adx_threshold,
        sar_acceleration=sar_acceleration,
        sar_maximum=sar_maximum,
        use_dmp_cross=use_dmp_cross,
    )

    stop_df, exits = trailing_stop_and_exits(
        entries=entries,
        close=close,
        high=high,
        low=low,
        atr_length=atr_length,
        atr_multiplier=atr_multiplier,
    )

    price_for_orders = stop_fill_price(
        exits=exits,
        stop_df=stop_df,
        open_df=open_,
        low_df=low,
        high_df=high,
        base_price=close,  # use Close elsewhere; override only on exit bars
    )

    return entries, exits, stop_df, price_for_orders
