import pandas as pd
import vectorbt as vbt
import numpy as np
from numba import njit
from utils.KrakenHistoricalData import KrakenHistoricalData
from vectorbt.portfolio import nb

k = KrakenHistoricalData()

symbols = ["BTC"]
interval = "4h"
end = pd.to_datetime("2025-09-30").tz_localize('UTC')
start = end - pd.Timedelta(days=30 * 6)

df_multi = k.get_ohlcv_df(symbols, interval=interval)
# remove multi-index

# Drop top-level symbol index if present (BTC-only)
if isinstance(df_multi.columns, pd.MultiIndex):
    df_multi = df_multi.xs(symbols[0], axis=1, level=0)

# Slice AND copy to avoid SettingWithCopy
df = df_multi.loc[start:end].copy()

# Ensure OHLC are strict float64
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

close = df["close"]
open_ = df["open"]
high = df["high"]
low = df["low"]

print(df.dtypes)        # debug: should all show float64 for OHLC
print(df.head())        # optional sanity check


atr_len = 14
atr = vbt.IndicatorFactory.from_talib('ATR').run(
    high, low, close, timeperiod=atr_len
).real

psar = vbt.IndicatorFactory.from_talib('SAR').run(
    high,
    low,
    acceleration=0.02,  # tweakable
    maximum=0.2  # tweakable
).real

psar_below = psar < close
psar_above = psar > close

# Raw bullish flip: was above, now below
long_flip = (psar_below & psar_above.shift(1).fillna(False)).astype(bool)


# Shift to avoid lookahead: enter on next bar after the signal
entries = long_flip.vbt.fshift(1)

# only exit on ATR
exits = pd.Series(False, index=close.index)

atr_mult = 3.0  # e.g. 3 * ATR


@njit
def atr_trailing_sl_nb(c, atr_arr, atr_mult):
    # c: AdjustSLContext
    # atr_arr: 1D numpy array, same length as close

    # Only apply when in a long position
    if c.position_now <= 0:
        return c.curr_stop, c.curr_trail

    # current ATR value at this index
    atr_now = atr_arr[c.i]

    if np.isnan(atr_now):
        return c.curr_stop, c.curr_trail

    # ATR-based trailing stop price
    desired_stop_price = c.val_price_now - atr_mult * atr_now

    # Convert desired stop price to sl_stop fraction relative to entry
    new_sl_stop = 1.0 - desired_stop_price / c.init_price

    # Clamp to sane range
    if new_sl_stop < 0.0:
        new_sl_stop = 0.0
    elif new_sl_stop > 1.0:
        new_sl_stop = 1.0

    # If no trailing active yet, initialize
    if not c.curr_trail:
        return new_sl_stop, True

    # Trailing: only tighten (for longs: smaller sl_stop = closer to entry)
    if new_sl_stop < c.curr_stop:
        return new_sl_stop, True

    return c.curr_stop, True



# Make sure `atr` is a NumPy array for numba
atr_arr = atr.to_numpy() if hasattr(atr, "to_numpy") else np.asarray(atr)

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,  # can be False if you only want stops to exit
    direction='longonly',
    init_cash=1000.0,

    # Positioning: 100% in when in trade, compounds automatically
    size=1.0,
    size_type='percent',

    # Provide OHLC so stops are evaluated intrabar
    open=open_,
    high=high,
    low=low,

    # Enable stops: initial wide stop; will be tightened by our function
    sl_stop=1.0,  # effectively "very wide" initial, will be reduced
    sl_trail=True,
    use_stops=True,

    # Hook in our ATR trailing logic
    adjust_sl_func_nb=atr_trailing_sl_nb,
    adjust_sl_args=(vbt.Rep('atr_arr'), atr_mult),
    broadcast_named_args={'atr_arr': atr_arr},

    fees=0.001,
)

stats = pf.stats()

print(stats)

fig = pf.plot(width=1400, height=600)
fig.show()
