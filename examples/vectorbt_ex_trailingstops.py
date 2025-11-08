import pandas as pd
import pandas_ta as ta
import vectorbt as vbt

# -----------------------------
# 1. Get example OHLCV data via vectorbt
# -----------------------------
symbol = "BTC-USD"

data = vbt.YFData.download(
    symbol,
    start="2022-01-01",
    end="2023-01-01"
).get()

open_ = data["Open"]
high = data["High"]
low = data["Low"]
close = data["Close"]

# -----------------------------
# 2. PSAR for entry direction
# -----------------------------
psar = ta.psar(high=high, low=low, close=close)

print("PSAR columns:", psar.columns.tolist())

psar_long_col = [c for c in psar.columns if "PSARl" in c][0]
psar_short_col = [c for c in psar.columns if "PSARs" in c][0]

psar_long = psar[psar_long_col]
psar_short = psar[psar_short_col]

# Bullish regime when price is above long PSAR (SAR under price)
bullish = (close > psar_long) & psar_long.notna()

# Base PSAR flip entry: flip into long when we newly become bullish
psar_flip_long = bullish & ~bullish.shift(1, fill_value=False)

# -----------------------------
# 3. ADX + DMP/DMN filter
# -----------------------------
adx_len = 14
adx = ta.adx(high=high, low=low, close=close, length=adx_len)

print("ADX columns:", adx.columns.tolist())
# Typical columns: ['DMP_14', 'DMN_14', 'ADX_14']
dmp = adx[[c for c in adx.columns if "DMP_" in c][0]]
dmn = adx[[c for c in adx.columns if "DMN_" in c][0]]
adx_val = adx[[c for c in adx.columns if "ADX_" in c][0]]

# Condition: +DI (DMP) > -DI (DMN)
trend_filter = (dmp > dmn)

# (Optional common practice) Require "decent" trend strength; you can tweak/remove.
# adx_strength = adx_val > 20

# Final entry condition: PSAR flip long + DMP > DMN
entries = psar_flip_long & trend_filter & dmp.notna() & dmn.notna()

# If you want ADX threshold too, use:
# entries = psar_flip_long & trend_filter & adx_strength & dmp.notna() & dmn.notna()

# -----------------------------
# 4. Chandelier Exit for exits
# -----------------------------
length = 22
atr_length = 22
multiplier = 3.0

ce = ta.chandelier_exit(
    high=high,
    low=low,
    close=close,
    high_length=length,
    low_length=length,
    atr_length=atr_length,
    multiplier=multiplier
)

print("Chandelier Exit columns:", ce.columns.tolist())
long_stop = ce.iloc[:, 0]  # assume first column is long stop line

# Exit when close crosses below CE stop
exits = (close < long_stop) & (close.shift(1) >= long_stop.shift(1))
# -----------------------------
# 5. Backtest with vectorbt
# -----------------------------
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    direction="longonly",
    init_cash=10_000,
    fees=0.004,
    slippage=0.001
)

print(pf.stats())
# -----------------------------
# 6. (Optional) Plot price + CE + trades
# -----------------------------
# This will open an interactive plot if environment supports it.
pf.plot(width=1600, height=1500).show()
# Overlay CE line for visual debug
