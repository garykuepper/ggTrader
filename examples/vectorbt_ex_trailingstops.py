import numpy as np
import pandas as pd
from numba import njit
import vectorbt as vbt
import pandas_ta as ta

# ========= 1. Data =========
tickers = ["BTC-USD", "ETH-USD"]
data = vbt.YFData.download(tickers, start="2024-01-01")
close = data.get("Close").astype(np.float64).copy(deep=True)
high  = data.get("High").astype(np.float64).copy(deep=True)
low   = data.get("Low").astype(np.float64).copy(deep=True)


atr = vbt.ta('AverageTrueRange').run(high, low, close, window=14)
psar = vbt.ta('PSARIndicator').run(high, low,close, step=0.02, max_step=0.2)

print(atr.average_true_range)
print(psar.psar_up_indicator)

atr_talib = vbt.talib('ATR').run(high, low, close, timeperiod=14)
print(atr_talib.real)
sar_talib = vbt.talib('SAR').run(high, low, acceleration=0.02, maximum=0.2)
print(sar_talib.real)

sar_pandas_ta = vbt.pandas_ta('psar').run(high, low, close=close, acceleration=0.02, maximum=0.2)
print(sar_pandas_ta.psarl)
buy = sar_pandas_ta.psarl_below(close)
print(buy[buy['BTC-USD']])

adx_pandas_ta = vbt.pandas_ta('adx').run(high, low, close, length=14)

buy2 = adx_pandas_ta.dmp_above(adx_pandas_ta.dmn)
print(buy2.info())
buy2.columns = buy2.columns.droplevel(0)
print(buy2)
print(buy2[buy2['BTC-USD']])

ce = vbt.pandas_ta('chandelier_exit').run(high, low, close, multiplier=3)

print(dir(ce))
print(ce.chdlrextl)
print(ce.chdlrextd)