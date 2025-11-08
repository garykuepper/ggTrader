import numpy as np
import pandas as pd
from numba import njit
import vectorbt as vbt
import pandas_ta as ta

# ========= 1. Data =========
tickers = ["BTC-USD", "ETH-USD"]
data = vbt.YFData.download(tickers, start="2025-01-01", interval='4h')

close = data.get("Close").astype(np.float64).copy(deep=True)
high  = data.get("High").astype(np.float64).copy(deep=True)
low   = data.get("Low").astype(np.float64).copy(deep=True)

# ========= 2. Signal Data Comparison =========
print(f"\npandas_ta")
sar_pandas_ta = vbt.pandas_ta('psar').run(high, low, close=close, acceleration=0.02, maximum=0.2)
print(sar_pandas_ta.psarl.tail())
sar_buy = sar_pandas_ta.psarl_below(close)

adx_pandas_ta = vbt.pandas_ta('adx').run(high, low, close, length=14)

adx_buy= adx_pandas_ta.dmp_above(adx_pandas_ta.dmn)
# adx_buy.columns = buy2.columns.droplevel(0)
print(f"\ndmp > dmn")
print(adx_buy.tail())

ce = vbt.pandas_ta('chandelier_exit').run(high, low, close, multiplier=3)
print(f"\nChandelier Exit")
ce_signal = ce.chdlrextd > 0

print(ce_signal.tail())

entries = sar_buy & adx_buy

exits = ce_signal

print(f"\nentries")
print(entries.tail())


pf = vbt.Portfolio.from_signals(close,
                                entries=entries,
                                exits=exits,
                                size=1,
                                direction='longonly',
                                sl_stop=0.1,
                                sl_trail=True,
                                freq='4h',
                                init_cash = 1000,
                                )

pf_btc = pf[pf.wrapper.columns[0]]
pf_btc.plot().show()

print(pf_btc.stats())

