import vectorbt as vbt
import pandas as pd
import numpy as np
from ggTrader.Signals import Signals
from utils.KrakenHistoricalData import KrakenHistoricalData

k = KrakenHistoricalData()

symbols = ["BTC"]
interval = "4h"
init_cash = 1000.0
transaction_fee = 0.004
start = pd.to_datetime("2025-01-01").tz_localize('UTC')
end = pd.to_datetime("2025-09-30").tz_localize('UTC')

df_multi = k.get_ohlcv_df(symbols, interval=interval)
df_multi.columns = df_multi.columns.droplevel(0)
df = df_multi.loc[start:end]

close = df['close']     # your close Series
# Indicator Params
params = {'sar_acceleration': 0.02,
          'sar_maximum': 0.2,
          'atr_multiplier': 3,
          'adx_threshold': 20,
          'adx_length': 14,
          'ce_high_length': 22,
          'ce_low_length': 22,
          'atr_length': 14,  # atr multiplier for chandelier exit
          }
s = Signals(**params)
signals = s.calc_signals(df.copy())
entries = signals['entry_signal']
exits = signals['exit_signal']

# Build target allocation:
# 1.0 = 100% of portfolio in asset
# 0.0 = 100% in cash
size = pd.Series(np.nan, index=close.index)
size[entries] = 1.0   # on buy signal: go all-in
size[exits] = 0.0     # on sell signal: go flat

pf = vbt.Portfolio.from_orders(
    close=close,
    size=size,
    size_type='targetpercent',  # interpret `size` as target weight
    direction='longonly',
    init_cash=init_cash,           # starting cash
    fees=transaction_fee,                 # 0.1% fee (optional)
    slippage=0.0005,            # optional
)

stats = pf.stats()

print(stats)

pf.plot().show()
