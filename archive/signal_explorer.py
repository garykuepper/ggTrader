import numpy as np
import pandas as pd
from utils.KrakenHistoricalData import KrakenHistoricalData
from ggTrader.Backtest import Backtest
from tabulate import tabulate
from ggTrader.Signals import Signals

k = KrakenHistoricalData()


symbols = ["BTC"]
interval = "4h"
end = pd.to_datetime("2025-09-30").tz_localize('UTC')
start = end - pd.Timedelta(days=30 * 6)
# start = pd.to_datetime("2023-01-01").tz_localize('UTC')
# Indicator Params
sar_acceleration = 0.02
sar_maximum = 0.2
atr_multiplier = 3  #atr multiplier for chandelier exit
adx_threshold = 25  #trend strength
ce_high_length = 22  # look back length

s = Signals(adx_threshold=adx_threshold,
            atr_multiplier=atr_multiplier,
            ce_high_length=ce_high_length)
# Fees
maker_fee = 0.0025
taker_fee = 0.004
transaction_fee = (maker_fee + taker_fee) / 2

# k.use_remote("https://garygigabytes.com/kraken/parquet")  # no trailing slash
# df_multi = k.get_ohlcv_df_remote(symbols, interval=interval)
df_multi = k.get_ohlcv_df(symbols, interval=interval)

df_multi.columns = df_multi.columns.droplevel(0)
df = df_multi.loc[start:end]

signals = s.calc_signals(df.copy())

print(f"\nSignals:")
print(signals.info())
print(signals.head())
print(signals.tail())

bt = Backtest(signals, interval=interval, transaction_fee=transaction_fee)
stats, profit_df = bt.run()


print(f"\nStats:")
print(tabulate(stats.items(),tablefmt="github"))