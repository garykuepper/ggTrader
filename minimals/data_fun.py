import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator


def get_sample_data(ticker: str):
    df = yf.download(ticker,
                     period='7d',
                     interval='4h',
                     auto_adjust=True,
                     progress=False,
                     multi_level_index=False)
    df.columns = [col.lower() for col in df.columns]
    return df


# end_date = datetime.now(timezone.utc)
# start_date = end_date - timedelta(days=30)

symbol = 'BTC-USD'
df = get_sample_data(symbol)
signals = pd.DataFrame()
signals['close'] = df['close'].copy()
# ... existing code above ...

# Compute EMA signals (as you currently do)
signals['ema_slow'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
signals['ema_fast'] = EMAIndicator(close=df['close'], window=15).ema_indicator()
signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
prev = signals['crossover'].shift(1)
signal_short = signals[(signals['crossover'] != prev) & (signals['crossover'].isin([1, -1]))]

# Align the crossover signal to the main dataframe index
crossover_aligned = signal_short['crossover'].reindex(df.index)

# Separate bullish (1) and bearish (-1) signals
bullish = signals['close'].where(crossover_aligned == 1)
bearish = signals['close'].where(crossover_aligned == -1)

print("\n Data")
print(tabulate(df, headers='keys', tablefmt='github'))
print("\n Signals")
print(tabulate(signals, headers='keys', tablefmt='github'))
print("\n Signal short Table")
print(tabulate(signal_short, headers='keys', tablefmt='github'))


# Build addplots with safe handling for empty signals
apds = [
    mpf.make_addplot(signals['ema_slow'], color='blue', width=1.0, linestyle='-'),
    mpf.make_addplot(signals['ema_fast'], color='orange', width=1.0, linestyle='-'),
]

# Only add bullish markers if there are actual signals
if bullish.notna().any():
    apds.append(
        mpf.make_addplot(bullish, type="scatter", marker="^", color="green", markersize=80,
                         edgecolors="black", linewidths=1.5, panel=0)
    )

# Only add bearish markers if there are actual signals
if bearish.notna().any():
    apds.append(
        mpf.make_addplot(bearish, type="scatter", marker="v", color="red", markersize=80,
                         edgecolors="black", linewidths=1.5, panel=0)
    )

mpf.plot(df,
         type='candle',
         volume=True,
         style='yahoo',
         addplot=apds)