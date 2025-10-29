import talib as ta
import numpy as np
import yfinance as yf
import pandas as pd
import mplfinance as mpf
from tabulate import tabulate

c = np.random.randn(100)

sar_acceleration = 0.02
sar_maximum = 0.2
atr_multiplier = 3.0
df = yf.download("BTC-USD", period="60d", interval="1d", multi_level_index=False)

print(df.head())

df['rsi'] = ta.RSI(df['Close'])
df['adx'] = ta.ADX(df['High'], df['Low'], df['Close'])

df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'])
df['atr_stop_loss'] = df['Close'] - (atr_multiplier * df['atr'].shift(1))

# SAR Signal
df['sar'] = ta.SAR(df['High'],
                   df['Low'],
                   acceleration=sar_acceleration,
                   maximum=sar_maximum)
cross_up = (df['Close'] > df['sar']) & (df['Close'].shift(1) <= df['sar'].shift(1))
cross_down = (df['Close'] < df['sar']) & (df['Close'].shift(1) >= df['sar'].shift(1))

signals = pd.Series(0, index=df.index, dtype=int)
signals[cross_up] = 1
signals[cross_down] = -1
df['signals'] = signals

print(tabulate(df.tail(30), headers="keys", tablefmt="github"))

signal_markers = []
for idx, s in enumerate(df.get('signals', pd.Series([0] * len(df), index=df.index))):
    if s == 1:
        signal_markers.append(
            dict(type='scatter', x=df.index[idx], y=df['Close'].iloc[idx], marker='^', color='green', markersize=100))
    elif s == -1:
        signal_markers.append(
            dict(type='scatter', x=df.index[idx], y=df['Close'].iloc[idx], marker='v', color='red', markersize=100))

addplots = [
    mpf.make_addplot(df['sar'], color='orange', width=1.0, label='SAR'),
    mpf.make_addplot(df['atr_stop_loss'], color='blue', width=1.0, label='ATR'),
]

mpf.plot(
    df,
    type="candle",
    volume=True,
    addplot=addplots,
    figsize=(16, 10),
    title="BTC-USD with SAR and ATR",
    style="yahoo")  # optional lines
