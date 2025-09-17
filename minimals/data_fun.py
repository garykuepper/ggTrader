import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator
import plotly.graph_objects as go

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
signals['ema_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
signals['ema_slow'] = EMAIndicator(close=df['close'], window=15).ema_indicator()
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
#
# # Use aligned signals for Plotly buy/sell markers to match the same timestamps as `df`
# fig = go.Figure(data=[go.Candlestick(x=df.index,
#                 open=df['open'],
#                 high=df['high'],
#                 low=df['low'],
#                 close=df['close'])])
# fig.add_trace(go.Scatter(x=df.index, y=signals['ema_slow'],
#                          mode='lines', name='EMA Slow', line=dict(color='blue')))
# fig.add_trace(go.Scatter(x=df.index, y=signals['ema_fast'],
#                          mode='lines', name='EMA Fast', line=dict(color='orange')))
#
# # Use the aligned crossover series for points
# buy_points_aligned = df.index[crossover_aligned == 1]
# sell_points_aligned = df.index[crossover_aligned == -1]
#
# fig.add_trace(go.Scatter(
#     x=buy_points_aligned, y=df.loc[buy_points_aligned]['close'],
#     mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
#     name='Buy'
# ))
# fig.add_trace(go.Scatter(
#     x=sell_points_aligned, y=df.loc[sell_points_aligned]['close'],
#     mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
#     name='Sell'
# ))
#
# fig.show()