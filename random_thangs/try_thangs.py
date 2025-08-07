import pandas as pd
from mplfinance import figure
from ta.trend import EMAIndicator, SMAIndicator

import matplotlib.pyplot as plt
from tabulate import tabulate
import mplfinance as mpf

# Load CSV data into a DataFrame
df = pd.read_csv("../dl_data/yf_UPRO_5y.csv")

# Convert the 'date' column to datetime objects for proper time series handling
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the DataFrame index for easier plotting and time-based operations
df = df.set_index('date')

## EMA
# Calculate the fast EMA (shorter window, reacts quicker to price changes)
df['ema_fast'] = EMAIndicator(df['close'], window=17).ema_indicator()
df['ema_slow'] = EMAIndicator(df['close'], window=35).ema_indicator()

# Create a boolean Series that is True when fast EMA is above slow EMA, False otherwise
ema_fast_above = df['ema_fast'] > df['ema_slow']

# Detect bullish crossovers:
# Condition: fast EMA is currently above slow EMA (True)
# AND fast EMA was NOT above slow EMA in the previous time step (False)
# This means the fast EMA just crossed above the slow EMA (bullish signal)
cross_up = (ema_fast_above) & (~ema_fast_above.shift(1).fillna(False).astype(bool))

# Detect bearish crossovers:
# Condition: fast EMA is currently below slow EMA (False)
# AND fast EMA was above slow EMA in the previous time step (True)
# This means the fast EMA just crossed below the slow EMA (bearish signal)
cross_down = (~ema_fast_above) & (ema_fast_above.shift(1).fillna(False).astype(bool))

## SMA
# Calculate the fast EMA (shorter window, reacts quicker to price changes)
df['sma_fast'] = SMAIndicator(df['close'], window=50).sma_indicator()
df['sma_slow'] = SMAIndicator(df['close'], window=200).sma_indicator()

# Create a boolean Series that is True when fast EMA is above slow EMA, False otherwise
sma_fast_above = df['sma_fast'] > df['sma_slow']

# Detect bullish crossovers:
# Condition: fast EMA is currently above slow EMA (True)
# AND fast EMA was NOT above slow EMA in the previous time step (False)
# This means the fast EMA just crossed above the slow EMA (bullish signal)
cross_up_sma = (sma_fast_above) & (~sma_fast_above.shift(1).fillna(False).astype(bool))

# Detect bearish crossovers:
# Condition: fast EMA is currently below slow EMA (False)
# AND fast EMA was above slow EMA in the previous time step (True)
# This means the fast EMA just crossed below the slow EMA (bearish signal)
cross_down_sma = (~sma_fast_above) & (sma_fast_above.shift(1).fillna(False).astype(bool))

# Select columns to plot: closing price and both EMAs
columns = ['close', 'ema_fast', 'ema_slow']
woot = df[columns]

buy = woot[cross_up].copy()
buy['signal'] = 'BUY'

sell = woot[cross_down].copy()
sell['signal'] = 'SELL'

signals = pd.concat([buy, sell]).sort_index()

print(tabulate(signals, headers='keys', tablefmt='github'))
print("\n")
buy_signals = df.loc[cross_up, ['close']].reset_index().rename(columns={'date': 'buy_date', 'close': 'buy_price'})
sell_signals = df.loc[cross_down, ['close']].reset_index().rename(columns={'date': 'sell_date', 'close': 'sell_price'})

# Merge buy with next sell after buy_date
trades = pd.merge_asof(buy_signals.sort_values('buy_date'),
                       sell_signals.sort_values('sell_date'),
                       left_on='buy_date', right_on='sell_date',
                       direction='forward')

# Calculate profit
starting_cash = 10000
profit = []
cash = starting_cash

for i, row in trades.iterrows():
    shares_bought = cash / row['buy_price']
    sell_value = shares_bought * row['sell_price']
    profit.append(sell_value - cash)
    cash = sell_value

trades['profit'] = profit
trades['cum_profit'] = trades['profit'].cumsum()
print(tabulate(trades, headers='keys', tablefmt='github'))
print(f"Total Profit: {trades['profit'].sum():.2f}")

# # Plot the selected columns
# ax = woot.plot(title="EMA Crossovers (Boolean Method)")
#
# # Plot green upward triangles at bullish crossover points on the fast EMA line
# ax.plot(df.index[cross_up], df['ema_fast'][cross_up], '^', color='green', markersize=10, label='Bullish Cross')
#
# # Plot red downward triangles at bearish crossover points on the fast EMA line
# ax.plot(df.index[cross_down], df['ema_fast'][cross_down], 'v', color='red', markersize=10, label='Bearish Cross')
#
# # Add legend to the plot
# ax.legend()
#
# # Show the plot
# plt.show()


import mplfinance as mpf
import numpy as np
import pandas as pd


df_mpf = df

# Create addplots for EMAs
ema_fast_plot = mpf.make_addplot(df_mpf['ema_fast'], color='blue', width=1.0)
ema_slow_plot = mpf.make_addplot(df_mpf['ema_slow'], color='orange', width=1.0)

# Prepare scatter markers for bullish and bearish crossovers
bullish_markers = pd.Series(np.nan, index=df_mpf.index)
bullish_markers.loc[cross_up] = df_mpf.loc[cross_up, 'ema_fast']

bearish_markers = pd.Series(np.nan, index=df_mpf.index)
bearish_markers.loc[cross_down] = df_mpf.loc[cross_down, 'ema_fast']

bullish_scatter = mpf.make_addplot(bullish_markers, type='scatter', markersize=70, marker='^', color='green')
bearish_scatter = mpf.make_addplot(bearish_markers, type='scatter', markersize=70, marker='v', color='red')

# Plot with mplfinance
mpf.plot(
    df_mpf,
    type='line',
    style='yahoo',
    addplot=[ema_fast_plot, ema_slow_plot, bullish_scatter, bearish_scatter],
    title='EMA Crossovers (mplfinance)',
    volume=True,
    figsize=(12, 8),
    tight_layout=True,
    block=False
)

# Create addplots for EMAs
sma_fast_plot = mpf.make_addplot(df_mpf['sma_fast'], color='blue', width=1.0)
sma_slow_plot = mpf.make_addplot(df_mpf['sma_slow'], color='orange', width=1.0)

# Prepare scatter markers for bullish and bearish crossovers
bullish_markers_sma = pd.Series(np.nan, index=df_mpf.index)
bullish_markers_sma.loc[cross_up_sma] = df_mpf.loc[cross_up_sma, 'sma_fast']

bearish_markers_sma = pd.Series(np.nan, index=df_mpf.index)
bearish_markers_sma.loc[cross_down_sma] = df_mpf.loc[cross_down_sma, 'sma_fast']

bullish_scatter_sma = mpf.make_addplot(bullish_markers_sma, type='scatter', markersize=70, marker='^', color='green')
bearish_scatter_sma = mpf.make_addplot(bearish_markers_sma, type='scatter', markersize=70, marker='v', color='red')

# Plot with mplfinance
mpf.plot(
    df_mpf,
    type='line',
    style='yahoo',
    addplot=[sma_fast_plot, sma_slow_plot, bullish_scatter_sma, bearish_scatter_sma],
    title='SMA Crossovers (mplfinance)',
    volume=True,
    figsize=(12, 8),
    tight_layout=True,
    block=True

)
