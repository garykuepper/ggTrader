import pandas as pd
from ta.trend import EMAIndicator
from tabulate import tabulate
from datetime import datetime, timedelta
from old.ggTrader_old.data_manager import UniversalDataManager

pd.set_option('future.no_silent_downcasting', True)

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variables
mongo_uri = os.getenv('MONGO_URI')
db_name = os.getenv('DB_NAME')

# Load CSV data into a DataFrame
df = pd.read_csv("../dl_data/yf_BTC_1m_5d.csv")

# Convert the 'date' column to datetime objects for proper time series handling
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the DataFrame index for easier plotting and time-based operations
df = df.set_index('date')

symbol = "XRPUSDT"
interval = "5m"
end_date = datetime.now() - timedelta(hours=1)
start_date = end_date - timedelta(days=7)
time_diff = end_date - start_date
num_days = time_diff.days
print(F"From {start_date} to {end_date}")
marketType = "crypto"
ema_fast_range = 8
ema_slow_range = 10
# dm = UniversalDataManager()

# Change these lines:
# latest = dm.get_latest_optimization_parameters(symbol, "ema_crossover", interval)
# ema_fast = latest["parameters"]["ema_fast"]  # Use ema_fast, not ema_fast_range
# ema_slow = latest["parameters"]["ema_slow"]  # Use ema_slow, not ema_slow_range



# Get the specific data manager for this symbol/market
# specific_dm = dm.get_manager(symbol, interval, marketType)
#
# df = dm.load_or_fetch(symbol, interval, start_date, end_date, market=marketType)

# And update the EMA calculations:
df['ema_fast'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
df['ema_slow'] = EMAIndicator(df['close'], window=ema_slow).ema_indicator()

# Create a boolean Series that is True when fast EMA is above slow EMA, False otherwise
ema_fast_above = df['ema_fast'] > df['ema_slow']

cross_up = (ema_fast_above) & (~ema_fast_above.shift(1).fillna(False).astype(bool))
cross_down = (~ema_fast_above) & (ema_fast_above.shift(1).fillna(False).astype(bool))

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

print(df.columns)
buy_signals = df.loc[cross_up, ['close']].reset_index().rename(columns={'datetime': 'buy_date', 'close': 'buy_price'})
sell_signals = df.loc[cross_down, ['close']].reset_index().rename(
    columns={'datetime': 'sell_date', 'close': 'sell_price'})

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
totalProfit = trades['profit'].sum()
dailyProfit = totalProfit / num_days
trades['cum_profit'] = trades['profit'].cumsum()
print(tabulate(trades, headers='keys', tablefmt='github'))
print(f"Total Profit: ${totalProfit:.2f}")
print(f"Daily Profit: ${dailyProfit:.2f}")
print(f"Start Cash:   ${starting_cash:.2f}")
print(f"Num of Days:   {num_days}")
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

# Rename columns to mplfinance expected names
df_mpf = df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert Volume to numeric, replacing non-numeric values with NaN
df_mpf['Volume'] = pd.to_numeric(df_mpf['Volume'], errors='coerce')

# Drop rows where Volume is NaN if needed, or fill with 0
df_mpf['Volume'] = df_mpf['Volume'].fillna(0)

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
# mpf.plot(
#     df_mpf,
#     type='line',
#     style='yahoo',
#     addplot=[ema_fast_plot, ema_slow_plot, bullish_scatter, bearish_scatter],
#     title=str(F"{symbol}: EMA"),
#     volume=True,
#     figsize=(12,8),
#     tight_layout=True
# )
# At the end of your script, before it exits:
try:
    dm.close()
except:
    pass