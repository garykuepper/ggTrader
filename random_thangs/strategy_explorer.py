import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator
import mplfinance as mpf
from tabulate import tabulate
from data_manager import CryptoDataManager, StockDataManager
from datetime import datetime, timedelta

# --- Helper functions ---
def calculate_crossover(df, fast_col, slow_col):
    """Return boolean Series for bullish and bearish crossovers."""
    fast_above = df[fast_col] > df[slow_col]
    cross_up = fast_above & (~fast_above.shift(1).fillna(False).astype(bool))
    cross_down = (~fast_above) & (fast_above.shift(1).fillna(False).astype(bool))
    return cross_up, cross_down

def add_indicator(df, col_name, indicator_class, window):
    """Add indicator column to DataFrame."""
    df[col_name] = indicator_class(df['close'], window=window).ema_indicator() \
        if indicator_class == EMAIndicator else indicator_class(df['close'], window=window).sma_indicator()

def create_marker_series(df, cross_mask, value_col):
    """Create a Series with marker values at crossover points, NaN elsewhere."""
    markers = pd.Series(np.nan, index=df.index)
    markers.loc[cross_mask] = df.loc[cross_mask, value_col]
    return markers

def plot_crossovers(df, fast_col, slow_col, cross_up, cross_down, title, style='yahoo'):
    """Plot price, indicators, and crossover markers using mplfinance."""
    df_mpf = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })

    fast_plot = mpf.make_addplot(df_mpf[fast_col], color='blue', width=1.0)
    slow_plot = mpf.make_addplot(df_mpf[slow_col], color='orange', width=1.0)

    bullish_markers = create_marker_series(df_mpf, cross_up, fast_col)
    bearish_markers = create_marker_series(df_mpf, cross_down, fast_col)

    addplots = [fast_plot, slow_plot]
    if not bullish_markers.dropna().empty:
        addplots.append(mpf.make_addplot(bullish_markers, type='scatter', markersize=70, marker='^', color='green'))
    if not bearish_markers.dropna().empty:
        addplots.append(mpf.make_addplot(bearish_markers, type='scatter', markersize=70, marker='v', color='red'))

    mpf.plot(
        df_mpf,
        type='line',
        style=style,
        addplot=addplots,
        title=title,
        volume=True,
        figsize=(12,8),
        tight_layout=True
    )

# --- Main workflow ---

# Load data
# df = pd.read_csv("yf_ltc_5y.csv")
# df['date'] = pd.to_datetime(df['date'])
# df = df.set_index('date')
symbol = 'SPY'
interval = '1h'
end_date = datetime(2025, 8, 5)
start_date = end_date - timedelta(days=3)

df = StockDataManager().get_stock_data(symbol, interval, start_date, end_date)

# Add indicators
add_indicator(df, 'ema_fast', EMAIndicator, 8)
add_indicator(df, 'ema_slow', EMAIndicator, 20)
add_indicator(df, 'sma_fast', SMAIndicator, 20)
add_indicator(df, 'sma_slow', SMAIndicator, 50)

# Calculate crossovers
cross_up_ema, cross_down_ema = calculate_crossover(df, 'ema_fast', 'ema_slow')
cross_up_sma, cross_down_sma = calculate_crossover(df, 'sma_fast', 'sma_slow')

# --- Trade simulation (unchanged, but could be refactored similarly) ---
columns = ['close', 'ema_fast', 'ema_slow']
woot = df[columns]

buy = woot[cross_up_ema].copy()
buy['signal'] = 'BUY'
sell = woot[cross_down_ema].copy()
sell['signal'] = 'SELL'
signals = pd.concat([buy, sell]).sort_index()
print(tabulate(signals, headers='keys', tablefmt='github'))
print("\n")

buy_signals = df.loc[cross_up_ema, ['close']].reset_index().rename(columns={'date':'buy_date', 'close':'buy_price'})
sell_signals = df.loc[cross_down_ema, ['close']].reset_index().rename(columns={'date':'sell_date', 'close':'sell_price'})

trades = pd.merge_asof(buy_signals.sort_values('buy_date'),
                       sell_signals.sort_values('sell_date'),
                       left_on='buy_date', right_on='sell_date',
                       direction='forward')

starting_cash = 1000
profit = []
buy_v = []
sell_v = []
cash = starting_cash
for i, row in trades.iterrows():
    shares_bought = cash/row['buy_price']
    buy_v.append(cash)
    sell_value = shares_bought * row['sell_price']
    sell_v.append(sell_value)
    profit.append(sell_value - cash)
    cash = sell_value


trades['buy_value'] = buy_v
trades['sell_value'] = sell_v
trades['profit'] = profit
trades['cum_profit'] = trades['profit'].cumsum()
print(tabulate(trades,headers='keys',tablefmt='github'))
print(f"Total Profit: {trades['profit'].sum():.2f}")
print(f"Daily Profit: {(trades['profit'].sum())/len(df):.2f} ")

# --- Plotting ---
plot_crossovers(df, 'ema_fast', 'ema_slow', cross_up_ema, cross_down_ema, 'EMA Crossovers (mplfinance)')
plot_crossovers(df, 'sma_fast', 'sma_slow', cross_up_sma, cross_down_sma, 'SMA Crossovers (mplfinance)')