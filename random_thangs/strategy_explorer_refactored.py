import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator
import mplfinance as mpf
from tabulate import tabulate
from data_manager import CryptoDataManager, StockDataManager
from datetime import datetime, timedelta, timezone

# --- Helper functions ---
def add_indicator(df, col_name, indicator_class, window):
    df[col_name] = indicator_class(df['close'], window=window).ema_indicator() \
        if indicator_class == EMAIndicator else indicator_class(df['close'], window=window).sma_indicator()

def calculate_crossover(df, fast_col, slow_col):
    fast_above = df[fast_col] > df[slow_col]
    cross_up = fast_above & (~fast_above.shift(1).fillna(False).astype(bool))
    cross_down = (~fast_above) & (fast_above.shift(1).fillna(False).astype(bool))
    return cross_up, cross_down

def plot_crossovers(df, fast_col, slow_col, cross_up, cross_down, title, style='yahoo'):
    # Rename columns for mplfinance format
    df_mpf = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

    # Prepare addplots for fast and slow lines
    fast_plot = mpf.make_addplot(df_mpf[fast_col], color='blue', width=1.0)
    slow_plot = mpf.make_addplot(df_mpf[slow_col], color='orange', width=1.0)

    # Markers for crossovers
    bullish_markers = pd.Series(np.nan, index=df_mpf.index)
    bearish_markers = pd.Series(np.nan, index=df_mpf.index)
    bullish_markers.loc[cross_up] = df_mpf.loc[cross_up, fast_col]
    bearish_markers.loc[cross_down] = df_mpf.loc[cross_down, fast_col]

    addplots = [fast_plot, slow_plot]
    if not bullish_markers.dropna().empty:
        addplots.append(mpf.make_addplot(bullish_markers, type='scatter', markersize=70, marker='^', color='green'))
    if not bearish_markers.dropna().empty:
        addplots.append(mpf.make_addplot(bearish_markers, type='scatter', markersize=70, marker='v', color='red'))

    # Use returnfig=True to get the matplotlib figure object
    fig, axes = mpf.plot(df_mpf,
                         type='candle',
                         style=style,
                         addplot=addplots,
                         volume=True,
                         figsize=(12, 8),
                         tight_layout=True,
                         returnfig=True)

    # Add title above the plot area using suptitle
    fig.suptitle(title, fontsize=16, y=0.98)

    # Show the plot
    mpf.show()

def simulate_trades(df, cross_up, cross_down, trailing_pct=None, min_hold_bars=0):
    # Prepare buy and sell signals
    buy_signals = df.loc[cross_up, ['close']].reset_index().rename(columns={'date': 'buy_date', 'close': 'buy_price'})
    sell_signals = df.loc[cross_down, ['close']].reset_index().rename(columns={'date': 'sell_date', 'close': 'sell_price'})

    actual_trades = []
    for _, buy_row in buy_signals.iterrows():
        buy_date = buy_row['buy_date']
        buy_price = buy_row['buy_price']

        # Find next sell after buy
        possible_sells = sell_signals[sell_signals['sell_date'] > buy_date].sort_values('sell_date')
        if possible_sells.empty:
            # No sell after buy, skip or close at last available price
            continue
        planned_sell = possible_sells.iloc[0]
        sell_date = planned_sell['sell_date']
        planned_sell_price = planned_sell['sell_price']

        df_slice = df.loc[buy_date:sell_date]
        highest_price = buy_price
        trailing_stop_price = highest_price * (1 - trailing_pct) if trailing_pct else None

        triggered = False
        actual_sell_price = planned_sell_price
        actual_sell_date = sell_date

        if trailing_pct is not None:
            for i, (dt, row) in enumerate(df_slice.iterrows()):
                if i < min_hold_bars:
                    continue  # enforce minimum holding period
                price_high = row['high']
                price_low = row['low']

                if price_high > highest_price:
                    highest_price = price_high
                    trailing_stop_price = highest_price * (1 - trailing_pct)

                if price_low <= trailing_stop_price:
                    # Trailing stop triggered - sell at open price of this bar for realism
                    actual_sell_price = row['open']
                    actual_sell_date = dt
                    triggered = True
                    break

        actual_trades.append({
            'buy_date': buy_date,
            'buy_price': buy_price,
            'sell_date': actual_sell_date,
            'sell_price': actual_sell_price,
            'trailing_stop_triggered': triggered
        })

    trades_df = pd.DataFrame(actual_trades).sort_values('buy_date').reset_index(drop=True)

    # Calculate profit and running cash
    starting_cash = 1000
    cash = starting_cash
    buy_values, sell_values, profits = [], [], []

    for _, trade in trades_df.iterrows():
        shares = cash / trade['buy_price']
        buy_values.append(cash)
        sell_val = shares * trade['sell_price']
        sell_values.append(sell_val)
        profits.append(sell_val - cash)
        cash = sell_val

    trades_df['buy_value'] = buy_values
    trades_df['sell_value'] = sell_values
    trades_df['profit'] = profits
    trades_df['cum_profit'] = trades_df['profit'].cumsum()
    return trades_df

# --- Main ---
symbol = 'BTC'
interval = '4h'
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=30*6)
ema_fast_window = 10
ema_slow_window = 25
trailing_pct = 0.05
min_hold_bars = 5
df = CryptoDataManager().get_crypto_data(symbol, interval, start_date, end_date)
# df = StockDataManager().get_stock_data(symbol, interval, start_date, end_date)
# Add indicators
add_indicator(df, 'ema_fast', EMAIndicator, ema_fast_window )
add_indicator(df, 'ema_slow', EMAIndicator, ema_slow_window)

# Calculate crossovers
cross_up_ema, cross_down_ema = calculate_crossover(df, 'ema_fast', 'ema_slow')

# Simulate trades without trailing stop
trades_no_stop = simulate_trades(df, cross_up_ema, cross_down_ema)

print(tabulate(trades_no_stop, headers='keys', tablefmt='github'))
print(f"Total Profit without trailing stop: {trades_no_stop['profit'].sum():.2f}")
print(f"Daily Profit: {(trades_no_stop['profit'].sum()) / len(df):.2f}\n")

# Simulate trades with 3% trailing stop and 3-bar minimum hold
trades_with_stop = simulate_trades(df, cross_up_ema, cross_down_ema, trailing_pct=trailing_pct, min_hold_bars=min_hold_bars)

print(tabulate(trades_with_stop, headers='keys', tablefmt='github'))
print(f"Total Profit with trailing stop: {trades_with_stop['profit'].sum():.2f}")

# Plot EMA crossovers
title = f"{symbol}| EMA: {ema_fast_window} fast, {ema_slow_window} slow, "
plot_crossovers(df, 'ema_fast', 'ema_slow', cross_up_ema, cross_down_ema, title)
