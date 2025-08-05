from datetime import datetime, timedelta
from data_manager import CryptoDataManager
from ema_trailing_strategy import EmaTrailingStrategy
from tabulate import tabulate
import mplfinance as mpf
import pandas as pd
import numpy as np


# --- Plot function ---
def plot_strategy(df, trades, fast_col='ema_fast', slow_col='ema_slow', title='EMA Strategy with Trailing Stop'):
    df_plot = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

    ema_fast_plot = mpf.make_addplot(df_plot[fast_col], color='blue', width=1)
    ema_slow_plot = mpf.make_addplot(df_plot[slow_col], color='orange', width=1)

    buy_marker = pd.Series(np.nan, index=df_plot.index)
    sell_marker = pd.Series(np.nan, index=df_plot.index)

    for _, row in trades.iterrows():
        buy_dt = row['buy_date']
        sell_dt = row['sell_date']
        if buy_dt in buy_marker.index:
            buy_marker.loc[buy_dt] = df_plot.loc[buy_dt, 'Low'] * 0.995
        if pd.notna(sell_dt) and sell_dt in sell_marker.index:
            sell_marker.loc[sell_dt] = df_plot.loc[sell_dt, 'High'] * 1.005

    buy_plot = mpf.make_addplot(buy_marker, type='scatter', markersize=100, marker='^', color='green')
    sell_plot = mpf.make_addplot(sell_marker, type='scatter', markersize=100, marker='v', color='red')

    addplots = [ema_fast_plot, ema_slow_plot, buy_plot, sell_plot]

    mpf.plot(df_plot, type='candle', style='yahoo', addplot=addplots, title=title, volume=True, figsize=(14, 8))


# --- Main ---
symbol = 'LTCUSDT'
interval = '4h'
end_date = datetime(2025, 8, 5)
start_date = end_date - timedelta(days=30)

df = CryptoDataManager().get_crypto_data(symbol, interval, start_date, end_date)

strat = EmaTrailingStrategy(df, fast_window=5, slow_window=12, trailing_pct=0.05, min_hold_bars=4, starting_cash=1000)
trades, df_with_indicators = strat.run()

print(tabulate(trades, headers='keys', tablefmt='github'))

# Use df_with_indicators for plotting (has ema_fast, ema_slow)
plot_strategy(df_with_indicators, trades, fast_col='ema_fast', slow_col='ema_slow', title='EMA Trailing Stop Strategy')
