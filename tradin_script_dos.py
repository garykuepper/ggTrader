# Python
from trading_strategy import EMAStrategy
from data_manager import CryptoDataManager
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import pandas as pd
import numpy as np
import mplfinance as mpf

def align_to_binance_4h(dt):
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=(dt.hour // 4) * 4)


def plot_candles_with_trades(price_df: pd.DataFrame, result: dict, title: str = "Chart"):
    """
    price_df: DataFrame with columns ['open','high','low','close','volume'] indexed by datetime
    result: dict from your strategy.backtest() with keys:
        - 'signal_df': DataFrame including 'ema_fast','ema_slow' (optional)
        - 'trades': list of TradeResult objects with buy_time/exit_time and prices
    """
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)

    # mplfinance expects specific column names
    plot_df = price_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }).copy()

    # Prepare EMA overlays if available in signal_df
    apds = []
    sig_df = result.get('signal_df', pd.DataFrame())
    if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
        # align to plot_df index
        em_fast = sig_df.reindex(plot_df.index)['ema_fast'] if 'ema_fast' in sig_df else None
        em_slow = sig_df.reindex(plot_df.index)['ema_slow'] if 'ema_slow' in sig_df else None
        if em_fast is not None:
            apds.append(mpf.make_addplot(em_fast, color='tab:blue', width=1.0))
        if em_slow is not None:
            apds.append(mpf.make_addplot(em_slow, color='tab:orange', width=1.0))

    # Build buy/sell marker series aligned to index
    buy_y = pd.Series(np.nan, index=plot_df.index)
    sell_y = pd.Series(np.nan, index=plot_df.index)

    for tr in result.get('trades', []):
        # Ensure timestamps match index frequency
        bt = pd.Timestamp(tr.buy_time)
        et = pd.Timestamp(tr.exit_time)

        # If the exact timestamp exists in index, place marker at trade price;
        # fallback to candle Close if not.
        if bt in plot_df.index:
            buy_y.loc[bt] = float(tr.buy_price)
        else:
            # fallback: find nearest index at/after bt
            idx = plot_df.index.searchsorted(bt)
            if idx < len(plot_df.index):
                buy_y.iloc[idx] = float(tr.buy_price)

        if et in plot_df.index:
            sell_y.loc[et] = float(tr.exit_price)
        else:
            idx = plot_df.index.searchsorted(et)
            if idx < len(plot_df.index):
                sell_y.iloc[idx] = float(tr.exit_price)

    # Add scatter markers
    apds.append(mpf.make_addplot(buy_y, type='scatter', marker='^', color='green', markersize=80))
    apds.append(mpf.make_addplot(sell_y, type='scatter', marker='v', color='red', markersize=80))

    # Plot
    mpf.plot(
        plot_df[['Open', 'High', 'Low', 'Close', 'Volume']],
        type='candle',
        style='yahoo',
        addplot=apds,
        volume=True,
        title=title,
        ylabel='Price',
        ylabel_lower='Volume',
        figratio=(16, 9),
        figscale=1.1
    )
cm = CryptoDataManager()
symbol, interval = "BTCUSDT", "4h"
end_date = align_to_binance_4h(datetime.now(timezone.utc))
start_date = end_date - timedelta(days=60)
df = cm.get_crypto_data(symbol, interval, start_date, end_date)

strategy = EMAStrategy(
    name="ema_crossover",
    params={'ema_fast': 16, 'ema_slow': 35},
    trailing_pct=0.01  # 5% trailing
)

result = strategy.backtest(df, starting_cash=1000,
                           min_hold_bars=10,
                           use_trailing=True,
                           trailing_price_mode="ohlc4")

print(f"Final cash: {result['final_cash']:.2f} (Return: {result['total_return_pct']:.2f}%)")
# Inspect signals or trades
signals = result['signal_df']
# print(tabulate(signals.tail(10), headers='keys', tablefmt='github'))
print(f"Trades: {len(result['trades'])}")
for t in result['trades'][-3:]:
    print(f"{t.buy_time} buy {t.buy_price:.4f} -> {t.exit_time} {t.exit_price:.4f} ({t.exit_reason})")


plot_candles_with_trades(df, result, f"{symbol} Chart")