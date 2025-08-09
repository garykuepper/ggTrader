import pandas as pd
import mplfinance as mpf

from portfolio import Portfolio
from position import Position
from datetime import datetime, timedelta, timezone
from data_manager import CryptoDataManager
from tabulate import tabulate
import numpy as np

import ta


def calc_emas(ema_fast: int, ema_slow: int, data: pd.DataFrame):
    data_for_ema = data['close']
    data['ema_fast'] = ta.trend.EMAIndicator(close=data_for_ema,
                                             window=ema_fast).ema_indicator()
    data['ema_slow'] = ta.trend.EMAIndicator(close=data_for_ema,
                                             window=ema_slow).ema_indicator()
    return data


def crossover(data: pd.DataFrame):
    # Calculate the difference
    data['ema_diff'] = data['ema_fast'] - data['ema_slow']

    # Find crossovers: 1 for golden cross, -1 for death cross, 0 otherwise
    data['crossover'] = np.where(
        (data['ema_diff'].shift(1) < 0) & (data['ema_diff'] > 0), 1,
        np.where(
            (data['ema_diff'].shift(1) > 0) & (data['ema_diff'] < 0), -1, 0
        )
    )

    return data


def get_crossover_table(data: pd.DataFrame):
    return data[data['crossover'] != 0]


def print_crossover(data: pd.DataFrame):
    cross_df = get_crossover_table(data)
    print("Crossover:")
    print(tabulate(cross_df, headers='keys', tablefmt='github'))


def plot_crossover(data: pd.DataFrame):
    symbol = data['symbol'].iloc[0]
    plot_data = data.copy().rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    plot_data.index = pd.to_datetime(plot_data.index)

    # Full-length Series with NaNs except at crossover points
    cross_up_y = plot_data['Close'].where(plot_data['crossover'] == 1)
    cross_dn_y = plot_data['Close'].where(plot_data['crossover'] == -1)

    apds = [
        mpf.make_addplot(plot_data['ema_fast']),
        mpf.make_addplot(plot_data['ema_slow']),
        mpf.make_addplot(cross_up_y, type='scatter', marker='^', markersize=80, color='green'),
        mpf.make_addplot(cross_dn_y, type='scatter', marker='v', markersize=80, color='red'),
    ]

    mpf.plot(
        plot_data[['Open', 'High', 'Low', 'Close', 'Volume']],  # be explicit
        type='candle',
        style='yahoo',
        addplot=apds,
        volume=True,
        title=f'{symbol} EMA Crossovers',
        ylabel='Price'
    )


def get_latest_signal(data: pd.DataFrame):
    cross_df = data[data['crossover'] != 0]
    return cross_df.iloc[-1]['crossover']


def simulate(symbol, start, end):
    pass


def align_to_binance_4h(dt):
    """Align a datetime to the previous 4h boundary (UTC)."""
    dt = dt.replace(minute=0, second=0, microsecond=0)
    aligned_hour = (dt.hour // 4) * 4
    return dt.replace(hour=aligned_hour)


cm = CryptoDataManager()

p = Portfolio()

symbol = "BTCUSDT"
interval = "4h"
# get closest hour
end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
end_date = align_to_binance_4h(end_date)
start_date = end_date - timedelta(days=30 * 2)

# ema
ema_fast = 10
ema_slow = 50

data = cm.get_crypto_data(symbol, interval, start_date, end_date)

data = calc_emas(ema_fast, ema_slow, data)
data = crossover(data)

print(tabulate(data.tail(10), headers='keys', tablefmt='github'))

print_crossover(data)
plot_crossover(data)
signal = get_latest_signal(data)

print(signal)

signal_table = get_crossover_table(data).copy()
p.print_acct()
for row in signal_table.itertuples():
    # print(f"{row.Index} : {row.crossover}")
    if p.has_position(symbol):
        p.update_position_price(symbol, row.close)
    if row.crossover == -1:
        p.close_position(symbol)

    if row.crossover == 1:
        qty = p.cash / row.close
        bought_price = row.close
        p.open_position(Position(symbol, qty, bought_price, row.Index))

p.update_position_price(symbol, data['close'].iloc[-1])
p.print_trade_history()
p.print_acct()
p.print_positions()
