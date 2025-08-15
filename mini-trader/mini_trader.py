from datetime import datetime, timedelta

import yfinance as yf
import mplfinance as mplf
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from ta.trend import EMAIndicator
from tabulate import tabulate


class MiniTrader:

    def __init__(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime,):
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        self.signal_data = pd.DataFrame()

    def get_data(self):
        self.data = yf.download(self.symbol,
                                interval=self.interval,
                                start=self.start_date,
                                end=self.end_date,
                                multi_level_index=False,
                                auto_adjust=True)

    def calc_signals(self, ema_fast, ema_slow):
        self.signal_data['ema_fast'] = EMAIndicator(close=self.data["Close"], window=ema_fast, fillna=False).ema_indicator()
        self.signal_data['ema_slow'] = EMAIndicator(close=self.data["Close"], window=ema_slow, fillna=False).ema_indicator()
        # Compute crossover points: +1 when fast crosses above slow (bullish), -1 when below (bearish)

        signal = (self.signal_data['ema_fast'] > self.signal_data['ema_slow']).astype(int)
        cross = signal.diff()
        cross_up = cross == 1
        cross_down = cross == -1

        # Create series for markers positioned at the price level on crossover bars
        self.signal_data['buy_marker'] = self.data["Close"].where(cross_up)
        self.signal_data['sell_marker'] = self.data["Close"].where(cross_down)

    def plot_data(self):

        ema_fast = self.signal_data['ema_fast']
        ema_slow = self.signal_data['ema_slow']
        buy_markers = self.signal_data['buy_marker']
        sell_markers = self.signal_data['sell_marker']
        apds = [
            mpf.make_addplot(ema_fast, color="#ff9800", width=1.2, panel=0, label="EMA12"),
            mpf.make_addplot(ema_slow, color="#9c27b0", width=1.2, panel=0, label="EMA26"),
            mpf.make_addplot(buy_markers, type="scatter", marker="^", color="green", markersize=80, panel=0,
                             label="Bull X"),
            mpf.make_addplot(sell_markers, type="scatter", marker="v", color="red", markersize=80, panel=0,
                             label="Bear X"),
        ]

        mplf.plot(self.data,
                  type='candle',
                  addplot=apds,
                  style='yahoo',
                  volume=True,
                  figsize=(13, 7),
                  tight_layout=True
                  )

class Position:
    def __init__(self, symbol, shares, price, date):
        pass

class Portfolio:
    def __init__(self):
        pass

symbol = "BTC-USD"
interval = "1d"
period = "180d"
end_date = datetime(2025, 8, 1)
start_date = end_date - timedelta(days=180)
ema_fast = 12
ema_slow = 30

mt = MiniTrader(symbol, interval, start_date, end_date)
mt.get_data()
mt.calc_signals(ema_fast, ema_slow)
mt.plot_data()
