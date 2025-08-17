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
                 end_date: datetime, ):
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
        self.signal_data['ema_fast'] = EMAIndicator(close=self.data["Close"], window=ema_fast,
                                                    fillna=False).ema_indicator()
        self.signal_data['ema_slow'] = EMAIndicator(close=self.data["Close"], window=ema_slow,
                                                    fillna=False).ema_indicator()
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

    def backtest(self):
        portfolio = Portfolio()

        for row in mt.signal_data.itertuples():
            price = mt.data.loc[row.Index, 'Close']
            date = row.Index
            if portfolio.in_position(symbol):
                portfolio.update_position_price(symbol, price)

            if pd.notna(row.buy_marker):
                qty = portfolio.cash / price
                portfolio.add_position(Position(symbol, qty, price, date))

            elif pd.notna(row.sell_marker) and portfolio.in_position(symbol):
                portfolio.remove_position(portfolio.get_position(symbol), date=date)

        portfolio.print_trades()
        portfolio.print_positions()
        print("\nPerformance:")
        print(f"Total value: $ {portfolio.get_total_value():.2f}")
        print(f"Total cash:  $ {portfolio.cash:.2f}")
        print(f"Total profit: $ {portfolio.profit:.2f}")


class Position:
    def __init__(self, symbol: str, qty: float, price: float, date: datetime):
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.date = date
        self.cost = qty * price
        self.current_value = qty * price
        self.profit = 0
        self.status = "open"

    def open_position(self):
        pass

    def close_position(self, date: datetime):
        self.status = "closed"
        self.date = date

    def update_price(self, price: float):
        self.price = price
        self.current_value = self.qty * price
        self.profit = self.current_value - self.cost

    def update_date(self, date: datetime):
        self.date = date


class Portfolio:
    def __init__(self, cash=1000):
        self.trades = []
        self.positions = []
        self.cash = cash
        self.profit = 0
        self.start_cash = cash

    def add_position(self, position: Position):
        self.cash -= position.cost
        self.positions.append(position)
        self.trades.append(position.__dict__.copy())

    def remove_position(self, position: Position, date: datetime):
        self.cash += position.current_value

        position.close_position(date=date)
        self.trades.append(position.__dict__.copy())
        self.positions.remove(position)

    def update_position_price(self, symbol: str, price: float):
        position = self.get_position(symbol)
        position.update_price(price)
        self.profit = self.get_total_value() - self.start_cash

    def get_position(self, symbol: str):
        for position in self.positions:
            if position.symbol == symbol:
                return position
        print(f"Position for {symbol} not found")
        return None

    def in_position(self, symbol: str):
        for position in self.positions:
            if position.symbol == symbol:
                return True
        return False

    def print_positions(self):
        pos = []
        for position in self.positions:
            pos.append(position.__dict__)
        print("\nPositions:")
        print(tabulate(pos, headers="keys", tablefmt="github"))

    def print_trades(self):
        trades = []
        for trade in self.trades:
            trades.append(trade)
        print("\nTrades:")
        print(tabulate(trades, headers="keys", tablefmt="github"))

    def get_total_value(self):
        total_value = self.cash
        for position in self.positions:
            total_value += position.current_value
        return total_value


symbol = "BTC-USD"
interval = "1d"
end_date = datetime(2025, 8, 1)
start_date = end_date - timedelta(days=365)
ema_fast = 12
ema_slow = 30

mt = MiniTrader(symbol, interval, start_date, end_date)
mt.get_data()
mt.calc_signals(ema_fast, ema_slow)
# mt.plot_data()
mt.backtest()
