import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import time as t
import math
import optuna
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
        self.total_profit = 0

    def get_data(self):
        self.data = self.get_yf_data(self.symbol, self.interval, self.start_date, self.end_date)
        return self.data

    def update_data(self, data: pd.DataFrame):
        self.data = data

    @staticmethod
    def get_yf_data(symbol: str, interval: str, start_date: datetime, end_date: datetime):
        return yf.download(symbol,
                           interval=interval,
                           start=start_date,
                           end=end_date,
                           multi_level_index=False,
                           auto_adjust=True)

    @staticmethod
    def calc_signals(ema_fast: int, ema_slow: int, data: pd.DataFrame):
        signal_data = pd.DataFrame(index=data.index)
        signal_data['ema_fast'] = EMAIndicator(close=data["Close"], window=ema_fast,
                                               fillna=False).ema_indicator()
        signal_data['ema_slow'] = EMAIndicator(close=data["Close"], window=ema_slow,
                                               fillna=False).ema_indicator()
        # Compute crossover points: +1 when fast crosses above slow (bullish), -1 when below (bearish)

        signal = (signal_data['ema_fast'] > signal_data['ema_slow']).astype(int)
        cross = signal.diff()
        cross_up = cross == 1
        cross_down = cross == -1

        # Create series for markers positioned at the price level on crossover bars
        signal_data['buy_marker'] = data["Close"].where(cross_up)
        signal_data['sell_marker'] = data["Close"].where(cross_down)
        return signal_data

    def plot_data(self, num_of_pts=200):

        if self.data.empty or self.signal_data.empty:
            print("Error: No data available to plot.")
            return

        total_pts = len(self.signal_data)
        if total_pts == 0:
            print("Error: Signal data is empty. Cannot generate plot.")
            return

        start_plot = max(0, total_pts - num_of_pts)
        if start_plot >= total_pts - 1:
            print("Error: Not enough data points to plot.")
            return

        data_slice = self.data.iloc[start_plot:]
        signals_slice = self.signal_data.iloc[start_plot:]

        # Additional checks for empty or all-NaN data
        if data_slice.empty or not data_slice['Close'].notna().any():
            print("Error: Sliced data is empty or has no non-NaN Close values.")
            return
        if signals_slice.empty:
            print("Error: Sliced signals are empty.")
            return

        apds = [
            mpf.make_addplot(signals_slice['ema_fast'], width=1.2),
            mpf.make_addplot(signals_slice['ema_slow'], width=1.2),
            mpf.make_addplot(
                signals_slice['buy_marker'],
                type="scatter",
                marker="^",
                color="green",
                markersize=80,
                edgecolors="black",
                linewidths=1.5
            ),
            mpf.make_addplot(
                signals_slice['sell_marker'],
                type="scatter",
                marker="v",
                color="red",
                markersize=80,
                edgecolors="black",
                linewidths=1.5
            ),
        ]

        try:
            mpf.plot(
                data_slice,
                type='candle',
                addplot=apds,
                style='yahoo',
                title=f"Trading Chart for {self.symbol} ({self.interval})",
                volume=True,
                figsize=(13, 7),
                tight_layout=True
            )
        except Exception as e:
            print(f"Error generating plot: {e}")
            print(f"data_slice shape: {data_slice.shape}, signals_slice shape: {signals_slice.shape}")

    @staticmethod
    def backtest(signal_data: pd.DataFrame,
                 data: pd.DataFrame,
                 symbol: str,
                 trail_percentage: float=3,
                 hold_min: int=5,
                 print_position=False,
                 print_trades=False,):
        portfolio = Portfolio(hold_min=hold_min)

        for row in signal_data.itertuples():
            price = data.loc[row.Index, 'Close']
            date = row.Index
            if portfolio.in_position(symbol):
                portfolio.update_position_price(symbol, price, date, trail_percentage)

            if pd.notna(row.buy_marker) and not portfolio.in_position(symbol):
                qty = portfolio.cash / price
                portfolio.add_position(Position(symbol, qty, price, date))

            elif pd.notna(row.sell_marker) and portfolio.in_position(symbol):
                portfolio.remove_position(portfolio.get_position(symbol), date=date)
        if print_position:
            portfolio.print_positions()
        if print_trades:
            portfolio.print_trades()
        return portfolio.profit

    @staticmethod
    def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.001) -> float:
        """
        Calculate Sharpe ratio from a list of returns.

        Args:
            returns: List of profit/return values
            risk_free_rate: Risk-free rate for excess return calculation

        Returns:
            Sharpe ratio (float)
        """
        if len(returns) == 0:
            return -float("inf")

        returns_series = pd.Series(returns)
        avg_return = returns_series.mean()
        std_dev = returns_series.std()
        excess_return = avg_return - risk_free_rate

        epsilon = 0.0001
        sharpe_ratio = excess_return / max(std_dev, epsilon) if std_dev > epsilon else 0

        return sharpe_ratio



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
        self.trailing_stop = None
        self.ts_triggered = False
        self.ts_consec_hits = 0


    def open_position(self):
        pass

    def close_position(self, date: datetime):
        self.status = "closed"
        self.date = date

    def update_price(self, new_price: float, trail_percentage: float = 0):
        self.price = new_price
        self.current_value = self.qty * new_price
        self.profit = self.current_value - self.cost
        if trail_percentage > 0:
            trailing_stop_candidate = new_price * (1 - trail_percentage / 100)
            if self.trailing_stop is None or trailing_stop_candidate > self.trailing_stop:
                self.trailing_stop = trailing_stop_candidate

    def update_date(self, date: datetime):
        self.date = date


class Portfolio:
    def __init__(self, cash=1000, hold_min=5):
        self.trades = []
        self.positions = []
        self.cash = cash
        self.profit = 0
        self.start_cash = cash
        self.hold_min = hold_min
        self.transaction_fee = 0.004  # max maker fee

    def add_position(self, position: Position):
        self.cash -= position.cost * (1 + self.transaction_fee)
        self.positions.append(position)
        self.trades.append(position.__dict__.copy())

    def remove_position(self, position: Position, date: datetime):
        self.cash += position.current_value - (position.current_value * self.transaction_fee)

        position.close_position(date=date)
        self.trades.append(position.__dict__.copy())
        self.positions.remove(position)

    def update_position_price(self, symbol: str, price: float, date: datetime, trail_percentage: float = 0):
        position = self.get_position(symbol)
        if position:
            position.update_price(price, trail_percentage)
        # Check for trailing stop trigger
        self.check_trailing_stop(position, date)
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

    def check_trailing_stop(self, position: Position, date: datetime):
        """
        Check if the current price has hit the trailing stop for the position.
        If the stop is hit, automatically close the position.
        """

        if position.trailing_stop and position.price <= position.trailing_stop:
            # print(f"Trailing stop hit for {position.symbol} at {position.price:.2f}. Closing position.")
            position.ts_consec_hits += 1
            if position.ts_consec_hits >= self.hold_min:
                position.ts_triggered = True
                self.remove_position(position, date)
        else:
            position.ts_consec_hits = 0


def objective(trial):
    max_window = 200
    max_fast_window = math.floor(max_window / 3)
    fast_window = trial.suggest_int('fast_window', 12, max_fast_window, step=2)
    slow_window = trial.suggest_int('slow_window', fast_window + 10, max_window, step=2)
    trail_pct = trial.suggest_int('trail_pct', 3, 6)
    hold_min = trial.suggest_int('hold_min', 2, 6)
    signals = mt.calc_signals(fast_window, slow_window, data)

    # Rolling window backtesting
    num_of_pts = len(data)
    window_min_pts = math.floor(num_of_pts / 2)
    step = math.floor(window_min_pts / 10)
    returns = []

    for i in range(num_of_pts - 1, window_min_pts, -step):
        profit = mt.backtest(
            signals.iloc[i - window_min_pts:i],
            data.iloc[i - window_min_pts:i],
            symbol=symbol,
            trail_percentage=trail_pct,
            hold_min=hold_min
        )
        returns.append(profit)

    # Use the static method
    sharpe_ratio = mt.calculate_sharpe_ratio(returns)

    total_profit = mt.backtest(signals, data, symbol, trail_percentage=trail_pct, hold_min=hold_min)

    if total_profit <= 0 or sharpe_ratio <= 0:
        return -float("inf")

    return sharpe_ratio



def days_min(pts_per_day, num_pts):
    return int(math.floor(num_pts / pts_per_day))


symbol = "BTC-USD"
interval = "4h"

pts_per_day = {"1d": 1, "1h": 24, "4h": 6}
end_date = datetime(2025, 8, 1)

days = days_min(pts_per_day[interval], 365 * 5)
start_date = end_date - timedelta(days=days)

mt = MiniTrader(symbol, interval, start_date, end_date)
data = mt.get_data()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=-1)

t.sleep(1)

signals = mt.calc_signals(study.best_params['fast_window'], study.best_params['slow_window'], data)
profit = mt.backtest(signals, data, symbol,
                     trail_percentage=study.best_params['trail_pct'],
                     hold_min=study.best_params['hold_min'],
                     print_trades=True,
                     print_position=True,)
best_parameters = {**study.best_params, "Best Sharpe Ratio": study.best_value, "Total Profit": profit}

print(f"Start date:  {start_date}")
print(f"End date:    {end_date}")
print(f"Num of Days: {days}")

print("Best parameters:")
print(tabulate(best_parameters.items(), headers=["Parameter", "Value"], tablefmt="github"))

mt.signal_data = signals
mt.plot_data(num_of_pts=400)
