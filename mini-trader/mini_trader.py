import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta, timezone
import time
import math
import optuna
from ta.trend import EMAIndicator
from tabulate import tabulate


# TODO:  Turn Backtest into a class
# TODO:  Add data as own class
# TODO: Make position data class
# TODO: Make generic StopPolicy class for trailing stops
# TODO: Strategy Abstract class, with generate_signals() method
# TODO: Performance Metrics and Plotting
# TODO: Optimize class for optuna

class MiniTrader:

    def __init__(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime):

        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        self.signal_data = pd.DataFrame()

    def get_data(self):
        self.data = self.get_yf_data(self.symbol, self.interval, self.start_date, self.end_date)
        if self.data.empty:
            raise ValueError("No data available.")
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
    def plot_data(data: pd.DataFrame, signal_data: pd.DataFrame, symbol: str, num_of_pts=200):

        if data.empty or signal_data.empty:
            print("Error: No data available to plot.")
            return

        total_pts = len(data)
        if total_pts == 0:
            print("Error: Data is empty. Cannot generate plot.")
            return

        start_plot = max(0, total_pts - num_of_pts)
        if start_plot >= total_pts - 1:
            print("Error: Not enough data points to plot.")
            return

        data_slice = data.iloc[start_plot:]
        signals_slice = signal_data.reindex(data_slice.index)

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
        ]
        # Only add scatter plots if there is at least one non-NaN point
        if 'buy_marker' in signals_slice and signals_slice['buy_marker'].notna().any():
            apds.append(
                mpf.make_addplot(
                    signals_slice['buy_marker'],
                    type="scatter",
                    marker="^",
                    color="green",
                    markersize=80,
                    edgecolors="black",
                    linewidths=1.5
                )
            )
        if 'sell_marker' in signals_slice and signals_slice['sell_marker'].notna().any():
            apds.append(
                mpf.make_addplot(
                    signals_slice['sell_marker'],
                    type="scatter",
                    marker="v",
                    color="red",
                    markersize=80,
                    edgecolors="black",
                    linewidths=1.5
                )
            )

        try:
            mpf.plot(
                data_slice,
                type='candle',
                addplot=apds,
                style='yahoo',
                title=f"Trading Chart for {symbol} ",
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
                 trail_percentage: float = 3,
                 hold_min: int = 5,
                 print_position=False,
                 print_trades=False,
                 position_share_pct: float = 1.0, ):
        portfolio = Portfolio()

        for row in signal_data.itertuples():
            price = data.loc[row.Index, 'Close']
            date = row.Index
            if portfolio.in_position(symbol):
                portfolio.update_position_price(symbol, price, date, trail_percentage)

            # Open position
            if pd.notna(row.buy_marker) and not portfolio.in_position(symbol):
                # Allocate only a percentage of current equity; cap by available cash (no margin).
                target_allocation = portfolio.get_total_value() * max(0.0, min(1.0, position_share_pct))
                invest_amount = min(portfolio.cash, target_allocation)
                if invest_amount > 0:
                    qty = invest_amount / price
                    portfolio.add_position(
                        Position(symbol, qty, price, date, trail_percentage, hold_min, pos_pct=position_share_pct))
            # Close Position
            elif pd.notna(row.sell_marker) and portfolio.in_position(symbol):
                portfolio.remove_position(portfolio.get_position(symbol), date=date)

        if print_position:
            portfolio.print_positions()
        if print_trades:
            portfolio.print_trades()
        return portfolio.profit

    @staticmethod
    def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.01) -> float:
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
    def __init__(self, symbol: str,
                 qty: float,
                 price: float,
                 date: datetime,
                 trail_pct: float,
                 hold_min: int,
                 pos_pct: float = 1.0,
                 ):
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.date = date
        self.cost = qty * price
        self.current_value = qty * price
        self.profit = 0
        self.status = "open"
        self.position_pct = pos_pct
        self.trailing_stop = TrailingStop(ts_pct=trail_pct, hold_min=hold_min)

    def open_position(self):
        pass

    def close_position(self, date: datetime):
        self.status = "closed"
        self.date = date

    def update_price(self, new_price: float, trail_percentage: float = 0):
        self.price = new_price
        self.current_value = self.qty * new_price
        self.profit = self.current_value - self.cost
        if self.trailing_stop:
            self.trailing_stop.update(new_price)

    def update_date(self, date: datetime):
        self.date = date


class Portfolio:
    def __init__(self, cash: int = 1000, transaction_fee: float = 0.004):
        self.trades = []
        self.positions = []
        self.cash = cash
        self.profit = 0
        self.start_cash = cash
        self.transaction_fee = transaction_fee  # max maker fee

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
        If the stop is hit for 'hold_min' consecutive bars, automatically close the position.
        """
        if not position:
            return
        if position.trailing_stop and position.trailing_stop.check(position.price):
            self.remove_position(position, date)


class TrailingStop:
    def __init__(self, ts_pct: int = 5, hold_min: int = 4):
        self.trailing_stop_pct = ts_pct
        self.hold_min = hold_min
        self.level = None
        self.consec_hits = 0
        self.triggered = False

    def __repr__(self):
        level_str = "None" if self.level is None else f"{self.level:.2f}"
        return f"Triggered: {self.triggered}, Level: {level_str}, Consec Hits: {self.consec_hits}"

    def update(self, price: float):
        """
        Update the trailing stop level based on the latest price.
        Only ratchets upward (for long positions).
        """
        candidate = price * (1 - self.trailing_stop_pct / 100.0)
        if self.level is None or candidate > self.level:
            self.level = candidate

    def check(self, price: float) -> bool:
        """
        Check if current price has hit or fallen below the trailing stop level.
        Requires 'hold_min' consecutive hits to trigger.
        Returns True when the stop is triggered for exit.
        """
        if self.level is None:
            return False

        if price <= self.level:
            self.consec_hits += 1
            if self.consec_hits >= self.hold_min:
                self.triggered = True
                return True
        else:
            # Reset consecutive hit counter if price recovers above the stop
            self.consec_hits = 0

        return False


class Strategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.signal_data = pd.DataFrame(index=df.index)

    def calc_signals(self):
        pass


class EMAStrategy(Strategy):
    def __init__(self, df: pd.DataFrame, fast_window: int, slow_window: int):
        super().__init__(df)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def calc_signals(self):
        self.signal_data['ema_fast'] = EMAIndicator(close=self.df["Close"], window=self.fast_window,
                                                    fillna=False).ema_indicator()
        self.signal_data['ema_slow'] = EMAIndicator(close=self.df["Close"], window=self.slow_window,
                                                    fillna=False).ema_indicator()
        # Compute crossover points: +1 when fast crosses above slow (bullish), -1 when below (bearish)

        signal = (self.signal_data['ema_fast'] > self.signal_data['ema_slow']).astype(int)
        cross = signal.diff()
        cross_up = cross == 1
        cross_down = cross == -1

        # Create series for markers positioned at the price level on crossover bars
        self.signal_data['buy_marker'] = self.df["Close"].where(cross_up)
        self.signal_data['sell_marker'] = self.df["Close"].where(cross_down)
        return self.signal_data


def objective(trial):
    max_window = 100
    min_fast_window = 10
    max_fast_window = int(math.floor(max_window * 0.5))
    fast_window = trial.suggest_int('fast_window', min_fast_window, max_fast_window, step=2)
    min_slow_window = int(math.floor((fast_window * 1.4)/2.0)*2)
    slow_window = trial.suggest_int('slow_window', min_slow_window, max_window, step=2)
    trail_pct = trial.suggest_int('trail_pct', 2, 8)
    hold_min = trial.suggest_int('hold_min', 2, 8)
    # trail_pct = 5
    # hold_min = 4
    ema_strategy = EMAStrategy(data, fast_window, slow_window)
    signals = ema_strategy.calc_signals()

    # Rolling window backtesting to prevent overfitting
    num_of_pts = len(data)
    window_min_pts = math.floor(num_of_pts / 4)
    step = math.floor(window_min_pts / 10)
    returns = []

    for start_idx in range(0, num_of_pts - window_min_pts, step):
        end_idx = start_idx + window_min_pts
        profit = mt.backtest(
            signals.iloc[start_idx:end_idx],
            data.iloc[start_idx:end_idx],
            symbol=symbol,
            trail_percentage=trail_pct,
            hold_min=hold_min
        )
        returns.append(profit)

    # Use the static method
    sharpe_ratio = mt.calculate_sharpe_ratio(returns)

    total_profit = mt.backtest(signals, data, symbol, trail_percentage=trail_pct, hold_min=hold_min)

    # if total_profit <= 0:
    #     return -1e10

    return round(sharpe_ratio, 3)
    # return total_profit


def days_min(pts_per_day, num_pts):
    return int(math.floor(num_pts / pts_per_day))


def nearest_4hr(date: datetime):
    hour = date.hour
    floored_hour = (hour // 4) * 4
    return date.replace(hour=floored_hour)


symbol = "BTC-USD"
interval = "4h"

pts_per_day = {"1d": 1, "1h": 24, "4h": 6}
# end_date = datetime(2025, 8, 1)
end_date = nearest_4hr(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))

days = days_min(pts_per_day[interval], 365 * 5)
start_date = end_date - timedelta(days=days)

mt = MiniTrader(symbol, interval, start_date, end_date)
data = mt.get_data()

storage = "sqlite:///ema_optuna.db"  # file-based SQLite

name = f"{symbol}-{interval}-{end_date.strftime('%Y-%m-%d-%H')}"
study = optuna.create_study(direction="maximize",
                            storage=storage,
                            study_name=name,
                            load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=-1)

# pause to let study print
time.sleep(1)

ema_strategy = EMAStrategy(data, study.best_params['fast_window'], study.best_params['slow_window'])
signals = ema_strategy.calc_signals()
profit = mt.backtest(signals, data, symbol,
                     trail_percentage=study.best_params['trail_pct'],
                     hold_min=study.best_params['hold_min'],
                     print_trades=True,
                     print_position=True, )
best_parameters = {**study.best_params,
                   "Best Sharpe Ratio": round(study.best_value, 3),
                   "Total Profit, $": round(profit, 2),
                   "Daily Profit, $": round(profit / days, 2)}

print(f"Start date:  {start_date}")
print(f"End date:    {end_date}")
print(f"Num of Days: {days}")
print(f"Study name:  {name}")
print("Best parameters:")
print(tabulate(best_parameters.items(), headers=["Parameter", "Value"], tablefmt="github"))
# print(tabulate(data.tail(10), headers="keys", tablefmt="github"))
# mt.plot_data(data, signals, symbol, num_of_pts=400)
