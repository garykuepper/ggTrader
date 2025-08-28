import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta, timezone
import time
import math
import optuna
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from tabulate import tabulate


def get_yf_data(symbol: str, interval: str, start_date: datetime, end_date: datetime):
    return yf.download(symbol,
                       interval=interval,
                       start=start_date,
                       end=end_date,
                       multi_level_index=False,
                       auto_adjust=True)


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


def backtest(signal_data: pd.DataFrame,
             data: pd.DataFrame,
             symbol: str,
             trail_percentage: float = 3,
             hold_min: int = 5,
             print_position=False,
             print_trades=False,
             position_share_pct: float = 1.0,
             starting_cash: int = 1000, ):
    portfolio = Portfolio(cash=starting_cash)

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


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.01) -> float:
    """
    Calculate Sharpe ratio from a list of returns.

    Args:
        returns: List of profit/return pct values
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
                 share_pct: float = 1.0,
                 ):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = price
        self.entry_date = date
        self.exit_price = None
        self.exit_date = None
        self.current_price = price
        self.current_value = qty * price
        self.cost = qty * price
        self.profit = 0
        self.profit_pct = self.profit / self.cost
        self.status = "open"
        self.share_pct = share_pct
        self.trailing_stop = FixedPctTrailingStop(ts_pct=trail_pct, hold_min=hold_min)

    # TODO: Add explicit immutable entry fields
    # TODO: Add Position.entry_date and Position.entry_price on creation to make trade snapshots unambiguous
    # TODO: Consider adding a unique trade_id here for reliable entry↔exit matching in trade records

    def open_position(self):
        pass

    def close_position(self, date: datetime):
        self.status = "closed"
        self.exit_date = date

    def update_price(self, new_price: float, date: datetime = None):
        self.current_price = new_price
        self.current_value = self.qty * new_price
        self.profit = self.current_value - self.cost
        self.profit_pct = self.profit / self.cost
        if self.trailing_stop:
            self.trailing_stop.update(new_price, date)


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
        self.trades.append(position)

    # TODO: Replace Position.__dict__ snapshots with immutable trade records (dicts with primitives only)
    # TODO: When opening a position, append an entry snapshot with fields:
    # TODO:   {symbol, qty, entry_price, entry_date, exit_price=None, exit_date=None, profit=None, fees, position_pct, trailing_stop_level, trailing_stop_consec_hits, trailing_stop_triggered}
    # TODO: Snapshot trailing-stop primitive fields at entry (do NOT store the TrailingStop object)
    # TODO: Consider using a unique trade_id for each trade so that the entry record can be updated in-place at exit

    def remove_position(self, position: Position, date: datetime):
        self.cash += position.current_value - (position.current_value * self.transaction_fee)
        position.exit_date = date
        position.status = 'closed'
        position.exit_price = position.current_price
        # self.trades.append(position.__dict__.copy())
        self.positions.remove(position)

    # TODO: When closing a position, UPDATE the corresponding entry snapshot in self.trades (preferred) OR append an exit snapshot that contains only primitive fields
    # TODO: Exit snapshot must include: exit_price, exit_date, profit, fees (buy_fee, sell_fee), net_profit, trailing_stop_level_at_exit, trailing_stop_consec_hits_at_exit, trailing_stop_triggered_at_exit
    # TODO: Do NOT append live objects (e.g., TrailingStop) to trades — snapshot primitive values instead
    # TODO: Add unit tests to ensure trade snapshots remain immutable after TrailingStop updates

    def update_position_price(self, symbol: str, price: float, date: datetime, trail_percentage: float = 0):
        position = self.get_position(symbol)
        if position:
            position.update_price(price, date)
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
            trades.append(trade.__dict__)
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
        if position.trailing_stop and position.trailing_stop.check(position.current_price):
            self.remove_position(position, date)


class TrailingStop:
    def __init__(self, hold_min: int = 4):
        self.level = None
        self.consec_hits = 0
        self.triggered = False
        self.hold_min = hold_min
        self.date = None

    def __repr__(self):
        level_str = "None" if self.level is None else f"{self.level:.2f}"
        return f"Triggered: {self.triggered}, Level: {level_str}, Consec Hits: {self.consec_hits}"

    def update(self, price: float, date: datetime):
        """
        Base update method to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def check(self, price: float) -> bool:
        if self.level is None:
            return False

        if price <= self.level:
            self.consec_hits += 1
            if self.consec_hits >= self.hold_min:
                self.triggered = True
                return True
        else:
            self.consec_hits = 0

        return False


class FixedPctTrailingStop(TrailingStop):
    def __init__(self, ts_pct: float = 5, hold_min: int = 4):
        super().__init__(hold_min=hold_min)
        self.trailing_stop_pct = ts_pct

    def update(self, price: float, date: datetime):
        candidate = price * (1 - self.trailing_stop_pct / 100.0)
        if self.level is None or candidate > self.level:
            self.level = candidate
            self.date = date


class ATRTrailingStop(TrailingStop):
    def __init__(self, df, atr_window=14, atr_multiplier=3, hold_min=4):
        super().__init__(hold_min=hold_min)
        self.df = df
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.atr_values = self._calculate_atr()

    def _calculate_atr(self):
        atr_indicator = AverageTrueRange(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=self.atr_window
        )
        return atr_indicator.average_true_range()

    def update(self, price: float, date: datetime):
        if self.atr_values.index.get_loc(pd.Timestamp(date)) < self.atr_window:
            return

        atr = self.atr_values.loc[date]
        close_price = price

        candidate = close_price - (atr * self.atr_multiplier)

        if self.level is None or candidate > self.level:
            self.level = candidate
            self.date = date


class Strategy:
    def __init__(self, ohlc_data: pd.DataFrame):
        self.ohlc_data = ohlc_data
        self.signal_data = pd.DataFrame(index=ohlc_data.index)

    def calc_signals(self):
        pass


class EMAStrategy(Strategy):
    def __init__(self, ohlc_data: pd.DataFrame, fast_window: int, slow_window: int):
        super().__init__(ohlc_data)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def calc_signals(self):
        self.signal_data['ema_fast'] = EMAIndicator(close=self.ohlc_data["Close"], window=self.fast_window,
                                                    fillna=False).ema_indicator()
        self.signal_data['ema_slow'] = EMAIndicator(close=self.ohlc_data["Close"], window=self.slow_window,
                                                    fillna=False).ema_indicator()
        # Compute crossover points: +1 when fast crosses above slow (bullish), -1 when below (bearish)

        signal = (self.signal_data['ema_fast'] > self.signal_data['ema_slow']).astype(int)
        cross = signal.diff()
        cross_up = cross == 1
        cross_down = cross == -1

        # Create series for markers positioned at the price level on crossover bars
        self.signal_data['buy_marker'] = self.ohlc_data["Close"].where(cross_up)
        self.signal_data['sell_marker'] = self.ohlc_data["Close"].where(cross_down)
        self.signal_data['signal'] = cross.where(cross.isin([1, -1]), 0)

        return self.signal_data


from dataclasses import dataclass


@dataclass
class TickerParameters:
    symbol: str
    interval: str = "1d"
    cooldown_period: int = 2
    hold_min_periods: int = 2
    ema_fast_window: int = 12
    ema_slow_window: int = 26
    trailing_stop_pct: int = 3
    position_share_pct: float = .1


class Backtest:

    def __init__(self, symbols: list, interval: str, start_date: datetime, end_date: datetime):
        self.symbols = symbols
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_params = {}
        self.ohlc_data_dict = {}
        self.signal_data_dict = {}
        self.portfolio = Portfolio(cash=1000)

    def fetch_ohlc_data(self):
        for symbol in self.symbols:
            self.ohlc_data_dict[symbol] = get_yf_data(symbol, self.interval, self.start_date, self.end_date)

    def calc_signals(self):
        for symbol in self.symbols:
            self.signal_data_dict[symbol] = EMAStrategy(self.ohlc_data_dict[symbol], 20, 50).calc_signals()

    def run(self):
        # get ohlc and signal data for each symbol
        self.fetch_ohlc_data()
        self.calc_signals()
        # loop through each date in the ohlc data
        first_ohlc = next(iter(self.ohlc_data_dict.values()))
        date_index = first_ohlc.index
        for date in date_index:
            for symbol in self.symbols:
                # check if in position
                signal = self.signal_data_dict[symbol].loc[date, 'signal']
                close = self.ohlc_data_dict[symbol].loc[date, 'Close']

                if self.portfolio.in_position(symbol):

                    self.portfolio.update_position_price(symbol, close, date)  # Also checks trailing_stop
                    if self.should_exit(signal):
                        self.portfolio.remove_position(self.portfolio.get_position(symbol), date=date)

                # if not in position check if should enter
                else:
                    if self.should_enter(signal):
                        position_share_pct = .2
                        qty = self.position_sizing(position_share_pct, close)
                        if qty == 0:
                            continue
                        else:
                            self.portfolio.add_position(Position(symbol, qty, close, date, 3, 3,position_share_pct))

    def position_sizing(self, position_share_pct, price):
        target_allocation = self.portfolio.get_total_value() * max(0.0, min(1.0, position_share_pct))
        invest_amount = min(self.portfolio.cash, target_allocation)
        if invest_amount > 0:
            qty = invest_amount / price
            return qty
        return 0

    @staticmethod
    def should_exit(signal):
        # TODO: Trailing stop triggers when price is updated.  Maybe I should make this more clear?
        if signal == -1:
            return True
        return False

    @staticmethod
    def should_enter(signal):
        # TODO: Add re-entry logic.  if today - exit date > cooldown and if still buy --> reenter
        # exit date --> find index integer,  then cooldown would be the number of periods determined by the interval
        if signal == 1:
            return True
        return False

    def print_trades(self):
        self.portfolio.print_trades()

    def reentry(self):
        pass


symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'ADA-USD', 'SOL-USD','XRP-USD']
# symbols = ['BTC-USD', 'ETH-USD']
end_date = datetime(2025, 8, 1)
start_date = end_date - timedelta(days=30)
backtest = Backtest(symbols, '4h', start_date, end_date)
backtest.run()
backtest.print_trades()
profit = backtest.portfolio.profit
profit_pct = profit / backtest.portfolio.start_cash  * 100
print(f"Start: {start_date}")
print(f"End:   {end_date}")
print(f"Total Profit: $ {profit:.2f}")
print(f"Profit Pct:   % {profit_pct:.2f}")
