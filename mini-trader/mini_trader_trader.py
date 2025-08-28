import numpy as np
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
from dataclasses import dataclass


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


def sharpe_ratio(equity_curve: pd.Series, periods_per_year=6 * 365, rf_annual=0.01) -> float:
    """
    Compute Sharpe from equity curve (per-bar equity). Uses log returns.
    periods_per_year: for 4H bars ~ 6/day * 365 = 2190
    """
    if len(equity_curve) < 3:
        return 0.0
    rets = np.log(equity_curve / equity_curve.shift(1)).dropna()
    rf_per_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = rets - rf_per_period
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    return 0.0 if sigma == 0 or np.isnan(sigma) else float((mu / sigma) * np.sqrt(periods_per_year))


# Compute Sharpe from equity curve
def _periods_per_year_from_interval(interval: str) -> int:
    # Handles "4h", "1h", "1d" and similar
    if interval.endswith("h"):
        hours = int(interval[:-1])
        per_day = 24 // max(1, hours)
        return per_day * 365
    if interval.endswith("d"):
        days = int(interval[:-1])
        per_day = 1 // max(1, days) if days > 0 else 1
        return per_day * 365
    # Fallbacks for your common choices
    mapping = {"4h": 6 * 365, "1h": 24 * 365, "1d": 365}
    return mapping.get(interval, 6 * 365)


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
                 share_pct: int = 100,
                 trailing_stop=None,
                 ):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = price
        self.entry_date = date
        self.exit_price = None
        self.exit_date = None
        self.current_price = price
        self.status = "open"
        self.share_pct = share_pct
        # Accept an injected trailing_stop. If none provided, fall back to FixedPctTrailingStop
        if trailing_stop is None:
            self.trailing_stop = FixedPctTrailingStop(ts_pct=trail_pct, hold_min=hold_min)
        else:
            self.trailing_stop = trailing_stop

    @property
    def cost(self) -> float:
        return self.qty * self.entry_price

    @property
    def current_value(self) -> float:
        return self.qty * self.current_price

    @property
    def profit(self) -> float:
        return self.current_value - self.cost

    @property
    def profit_pct(self) -> float:
        return self.profit / self.cost

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
        if self.trailing_stop:
            self.trailing_stop.update(new_price, date)

    def as_dict(self):
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "cost": self.cost,
            "current_value": self.current_value,
            "current_price": self.current_price,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "status": self.status,
            "trailing_triggered": self.trailing_stop.triggered,
        }


class Portfolio:
    def __init__(self, cash: int = 1000, transaction_fee: float = 0.004):
        self.trades: list[Position] = []
        self.positions: list[Position] = []
        self.cash = cash
        self.start_cash = cash
        self.transaction_fee = transaction_fee  # max maker fee
        self.equity_curve = pd.Series(dtype=float)
        self.profit_per_symbol = {}

    def add_position(self, position: Position):
        self.cash -= position.cost * (1 + self.transaction_fee)
        self.positions.append(position)
        self.trades.append(position)

    # TODO: Replace Position.__dict__ snapshots with immutable trade records (dicts with primitives only)
    # TODO: When opening a position, append an entry snapshot with fields:
    # TODO:   {symbol, qty, entry_price, entry_date, exit_price=None, exit_date=None, profit=None, fees, position_pct, trailing_stop_level, trailing_stop_consec_hits, trailing_stop_triggered}
    # TODO: Snapshot trailing-stop primitive fields at entry (do NOT store the TrailingStop object)
    # TODO: Consider using a unique trade_id for each trade so that the entry record can be updated in-place at exit

    def close_position(self, position: Position, date: datetime):
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

    def update_position_price(self, symbol: str, price: float, date: datetime):
        position = self.get_position(symbol)
        if position:
            position.update_price(price, date)
        # Check for trailing stop trigger
        self.check_trailing_stop(position, date)

    @property
    def profit(self):
        return self.get_total_value() - self.start_cash

    @property
    def profit_pct(self):
        return self.profit / self.start_cash

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
            pos.append(position.as_dict())
        print("\nPositions:")
        print(tabulate(pos, headers="keys", tablefmt="github"))

    def print_trades(self):
        trades = []
        for trade in self.trades:
            trades.append(trade.as_dict())
        print("\nTrades:")
        print(tabulate(trades, headers="keys", tablefmt="github"))

    def get_total_value(self):
        total_value = self.cash
        for position in self.positions:
            total_value += position.current_value
        return total_value


    def get_profit_per_symbol(self):
        from collections import defaultdict

        profit_per_symbol = defaultdict(float)

        for trade in self.trades:
            symbol = trade.symbol
            profit_per_symbol[symbol] += trade.profit

        return dict(profit_per_symbol)

    def print_profit_per_symbol(self):
        print("\nProfit per Symbol:")
        profits = self.get_profit_per_symbol()
        if not profits:
            print("  (no trades)")
            return

        table = []
        for symbol,profit in sorted(profits.items(), key=lambda x: x[1], reverse=True):
            table.append([symbol, f"${profit:,.2f}"])
        print(tabulate(table, headers=["Symbol", "Profit"], tablefmt="github"))

    def check_trailing_stop(self, position: Position, date: datetime):
        """
        Check if the current price has hit the trailing stop for the position.
        If the stop is hit for 'hold_min' consecutive bars, automatically close the position.
        """
        if not position:
            return
        if position.trailing_stop and position.trailing_stop.check(position.current_price):
            self.close_position(position, date)

    def record_equity(self, date: datetime):
        """
        Snapshot total equity at the end of a bar.
        """
        ts = pd.Timestamp(date)
        total = self.get_total_value()
        # Ensure monotonic index insertion
        self.equity_curve.loc[ts] = float(total)


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
    def __init__(self, df, atr_window=14, atr_multiplier=3.0, hold_min=4):
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
        ts = pd.Timestamp(date)
        # Guard: if timestamp not in ATR series (tz or missing), skip update
        if ts not in self.atr_values.index:
            # Optionally try to align by normalizing tz or using asof:
            # ts = self.atr_values.index.asof(ts)
            return
        idx = self.atr_values.index.get_loc(ts)
        if idx < self.atr_window:
            return

        atr = self.atr_values.iloc[idx]
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

    def __init__(self,
                 symbols: list,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime,
                 cooldown_period: int = 2,
                 hold_min_periods: int = 2,
                 trail_pct: int = 3,
                 use_atr_trailing_stop: bool = False,
                 atr_window: int = 14,
                 atr_multiplier: float = 3.0, ):

        self.symbols = symbols
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_params = {}
        self.ohlc_data_dict = {}
        self.signal_data_dict = {}
        self.portfolio = Portfolio(cash=1000)
        self.next_entry_time = {sym: datetime.min for sym in self.symbols}
        self.bar_delta = None
        self.cooldown_period = cooldown_period
        self.hold_min = hold_min_periods
        self.trail_pct = trail_pct
        # ATR trailing stop config
        self.use_atr_trailing_stop = use_atr_trailing_stop
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier

    def fetch_ohlc_data(self):
        for symbol in self.symbols:
            self.ohlc_data_dict[symbol] = get_yf_data(symbol, self.interval, self.start_date, self.end_date)
        first_ohlc = next(iter(self.ohlc_data_dict.values()))
        date_index = first_ohlc.index
        self.bar_delta = date_index[1] - date_index[0]

    def calc_signals(self, fast_window=25, slow_window=80):
        for symbol in self.symbols:
            self.signal_data_dict[symbol] = EMAStrategy(self.ohlc_data_dict[symbol], fast_window,
                                                        slow_window).calc_signals()

    def prep_data(self, fast_window=25, slow_window=80):
        self.fetch_ohlc_data()
        self.calc_signals(fast_window, slow_window)

    def run(self):
        # get ohlc and signal data for each symbol

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
                    # If trailing stop auto-closed the position, skip further exit handling for this symbol/date
                    if not self.portfolio.in_position(symbol):
                        # mark cooldown starting from the close timestamp if the position just closed
                        # record when we can re-enter this symbol
                        self.next_entry_time[symbol] = date + (self.bar_delta * self.cooldown_period)
                        continue
                    if self.should_exit(signal):
                        position = self.portfolio.get_position(symbol)
                        if position:
                            self.portfolio.close_position(position, date=date)
                            # set cooldown following the explicit close as well
                            self.next_entry_time[symbol] = date + (self.bar_delta * self.cooldown_period)

                # if not in position check if should enter
                else:
                    if self.should_enter(signal, symbol, date):
                        position_share_pct = 10
                        qty = self.position_sizing(position_share_pct, close)
                        if qty == 0:
                            continue
                        else:
                            # Construct trailing stop according to configuration
                            trailing_stop_obj = None
                            if self.use_atr_trailing_stop:
                                # Create ATRTrailingStop using this symbol's OHLC df and current ATR params
                                df = self.ohlc_data_dict[symbol]
                                trailing_stop_obj = ATRTrailingStop(df,
                                                                    atr_window=self.atr_window,
                                                                    atr_multiplier=self.atr_multiplier,
                                                                    hold_min=self.hold_min)
                            # If not using ATR trailing stop, Position will construct a FixedPctTrailingStop internally
                            self.portfolio.add_position(
                                Position(symbol, qty, close, date, self.trail_pct, self.hold_min, position_share_pct,
                                         trailing_stop=trailing_stop_obj))

            # record equity at the end of the bar
            self.portfolio.record_equity(date)

    def position_sizing(self, position_share_pct, price):
        # position_share_pct expected as percent (0-100)
        pct = max(0.0, min(100.0, position_share_pct)) / 100.0
        target_allocation = self.portfolio.get_total_value() * pct
        # Account for buy fees so we don't over-allocate causing negative cash
        max_affordable = self.portfolio.cash / (1.0 + self.portfolio.transaction_fee)
        invest_amount = min(max_affordable, target_allocation)
        if invest_amount > 0:
            qty = invest_amount / price
            return qty
        return 0

    # Helper: coerce various datetime-like objects to timezone-aware pandas.Timestamp in UTC
    def _to_timestamp_utc(self, obj):
        """
        Convert obj (datetime, pd.Timestamp, etc.) to a timezone-aware pd.Timestamp in UTC.
        If obj is None, returns None.
        """
        if obj is None:
            return None
        ts = pd.Timestamp(obj)
        # Localize naive -> UTC, convert aware -> UTC
        if ts.tz is None:
            try:
                ts = ts.tz_localize('UTC')
            except Exception:
                # In case Timestamp.min or other extreme values raise; fallback to UTC attach
                ts = pd.Timestamp(ts.to_pydatetime()).tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts

    def _sentinel_next_entry(self):
        # Return a safe timezone-aware minimal timestamp sentinel (UTC)
        # Use a small fixed early date rather than pd.Timestamp.min which can be problematic across versions
        return pd.Timestamp("1970-01-01T00:00:00Z")

    def should_exit(self, signal):
        if signal == -1:
            return True
        return False

    def should_enter(self, signal, symbol, date):
        # Only enter on buy signal AND after the cooldown has expired for this symbol
        if signal != 1:
            return False

        # Get next allowed entry time; use timezone-aware sentinel if missing
        raw_next_allowed = self.next_entry_time.get(symbol, None)
        if raw_next_allowed is None:
            next_allowed_ts = self._sentinel_next_entry()
        else:
            next_allowed_ts = self._to_timestamp_utc(raw_next_allowed)

        # If bar_delta is None (not yet computed) or cooldown is zero, allow entry
        if self.bar_delta is None or self.cooldown_period <= 0:
            return True

        # Normalize current bar timestamp to UTC and compare
        date_ts = self._to_timestamp_utc(date)
        if date_ts is None:
            # Defensive: if we couldn't convert, disallow to be safe
            return False

        return date_ts >= next_allowed_ts

    def print_trades(self):
        self.portfolio.print_trades()

    def print_positions(self):
        self.portfolio.print_positions()


def days_min(pts_per_day, num_pts):
    return int(math.floor(num_pts / pts_per_day))


def nearest_4hr(date: datetime):
    hour = date.hour
    floored_hour = (hour // 4) * 4
    return date.replace(hour=floored_hour)


def main():
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'LTC-USD', 'SHIB-USD', 'XLM-USD',
               'LINK-USD']

    # symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD']
    # symbols = ['BTC-USD', 'ETH-USD']
    # symbols = ['BTC-USD']
    end_date = nearest_4hr(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
    start_date = end_date - timedelta(days=180)
    backtest = Backtest(symbols,
                        '4h',
                        start_date,
                        end_date,
                        cooldown_period=4,
                        hold_min_periods=4,
                        trail_pct=4,
                        use_atr_trailing_stop=False,  # Enable ATR trailing stop here
                        atr_window=14,
                        atr_multiplier=3.0)

    backtest.fetch_ohlc_data()
    backtest.calc_signals(fast_window=44, slow_window=82)
    backtest.run()
    backtest.print_trades()
    backtest.print_positions()
    profit = backtest.portfolio.profit
    profit_pct = backtest.portfolio.profit_pct * 100

    ppyear = _periods_per_year_from_interval('4h' if backtest.interval is None else backtest.interval)
    sr = sharpe_ratio(backtest.portfolio.equity_curve.sort_index(), periods_per_year=ppyear, rf_annual=0.01)

    print(f"Start: {start_date}")
    print(f"End:   {end_date}")
    print(f"Total Profit: $ {profit:.2f}")
    print(f"Profit Pct:   % {profit_pct:.2f}")
    print(f"Sharpe Ratio: {sr:.3f}")
    backtest.portfolio.print_profit_per_symbol()

if __name__ == "__main__":
    main()
