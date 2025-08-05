import pandas as pd
import ta  # Technical Analysis library


class TradingStrategy:
    def __init__(self, capital, short_window=5, long_window=20, trailing_stop_pct=0.05):
        self.capital = capital
        self.position = 0.0
        self.entry_price = None
        self.stop_loss = None
        self.in_position = False

        self.short_window = short_window
        self.long_window = long_window
        self.trailing_stop_pct = trailing_stop_pct
        self.prices = pd.Series(dtype="float64")
        self.highest_price_since_entry = None

    def on_new_data(self, df: pd.DataFrame):
        """
        Accepts a DataFrame with a DatetimeIndex and 'close' price column.
        Iterates over the DataFrame rows, processing each price with its timestamp.
        """
        for timestamp, row in df.iterrows():
            price = row['close']
            self._process_new_price(price, timestamp)

    def _process_new_price(self, price, timestamp):
        self._update_price_history(price, timestamp)
        if len(self.prices) < self.long_window:
            return

        short_ind, long_ind = self._calculate_indicators()

        if not self.in_position and short_ind > long_ind:
            self._enter_trade(price, timestamp)
        elif self.in_position:
            self._manage_open_position(price, short_ind, long_ind, timestamp)

    def _update_price_history(self, price, timestamp):
        new_series = pd.Series([price], index=[timestamp])
        if self.prices.empty:
            self.prices = new_series
        else:
            self.prices = pd.concat([self.prices, new_series])

    def _calculate_indicators(self):
        """
        Must be implemented by subclasses.
        Returns: (short_indicator_value, long_indicator_value)
        """
        raise NotImplementedError

    def _enter_trade(self, price, timestamp):
        size = self.capital / price
        self.enter_position(price, size, timestamp)
        self.highest_price_since_entry = price
        self._update_trailing_stop(price, timestamp)

    def _manage_open_position(self, price, short_ind, long_ind, timestamp):
        self._update_highest_price(price)
        self._update_trailing_stop(price, timestamp)
        if short_ind < long_ind:
            self.exit_position(price, timestamp)
            self.highest_price_since_entry = None
            self.stop_loss = None
        else:
            self.check_stop_loss(price, timestamp)

    def _update_highest_price(self, price):
        if price > self.highest_price_since_entry:
            self.highest_price_since_entry = price

    def _update_trailing_stop(self, price, timestamp):
        new_stop = self.highest_price_since_entry * (1 - self.trailing_stop_pct)
        if self.stop_loss is None or new_stop > self.stop_loss:
            self.set_stop_loss(new_stop, timestamp)

    def enter_position(self, price, size, timestamp):
        self.position = size
        self.entry_price = price
        valued_position = self.position * self.entry_price
        self.in_position = True
        print(f"{timestamp} Entered position: size={size:.4f}, price={price:.2f}, value={valued_position:.2f}")

    def exit_position(self, price, timestamp):
        profit_loss = (price - self.entry_price) * self.position
        self.capital += profit_loss
        print(f"{timestamp} Exited position at price={price:.2f}, P/L={profit_loss:.2f}, New capital={self.capital:.2f}")
        self.position = 0.0
        self.entry_price = None
        self.stop_loss = None
        self.in_position = False

    def set_stop_loss(self, price, timestamp):
        self.stop_loss = price
        print(f"{timestamp} Stop loss set at {price:.2f}")

    def check_stop_loss(self, price, timestamp):
        if self.in_position and price <= self.stop_loss:
            print(f"{timestamp} Stop loss hit at {price:.2f}")
            self.exit_position(price, timestamp)

    def get_position_value(self, current_price=None):
        """
        Returns the current market value of the open position.
        If no position is held, returns 0.
        current_price: If None, uses the last price in self.prices.
        """
        if not self.in_position or self.position == 0:
            return 0.0
        if current_price is None:
            current_price = self.prices.iloc[-1]
        return self.position * current_price

class MovingAverageStrategy(TradingStrategy):
    def _calculate_indicators(self):
        short_ma = ta.trend.SMAIndicator(self.prices, window=self.short_window).sma_indicator().iloc[-1]
        long_ma = ta.trend.SMAIndicator(self.prices, window=self.long_window).sma_indicator().iloc[-1]
        return short_ma, long_ma


class EMA_Strategy(TradingStrategy):
    def _calculate_indicators(self):
        short_ema = ta.trend.EMAIndicator(self.prices, window=self.short_window).ema_indicator().iloc[-1]
        long_ema = ta.trend.EMAIndicator(self.prices, window=self.long_window).ema_indicator().iloc[-1]
        return short_ema, long_ema
