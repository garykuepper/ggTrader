from tabulate import tabulate

from trailing_stop import TrailingStop


class Portfolio:

    def __init__(self, cash=1000):
        self.positions = {}
        self.account_value = 0
        self.cash = cash

    def enter_position(self, position):
        """
        Deduct cash for the position's cost and add the position if sufficient cash.
        Returns True if entered, False otherwise.
        """
        cost = position.qty * position.bought_price
        if cost > self.cash + 1e-9:
            return False
        self.cash -= cost
        self.positions[position.symbol] = position
        return True

    def get_position_by_symbol(self, symbol):
        return self.positions.get(symbol, None)

    def position_exists(self, symbol):
        return self.get_position_by_symbol(symbol) is not None

    def exit_position_by_symbol(self, symbol):
        position = self.get_position_by_symbol(symbol)
        if position:
            self.cash += position.current_value
            del self.positions[symbol]
            return True
        return False

    def update_position(self, symbol, qty=None, bought_price=None, date=None, current_price=None):
        """
        Use Position.update_price when changing current price to keep trailing stop in sync.
        """
        position = self.positions.get(symbol)
        if not position:
            return False
        if qty is not None:
            position.qty = qty
        if bought_price is not None:
            position.bought_price = bought_price
            position.current_price = bought_price
        if date is not None:
            position.date = date
        if current_price is not None:
            position.update_price(current_price)
        position.cost = position.qty * position.bought_price
        position.current_value = position.qty * position.current_price
        return True

    def get_all_positions(self):
        return self.positions

    def get_position_value(self, symbol):
        position = self.get_position_by_symbol(symbol)
        return position.current_value if position else 0.0

    def print_all_positions(self):
        rows = []
        for pos in self.positions.values():
            rows.append({
                "Symbol": pos.symbol,
                "Qty": pos.qty,
                "Bought Price": pos.bought_price,
                "Date": pos.date,
                "Current Price": pos.current_price,
                "Cost": pos.cost,
                "Current Value": pos.current_value
            })
        print(tabulate(rows, headers="keys", tablefmt="github"))

    def total_positions_value(self):
        return sum(p.current_value for p in self.positions.values())

    def total_equity(self):
        return self.cash + self.total_positions_value()

    def check_all_stops(self, current_prices):
        """
        Check all positions for stop triggers (e.g., trailing stops).
        current_prices: dict mapping symbol to current price
        """
        for symbol, position in list(self.positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue
            position.update_price(price)
            if position.trailing_stop and position.trailing_stop.is_triggered(price):
                print(f"STOP triggered for {symbol} at {price:.2f} (stop: {position.trailing_stop.stop_price:.2f})")
                self.exit_position_by_symbol(symbol)


class Position:

    def __init__(self, symbol, qty, bought_price, date, trailing_pct=None):
        self.symbol = symbol
        self.qty = qty
        self.bought_price = bought_price
        self.date = date
        self.current_price = bought_price
        self.cost = qty * bought_price
        self.current_value = qty * self.current_price
        self.trailing_stop = TrailingStop(trailing_pct, bought_price) if trailing_pct else None

    def update_price(self, price):
        self.current_price = price
        self.current_value = self.qty * self.current_price
        if self.trailing_stop:
            self.trailing_stop.update(price)

    def __repr__(self):
        return (f"Position(symbol={self.symbol}, qty={self.qty}, "
                f"bought_price={self.bought_price}, date={self.date}, "
                f"current_price={self.current_price}, current_value={self.current_value})")


class StockPosition(Position):
    def __init__(self, symbol, qty, bought_price, date, trailing_pct=None):
        super().__init__(symbol, qty, bought_price, date, trailing_pct)
        # Add stock-specific attributes or methods here


class OptionPosition(Position):
    def __init__(self, symbol, qty, bought_price, date, expiry, strike, option_type, trailing_pct=None):
        super().__init__(symbol, qty, bought_price, date, trailing_pct)
        self.expiry = expiry
        self.strike = strike
        self.option_type = option_type  # 'call' or 'put'
        # Add option-specific logic here


class CryptoPosition(Position):
    def __init__(self, symbol, qty, bought_price, date, trailing_pct=None):
        super().__init__(symbol, qty, bought_price, date, trailing_pct)
        # Add crypto-specific attributes or methods here


class FixedStopPosition(Position):
    def __init__(self, symbol, qty, bought_price, date, stop_price):
        super().__init__(symbol, qty, bought_price, date)
        self.stop_price = stop_price

    def is_stop_triggered(self, current_price):
        return current_price is not None and current_price <= self.stop_price


class TimeStopPosition(Position):
    def __init__(self, symbol, qty, bought_price, date, max_days):
        super().__init__(symbol, qty, bought_price, date)
        self.max_days = max_days
        self.days_held = 0

    def update_days_held(self):
        self.days_held += 1

    def is_stop_triggered(self):
        return self.days_held >= self.max_days
