from Position import Position
from tabulate import tabulate

from datetime import datetime
import pandas as pd


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

    def close_position(self, position: Position, date: datetime):
        self.cash += position.current_value - (position.current_value * self.transaction_fee)
        position.exit_date = date
        position.status = 'closed'
        position.exit_price = position.current_price
        # self.trades.append(position.__dict__.copy())
        self.positions.remove(position)

    def update_position_price(self, symbol: str, price: float, date: datetime):
        position = self.get_position(symbol)
        if position:
            position.update_price(price, date)

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
        for symbol, profit in sorted(profits.items(), key=lambda x: x[1], reverse=True):
            table.append([symbol, f"${profit:,.2f}"])
        print(tabulate(table, headers=["Symbol", "Profit"], tablefmt="github"))

    def record_equity(self, date: datetime):
        """
        Snapshot total equity at the end of a bar.
        """
        ts = pd.Timestamp(date)
        total = self.get_total_value()
        # Ensure monotonic index insertion
        self.equity_curve.loc[ts] = float(total)
