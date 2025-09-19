from ggTrader.Position import Position
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

        # NEW: track realized profit separately from unrealized
        self.realized_profit: float = 0.0

    def add_position(self, position: Position):
        self.cash -= position.cost * (1 + self.transaction_fee)
        self.positions.append(position)
        self.trades.append(position)

    def close_position(self, position: Position, date: datetime):
        # LOCK IN REALIZED PROFIT (accounting for fees)
        exit_value = position.current_value
        transaction_cost = exit_value * self.transaction_fee
        self.realized_profit += position.profit - transaction_cost

        self.cash += exit_value - transaction_cost
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
        # DEV: expose total profit as realized + unrealized
        return self.total_value - self.start_cash

    @property
    def profit_pct(self):
        return self.profit / self.start_cash

    # NEW: unrealized gain/loss for open positions
    @property
    def unrealized_profit(self) -> float:
        total = 0.0
        for position in self.positions:
            total += position.profit
        return total

    # NEW: realized + unrealized


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
        print(tabulate(pos, headers="keys", tablefmt="github", showindex=True))

    def print_trades(self):
        trades = []
        for trade in self.trades:
            trades.append(trade.as_dict())
        print("\nTrades:")
        print(tabulate(trades, headers="keys", tablefmt="github", showindex=True))

    @property
    def total_value(self):
        return self.cash + self.total_position_value

    @property
    def total_position_value(self):
        total = 0.0
        for position in self.positions:
            total += position.current_value
        return total

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
        total = self.total_value
        # Ensure monotonic index insertion
        self.equity_curve.loc[ts] = float(total)

    def print_stats(self):
        print("\nStats:")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Total Position Value: ${self.total_position_value:,.2f}")
        print(f"Total Value: ${self.total_value:,.2f}")
        print(f"Realized Profit: ${self.realized_profit:,.2f}")
        print(f"Unrealized Profit: ${self.unrealized_profit:,.2f}")
        print(f"Total Profit: ${self.profit:,.2f}")
        print(f"Profit Pct: {self.profit_pct * 100:.2f}%")