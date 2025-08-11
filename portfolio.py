from datetime import datetime, timezone
from tabulate import tabulate
from position import Position


class Portfolio:

    def __init__(self, cash=1000):
        self.cash = cash
        self.positions = []
        self.trades = []
        self.total_value = cash
        self.positions_value = 0

    def deposit_cash(self, amount: float) -> None:
        self.cash += amount

    def withdraw_cash(self, amount: float) -> None:
        self.cash -= amount

    def open_position(self, position: Position) -> None:
        if not self.check_sufficient_cash(position.cost):
            print("Insufficient cash")
            return
        self.positions.append(position)
        self.cash -= position.value
        self.calc_positions_value()
        self.trades.append(position)

    def check_sufficient_cash(self, amount: float) -> bool:
        return self.cash >= amount

    def close_position(self, symbol: str, date: datetime) -> None:
        position = self.get_position(symbol)
        if position is None:
            print("No position")
            return
        position.status = 'closed'
        position.sold_date = date
        self.cash += position.value
        self.positions = [p for p in self.positions if p.symbol != symbol]
        self.calc_positions_value()

    def get_position(self, symbol: str) -> Position:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    def has_position(self, symbol: str) -> bool:
        if self.get_position(symbol) is None:
            return False
        return True

    def update_position_price(self, ticker: str, price: float) -> None:
        position = self.get_position(ticker)
        position.update_position_value(price)
        self.calc_positions_value()
        self.calc_portfolio_value()

    def print_positions(self):
        table = []
        for position in self.positions:
            table.append(position.__dict__)
        print("\nPositions:")
        print(tabulate(table, headers='keys', tablefmt='github'))

    def print_trade_history(self):
        table = []
        for trade in self.trades:
            table.append(trade.__dict__)
        print("\nTrade History:")
        print(tabulate(table, headers='keys', tablefmt='github'))

    def calc_positions_value(self):
        self.positions_value = 0
        for position in self.positions:
            self.positions_value += position.value

    def calc_portfolio_value(self):
        self.total_value = self.cash + self.positions_value

    def print_acct(self):

        data = [
            ["Cash", self.cash],
            ["Position Value", self.positions_value],
            ["Total Value", self.total_value]
        ]
        print("\nAccount Summary:")
        print(tabulate(data))

# position1 = Position('AAPL', 2, 74,
#                      datetime(2024, 8, 1).replace(tzinfo=timezone.utc))
# position2 = Position('GOOG', 3.1, 24,
#                      datetime(2024, 9, 1).replace(tzinfo=timezone.utc))
# position3 = Position('MSFT', 2.45, 34,
#                      datetime(2024, 10, 1).replace(tzinfo=timezone.utc))
# port = Portfolio()
#
# port.open_position(position1)
# port.open_position(position2)
# port.open_position(position3)
#
# port.print_positions()
# port.print_acct()
#
# port.update_position_price('GOOG', 100)
#
# port.print_positions()
# port.print_acct()
#
# port.close_position('GOOG')
#
# port.print_positions()
# port.print_acct()
#
# port.print_trade_history()
