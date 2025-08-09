from datetime import datetime, timezone
from tabulate import tabulate
from position import Position


class Portfolio:

    def __init__(self, cash=1000):
        self.cash = cash
        self.positions = []
        self.trades = []
        self.portfolio_value = cash
        self.positions_value = 0

    def add_position(self, position: Position) -> None:
        self.positions.append(position)
        self.cash -= position.value
        self.calc_positions_value()


    def remove_position(self, symbol):
        position = self.get_position(symbol)
        self.cash += position.value
        del position

    def get_position(self, symbol):
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    def update_position_price(self, ticker, price):
        position = self.get_position(ticker)
        position.update_position_value(price)
        print(position.value)
        self.calc_positions_value()
        self.calc_portfolio_value()

    def print_positions(self):
        table = []
        for position in self.positions:
            table.append(position.as_dict())

        print(tabulate(table, headers='keys', tablefmt='github'))

    def calc_positions_value(self):
        self.positions_value = 0
        for position in self.positions:
            self.positions_value += position.value

    def calc_portfolio_value(self):
        self.portfolio_value = self.cash + self.positions_value

    def print_acct(self):

        data = [
            ["Cash", self.cash],
            ["Position Value", self.positions_value],
            ["Portfolio Value", self.portfolio_value]
        ]
        print(tabulate(data))


position1 = Position('AAPL', 2, 74, datetime(2024, 8, 1))
position2 = Position('GOOG', 2, 74, datetime(2024, 9, 1))
port = Portfolio()

port.add_position(position1)
port.add_position(position2)

port.print_positions()
port.print_acct()

port.update_position_price('GOOG', 100)

port.print_positions()
port.print_acct()
