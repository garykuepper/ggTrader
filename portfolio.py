from datetime import datetime, timezone, timedelta
from tabulate import tabulate
from data_manager import CryptoDataManager

class Portfolio:

    def __init__(self, cash=1000):
        self.cash = cash
        self.positions = []
        self.trades = []
        self.portfolio_value = 0

    def add_position(self, symbol, qty, entry_price, date):
        cost = qty * entry_price
        if not self.has_sufficient_cash(cost):
            print(f"Not enough cash to buy {symbol}. Cash: {self.cash}, Required: {cost}")
            return

        current_value = cost
        self.cash -= cost
        position = {'date': date,
                   'symbol': symbol,
                   'qty': qty,
                   'entry_price': entry_price,
                   'cost': cost,
                   'current_value': current_value}
        self.positions.append(position)
        self.trades.append(position | {'trade_type': 'BUY'})


    def has_sufficient_cash(self, cost):
        # Can be simplified to:
        return cost <= self.cash

    def remove_position(self, ticker):
        position = self.get_position(ticker)
        if position:
            self.cash += position['current_value']
            self.positions = [pos for pos in self.positions if pos['symbol'] != ticker]
            print(f"Removed {position['qty']} {position['symbol']} from portfolio. Cash: {self.cash}")
            self.trades.append(position | {'trade_type': 'SELL'})

    def update_position(self, ticker, qty, current_price, entry_price):
        for position in self.positions:
            if position['symbol'] == ticker:
                position['qty'] = qty
                position['current_value'] = qty * current_price
                position['entry_price'] = entry_price

    def update_position_value(self, ticker, current_price):
        for position in self.positions:
            if position['symbol'] == ticker:
                position['current_value'] = position['qty'] * current_price

    def get_position(self, ticker):
        for position in self.positions:
            if position['symbol'] == ticker:
                return position
        return None

    def get_position_value(self):
        self.portfolio_value = 0
        for position in self.positions:
            self.portfolio_value += position['current_value']
        return self.portfolio_value

    def print_positions(self):
        print(tabulate(self.positions, headers='keys', tablefmt='github'))

    def get_cash(self):
        return self.cash

    def get_total_value(self):
        return self.portfolio_value + self.cash

    def print_trades(self):
        print(tabulate(self.trades, headers='keys', tablefmt='github'))

cm = CryptoDataManager()
porto = Portfolio()
symbol = 'BTCUSDT'
symbol2 = 'ETHUSDT'
porto.add_position(symbol,
                   4.3,
                   36.54,
                   datetime.now(timezone.utc))

porto.add_position(symbol2,
                   5.3,
                   24.4,
                   datetime.now(timezone.utc) - timedelta(days=1)
                   )

porto.add_position('LTCUSDT',
                   1.3,
                   2.4,
                   datetime.now(timezone.utc) - timedelta(days=2)
                   )

porto.print_positions()
print(f"Portfolio value: ${porto.get_position_value()}")

eth_qty = porto.get_position(symbol2)['qty']
eth_entry = porto.get_position(symbol2)['entry_price']
porto.update_position(symbol2, eth_qty, 27.8, eth_entry)

porto.print_positions()
print(f"Portfolio value: ${porto.get_position_value():.2f}")

porto.update_position_value(symbol2, 32.8)
porto.remove_position(symbol2)
porto.print_positions()
print(f"Portfolio value: ${porto.get_position_value():.2f}")
print(f"Total value: ${porto.get_total_value():.2f}")


porto.print_trades()
quote = cm.get_market_day(symbol,datetime(2025,8,1))
close = quote['close'].iloc[0]
print(close)
print(type(close))