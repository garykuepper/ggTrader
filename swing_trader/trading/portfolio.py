
from datetime import datetime

class Portfolio:
    def __init__(self, cash=10000.0, name="portfolio"):
        self.name = name
        self.positions = {}
        self.cash = cash  # Use initial cash passed
        self.history = []
        self.initial_cash = cash  # Store initial cash for calculations
        self.start_date = None
        self.end_date = None

    # TODO: Add a history of transactions to the portfolio
    #  Add a method to get the history of transactions
    #  Add a method to get the history of positions
    #  Add a method to calc cagr for the portfolio? Would need to store the start and end date

    def add_position(self, ticker, quantity, bought_price, date=None):

        if self.start_date is None and date is not None:
            self.start_date = date
        if date is not None:
            self.end_date = date

        if ticker in self.positions:
            pos = self.positions[ticker]
            old_qty = pos['quantity']
            old_cost = pos['cost_basis'] * old_qty
            new_qty = old_qty + quantity
            if new_qty == 0:
                # Position fully closed
                del self.positions[ticker]
            else:
                # Calculate new weighted average cost basis only if increasing position
                if quantity > 0:
                    pos['cost_basis'] = (old_cost + quantity * bought_price) / new_qty
                pos['quantity'] = new_qty
                pos['current_price'] = bought_price
                pos['total_value'] = pos['quantity'] * pos['current_price']
        else:
            if quantity > 0:
                self.positions[ticker] = {
                    'quantity': quantity,
                    'cost_basis': bought_price,
                    'current_price': bought_price,
                    'total_value': quantity * bought_price
                }
            else:
                raise ValueError("Cannot add negative quantity for new position")

        self.cash -= quantity * bought_price
        self.record_transaction("BUY", ticker, quantity, bought_price, date)

    def remove_position(self, ticker, quantity, price, date=None):
        if ticker in self.positions:
            pos = self.positions[ticker]
            if quantity >= pos['quantity']:
                del self.positions[ticker]
            else:
                pos['quantity'] -= quantity
                pos['total_value'] = pos['quantity'] * pos['current_price']
            self.cash += quantity * price
            self.record_transaction("SELL", ticker, quantity, price, date)
        else:
            raise ValueError(f"Position for {ticker} does not exist.")

    def update_prices(self, ticker, price):
        if ticker in self.positions:
            pos = self.positions[ticker]
            pos['current_price'] = price
            pos['total_value'] = pos['quantity'] * price
        else:
            # raise ValueError(f"Position for {ticker} does not exist.")
            print(f"Position for {ticker} does not exist.")

    def total_portfolio_value(self):
        positions_value = sum(pos['total_value'] for pos in self.positions.values())
        return self.cash + positions_value

    def get_profit_loss(self):
        total_value = self.total_portfolio_value()
        profit_loss = total_value - self.initial_cash
        return profit_loss

    def get_profit_loss_percentage(self):
        profit_loss = self.get_profit_loss()
        if self.initial_cash == 0:
            return 0.0
        return (profit_loss / self.initial_cash) * 100

    def get_cash(self):
        return self.cash

    def get_initial_cash(self):
        return self.initial_cash

    def get_positions(self):
        return self.positions

    def update_all_prices(self, price_lookup: dict):
        for ticker, price in price_lookup.items():
            self.update_prices(ticker, price)


    def get_history(self):
        return self.history

    def __str__(self):
        return f"Portfolio(name={self.name}, cash={self.cash}, positions={self.positions})"

    def __repr__(self):
        return self.__str__()

    def record_transaction(self, action, ticker, quantity, price, date=None):
        self.history.append({
            "action": action,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "cash_after": self.cash,
            "date": date
        })

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date


    def get_cagr(self):
        if not self.start_date or not self.end_date:
            return None

        # Convert strings to datetime objects
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        if isinstance(self.end_date, str):
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        delta_days = (self.end_date - self.start_date).days
        years = delta_days / 365.25

        if self.initial_cash <= 0 or years <= 0:
            return None

        end_value = self.total_portfolio_value()
        return (end_value / self.initial_cash) ** (1 / years) - 1

