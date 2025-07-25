class Portfolio:
    def __init__(self, cash=10000.0, name="portfolio"):
        self.name = name
        self.positions = {}
        self.cash = cash  # Use initial cash passed
        self.history = []
        self.initial_cash = cash  # Store initial cash for calculations

    # TODO: Add a history of transactions to the portfolio
    #  Add a method to get the history of transactions
    #  Add a method to get the history of positions
    #  Add a method to calc cagr for the portfolio? Would need to store the start and end date

    def get_cash(self):
        return self.cash

    def add_position(self, ticker, quantity, bought_price):
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

    def remove_position(self, ticker, quantity, price):
        if ticker in self.positions:
            pos = self.positions[ticker]
            if quantity >= pos['quantity']:
                del self.positions[ticker]
            else:
                pos['quantity'] -= quantity
                pos['total_value'] = pos['quantity'] * pos['current_price']
            self.cash += quantity * price
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
