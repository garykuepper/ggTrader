class Portfolio:

    def __init__(self, cash=1000):
        self.cash = cash
        self.positions = {}
        self.trades = []
        self.portfolio_value = 0

    def add_position(self, ticker, shares, entry_price):
        pass

    def remove_position(self, ticker):
        pass

    def update_position(self, ticker, shares, current_price, entry_price):
        pass

    def get_position_value(self, ticker):
        pass

    def get_portfolio_value(self):
        self.portfolio_value = self.cash + sum(self.positions.values())

    def get_trades(self):
        return self.trades

    def get_positions(self):
        return self.positions

    def get_cash(self):
        return self.cash

    def get_ticker_shares(self, ticker):
        return self.positions[ticker]





