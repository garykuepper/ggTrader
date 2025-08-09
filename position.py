class Position:
    def __init__(self, symbol, qty, bought_price, date):
        self.symbol = symbol
        self.qty = qty
        self.bought_price = bought_price
        self.date = date
        self.current_price = bought_price
        self.value = qty * bought_price

    def as_dict(self):
        return {
            'date': self.date,
            'symbol': self.symbol,
            'qty': self.qty,
            'bought_price': self.bought_price,
            'position_value': self.value
        }

    def update_position_value(self, current_price):
        self.current_price = current_price
        self.value = self.qty * current_price