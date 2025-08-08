class TrailingStop:
    def __init__(self, trailing_pct, initial_price):
        self.trailing_pct = trailing_pct
        self.highest_price = initial_price
        self.stop_price = initial_price * (1 - trailing_pct)

    def update(self, current_price):
        if current_price is None:
            return self.stop_price
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_price = self.highest_price * (1 - self.trailing_pct)
        return self.stop_price

    def is_triggered(self, current_price):
        if current_price is None:
            return False
        return current_price <= self.stop_price