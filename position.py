from datetime import datetime

class Position:
    def __init__(self, symbol:str, qty:float, bought_price:float, date: datetime):
        self.symbol = symbol
        self.qty = qty
        self.bought_price = bought_price
        self.bought_date = date
        self.current_price = bought_price
        self.value = qty * bought_price
        self.cost = qty * bought_price
        self.sold_date = None
        self.status = 'open'  # 'open' or 'closed'
        self.profit_loss = 0


    def update_position_value(self, current_price: float):
        self.current_price = current_price
        self.value = self.qty * current_price
        self.profit_loss = self.value - self.cost