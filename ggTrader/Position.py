from datetime import datetime


class Position:
    def __init__(self, symbol: str,
                 qty: float,
                 price: float,
                 date: datetime,
                 trail_pct: float = 0.0,
                 hold_min: int = 3,
                 share_pct: int = 100,

                 ):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = price
        self.entry_fee = 0.0
        self.entry_date = date
        self.exit_price = None
        self.exit_date = None
        self.exit_fee = 0.0
        self.current_price = price
        self.status = "open"
        self.share_pct = share_pct
        self.stop_loss_triggered = False

    @property
    def cost(self) -> float:
        return self.qty * self.entry_price

    @property
    def current_value(self) -> float:
        return self.qty * self.current_price

    @property
    def profit(self) -> float:
        return self.current_value - self.cost

    @property
    def profit_pct(self) -> float:
        return self.profit / self.cost

    def open_position(self):
        pass

    def close_position(self, date: datetime):
        self.status = "closed"
        self.exit_date = date

    def update_price(self, new_price: float, date: datetime = None):
        self.current_price = new_price

    def as_dict(self):
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "cost": self.cost,
            "current_value": self.current_value,
            "current_price": self.current_price,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "status": self.status,
            "stop_loss_triggered": self.stop_loss_triggered,
        }
