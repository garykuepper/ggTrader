# swing_trader/strategy/strategy.py

class Strategy:
    def __init__(self, symbol):
        self.symbol = symbol

    def generate_signals(self):
        raise NotImplementedError("Subclasses must implement this method.")