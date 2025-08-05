import pandas as pd
import ta
class TradingStrategy:

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def latest_signal(self):
        pass

    def generate_signals(self, df):
        pass
class EMACrossover(TradingStrategy):

    def __init__(self, name, fast_ema, slow_ema):
        super().__init__(name)
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def update_signals(self, prices):
        pass

    def latest_signal(self):
        pass

    def get_name(self):
        return self.name

    def get_fast_ema(self):
        return self.fast_ema

    def get_slow_ema(self):
        return self.slow_ema

    def set_fast_ema(self, fast_ema):
        self.fast_ema = fast_ema

    def set_slow_ema(self, slow_ema):
        self.slow_ema = slow_ema

    def generate_signals(self, prices):
        pass