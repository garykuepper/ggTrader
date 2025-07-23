import pandas as pd

class Backtester:
    def __init__(self, db, signal_ticker, signal_field="strat_swing_signal", target_field="strat_swing_target", initial_cash=10000):
        self.db = db
        self.signal_ticker = signal_ticker
        self.signal_field = signal_field
        self.target_field = target_field
        self.initial_cash = initial_cash
        self.collection = db["stock_data"]
        self.prices = {}

