
import pandas as pd

class Backtester:
    def __init__(self, signals_df):
        self.signals_df = signals_df
        self.trades = []
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = 0

    def run(self):
        for _, row in self.signals_df.iterrows():
            if row['signal'] == 1 and self.position == 0:
                self.entry_price = row['close']
                self.position = 1
                self.trades.append({'action': 'buy', 'price': self.entry_price, 'date': row['date']})
            elif row['signal'] == -1 and self.position == 1:
                exit_price = row['close']
                self.position = 0
                self.trades.append({'action': 'sell', 'price': exit_price, 'date': row['date']})
        return self.trades

    # Placeholder for future PnL calculation
    def calculate_pnl(self):
        pnl = 0
        for i in range(0, len(self.trades), 2):
            if i+1 < len(self.trades):
                buy = self.trades[i]
                sell = self.trades[i+1]
                pnl += sell['price'] - buy['price']
        return pnl