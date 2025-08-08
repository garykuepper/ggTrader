import pandas as pd
import ta
from tabulate import tabulate

class TradingStrategy:
    def __init__(self, name, params, trailing_pct):
        self.name = name
        self.params = params
        self.trailing_pct = trailing_pct
        self.signal_df = pd.DataFrame()


class EMAStrategy(TradingStrategy):
    def __init__(self, name, params, trailing_pct):
        super().__init__(name, params, trailing_pct)
        self.ema_fast = params['ema_fast']
        self.ema_slow = params['ema_slow']

    def calculate_emas(self, price_series):
        """
        Calculate fast and slow EMAs and return as a DataFrame.
        """
        df = price_series.copy()
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
        return df

    def find_crossovers(self, price_series):
        """
        Build signals:
        1 for bullish crossover (fast crosses above slow),
        -1 for bearish crossover (fast crosses below slow),
        0 otherwise. Avoid signals during EMA warm-up (NaNs).
        """
        df = self.calculate_emas(price_series)
        df['signal'] = 0

        fast = df['ema_fast']
        slow = df['ema_slow']

        bullish = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        bearish = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        # Only consider rows where both EMAs are valid
        valid = fast.notna() & slow.notna()

        df.loc[valid & bullish, 'signal'] = 1
        df.loc[valid & bearish, 'signal'] = -1

        self.signal_df = df

    def print_signals(self):
        df = self.signal_df
        new_df = df[df['signal'] != 0][['close', 'ema_fast', 'ema_slow', 'signal']]
        print(tabulate(new_df, headers='keys', tablefmt='github'))

    def get_latest_signal_row(self):
        df = self.signal_df
        signal_row = df[df['signal'] != 0]
        if not signal_row.empty:
            return signal_row.iloc[-1]
        return None

    def get_latest_signal(self):
        row = self.get_latest_signal_row()
        return row['signal'] if row is not None else 0
