# Python
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, SMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import AverageTrueRange
from tabulate import tabulate


class Signals:
    def __init__(self,
                 ema_fast: int = 20,
                 ema_slow: int = 50,
                 atr_multiplier: float = 2.0,
                 atr_window: int = 14):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_multiplier = atr_multiplier
        self.atr_window = atr_window
        self.signals = pd.DataFrame()
        self.ohlcv = pd.DataFrame()  # original OHLCV data

    # TODO: Use https://github.com/TA-Lib/ta-lib-python library

    # TODO: Class Signals, holds cache of all signals in a long form
    # Details: signals are stored in a long form DataFrame with columns:
    #  - date
    #  - symbol
    #  - close
    #  - high
    #  - low
    #  - quote
    #  - interval
    #  - signal (1=buy, -1=sell, 0=hold)
    #  - SAR
    #  - ATR
    #  - stop-loss

    @classmethod
    def generate_fake_data(cls,
                           rows: int = 40,
                           seed: int = None,
                           start: float = 100.0,
                           drift: float = 0.5,
                           vol: float = 2.0) -> pd.DataFrame:
        """
        Generate a fake OHLCV DataFrame for testing.
        - rows: number of rows to generate
        - seed: random seed for reproducibility
        - start: starting price
        - drift: expected price drift per step
        - vol: volatility multiplier for random walk
        Returns a DataFrame with columns: close, high, low
        """
        if seed is not None:
            np.random.seed(seed)

        close = []
        high = []
        low = []
        price = start

        # Optional: include a date index (daily frequency) starting today
        ts = pd.Timestamp.today(tz='UTC').round("D")
        dates = pd.date_range(end=ts, periods=rows, freq='D')

        for i in range(rows):
            # simple stochastic process: price += drift + noise
            price = price + drift + float(np.random.randn()) * vol
            c = price
            h = c + abs(float(np.random.randn())) * vol * 0.8 + 0.5
            l = c - abs(float(np.random.randn())) * vol * 0.8 - 0.5
            close.append(round(c, 2))
            high.append(round(h, 2))
            low.append(round(l, 2))

        df = pd.DataFrame({'close': close, 'high': high, 'low': low})
        df.index = dates
        return df

    def get_signal_by_date(self, date: pd.Timestamp):
        return self.signals.loc[date]


if __name__ == "__main__":
    signals = Signals()
    df = signals.generate_fake_data(200)
    signals.compute(df)
    print(tabulate(signals.signals.tail(), headers='keys', tablefmt='github'))
