# Python
import numpy as np
import pandas as pd
import pandas_ta as pta
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

    @staticmethod
    def sign_crossovers(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Detect sign crossovers (neg->pos and pos->neg) in a single Series.
        Returns (cross_up, cross_down, signal), aligned with input index.
        """
        s = series.copy()
        s = s.where(np.isfinite(s))  # treat inf/-inf as NaN
        s = s.fillna(method="ffill")  # avoid false crosses due to NaN gaps

        prev = s.shift(1)
        cross_up = (prev < 0) & (s > 0)
        cross_down = (prev > 0) & (s < 0)
        signal = pd.Series(0, index=s.index, dtype="int8")
        signal[cross_up] = 1
        signal[cross_down] = -1
        return cross_up, cross_down, signal

    @staticmethod
    def crossover(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Detect crossovers between two Series a and b.
        Returns (cross_up, cross_down, signal), where:
          cross_up: a crosses above b
          cross_down: a crosses below b
          signal: +1 on cross_up, -1 on cross_down, else 0
        """
        a, b = a.align(b, join="inner")
        diff = (a - b).astype("float64")
        return sign_crossovers(diff)

if __name__ == "__main__":
    pass
