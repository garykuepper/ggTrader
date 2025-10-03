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

    def _build_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame()
        # signals = df.copy()
        # remove all NaN rows
        first_valid = df[['high', 'low', 'close']].dropna().index[0]
        # calc signals at first valid index
        signals = df.loc[first_valid:].copy()
        # EMA signals
        signals['ema_fast'] = EMAIndicator(close=signals['close'], window=self.ema_fast, fillna=False).ema_indicator()
        signals['ema_slow'] = EMAIndicator(close=signals['close'], window=self.ema_slow, fillna=False).ema_indicator()

        # MACD components
        signals['macd'] = MACD(close=signals['close'], window_slow=26, window_fast=12, window_sign=9,
                               fillna=False).macd()

        # Crossover-based signals
        signals['ema_crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
        signals['signal'] = signals['ema_crossover'].diff().fillna(0) / 2

        # ATR-based level
        signals['atr'] = AverageTrueRange(
            high=signals['high'],
            low=signals['low'],
            close=signals['close'],
            window=self.atr_window,
            fillna=False
        ).average_true_range()
        # Replace NaNs with 0 to avoid issues with division later on
        signals.loc[signals['atr'] == 0, 'atr'] = np.nan

        # ATR-based exit level
        signals['atr_sell'] = signals['close'] - signals['atr'] * self.atr_multiplier
        signals['atr_sell'] = signals['atr_sell'].shift(1)
        signals['atr_sell_signal'] = signals['close'] < signals['atr_sell']

        # PSAR
        signals['psar'] = PSARIndicator(high=signals['high'],
                                        low=signals['low'],
                                        close=signals['close'],
                                        step=0.02,
                                        max_step=0.20).psar()

        # ADX
        signals['adx'] = ADXIndicator(high=signals['high'],
                                      low=signals['low'],
                                      close=signals['close'],
                                      window=14,
                                      fillna=False).adx()

        # reindex signals to match original df data
        tmp = signals.reindex(df.index)
        first_valid = tmp.first_valid_index()

        if first_valid is not None:
            tmp = tmp.infer_objects(copy=False)
            tmp.loc[first_valid:] = tmp.loc[first_valid:].interpolate()
        signals = tmp

        return signals

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signals for the given OHLCV DataFrame.
        Expects df with at least columns: close, high, low.
        Returns a DataFrame with the same columns as in the original calc_signals.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        self.signals = self._build_signals(df)
        return self.signals

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


if __name__ == "__main__":
    signals = Signals()
    df = signals.generate_fake_data(200)
    signals.compute(df)
    print(tabulate(signals.signals.tail(), headers='keys', tablefmt='github'))