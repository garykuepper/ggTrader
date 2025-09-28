# Python
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, SMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import AverageTrueRange


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
        first_valid = df[['high', 'low', 'close']].dropna().index[0]
        signals = df.loc[first_valid:].copy()
        # EMA signals
        signals['ema_fast'] = EMAIndicator(close=signals['close'], window=self.ema_fast, fillna=False).ema_indicator()
        signals['ema_slow'] = EMAIndicator(close=signals['close'], window=self.ema_slow, fillna=False).ema_indicator()

        # MACD components
        signals['macd'] = MACD(close=signals['close'], window_slow=26, window_fast=12, window_sign=9,
                               fillna=False).macd()

        # Crossover-based signals
        signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
        signals['signal'] = signals['crossover'].diff().fillna(0) / 2

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
