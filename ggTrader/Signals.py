# Python
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


class Signals:
    def __init__(self):
        pass

class EMASignals:
    def __init__(self, ema_fast: int = 5, ema_slow: int = 20, atr_multiplier: float = 1.0, atr_window: int = 14):
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.atr_multiplier = float(atr_multiplier)
        self.atr_window = int(atr_window)

    def _build_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame()
        signals['close'] = df['close'].copy()

        # EMA signals
        signals['ema_fast'] = EMAIndicator(close=df['close'], window=self.ema_fast, fillna=False).ema_indicator()
        signals['ema_slow'] = EMAIndicator(close=df['close'], window=self.ema_slow, fillna=False).ema_indicator()

        # MACD components
        signals['macd'] = signals['ema_fast'] - signals['ema_slow']
        signals['macd_signal'] = signals['macd'].ewm(span=9, adjust=False).mean()
        signals['macd_cross'] = signals['macd'] - signals['macd_signal']

        # Higher-level EMA
        signals['ema_superslow'] = EMAIndicator(close=df['close'], window=self.ema_slow * 2, fillna=False).ema_indicator()

        # Crossover-based signals
        signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
        signals['signal'] = signals['crossover'].diff().fillna(0) / 2

        # ATR-based level
        signals['atr'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=self.atr_window, fillna=False
        ).average_true_range()
        signals.loc[signals['atr'] == 0, 'atr'] = np.nan

        # ATR-based exit level
        signals['atr_sell'] = df['close'] - signals['atr'] * self.atr_multiplier
        signals['atr_sell'] = signals['atr_sell'].shift(1)
        signals['atr_sell_signal'] = df['close'] < signals['atr_sell']

        return signals

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signals for the given OHLCV DataFrame.
        Expects df with at least columns: close, high, low.
        Returns a DataFrame with the same columns as in the original calc_signals.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        return self._build_signals(df)