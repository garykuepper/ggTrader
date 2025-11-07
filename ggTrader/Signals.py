import numpy as np
import pandas as pd
import pandas_ta as pta
from tabulate import tabulate


class Signals:
    def __init__(self,
                 adx_threshold: int = 25,
                 adx_length: int = 14,
                 atr_multiplier: float = 3.0,
                 ce_high_length: int = 22,
                 ce_low_length: int = 22,
                 atr_length: int = 14,
                 sar_acceleration: float = 0.02,
                 sar_maximum: float = 0.2,):
        self.ohlcv = pd.DataFrame()  # original OHLCV data
        self.adx_threshold = adx_threshold
        self.adx_length = adx_length
        self.atr_multiplier = atr_multiplier
        self.ce_high_length = ce_high_length
        self.ce_low_length = ce_low_length
        self.atr_length = atr_length
        self.sar_acceleration = sar_acceleration
        self.sar_maximum = sar_maximum
        # self.signals = pd.DataFrame()

    def calc_signals(self, df: pd.DataFrame):
        exit_signals = self.exit_signals(df,
                                         atr_multiplier=self.atr_multiplier,
                                         ce_high_length=self.ce_high_length,
                                         ce_low_length=self.ce_low_length,
                                         atr_length=self.atr_length,)

        entry_signals = self.entry_signals(df,
                                           exit_signals['ce_exit'],
                                           adx_threshold=self.adx_threshold,
                                           adx_length=self.adx_length,
                                           sar_maximum=self.sar_maximum,
                                           sar_acceleration=self.sar_acceleration,)

        signals = pd.concat([df, entry_signals, exit_signals], axis=1)
        signals = signals.loc[:, ~signals.columns.duplicated(keep='last')]
        signals = self.filter_signals(signals, entry_signals['entry_signal'], exit_signals['exit_signal'])
        return signals

    @staticmethod
    def entry_signals(df: pd.DataFrame,
                      ce_exit: pd.Series,
                      adx_threshold: int = 25,
                      adx_length: int = 14,
                      sar_maximum: float = 0.2,
                      sar_acceleration: float = 0.02,):
        signals = df.copy()

        # Entry: SAR Signal
        psar = pta.psar(df['high'],
                        df['low'],
                        close=df['close'],
                        af=sar_acceleration,
                        max_af=sar_maximum,
                        )

        signals['sar'] = psar.iloc[:, 0]
        signals['sar_s'] = df['close'] > signals['sar']

        # adx > 25 strong trend
        adx = pta.adx(df['high'],
                      df['low'],
                      df['close'],
                      length=adx_length,)
        signals['adx'] = adx.iloc[:, 0]
        signals['adx_d'] = adx.iloc[:, 1]
        signals['adx_dmp'] = adx.iloc[:, 2]
        signals['adx_dmn'] = adx.iloc[:, 3]

        signals['adx_s'] = np.where(signals['adx'] > adx_threshold, True, False)
        signals['adx_bullish'] = np.where(signals['adx_dmp'] > signals['adx_dmn'], True, False)
        # adx_bullish = adx_dmp > adx_dmn
        signals['adx_s'] = signals['adx_s'] & signals['adx_bullish']

        # Have ADX strength, and sar is a buy and ce is not an exit
        entry_series = signals['adx_s'] & signals['sar_s'] & ~ce_exit
        entry_rise = entry_series & (~entry_series.shift(1, fill_value=False))
        signals['entry_signal'] = entry_rise

        return signals

    @staticmethod
    def exit_signals(df: pd.DataFrame,
                     atr_multiplier: float = 3.0,
                     ce_high_length: int = 22,
                     ce_low_length: int = 22,
                     atr_length: int = 14,
                     ):

        signals = df.copy()
        # Exit: Chandlier Exit uses ATR
        signals['atr'] = pta.atr(df['high'], df['low'], df['close'])
        ce = pta.chandelier_exit(df['high'],
                                 df['low'],
                                 df['close'],
                                 multiplier=atr_multiplier,
                                 high_length=ce_high_length,
                                 low_length=ce_low_length,
                                 atr_length=atr_length)

        signals['ce_l'] = np.where(ce.iloc[:, 2] > 0, ce.iloc[:, 0], np.nan)
        signals['ce_sh'] = np.where(ce.iloc[:, 2] < 0, ce.iloc[:, 1], np.nan)
        signals['ce_exit'] = np.where(ce.iloc[:, 2] == 1, False, True)

        # exit
        exit_series = signals['ce_exit']

        exit_rise = exit_series & (~exit_series.shift(1, fill_value=False))
        signals['exit_signal'] = exit_rise

        return signals

    @staticmethod
    def filter_signals(signals: pd.DataFrame, entry_rise: pd.Series, exit_rise: pd.Series):
        in_pos = False
        filtered_entry = pd.Series(False, index=signals.index)
        filtered_exit = pd.Series(False, index=signals.index)
        in_position = pd.Series(False, index=signals.index)

        for ts in signals.index:
            if not in_pos and entry_rise.loc[ts]:
                filtered_entry.loc[ts] = True
                in_pos = True
            elif in_pos and exit_rise.loc[ts]:
                filtered_exit.loc[ts] = True
                in_pos = False
            in_position.loc[ts] = in_pos
        filtered = pd.DataFrame({
            'filtered_entry': filtered_entry,
            'filtered_exit': filtered_exit,
            'in_position': in_position
        })
        return pd.concat([signals, filtered], axis=1)


if __name__ == "__main__":
    pass
