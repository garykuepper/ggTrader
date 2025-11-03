# Python
import numpy as np
import pandas as pd
import pandas_ta as pta
from tabulate import tabulate


class Signals:
    def __init__(self):
        self.ohlcv = pd.DataFrame()  # original OHLCV data

    def calc_signals(self, df: pd.DataFrame):
        signals = self.entry_signals(df)
        return signals


    def entry_signals(self, df: pd.DataFrame, adx_threshold: int = 25):
        signals = df.copy()


        # Entry: SAR Signal
        psar = pta.psar(df['high'],
                        df['low'],
                        close=df['close'])

        signals['sar'] = psar.iloc[:, 0]
        signals['sar_s'] = signals['close'] > signals['sar']

        # adx > 25 strong trend
        adx = pta.adx(df['high'], df['low'], df['close'])
        signals['adx'] = adx.iloc[:, 0]
        signals['adx_d'] = adx.iloc[:, 1]
        signals['adx_dmp'] = adx.iloc[:, 2]
        signals['adx_dmn'] = adx.iloc[:, 3]

        signals['adx_s'] = np.where(signals['adx'] > adx_threshold, True, False)
        signals['adx_bullish'] = np.where(signals['adx_dmp'] > signals['adx_dmn'], True, False)
        # adx_bullish = adx_dmp > adx_dmn
        signals['adx_s'] = signals['adx_s'] & signals['adx_bullish']

        # Have ADX strength, and sar is a buy and ce is not an exit
        entry_series = signals['adx_s'] & signals['sar_s'] & ~signals['ce_exit']
        entry_rise = entry_series & (~entry_series.shift(1, fill_value=False))
        signals['entry_signal'] = entry_rise

        return signals

    def exit_signals(self):
        pass

    @staticmethod
    def filter_signals(signals: pd.DataFrame, entry_rise: pd.Series, exit_rise: pd.Series):

        in_pos = False
        filtered_entry = pd.Series(False, index=signals.index)
        filtered_exit = pd.Series(False, index=signals.index)

        for ts in signals.index:
            if not in_pos and entry_rise.loc[ts]:
                filtered_entry.loc[ts] = True
                in_pos = True
            elif in_pos and exit_rise.loc[ts]:
                filtered_exit.loc[ts] = True
                in_pos = False

        return pd.DataFrame({'entry': filtered_entry, 'exit': filtered_exit})

if __name__ == "__main__":
    pass
