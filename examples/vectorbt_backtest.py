import pandas as pd
import vectorbt as vbt
import numpy as np
from numba import njit
from utils.KrakenHistoricalData import KrakenHistoricalData
from vectorbt.portfolio import nb

# ... existing code ...
k = KrakenHistoricalData()

symbols = ["BTC"]
interval = "4h"
end = pd.to_datetime("2025-09-30").tz_localize('UTC')
start = end - pd.Timedelta(days=30 * 6)

# ... existing code ...

@njit
def atr_trailing_sl_nb(c, atr_arr, atr_mult):
    # Only apply when in a long position
    if c.position_now <= 0:
        return c.curr_stop, c.curr_trail

    atr_now = atr_arr[c.i]
    if np.isnan(atr_now):
        return c.curr_stop, c.curr_trail

    desired_stop_price = c.val_price_now - atr_mult * atr_now
    new_sl_stop = 1.0 - desired_stop_price / c.init_price

    if new_sl_stop < 0.0:
        new_sl_stop = 0.0
    elif new_sl_stop > 1.0:
        new_sl_stop = 1.0

    if not c.curr_trail:
        return new_sl_stop, True

    if new_sl_stop < c.curr_stop:
        return new_sl_stop, True

    return c.curr_stop, True
# ... existing code ...

class ATRPsarTrailingStopsBT:
    def __init__(
        self,
        symbols=("BTC",),
        interval="4h",
        start=None,
        end=None,
        atr_len=14,
        psar_accel=0.02,
        psar_max=0.2,
        atr_mult=3.0,
        init_cash=1000.0,
        fees=0.001,
        direction="longonly",
        position_size=1.0,          # 100% equity per trade
        size_type="percent",        # must be supported by from_signals
    ):
        self.symbols = list(symbols)
        self.interval = interval
        self.start = start
        self.end = end
        self.atr_len = atr_len
        self.psar_accel = psar_accel
        self.psar_max = psar_max
        self.atr_mult = atr_mult
        self.init_cash = init_cash
        self.fees = fees
        self.direction = direction
        self.position_size = position_size
        self.size_type = size_type

        self.k = KrakenHistoricalData()
        self.df = None
        self.pf = None
        self.stats_ = None

    def _load_data(self):
        df_multi = self.k.get_ohlcv_df(self.symbols, interval=self.interval)

        if isinstance(df_multi.columns, pd.MultiIndex):
            # assume first symbol if single-symbol run
            df_multi = df_multi.xs(self.symbols[0], axis=1, level=0)

        if self.start is not None or self.end is not None:
            df = df_multi.loc[self.start:self.end].copy()
        else:
            df = df_multi.copy()

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        self.df = df

    def _build_signals_and_indicators(self):
        df = self.df
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        atr = vbt.IndicatorFactory.from_talib("ATR").run(
            high, low, close, timeperiod=self.atr_len
        ).real.astype(np.float64)

        psar = vbt.IndicatorFactory.from_talib("SAR").run(
            high, low, acceleration=self.psar_accel, maximum=self.psar_max
        ).real

        psar_below = psar < close
        psar_above = psar > close
        long_flip = (psar_below & psar_above.shift(1).fillna(False)).astype(bool)
        entries = long_flip.vbt.fshift(1)
        exits = pd.Series(False, index=close.index)

        return open_, high, low, close, atr, entries, exits

    def run(self, plot=False):
        self._load_data()
        open_, high, low, close, atr, entries, exits = self._build_signals_and_indicators()

        atr_arr = atr.to_numpy(dtype=np.float64) if hasattr(atr, "to_numpy") else np.asarray(atr, dtype=np.float64)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            direction=self.direction,
            init_cash=self.init_cash,
            size=self.position_size,
            size_type=self.size_type,
            open=open_,
            high=high,
            low=low,
            sl_stop=1.0,
            sl_trail=True,
            use_stops=True,
            adjust_sl_func_nb=atr_trailing_sl_nb,
            adjust_sl_args=(vbt.Rep("atr_arr"), self.atr_mult),
            broadcast_named_args={"atr_arr": atr_arr},
            fees=self.fees,
        )

        self.pf = pf
        self.stats_ = pf.stats()

        if plot:
            fig = pf.plot(width=1400, height=600)
            fig.show()

        return pf, self.stats_

if __name__ == "__main__":
    # Example usage of the class with your existing parameters
    end = pd.to_datetime("2025-09-30").tz_localize("UTC")
    start = end - pd.Timedelta(days=30 * 6)

    bt = ATRPsarTrailingStopsBT(
        symbols=("BTC",),
        interval="4h",
        start=start,
        end=end,
        atr_len=14,
        psar_accel=0.02,
        psar_max=0.2,
        atr_mult=3.0,
        init_cash=1000.0,
        fees=0.001,
        direction="longonly",
        position_size=1.0,
        size_type="percent",  # supported by from_signals
    )

    pf, stats = bt.run(plot=True)
    print(stats)
