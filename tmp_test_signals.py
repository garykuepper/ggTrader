
import pandas as pd
import vectorbt as vbt
import numpy as np
from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.indicators.signals import Signals

def test():
    loader = TimescaleDBLoader()
    ohlcv = loader.fetch_ohlcv(["ADA-USD"], "4h", "2023-01-01", "2024-01-01")
    if ohlcv.empty:
        print("Empty data!")
        return

    close = ohlcv.xs("close", axis=1, level=1, drop_level=True)
    high = ohlcv.xs("high", axis=1, level=1, drop_level=True)
    low = ohlcv.xs("low", axis=1, level=1, drop_level=True)
    open_ = ohlcv.xs("open", axis=1, level=1, drop_level=True)

    entries = Signals.entry_signals(
        close, high, low,
        adx_length=14,
        adx_threshold=25,
        sar_acceleration=0.02,
        sar_maximum=0.2,
        use_dmp_cross=False
    )
    print(f"Num entries: {entries.sum()}")

    stop_arr, exits = Signals.trailing_stop_and_exits(
        entries, close, high, low,
        atr_length=14,
        atr_multiplier=3.0
    )
    print(f"Num exits: {exits.sum()}")

if __name__ == "__main__":
    test()
