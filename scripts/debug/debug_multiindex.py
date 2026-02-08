import pandas as pd
from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
from ggTrader.indicators.Signals import Signals
import os
import sys
import traceback

# Set PYTHONPATH to src to find ggTrader
sys.path.append(os.path.abspath('src'))

def debug():
    try:
        k_h = KrakenHistoricalData()
        symbols = ["BTC"]
        interval = "4h"
        start_dt = pd.to_datetime("2024-01-01").tz_localize('UTC')
        end_dt = pd.to_datetime("2024-01-10").tz_localize('UTC')
        
        print("Loading ohlcv_df...")
        ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)
        
        print(f"ohlcv_df.columns: {ohlcv_df.columns}")
        
        print("Calculating signals for BTC...")
        signals = Signals()
        # Note: ohlcv_df["BTC"] for a single symbol might be a simple DF if it was concatenated with one key
        # or it might have renamed columns.
        btc_ohlcv = ohlcv_df["BTC"]
        print(f"BTC OHLCV columns: {btc_ohlcv.columns}")
        
        res = signals._atr_trailing_stop_long_ohlc_touch_2d(btc_ohlcv)
        print("Signals calculation successful")
        print(res.head())
        
    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug()
