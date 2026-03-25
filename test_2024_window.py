
import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

from ggTrader.utils.setup import load_data_and_setup
from ggTrader.indicators.signals import Signals

def test_window_trades():
    # Window that failed in Fold 4 for some coin
    start_date = "2023-12-31"
    end_date = "2024-12-30"
    
    # Try multiple symbols from the top volume list
    symbols = ["ADA-USD", "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        config = {
            "SYMBOLS": [symbol],
            "INTERVAL": "4h",
            "START_DATE": start_date,
            "END_DATE": end_date,
        }
        try:
            ohlcv = load_data_and_setup(config)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
            continue
            
        # Keep 2D shape (n_time, 1)
        close = ohlcv.xs("close", axis=1, level=1).values
        high = ohlcv.xs("high", axis=1, level=1).values
        low = ohlcv.xs("low", axis=1, level=1).values
        # open_ = ohlcv.xs("open", axis=1, level=1).values

        print(f"Num bars: {len(close)}")

        # Test PSAR+ADX defaults
        entries = Signals.entry_signals(
            close, high, low,
            adx_length=14,
            adx_threshold=25,
            sar_acceleration=0.02,
            sar_maximum=0.2,
            use_dmp_cross=False
        )
        print(f"  PSAR+ADX Entries: {entries.sum()}")

        stop_arr, exits = Signals.trailing_stop_and_exits(
            entries, close, high, low,
            atr_length=14,
            atr_multiplier=3.0
        )
        print(f"  ATR Trailing Exits: {exits.sum()}")
        
        # Test Macd Cross (New)
        from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
        pre = IndicatorPrecomputer(close, high, low)
        macd_ind = pre.compute_macd(12, 26, 9)
        # MACD output names in vbt pandas_ta are usually 'macd', 'macdh', 'macds'
        macd = macd_ind.macd
        signal = macd_ind.macds # macds is the signal line
        
        # macd and signal are likely 2D (time, 1)
        macd_entries = (macd > signal) & (pd.Series(macd.flatten()).shift(1).values.reshape(-1,1) <= pd.Series(signal.flatten()).shift(1).values.reshape(-1,1))
        print(f"  MACD Cross Entries: {np.nansum(macd_entries)}")

if __name__ == "__main__":
    test_window_trades()
