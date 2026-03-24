import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer

def test_cache():
    print("Initializing dummy data (fixed seed)...")
    np.random.seed(42)
    size = 100
    close = np.random.random(size)
    high = close + 0.1
    low = close - 0.1
    
    pre = IndicatorPrecomputer(close, high, low)
    print(f"Data hash: {pre._data_hash}")
    
    print("\nComputing EMA(10) (First run - should save)...")
    ema1 = pre.compute_ema(10)
    print(f"EMA1 type: {type(ema1)}")
    
    # Check if cache file exists
    cache_files = os.listdir(".cache/indicators")
    print(f"Cache files found: {len(cache_files)}")
    for f in cache_files:
        if pre._data_hash in f:
            print(f" - {f} (Current)")
        else:
            print(f" - {f}")
        
    print("\nRe-initializing precomputer with SAME data...")
    pre2 = IndicatorPrecomputer(close, high, low)
    print(f"Data hash 2: {pre2._data_hash}")
    
    print("Computing EMA(10) (Second run - should load from disk)...")
    ema2 = pre2.compute_ema(10)
    print(f"EMA2 type: {type(ema2)}")
    
    if np.allclose(np.asarray(ema1.ema), np.asarray(ema2.ema), equal_nan=True):
        print("\nSUCCESS: EMA results match (including NaNs)!")
    else:
        print("\nFAILURE: EMA results mismatch.")
        print(f"EMA1 head: {np.asarray(ema1.ema)[:5]}")
        print(f"EMA2 head: {np.asarray(ema2.ema)[:5]}")

if __name__ == "__main__":
    test_cache()
