import os
import sys
import pandas as pd

# Add src to sys.path
sys.path.append(os.path.abspath('src'))

from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData

def extract_q4_2025():
    k_h = KrakenHistoricalData()
    input_dir = os.path.join(k_h.root_dir, 'data', 'raw', 'Kraken_OHLCVT_Q4_2025')
    
    print(f"Starting extraction for {input_dir}...")
    k_h.csvs_dir_to_parquet_parallel(input_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    extract_q4_2025()
