import os
import sys

# Add src to sys.path
sys.path.append(os.path.abspath('src'))

from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData

def main():
    """
    Standalone script to sync all new Kraken quarterly data folders 
    from data/raw into the data/parquet dataset.
    """
    print("--- Kraken Data Sync Utility ---")
    k_h = KrakenHistoricalData()
    k_h.sync_local_data()
    print("--------------------------------")

if __name__ == "__main__":
    main()
