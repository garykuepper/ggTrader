import os
import sys
import argparse
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ggTrader.data.kraken.historical_data import KrakenHistoricalData

def sync_kraken(sample=None, parallel=True):
    print(f"--- Kraken Data Sync Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    k_h = KrakenHistoricalData()
    
    # List available raw directories
    raw_dirs = k_h.list_quarter_dirs()
    if not raw_dirs:
        print("No raw data directories found in data/raw/")
        return

    print(f"Found {len(raw_dirs)} data directories.")
    
    # Use the facade's sync logic which handles manifest tracking
    k_h.sync_local_data(sample=sample)
    
    print(f"\n--- Sync Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kraken raw data to Parquet")
    parser.add_argument("--sample", type=int, help="Sample N files per directory for quick testing.")
    parser.add_argument("--no-parallel", action="store_true", help="Run sequentially instead of in parallel.")
    args = parser.parse_args()
    
    sync_kraken(sample=args.sample, parallel=not args.no_parallel)
