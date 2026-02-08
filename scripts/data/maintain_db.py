import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ggTrader.data.kraken.historical_data import KrakenHistoricalData

def maintain_database(sync=True, movers=True):
    print(f"--- Database Maintenance Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    k_h = KrakenHistoricalData()
    
    if sync:
        print("\n[1/2] Syncing local raw CSVs to Parquet...")
        try:
            k_h.sync_local_data()
            print("Sync complete.")
        except Exception as e:
            print(f"Sync failed: {e}")

    if movers:
        print("\n[2/2] Updating historical movers parquet...")
        try:
            print("Calculating top movers by volume (this may take a few minutes)...")
            k_h.save_historical_movers_to_parquet()
            print("Historical movers updated.")
        except Exception as e:
            print(f"Movers update failed: {e}")

    print(f"\n--- Maintenance Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ggTrader Database Maintenance")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing raw CSVs.")
    parser.add_argument("--no-movers", action="store_true", help="Skip updating historical movers.")
    args = parser.parse_args()
    
    maintain_database(sync=not args.no_sync, movers=not args.no_movers)
