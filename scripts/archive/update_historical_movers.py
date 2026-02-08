import os
import sys

# Add src to sys.path
sys.path.append(os.path.abspath('src'))

from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData

def main():
    """
    Standalone script to update the historical movers parquet file.
    This reads from the existing data/parquet dataset and identifies 
    top volume-movers for each day.
    """
    print("--- Updating Historical Movers ---")
    k_h = KrakenHistoricalData()
    print("Reading parquet data and calculating movers (this may take a minute)...")
    k_h.save_historical_movers_to_parquet()
    print("Historical movers updated successfully.")
    print("Path: data/historical_movers/historical_movers.parquet")
    print("---------------------------------")

if __name__ == "__main__":
    main()
