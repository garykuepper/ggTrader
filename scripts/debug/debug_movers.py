import sys
import os
import pandas as pd
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData

def test_movers():
    k_h = KrakenHistoricalData()
    date = pd.Timestamp("2024-01-01").tz_localize('UTC')
    
    print(f"Attempting to get movers for {date}...")
    try:
        movers = k_h.get_historical_movers_by_day(date, top_n=5)
        print(f"Success! Found {len(movers)} movers.")
        print(movers)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to get movers: {e}")

if __name__ == "__main__":
    test_movers()
