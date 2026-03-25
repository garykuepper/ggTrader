import pandas as pd
from ggTrader.utils.config import get_db_connection_string
from sqlalchemy import create_engine, text
import json
import os

def recover_full_day():
    engine = create_engine(get_db_connection_string())
    with engine.connect() as conn:
        # Get all runs from today
        query = text("""
            SELECT parameters, metadata, timestamp
            FROM runs
            WHERE timestamp > '2026-03-24 00:00:00'
            ORDER BY timestamp DESC
        """)
        df = pd.read_sql(query, conn)
        
    if df.empty:
        print("No runs found for today.")
        return

    all_results = {}
    master_metadata = {}
    
    for _, row in df.iterrows():
        p = row['parameters']
        m = row['metadata']
        if isinstance(p, str):
            p = json.loads(p)
        if isinstance(m, str):
            m = json.loads(m)
        
        # Merge metadata from the most recent one
        if not master_metadata:
            master_metadata = m
            
        # In per-coin WFO, we save { 'per_coin': { symbol: {...} } }
        # The key might be 'per_coin' or direct in 'parameters' depending on how it was saved
        # ResultDBManager uses self._safe_json_dumps(parameters)
        per_coin = p.get('per_coin', {})
        for sym, data in per_coin.items():
            if sym not in all_results:
                all_results[sym] = data
                
    print(f"Total Unique Assets Recovered: {len(all_results)}")
    
    os.makedirs("results/reconstructed_run", exist_ok=True)
    out_path = "results/reconstructed_run/run_results.json"
    
    # Wrap in the format expected by portfolio_analysis_standalone.py
    # master_results.get("metadata", {})
    # master_results.get("params", {}).get("per_coin", {})
    final_output = {
        "run_id": "RECONSTRUCTED",
        "timestamp": "2026-03-24T19:00:00",
        "metadata": master_metadata,
        "params": {
            "per_coin": all_results
        }
    }
    
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Saved reconstructed file to: {out_path}")

if __name__ == "__main__":
    recover_full_day()
