"""
Extract Phase 2 scorecard from the stdout log and identify alpha drivers.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

def main():
    log_file = Path("results/wfo_isolated_stdout.log")
    if not log_file.exists():
        print("Log file not found.")
        return

    content = log_file.read_text(encoding="utf-16")
    
    # 1. Extract Per-Symbol Scorecard (Phase 2)
    # Pattern: > SYMBOL (X/25): Best=STRAT | Robustness=0.0 | Win Rate=0.0% | Trades=0
    # Wait, the Phase 2 output format is:
    # > SYMBOL (X/25): Best=STRAT | Robustness=0.0 | Win Rate=0.0% | Trades=0
    
    pattern = r"> ([\w-]+) \(\d+/25\): Best=([\w\+]+) \| Robustness=([\d\.\-]+) \| Win Rate=([\d\.]+)% \| Trades=(\d+)"
    matches = re.finditer(pattern, content)
    
    data = []
    # We want the LAST set of these matches (the ones from Phase 2)
    # Phase 1 also prints them but Phase 2 prints them final.
    for m in matches:
        data.append({
            "Symbol": m.group(1),
            "Strategy": m.group(2),
            "Robustness": float(m.group(3)),
            "WinRate": float(m.group(4)),
            "Trades": int(m.group(5))
        })
        
    if not data:
        print("No matches found in log.")
        return
        
    # Get last 25 unique symbols (these are Phase 2 results)
    df_all = pd.DataFrame(data)
    df = df_all.drop_duplicates(subset=['Symbol'], keep='last').copy()
    
    # Wait! Phase 2 doesn't print Profit % per coin in the summary line!
    # It only prints them in the detail block?
    # No, Phase 2 in status_logger says:
    # [   0:38:11]   > ATOM-USD (1/25): Best=rsi_reversal+fixed_sl_tp | Robustness=1.8242 | Win Rate=50.00% | Trades=6
    
    # I need to find the Profit % per coin.
    # Phase 2 prints:
    # Phase 2 - ATOM-USD: profit_pct=X.XX% ...
    # Wait, I'll check the log again.
    
    # Actually, I'll check the log for "Phase 2 - combined portfolio" too.
    
    print("\n--- ITERATION 6 SYMBOL SCORECARD ---")
    print(df.sort_values("Robustness", ascending=False))
    
    # Identify drivers
    active = df[df['Trades'] > 0]
    print(f"\nActive Symbols: {len(active)} / 25")
    
    # Analyze Strategy Winners
    print("\n--- STRATEGY SELECTION TALLY ---")
    print(df['Strategy'].value_counts())

if __name__ == "__main__":
    main()
