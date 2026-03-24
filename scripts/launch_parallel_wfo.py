"""
Parallel WFO Launcher for Top-50 expansion.
Divides symbols into workers to utilize all CPU cores.
"""

import os
import sys
import subprocess
import time
from typing import List
from ggTrader.utils.config import load_symbols_from_json

def chunk_list(lst: List, n: int) -> List[List]:
    """Split list into n approximately equal chunks."""
    k, m = divmod(lst.__len__(), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main():
    symbols_file = "data/top_50_USD_2023-01-01_2025-12-31.json"
    n_workers = 5  # Optimal for 32GB RAM / 10-16 cores
    
    print(f"Loading symbols from {symbols_file}...")
    symbols = load_symbols_from_json(symbols_file)
    if not symbols:
        print("Failed to load symbols.")
        return
    
    # Take top 50 and ensure -USD suffix
    symbols = [s if s.endswith("-USD") else f"{s}-USD" for s in symbols[:50]]
    symbol_chunks = chunk_list(symbols, n_workers)
    
    python_exe = os.path.abspath(".venv/Scripts/python.exe")
    processes = []
    print(f"Launching {n_workers} parallel WFO workers...")
    
    handles = []
    for i, chunk in enumerate(symbol_chunks):
        chunk_str = ",".join(chunk)
        # Use separate log files for each worker to avoid race conditions
        log_file = f"results/wfo_worker_{i+1}.log"
        cmd = [
            python_exe, "-u",
            "scripts/run_walk_forward_optimization.py",
            "--symbols", chunk_str,
            "--train-metric", "composite",
            "--no-progress"  # Avoid mangling the UI
        ]
        
        f = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, bufsize=1)
        processes.append(p)
        handles.append(f)
            
    print("-" * 50)
    print("All workers launched. Monitoring...")
    
    try:
        while True:
            alive = [p.poll() is None for p in processes]
            if not any(alive):
                break
            
            # Simple heartbeat
            done_count = alive.count(False)
            print(f"Status: {done_count}/{n_workers} workers finished...", end="\r")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nTerminating workers...")
        for p in processes:
            p.terminate()
    finally:
        for f in handles:
            f.close()
            
    print("\nAll parallel workers finished.")

if __name__ == "__main__":
    # Ensure src is in path
    sys.path.append(os.path.join(os.getcwd(), "src"))
    main()
