"""
Iteration 7: Top-50 Universe Expansion WFO.
Targeting 'Unicorns' in the Ranks 26-50.
"""

import os
import sys
import subprocess
from datetime import datetime

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/iteration_7_top50_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Launching Iteration 7: Top-50 Universe Expansion")
    print(f"Results will be saved to: {results_dir}")
    print("-" * 50)
    
    # We use the main run_walk_forward_optimization.py script but with Top 50 overrides
    cmd = [
        sys.executable,
        "scripts/run_walk_forward_optimization.py",
        "--symbols-file", "data/top_50_USD_2023-01-01_2025-12-31.json",
        "--max-symbols", "50",
        "--train-metric", "composite"
    ]
    
    # We want to use the 'DETAILED' grids from Iteration 6
    # (The script run_walk_forward_optimization.py uses DETAILED_ENTRY_PARAM_GRIDS by default)
    
    # Set environment variables for the subprocess to ensure correct params if needed
    env = os.environ.copy()
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # We'll run this as a subprocess so we can capture output or just let it print
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    except KeyboardInterrupt:
        print("\nWFO Interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"Error during WFO: {e}")

if __name__ == "__main__":
    main()
