"""
Production Pipeline: Monthly Recalibration
1. Fetches live top 50 volume pairs via CCXT
2. Runs Phase 3 validation using Master WFO params on the last 6 months
3. Generates portfolio weights based on the new performance.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from ggTrader.utils.pipeline_phases import phase_3_recent_performance
from ggTrader.utils.pipeline_status_logger import StatusLogger


def main():
    parser = argparse.ArgumentParser(description="Live Trading Recalibration Pipeline")
    parser.add_argument("--master-results", type=str, required=True, help="Path to Master WFO run_results.json")
    parser.add_argument("--limit", type=int, default=50, help="Number of CCXT coins to pull.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    calib_dir = Path(f"results/recalibration_{timestamp}")
    calib_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Universe
    print(f"\n[{datetime.now()}] Step 1: Generating CCXT Universe...")
    universe_path = calib_dir / "top_50_ccxt_volume.json"
    subprocess.run([sys.executable, "scripts/update_universe_ccxt.py", "--limit", str(args.limit), "--out", str(universe_path)], check=True)

    # 2. Extract best params
    print(f"\n[{datetime.now()}] Step 2: Extracting Master Params for CCXT universe...")
    with open(args.master_results, "r") as f:
        master_data = json.load(f)

    master_config = master_data.get("configuration", {}).get("_raw_config", {})
    master_per_coin = master_config.get("per_coin_results", {})
    if not master_per_coin:
        master_per_coin = master_data.get("results", {}).get("per_coin_results", {})

    with open(universe_path, "r") as f:
        ccxt_coins = json.load(f)

    calib_params = {}
    for coin in ccxt_coins:
        if coin in master_per_coin:
            calib_params[coin] = master_per_coin[coin]
        else:
            print(f"  Warning: No Master WFO params for {coin}. Skipping this month.")

    # Modify raw_config for phase 3
    # Force use ccxt tail and look back e.g. 6 months
    master_config["RECENT_VALIDATION_USE_CCXT_TAIL"] = True
    master_config["RECENT_VALIDATION_START_DATE"] = (datetime.now() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    master_config["RECENT_VALIDATION_END_DATE"] = datetime.now().strftime("%Y-%m-%d")
    master_config["SYMBOLS"] = list(calib_params.keys())

    wfo_results = {"per_coin_results": calib_params}

    # 3. Run Phase 3
    print(f"\n[{datetime.now()}] Step 3: Running Phase 3 Validation on Recent Data...")
    logger = StatusLogger(calib_dir / "recalib_status.txt")
    phase_3_recent_performance(master_config, wfo_results, logger)

    # 4. Save new run_results.json
    print(f"\n[{datetime.now()}] Step 4: Saving recalibration results...")
    master_config["per_coin_results"] = wfo_results.get("per_coin_results", {})

    new_results_file = calib_dir / "run_results.json"
    dump_data = {
        "configuration": {
            "_raw_config": master_config
        },
        "results": wfo_results
    }
    with open(new_results_file, "w") as f:
        json.dump(dump_data, f, indent=4, default=str)

    # 5. Run Portfolio Analysis
    print(f"\n[{datetime.now()}] Step 5: Generating Portfolio Allocations...")
    subprocess.run([sys.executable, "scripts/portfolio_analysis_standalone.py", "--results-dir", str(calib_dir)], check=True)

    print(f"\n[{datetime.now()}] RECALIBRATION COMPLETE. Weights saved in {calib_dir}/portfolio_analysis/portfolio_weights.json")

if __name__ == "__main__":
    main()
