
"""Run a combined portfolio backtest using frozen (optimized) parameters."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from ggTrader.core.orchestrator import run_frozen_params_combined_backtest
from ggTrader.indicators.strategies import EXIT_REGISTRY
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers

def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest (Frozen Parameters)")
    parser.add_argument("--results-dir", type=str, required=True, help="Dir containing run_results.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / "run_results.json"
    
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        sys.exit(1)
        
    print(f"Loading optimization results from {results_path}...")
    with open(results_path, "r") as f:
        master_results = json.load(f)
        
    config = master_results.get("metadata", {})
    if "_raw_config" in config:
        config.update(config["_raw_config"])
        
    per_coin_results = master_results.get("params", {}).get("per_coin", {})
    
    if not per_coin_results:
        print("Error: No per_coin results found in run_results.json.")
        sys.exit(1)
        
    # Reload OHLCV for the backtest (ensuring full or latest range)
    print("Loading OHLCV data for backtest simulation...")
    ohlcv, _ = load_data_with_movers(config)
    
    rm = ResultsManager("backtest_simulation", pipeline_stage="backtest")
    
    # Use standard Orchestrator logic for replaying with frozen book
    out = run_frozen_params_combined_backtest(
        ohlcv,
        per_coin_results,
        config,
        exit_tournament=list(EXIT_REGISTRY.keys()),
        save_results=True,
        results_manager=rm,
        phase_title="PORTFOLIO BACKTEST SIMULATION (REPLAY)",
        combined_portfolio_label="Backtest Portfolio Performance",
    )
    
    print(f"\nBacktest complete. Results saved to: {rm.run_dir}")

if __name__ == "__main__":
    main()
