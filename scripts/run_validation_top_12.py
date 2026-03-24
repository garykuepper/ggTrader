"""
Phase 3 Validation: Run 'Top 12' robust basket on the last 60 days of market data.
Uses CCXT hybrid loader for the most recent data 'tail'.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ggTrader.utils.setup import load_hybrid_validation_ohlcv
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator import run_frozen_params_combined_backtest

def main():
    # 1. Selection: Top 12 Trading Symbols from Iteration 6
    TOP_12_SYMBOLS = [
        "DOGE-USD", "LDO-USD", "DOT-USD", "AVAX-USD", "ATOM-USD",
        "XLM-USD", "LINK-USD", "XRP-USD", "TRX-USD", "SOL-USD",
        "XMR-USD", "ETH-USD"
    ]
    
    # 2. Locate Iteration 6 results to get best params
    results_dir = Path("results")
    latest_run = "run_wfo_per_coin_multi_strategy_20260323_230341"
    run_path = results_dir / latest_run
    results_file = run_path / "run_results.json"
    
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)
    
    # Extract winners
    per_coin_results = data["strategy_parameters"]["per_coin"]
    raw_config = data["configuration"]["_raw_config"]
    
    # Filter for Top 12 and verify params exist
    filtered_winners = {}
    for sym in TOP_12_SYMBOLS:
        if sym in per_coin_results:
            filtered_winners[sym] = per_coin_results[sym]
        else:
            print(f"Warning: No best params for {sym} in results.")

    if not filtered_winners:
        print("Error: No winners found for the Top 12 list.")
        return

    # 3. Setup 60-day window with CCXT Tail
    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - pd.Timedelta(days=60)
    
    print(f"\n--- PHASE 3 VALIDATION: LAST 60 DAYS ({start_date.date()} to {end_date.date()}) ---")
    print(f"Symbols: {list(filtered_winners.keys())}")
    
    # Prepare mock config for loader
    mock_config = raw_config.copy()
    mock_config["SYMBOLS"] = list(filtered_winners.keys())
    mock_config["INTERVAL"] = "4h"

    print("Fetching hybrid OHLCV (DB + CCXT Kraken)...")
    try:
        ohlcv = load_hybrid_validation_ohlcv(
            mock_config, 
            start_date, 
            end_date, 
            use_ccxt_tail=True
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 4. Run Backtest with Flat 10%
    # Using run_frozen_params_combined_backtest to get the professional scorecard
    
    print("\nRunning Verification Backtest...")
    
    # Explicitly set Flat 10% configuration
    val_config = raw_config.copy()
    val_config["PORTFOLIO_SHARE"] = 0.1  # Flat 10% per signal
    val_config["USE_CASH_SHARING"] = True
    val_config["START_CASH"] = 10000.0
    
    # Extract exit tournament from config
    from ggTrader.indicators.strategies import EXIT_REGISTRY
    from ggTrader.pipeline.exit_tournament import parse_exit_tournament
    
    exit_tournament = parse_exit_tournament(
        val_config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY
    )

    out = run_frozen_params_combined_backtest(
        ohlcv,
        filtered_winners,
        val_config,
        exit_tournament=exit_tournament,
        save_results=False,
        phase_title="PHASE 3: TOP-12 RECENT PERFORMANCE (60 DAYS)",
        combined_portfolio_label="Top 12 Basket @ 10% Flat Share",
    )
    
    stats = out["final_stats"]
    
    print("\n" + "=" * 50)
    print("PHASE 3 SCORECARD (LAST 60 DAYS)")
    print("=" * 50)
    print(f"Total Return: {stats['profit_pct']:.2f}%")
    print(f"Trades Count: {stats['total_trades']}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print("=" * 50)

    # Save validation report
    report_dir = run_path / "phase_3_validation"
    report_dir.mkdir(exist_ok=True)
    pd.DataFrame([stats]).to_csv(report_dir / "latest_60d_top12_stats.csv")
    print(f"\nValidation complete. Stats saved to {report_dir}")

if __name__ == "__main__":
    main()
