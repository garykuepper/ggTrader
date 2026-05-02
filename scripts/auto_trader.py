"""Orchestrator for automated trading and periodic re-optimization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ggTrader.core.execution_engine import ExecutionEngine
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.run_config import full_pipeline_config, merge_run_config

# Project root for data/results paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def run_wfo(symbols: list[str], training_years: int) -> tuple[str | None, str | None]:
    """Run the full WFO pipeline and portfolio analysis to generate new parameters."""
    print(f"[{datetime.now().strftime('%X')}] STARTING RE-OPTIMIZATION (WFO)...")

    # Create temporary symbols file
    symbols_path = PROJECT_ROOT / "data" / "auto_trader_symbols.json"
    with open(symbols_path, "w") as f:
        json.dump(symbols, f)

    # Calculate dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    td_years = timedelta(days=365 * training_years)
    start_date = (datetime.now() - td_years).strftime("%Y-%m-%d")

    cmd = [
        sys.executable,
        "scripts/run_full_pipeline.py",
        "--symbols-file",
        str(symbols_path),
        "--max-symbols",
        str(len(symbols)),
        "--no-progress",
    ]

    # Environment variables to override config
    env = os.environ.copy()
    env["GGTRADER_START_DATE"] = start_date
    env["GGTRADER_END_DATE"] = end_date

    try:
        # 1. Run WFO
        subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

        # Find the latest results directory
        results_dir_base = PROJECT_ROOT / "results"
        pipeline_dirs = sorted(
            results_dir_base.glob("pipeline_*"), key=lambda x: x.name, reverse=True
        )
        if not pipeline_dirs:
            return None, None

        latest_results_dir = pipeline_dirs[0]
        latest_results = latest_results_dir / "run_results.json"

        if not latest_results.exists():
            return None, None

        # 2. Run Portfolio Analysis
        print(f"[{datetime.now().strftime('%X')}] STARTING PORTFOLIO ANALYSIS...")
        port_cmd = [
            sys.executable,
            "scripts/portfolio_analysis_standalone.py",
            "--results-dir",
            str(latest_results_dir),
        ]
        subprocess.run(port_cmd, capture_output=True, text=True, check=True)

        return str(latest_results), None

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Subprocess failed: {e.stderr}")
    except Exception as e:
        print(f"ERROR: Exception during re-optimization: {e}")

    return None, None


def main() -> None:
    """Main loop for auto-trading and re-optimization."""
    load_dotenv()

    # 1. Load Environment Config
    symbols_env = os.getenv("SYMBOLS", "BTC-USD,ETH-USD,SOL-USD")
    symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]

    interval = os.getenv("TRADING_INTERVAL", "4h")
    capital = float(os.getenv("CAPITAL_PER_TRADE", 25.0))
    reoptimize_days = int(os.getenv("REOPTIMIZE_DAYS", 30))
    training_years = int(os.getenv("TRAINING_YEARS", 3))
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"

    # State tracking
    last_wfo_time = 0.0
    active_results_path = os.getenv("RESULTS_PATH")

    # Initialize ResultsManager (creates run_id/folder if needed)
    ResultsManager("auto_trader")
    print(f"[{datetime.now().strftime('%X')}] AUTO-TRADER STARTED (Symbols: {len(symbols)})")

    while True:
        # 2. Check if it's time to re-optimize
        current_time = time.time()
        needs_wfo = (current_time - last_wfo_time) > (reoptimize_days * 86400)

        if needs_wfo or not active_results_path:
            new_results, _ = run_wfo(symbols, training_years)
            if new_results:
                active_results_path = new_results
                last_wfo_time = current_time
                print(
                    f"[{datetime.now().strftime('%X')}] "
                    f"RE-OPTIMIZATION SUCCESSFUL: {active_results_path}"
                )
            elif not active_results_path:
                print("CRITICAL: No results found and WFO failed. Sleeping 1h retry.")
                time.sleep(3600)
                continue

        # 3. Configure and Run One Trade Cycle
        config = full_pipeline_config()
        config = merge_run_config(
            config,
            DRY_RUN=dry_run,
            CAPITAL_PER_TRADE=capital,
            INTERVAL=interval,
            SYMBOLS=symbols,
        )

        try:
            # We initialize a new engine each cycle to pick up any result changes
            engine = ExecutionEngine(config, results_path=active_results_path)

            print(f"[{datetime.now().strftime('%X')}] HEARTBEAT: Starting trade cycle...")

            # Manually trigger one cycle
            ohlcv = engine._fetch_latest_data()
            signals = engine._compute_latest_signals(ohlcv)
            engine._execute_trade_logic(signals)

        except Exception as e:
            print(f"ERROR in trade cycle: {e}")

        # 4. Wait for next boundary
        now = datetime.now()
        current_hour = now.hour
        # Interval hours (e.g. 4 for 4h)
        match = "".join(filter(str.isdigit, interval))
        int_hours = int(match) if match else 4

        next_hour = ((current_hour // int_hours) + 1) * int_hours
        if next_hour >= 24:
            wait_seconds = (24 - current_hour) * 3600 - now.minute * 60 - now.second
        else:
            wait_seconds = (next_hour - current_hour) * 3600 - now.minute * 60 - now.second

        print(f"[{datetime.now().strftime('%X')}] Cycle complete. Sleeping {wait_seconds}s.")
        time.sleep(max(wait_seconds, 60))


if __name__ == "__main__":
    main()
