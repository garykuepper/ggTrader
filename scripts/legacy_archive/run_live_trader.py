"""Orchestrate live trading on Kraken using optimized WFO results."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from ggTrader.core.execution_engine import ExecutionEngine
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.run_config import full_pipeline_config, merge_run_config


def main() -> None:
    """Entry point for the live trading bot."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="ggTrader Live Trading Bot")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to run_results.json from a WFO run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders on the exchange",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=25.0,
        help="Capital allocated per trade in USD (default: 25.0)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Override polling interval (e.g. 1h, 4h)",
    )

    args = parser.parse_args()

    # Create ResultsManager to handle timestamped logging folder
    rm = ResultsManager("live_trader")
    print(f"[{rm.run_id}] Initializing live trader...")

    # Load base pipeline config and apply overrides
    config = full_pipeline_config()
    config = merge_run_config(
        config,
        DRY_RUN=args.dry_run,
        CAPITAL_PER_TRADE=args.capital,
        INTERVAL=args.interval,
    )

    # Save initial run metadata
    rm.save_run_results(params={}, metrics={}, metadata=config)

    print(f"Configuring live trader with results: {args.results}")
    if args.dry_run:
        print("!!! DRY RUN MODE ENABLED - No real orders will be placed !!!")

    # Initialize Execution Engine
    try:
        engine = ExecutionEngine(config, results_path=args.results)
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    # Start the event loop
    try:
        engine.run_event_loop()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"CRITICAL ERROR in event loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
