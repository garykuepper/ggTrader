"""CLI Command: Trade (Execution Engine)"""

import argparse
import sys

from ggTrader.utils.state_manager import get_latest_production_weights, get_latest_research_run


def register_trade_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'trade' subcommand."""
    parser = subparsers.add_parser("trade", help="Start the live execution engine")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Path to run_results.json for strategy params "
            "(default: auto-detect latest research run)"
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Path to portfolio_weights.json "
            "(default: auto-detect latest production weights)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without placing real orders on the exchange"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=25.0,
        help="Fallback capital per trade if weight is missing (default: 25.0)",
    )
    parser.add_argument(
        "--interval", type=str, default=None, help="Override polling interval (e.g. 1h, 4h)"
    )


def run_trade(args: argparse.Namespace):
    """Executes the live trading engine."""
    from dotenv import load_dotenv

    load_dotenv()

    # Import inside to prevent DB connections from initializing early
    from ggTrader.core.execution_engine import ExecutionEngine
    from ggTrader.utils.results_manager import ResultsManager
    from ggTrader.utils.run_config import full_pipeline_config, merge_run_config

    results_path = args.results
    if not results_path:
        print("Searching for latest Master Research results for parameters...")
        latest_res = get_latest_research_run()
        if not latest_res:
            print("Error: Could not automatically detect a valid run_results.json in results/")
            sys.exit(1)
        results_path = str(latest_res)
        print(f"Auto-detected latest research params: {results_path}")

    weights_path = args.weights
    if not weights_path:
        print("Searching for latest Production weights directory...")
        latest_weight = get_latest_production_weights()
        if latest_weight:
            weights_path = str(latest_weight)
            print(f"Auto-detected latest production weights: {weights_path}")
        else:
            print(
                "Warning: No portfolio_weights.json automatically found. "
                "Using fixed fallback capital."
            )

    rm = ResultsManager("live_trader")
    print(f"\n[{rm.run_id}] Initializing live trader...")

    config = full_pipeline_config()
    config = merge_run_config(
        config,
        DRY_RUN=args.dry_run,
        CAPITAL_PER_TRADE=args.capital,
        INTERVAL=args.interval,
    )

    rm.save_run_results(params={}, metrics={}, metadata=config)

    if args.dry_run:
        print("!!! DRY RUN MODE ENABLED - No real orders will be placed !!!")

    try:
        engine = ExecutionEngine(config, results_path=results_path)
        if weights_path:
            # Native support inside existing ExecutionEngine logic
            engine.load_portfolio_weights(weights_path)

        engine.run_event_loop()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"CRITICAL ERROR in event loop: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
