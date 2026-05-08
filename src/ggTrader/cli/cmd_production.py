"""CLI Command: Production (Monthly Recalibration)"""

import argparse
import sys
from pathlib import Path

from ggTrader.core.portfolio_optimizer import run_portfolio_competition
from ggTrader.utils.state_manager import get_latest_research_run


def register_production_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'production' subcommand."""
    parser = subparsers.add_parser(
        "production", help="Run the monthly recalibration to generate live trading weights"
    )
    parser.add_argument(
        "--master-results",
        type=str,
        default=None,
        help="Path to Master WFO run_results.json (default: auto-detect latest)",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of CCXT coins to pull (default: 50)"
    )


def run_production(args: argparse.Namespace):
    """Executes the production recalibration pipeline."""
    target_results = args.master_results
    if not target_results:
        print("Searching for latest Master Research results...")
        latest = get_latest_research_run()
        if not latest:
            print("Error: Could not automatically detect a valid run_results.json in results/")
            sys.exit(1)
        if not latest.run_dir:
            print(
                f"Error: Latest research run {latest.run_id} has no on-disk "
                f"directory; pass an explicit --master-results PATH."
            )
            sys.exit(1)
        target_results = str(latest.run_dir / "run_results.json")
        print(f"Auto-detected latest research run: {target_results}")

    results_dir = str(Path(target_results).parent)
    run_portfolio_competition(results_dir)
