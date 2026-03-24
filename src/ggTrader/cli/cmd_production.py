"""CLI Command: Production (Monthly Recalibration)"""

import argparse
import sys
import subprocess
from ggTrader.utils.state_manager import get_latest_research_run

def register_production_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'production' subcommand."""
    parser = subparsers.add_parser("production", help="Run the monthly recalibration to generate live trading weights")
    parser.add_argument(
        "--master-results", type=str, default=None,
        help="Path to Master WFO run_results.json (default: auto-detect latest)"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of CCXT coins to pull (default: 50)"
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
        target_results = str(latest)
        print(f"Auto-detected latest research run: {target_results}")

    cmd = [
        sys.executable, "scripts/run_recalibration_pipeline.py",
        "--master-results", target_results,
        "--limit", str(args.limit)
    ]
    subprocess.run(cmd, check=True)
