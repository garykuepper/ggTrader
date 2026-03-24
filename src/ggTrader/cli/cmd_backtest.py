"""CLI Command: Backtest (Portfolio Analysis Simulation)"""

import argparse
import sys
import subprocess
from ggTrader.utils.state_manager import get_latest_research_run

def register_backtest_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'backtest' subcommand."""
    parser = subparsers.add_parser("backtest", help="Simulate a portfolio backtest using research parameters")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Path to specific results directory (default: auto-detect latest)"
    )

def run_backtest(args: argparse.Namespace):
    """Executes the backtest simulation."""
    target_dir = args.run_id
    if not target_dir:
        print("Searching for latest research results...")
        latest = get_latest_research_run()
        if not latest:
            print("Error: Could not automatically detect a valid run_results.json in results/")
            sys.exit(1)
        target_dir = str(latest.parent)
        print(f"Auto-detected latest research run: {target_dir}")
        
    cmd = [
        sys.executable, "scripts/portfolio_analysis_standalone.py",
        "--results-dir", target_dir
    ]
    subprocess.run(cmd, check=True)
