"""CLI Command: Backtest (Portfolio Analysis Simulation)"""

import argparse
import subprocess
import sys

from ggTrader.utils.state_manager import get_latest_research_run, validate_results_asset_class


def register_backtest_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'backtest' subcommand."""
    parser = subparsers.add_parser(
        "backtest", help="Simulate a portfolio backtest using research parameters"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Path to specific results directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        default="crypto",
        choices=["crypto", "stocks"],
        help="Asset class to backtest (default: crypto)",
    )


def run_backtest(args: argparse.Namespace):
    """Executes the backtest simulation."""
    asset_class = getattr(args, "asset_class", "crypto")
    target_dir = args.run_id
    if target_dir:
        from pathlib import Path
        validate_results_asset_class(Path(target_dir) / "run_results.json", expected=asset_class)
    else:
        print(f"Searching for latest {asset_class} research results...")
        latest = get_latest_research_run(asset_class=asset_class)
        if not latest:
            print(
                f"Error: Could not automatically detect a valid run_results.json "
                f"for asset_class={asset_class!r} in results/"
            )
            sys.exit(1)
        target_dir = str(latest.parent)
        print(f"Auto-detected latest {asset_class} research run: {target_dir}")

    cmd = [sys.executable, "scripts/portfolio_analysis_standalone.py", "--results-dir", target_dir]
    subprocess.run(cmd, check=True)
