#!/usr/bin/env python3
"""
ggTrader Unified CLI (ggt)

Entry point for Research, Production, Backtesting, and Live Execution.
"""

import argparse
import sys
from pathlib import Path

# Provide native path resolutions for src
sys.path.append(str(Path(__file__).parent / "src"))

from ggTrader.cli.cmd_backtest import register_backtest_parser, run_backtest
from ggTrader.cli.cmd_cleanup import register_cleanup_parser, run_cleanup
from ggTrader.cli.cmd_db import register_db_parser, run_db
from ggTrader.cli.cmd_ingest import register_ingest_parser, run_ingest
from ggTrader.cli.cmd_production import register_production_parser, run_production
from ggTrader.cli.cmd_research import register_research_parser, run_research
from ggTrader.cli.cmd_trade import register_trade_parser, run_trade


def main():
    parser = argparse.ArgumentParser(
        prog="ggt", description="ggTrader: The Automated Crypto Trading & Optimization Engine"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Register command modules
    register_research_parser(subparsers)
    register_backtest_parser(subparsers)
    register_production_parser(subparsers)
    register_trade_parser(subparsers)
    register_cleanup_parser(subparsers)
    register_db_parser(subparsers)
    register_ingest_parser(subparsers)

    args = parser.parse_args()

    # Delegate to handlers
    if args.command == "research":
        run_research(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "production":
        run_production(args)
    elif args.command == "trade":
        run_trade(args)
    elif args.command == "cleanup":
        run_cleanup(args)
    elif args.command == "db":
        run_db(args)
    elif args.command == "ingest":
        run_ingest(args)


if __name__ == "__main__":
    main()
