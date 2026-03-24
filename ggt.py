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

from ggTrader.cli.cmd_research import register_research_parser, run_research
from ggTrader.cli.cmd_backtest import register_backtest_parser, run_backtest
from ggTrader.cli.cmd_production import register_production_parser, run_production
from ggTrader.cli.cmd_trade import register_trade_parser, run_trade

def main():
    parser = argparse.ArgumentParser(
        prog="ggt",
        description="ggTrader: The Automated Crypto Trading & Optimization Engine"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True
    
    # Register command modules
    register_research_parser(subparsers)
    register_backtest_parser(subparsers)
    register_production_parser(subparsers)
    register_trade_parser(subparsers)
    
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

if __name__ == "__main__":
    main()
