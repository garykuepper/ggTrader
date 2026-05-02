"""``ggt`` unified CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggt",
        description="ggTrader — algorithmic crypto trading toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # Register all subcommands
    from ggTrader.cli.cmd_backtest import register_backtest_parser, run_backtest
    from ggTrader.cli.cmd_cleanup import register_cleanup_parser, run_cleanup
    from ggTrader.cli.cmd_dashboard import register_dashboard_parser, run_dashboard
    from ggTrader.cli.cmd_db import register_db_parser, run_db
    from ggTrader.cli.cmd_ingest import register_ingest_parser, run_ingest
    from ggTrader.cli.cmd_pnl_daily import register_pnl_daily_parser, run_pnl_daily
    from ggTrader.cli.cmd_production import register_production_parser, run_production
    from ggTrader.cli.cmd_repair import register_repair_parser, run_repair
    from ggTrader.cli.cmd_report import register_report_parser, run_report
    from ggTrader.cli.cmd_research import register_research_parser, run_research
    from ggTrader.cli.cmd_status import register_status_parser, run_status
    from ggTrader.cli.cmd_trade import register_trade_parser, run_trade
    from ggTrader.cli.cmd_trade_report import register_trade_report_parser, run_trade_report

    register_research_parser(subparsers)
    register_status_parser(subparsers)
    register_report_parser(subparsers)
    register_backtest_parser(subparsers)
    register_trade_parser(subparsers)
    register_production_parser(subparsers)
    register_ingest_parser(subparsers)
    register_db_parser(subparsers)
    register_cleanup_parser(subparsers)
    register_dashboard_parser(subparsers)
    register_pnl_daily_parser(subparsers)
    register_repair_parser(subparsers)
    register_trade_report_parser(subparsers)

    _DISPATCH = {
        "research": run_research,
        "status": run_status,
        "backtest": run_backtest,
        "trade": run_trade,
        "production": run_production,
        "ingest": run_ingest,
        "db": run_db,
        "cleanup": run_cleanup,
        "dashboard": run_dashboard,
        "report": run_report,
        "pnl-daily": run_pnl_daily,
        "repair": run_repair,
        "trade-report": run_trade_report,
    }

    args = parser.parse_args()
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
