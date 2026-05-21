"""``ggt`` unified CLI entry point — Typer dispatcher to legacy argparse handlers.

Phase 0 migration: Typer owns the top-level command tree; each command shims
into the existing ``register_*_parser`` + ``run_*`` pair from ``cmd_*.py``.
Per-command flags are still defined by argparse (so handlers stay untouched);
Typer just routes ``ggt <cmd> <args...>`` to the right shim.

Run ``ggt --help`` for the command list, ``ggt <cmd> --help`` for per-command flags.
"""

from __future__ import annotations

import argparse
from typing import Callable

import typer

app = typer.Typer(
    name="ggt",
    help="ggTrader — algorithmic crypto trading toolkit",
    no_args_is_help=True,
    add_completion=False,
)


def _run_legacy(
    command: str,
    register: Callable[[argparse._SubParsersAction], None],
    handler: Callable[[argparse.Namespace], object],
    extra_args: list[str],
) -> None:
    """Parse ``extra_args`` against the legacy argparse subparser and dispatch."""
    parser = argparse.ArgumentParser(prog="ggt", add_help=False)
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)
    namespace = parser.parse_args([command, *extra_args])
    handler(namespace)


def _command(name: str, help_text: str, register_path: str, handler_path: str) -> None:
    """Register a Typer command that shims into a legacy argparse handler."""

    @app.command(
        name=name,
        help=help_text,
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        },
    )
    def _cmd(ctx: typer.Context) -> None:
        reg_mod, reg_attr = register_path.rsplit(":", 1)
        hnd_mod, hnd_attr = handler_path.rsplit(":", 1)
        register = getattr(__import__(reg_mod, fromlist=[reg_attr]), reg_attr)
        handler = getattr(__import__(hnd_mod, fromlist=[hnd_attr]), hnd_attr)
        _run_legacy(name, register, handler, list(ctx.args))


_command(
    "research",
    "Run the Grand Walk-Forward Optimization pipeline",
    "ggTrader.cli.cmd_research:register_research_parser",
    "ggTrader.cli.cmd_research:run_research",
)
_command(
    "status",
    "Show research/production status",
    "ggTrader.cli.cmd_status:register_status_parser",
    "ggTrader.cli.cmd_status:run_status",
)
_command(
    "report",
    "Generate reports from research results",
    "ggTrader.cli.cmd_report:register_report_parser",
    "ggTrader.cli.cmd_report:run_report",
)
_command(
    "backtest",
    "Run a single backtest",
    "ggTrader.cli.cmd_backtest:register_backtest_parser",
    "ggTrader.cli.cmd_backtest:run_backtest",
)
_command(
    "trade",
    "Live trading commands",
    "ggTrader.cli.cmd_trade:register_trade_parser",
    "ggTrader.cli.cmd_trade:run_trade",
)
_command(
    "production",
    "Run production portfolio competition",
    "ggTrader.cli.cmd_production:register_production_parser",
    "ggTrader.cli.cmd_production:run_production",
)
_command(
    "ingest",
    "Ingest OHLCV data into TimescaleDB",
    "ggTrader.cli.cmd_ingest:register_ingest_parser",
    "ggTrader.cli.cmd_ingest:run_ingest",
)
_command(
    "db",
    "TimescaleDB management commands",
    "ggTrader.cli.cmd_db:register_db_parser",
    "ggTrader.cli.cmd_db:run_db",
)
_command(
    "cleanup",
    "Remove old results, logs, and legacy code",
    "ggTrader.cli.cmd_cleanup:register_cleanup_parser",
    "ggTrader.cli.cmd_cleanup:run_cleanup",
)
_command(
    "dashboard",
    "Launch the research dashboard",
    "ggTrader.cli.cmd_dashboard:register_dashboard_parser",
    "ggTrader.cli.cmd_dashboard:run_dashboard",
)
_command(
    "pnl-daily",
    "Generate daily PnL report and push to notifiers",
    "ggTrader.cli.cmd_pnl_daily:register_pnl_daily_parser",
    "ggTrader.cli.cmd_pnl_daily:run_pnl_daily",
)
_command(
    "repair",
    "Repair corrupted state or data",
    "ggTrader.cli.cmd_repair:register_repair_parser",
    "ggTrader.cli.cmd_repair:run_repair",
)
_command(
    "trade-report",
    "Generate trade-level reports",
    "ggTrader.cli.cmd_trade_report:register_trade_report_parser",
    "ggTrader.cli.cmd_trade_report:run_trade_report",
)
_command(
    "signals",
    "Inspect live or backtest signals",
    "ggTrader.cli.cmd_signals:register_signals_parser",
    "ggTrader.cli.cmd_signals:run_signals",
)
_command(
    "backtest-strategy",
    "Backtest a new-architecture Strategy from a YAML config",
    "ggTrader.cli.cmd_backtest_strategy:register_backtest_strategy_parser",
    "ggTrader.cli.cmd_backtest_strategy:run_backtest_strategy",
)


def main() -> None:
    """Backwards-compatible entry point referenced by ``[project.scripts]``."""
    app()


if __name__ == "__main__":
    main()
