"""``ggt`` CLI — lab-first research toolkit."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="ggt",
    help="ggTrader — vectorbt research lab",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def lab(ctx: typer.Context) -> None:
    """Run a lab strategy walk-forward."""
    from ggTrader.lab.cli import run_lab

    run_lab(list(ctx.args))


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def db(ctx: typer.Context) -> None:
    """TimescaleDB management commands."""
    import argparse

    from ggTrader.cli.cmd_db import register_db_parser, run_db

    parser = argparse.ArgumentParser(prog="ggt", add_help=False)
    subs = parser.add_subparsers(dest="command")
    register_db_parser(subs)
    ns = parser.parse_args(["db", *ctx.args])
    run_db(ns)


@app.command()
def paper(
    live: bool = typer.Option(
        False, "--live", help="Submit real paper orders instead of dry-run logging."
    ),
    dry_run_writes: bool = typer.Option(
        False,
        "--dry-run-writes",
        help=(
            "Dry run only: also persist state (ledger, snapshot, peak, tape) as the pre-rebuild "
            "dry run did. Default is a read-only dry run that writes nothing."
        ),
    ),
    notify: bool = typer.Option(
        False,
        "--notify",
        help="Read-only dry run only: forward the DRY RUN lines to Telegram (prefixed).",
    ),
) -> None:
    """Run daily paper trading: generate ensemble signals and execute on Alpaca.

    Without --live this is a read-only dry run: intended orders are logged, nothing is
    submitted and nothing is written. The 12:45 cron stays disabled until the owner re-enables it.
    """
    if live and (dry_run_writes or notify):
        raise typer.BadParameter("--dry-run-writes and --notify only apply to a dry run")
    from ggTrader.paper.trader import LIVE_MEMBERSHIP_SPEC, run_paper_trading

    result = run_paper_trading(
        dry_run=not live,
        read_only=None if live else not dry_run_writes,
        notify=notify,
        membership=LIVE_MEMBERSHIP_SPEC,
    )
    print("Paper trading complete:")
    label = "Submitted" if live else "Planned"
    print(f"  {label} buys: {result['buys']}")
    print(f"  {label} sells: {result['sells']}")
    if result.get("untradable"):
        print(f"  Untradable (skipped): {result['untradable']}")
    if result.get("skipped_writes"):
        print(f"  Read-only: skipped writes {sorted(set(result['skipped_writes']))}")
    if result["errors"]:
        print(f"  Errors: {result['errors']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
