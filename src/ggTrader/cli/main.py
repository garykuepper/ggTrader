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
def ingest(ctx: typer.Context) -> None:
    """Ingest OHLCV data into TimescaleDB."""
    import argparse

    from ggTrader.cli.cmd_ingest import register_ingest_parser, run_ingest

    parser = argparse.ArgumentParser(prog="ggt", add_help=False)
    subs = parser.add_subparsers(dest="command")
    register_ingest_parser(subs)
    ns = parser.parse_args(["ingest", *ctx.args])
    run_ingest(ns)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
