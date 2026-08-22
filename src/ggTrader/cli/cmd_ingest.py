import argparse
import sys

PARKED_MESSAGE = (
    "ggt ingest is parked and does not run: crypto ingestion was shelved along with the rest "
    "of the crypto execution arc. Equity data loads on demand via CachedYFinanceLoader "
    "(src/ggTrader/data/live/cached_yfinance_loader.py) -- there is nothing to run by hand. "
    "See docs/cli_reference.md Section 3 for details."
)


def register_ingest_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'ingest' subcommand."""
    parser = subparsers.add_parser(
        "ingest", help="[PARKED, non-functional] historical crypto OHLCV ingestion"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to ingest (default: 30)"
    )


def run_ingest(args: argparse.Namespace) -> None:
    """Refuse to run: ingest is a parked, non-functional stub.

    This command never actually synced data -- it printed "Ingestion complete."
    while the real sync call was commented out. Rather than continue to lie
    about success, it now fails loudly and exits non-zero.
    """
    print(PARKED_MESSAGE, file=sys.stderr)
    sys.exit(1)
