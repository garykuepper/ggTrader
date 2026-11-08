"""``ggt db load-sp500`` and ``ggt db membership ...``: the CLI face of the membership database."""

from __future__ import annotations

import argparse
import gzip
import sys
from datetime import datetime
from pathlib import Path

from ggTrader.data.membership import FixtureMembershipSource, MembershipError
from ggTrader.data.universe_algebra import HistoryValidationError
from ggTrader.data.universe_db import (
    LOAD_SOURCES,
    LoadRefused,
    LoadReport,
    LoadRequest,
    apply_ddl,
    connect,
    git_sha,
    load_vintage,
    verify_vintage,
)
from ggTrader.utils.paths import REPO_ROOT

#: The committed vintage-1 file (immutable fixture; spec section 4).
DEFAULT_HISTORY_FILE = REPO_ROOT / "data" / "universe" / "sp500_constituents_history.csv.gz"
#: ``ggt db load-sp500`` exit codes.
EXIT_REFUSED = 2


def _parse_instant(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError("--retrieved-at needs a zone, e.g. ...T00:00:00Z")
    return value


def register_membership_parsers(db_subs: argparse._SubParsersAction) -> None:
    """Add ``load-sp500`` and ``membership`` to the ``ggt db`` subcommands."""
    load = db_subs.add_parser(
        "load-sp500",
        help="Load an S&P 500 membership file as the next vintage (dry run unless --commit)",
        description=(
            "Validate a date,tickers file and store it as a new vintage in schema universe. "
            "Dry run by default: prints the diff against the latest vintage. --commit writes."
        ),
    )
    load.add_argument(
        "--from-file", type=Path, required=True, help="CSV or CSV.GZ (upstream fetch is not built)"
    )
    load.add_argument("--source", choices=LOAD_SOURCES, default="fixture-import")
    load.add_argument("--source-url", default=None, help="Where the bytes came from")
    load.add_argument(
        "--retrieved-at",
        type=_parse_instant,
        required=True,
        help="ISO instant with a zone, e.g. 2026-06-09T00:00:00Z",
    )
    load.add_argument("--upstream-commit", default=None, help="40-hex upstream git commit")
    load.add_argument("--upstream-etag", default=None)
    load.add_argument("--expect-sha256", default=None, help="sha256 of the uncompressed bytes")
    load.add_argument("--note", default=None)
    load.add_argument(
        "--parent",
        default=None,
        help="Pin the latest vintage seen in the dry run (an id, or 'none' for an empty index)",
    )
    load.add_argument("--force", action="store_true", help="Allow a refused (suspicious) load")
    load.add_argument("--reason", default=None, help="Why --force is right (stored)")
    load.add_argument("--commit", action="store_true", help="Write the vintage")
    load.add_argument("--report-json", type=Path, default=None, help="Save the diff report")

    member = db_subs.add_parser("membership", help="Inspect and verify stored membership vintages")
    subs = member.add_subparsers(dest="membership_command", required=True)
    subs.add_parser("apply-ddl", help="Create/refresh schema universe (idempotent)")
    verify = subs.add_parser(
        "verify", help="Compare a stored vintage with a file, on every snapshot date"
    )
    verify.add_argument("--load-id", type=int, required=True)
    against = verify.add_mutually_exclusive_group()
    against.add_argument(
        "--against-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Constituent file to compare with (default: the committed vintage-1 file)",
    )
    against.add_argument(
        "--against-fixture",
        default=None,
        metavar="NAME",
        help="Compare with a fixture from data/universe/membership/MANIFEST.json "
        "(its hashes are checked first), e.g. sp500_v1_2026-06-02",
    )


def _read_input(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def _parent_pin(text: str | None) -> int | str | None:
    if text is None or text == "none":
        return text
    return int(text)


def print_load_report(report: LoadReport) -> None:
    """Print the human summary of a dry run or a committed load."""
    verb = "COMMITTED" if report.committed else "DRY RUN (nothing written)"
    print(f"{verb}: index={report.index_name} no_op={report.no_op}")
    if report.load_id is not None:
        print(f"  load_id            {report.load_id}")
    print(f"  parent             {report.parent_load_id}")
    print(f"  coverage           {report.coverage_start} .. {report.coverage_end}")
    print(f"  snapshot dates     {report.n_snapshot_dates}")
    print(f"  intervals          {report.n_intervals} ({report.n_symbols} symbols)")
    print(f"  members per date   {report.min_members} .. {report.max_members}")
    print(f"  rows added/superseded  {report.rows_added} / {report.rows_superseded}")
    print(f"  raw sha256         {report.raw_sha256}")
    print(f"  content sha256     {report.content_sha256}")
    if report.parent_load_id is not None:
        print(
            f"  dates only new/parent  {len(report.dates_only_in_new)} / "
            f"{len(report.dates_only_in_parent)}; changed dates {len(report.dates_changed)}"
        )
        print(f"  historical rewrites {report.historical_rewrites}")
        print(f"  tail changes        {report.tail_changes}")
        print(f"  renames             {report.renames}")
    for reason in report.refusal_reasons:
        print(f"  needs --force: {reason}")


def run_load_sp500(args: argparse.Namespace) -> int:
    """Run ``ggt db load-sp500``; returns the process exit code."""
    request = LoadRequest(
        raw=_read_input(args.from_file),
        source=args.source,
        source_url=args.source_url or f"file:{args.from_file}",
        retrieved_at=args.retrieved_at,
        upstream_commit=args.upstream_commit,
        upstream_etag=args.upstream_etag,
        note=args.note,
        expect_sha256=args.expect_sha256,
        parent=_parent_pin(args.parent),  # type: ignore[arg-type]
        force=args.force,
        force_reason=args.reason,
        commit=args.commit,
        loader_git_sha=git_sha(),
    )
    conn = connect()
    try:
        report = load_vintage(conn, request)
    except (LoadRefused, HistoryValidationError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    finally:
        conn.close()
    print_load_report(report)
    if args.report_json is not None:
        report.save(args.report_json)
        print(f"  report saved to {args.report_json}")
    return 0


def run_membership(args: argparse.Namespace) -> int:
    """Run ``ggt db membership <apply-ddl|verify>``; returns the process exit code."""
    if args.membership_command == "apply-ddl":
        conn = connect()
        try:
            apply_ddl(conn)
        finally:
            conn.close()
        print("schema universe is up to date")
        return 0
    path = args.against_file
    if args.against_fixture is not None:
        try:
            path = FixtureMembershipSource(args.against_fixture).path
        except MembershipError as exc:
            print(f"fixture {args.against_fixture!r} cannot be used: {exc}", file=sys.stderr)
            return 1
    conn = connect(read_only=True)
    try:
        report = verify_vintage(conn, args.load_id, path)
    finally:
        conn.close()
    for check in report.checks:
        print(f"  [{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    print(f"verify load {report.load_id}: {'OK' if report.ok else 'FAILED'}")
    return 0 if report.ok else 1
