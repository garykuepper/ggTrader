"""Command-line entry point for data integrity checks."""

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from psycopg2 import Error

from ggTrader.data import universe_db
from ggTrader.data.kaggle_import.guards import assert_connected_database, safe_error
from ggTrader.data.sec_import.verify import record, run_checks


def register_verify_data(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the verification command and its own options."""
    parser = subparsers.add_parser("verify-data", help="Check imported data and its source files")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Heavy check of file contents and stored rows. Run under scripts/run_capped.sh.",
    )
    parser.add_argument("--record", action="store_true", help="Save the result in the database")
    parser.add_argument("--out", type=Path, help="Write the full JSON report to this file")
    return parser


def run_verify_data(args: argparse.Namespace, url: str) -> int:
    """Print each check and optionally save the report to a file or the database."""
    if args.deep:
        print("Deep verification is a heavy job. Run it under scripts/run_capped.sh.", flush=True)
    try:
        url = universe_db.to_psycopg_url(url)
        conn = universe_db.connect(url)
    except Exception:  # noqa: BLE001 — connection errors may contain credentials
        print("Database connection failed.")
        return 1
    try:
        assert_connected_database(conn, unquote(urlparse(url).path.lstrip("/")))
        report = run_checks(conn, deep=args.deep)
        for check in report["checks"].values():
            outcome = "PASS" if check["ok"] else "FAIL"
            print(f"{outcome}  {check['detail']}")
        passed = sum(check["ok"] for check in report["checks"].values())
        print(f"{passed} of {len(report['checks'])} checks passed.")
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.record:
            check_id = record(conn, report, "deep" if args.deep else "ordinary")
            print(f"Saved verification report {check_id} in the database.")
        return 0 if report["ok"] else 1
    except (Error, ValueError, OSError) as exc:
        print(f"Verification command failed. {safe_error(exc)}")
        return 1
    finally:
        conn.close()
