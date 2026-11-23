#!/usr/bin/env python3
"""Restore a verified database backup into a disposable database and check its contents."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2 import sql

from ggTrader.data.kaggle_import.guards import (
    assert_connected_database,
    checked_pg_url,
    verify_sha256,
)

BACKUP = Path("/media/thesix/ggTrader-data/ggtraderdb-backup-20261122")
TABLES = (
    "public.ohlcv",
    "universe.membership",
    "public.lab_runs",
    "public.paper_trades",
    "public.form4_transactions",
)


def parse_expected(path: Path) -> dict[str, int]:
    """Read the five exact source counts."""
    values = [int(value) for value in path.read_text().split()]
    if len(values) != len(TABLES) or any(value < 0 for value in values):
        raise ValueError("Expected five nonnegative row counts")
    return dict(zip(TABLES, values, strict=True))


def compare_counts(expected: dict[str, int], actual: dict[str, int | None]) -> dict:
    """Report presence and equality for each required table."""
    return {
        name: {"expected": count, "actual": actual.get(name), "ok": actual.get(name) == count}
        for name, count in expected.items()
    }


def restore_command(name: str) -> list[str]:
    """Use the container's local postgres socket without passing credentials."""
    return [
        "docker",
        "exec",
        "-i",
        "-u",
        "postgres",
        "ggtrader_db",
        "pg_restore",
        "-d",
        name,
        "--no-owner",
        "--jobs",
        "1",
    ]


def check_database_volume(url: str, dump: Path, report: dict) -> None:
    """Require three dump sizes plus 40 GB on the matched local database volume."""
    target = urlparse(checked_pg_url(url))
    if target.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Cannot match the database target to the local container")
    ports = subprocess.run(
        ["docker", "port", "ggtrader_db", "5432/tcp"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    local_addresses = {"0.0.0.0", "::", target.hostname}
    if target.hostname == "localhost":
        local_addresses.update(("127.0.0.1", "::1"))
    matches = (line.rsplit(":", 1) for line in ports if ":" in line)
    if not any(
        address.strip("[]") in local_addresses and port == str(target.port or 5432)
        for address, port in matches
    ):
        raise ValueError("Cannot match the database target to the local container")
    result = subprocess.run(
        ["docker", "exec", "ggtrader_db", "df", "-P", "/var/lib/postgresql/data"],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        free = int(result.stdout.splitlines()[-1].split()[3]) * 1024
    except (IndexError, ValueError) as exc:
        raise ValueError("Cannot determine database volume free space") from exc
    size = dump.stat().st_size
    required = 3 * size + 40_000_000_000
    report["database_volume"] = {
        "container": "ggtrader_db",
        "free_bytes": free,
        "dump_bytes": size,
        "required_bytes": required,
        "formula": "3 * dump_bytes + 40,000,000,000 bytes",
    }
    if free < required:
        raise ValueError("Insufficient database volume free space")


def run(args: argparse.Namespace, report: dict) -> None:
    """Verify the dump, restore and check it, and clean up even after failure."""
    checked_pg_url(args.scratch_url)
    verify_sha256(args.dump, args.sums)
    expected = parse_expected(args.expected)
    check_database_volume(args.scratch_url, args.dump, report)
    name = "ggt_restore_drill_" + uuid.uuid4().hex[:8]
    admin = psycopg2.connect(args.scratch_url)
    admin.autocommit = True
    created = False
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        report["database"] = name
        target = urlunparse(urlparse(args.scratch_url)._replace(path="/" + name))
        checked_pg_url(target, expect_db=name)
        conn = psycopg2.connect(target)
        try:
            assert_connected_database(conn, name)
            conn.rollback()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cur.execute("SELECT timescaledb_pre_restore()")
            with args.dump.open("rb") as dump:
                subprocess.run(
                    restore_command(name),
                    stdin=dump,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            with conn.cursor() as cur:
                cur.execute("SELECT timescaledb_post_restore()")
                cur.execute("ANALYZE")
                actual = {}
                for table in TABLES:
                    cur.execute("SELECT to_regclass(%s)", (table,))
                    if cur.fetchone()[0] is None:
                        actual[table] = None
                        continue
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            *(sql.Identifier(part) for part in table.split("."))
                        )
                    )
                    actual[table] = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM timescaledb_information.hypertables")
                report["hypertables"] = cur.fetchone()[0]
            report["tables"] = compare_counts(expected, actual)
            report["ok"] = report["hypertables"] > 0 and all(
                item["ok"] for item in report["tables"].values()
            )
        finally:
            conn.close()
    finally:
        try:
            if created and not args.keep:
                with admin.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                            sql.Identifier(name)
                        )
                    )
        finally:
            admin.close()


def main(argv: list[str] | None = None) -> int:
    """Run the drill and write a JSON report, including failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=BACKUP / "ggtrader_20261122.dump")
    parser.add_argument("--expected", type=Path, default=BACKUP / "rowcounts_before.txt")
    parser.add_argument("--sums", type=Path, default=None)
    parser.add_argument("--scratch-url-env", default="GGT_SCRATCH_URL", metavar="NAME")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.scratch_url_env):
        parser.error("expected an environment variable name")
    args.scratch_url = os.environ.get(args.scratch_url_env)
    if not args.scratch_url:
        print(f"Environment variable {args.scratch_url_env} is unset or empty", file=sys.stderr)
        return 1
    if args.sums is None:
        args.sums = args.dump.parent / "SHA256SUMS"
    started = time.monotonic()
    report = {"ok": False, "free_bytes_before": shutil.disk_usage("/").free}
    try:
        run(args, report)
    except Exception as exc:  # noqa: BLE001 - exception text may contain credentials
        report["ok"] = False
        report["error"] = type(exc).__name__ + ": restore drill failed"
        print(report["error"], file=sys.stderr)
    report["elapsed_seconds"] = time.monotonic() - started
    report["free_bytes_end"] = shutil.disk_usage("/").free
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        print(f"{type(exc).__name__}: unable to write report", file=sys.stderr)
        return 1
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
