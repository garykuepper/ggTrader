"""``ggt db``: database administration and S&P 500 membership commands."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import text

from ggTrader.cli.cmd_membership import (
    register_membership_parsers,
    run_load_sp500,
    run_membership,
)
from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.db_engine import create_db_engine_or_exit


def _env_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise argparse.ArgumentTypeError("expected an environment variable name")
    return value


def register_db_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: PLR0915
    """Registers the 'db' subcommand."""
    parser = subparsers.add_parser("db", help="Database administration and maintenance")
    db_subs = parser.add_subparsers(dest="db_command", required=True)

    # Subcommand: ggt db diag
    db_subs.add_parser("diag", help="Check database and table sizes/stats")

    # Subcommand: ggt db clean
    clean = db_subs.add_parser("clean", help="Remove old or malformed OHLCV data")
    clean.add_argument("--yes", action="store_true", help="Confirm deletion")

    # Subcommand: ggt db truncate
    truncate = db_subs.add_parser("truncate", help="WIPE ALL OHLCV DATA (Fast)")
    truncate.add_argument("--yes-wipe-ohlcv", action="store_true", help="Confirm deletion")

    # Subcommand: ggt db compression
    comp = db_subs.add_parser("compression", help="Manage TimescaleDB compression")
    comp.add_argument("--enable", action="store_true", help="Enable compression policy")
    comp.add_argument("--status", action="store_true", help="Check compression stats")

    # Subcommand: ggt db export
    exp = db_subs.add_parser("export", help="Export DB via pg_dump")
    exp.add_argument("--jobs", "-j", type=int, default=4, help="Parallel jobs")
    exp.add_argument("--remote", "-r", type=str, help="Remote host for rsync")

    # Subcommands: ggt db load-sp500, ggt db membership (versioned S&P 500 membership)
    register_membership_parsers(db_subs)
    importer = db_subs.add_parser("import", help="Import market datasets")
    import_subs = importer.add_subparsers(dest="import_command", required=True)
    ddl = import_subs.add_parser("apply-ddl", help="Apply import schemas")
    mcgill = import_subs.add_parser("mcgill", help="Load McGill CSV")
    mcgill.add_argument("--file", type=Path, required=True)
    mcgill.add_argument("--sums", type=Path, required=True)
    mcgill_backup = mcgill.add_mutually_exclusive_group(required=True)
    mcgill_backup.add_argument("--backup-dir", type=Path)
    mcgill_backup.add_argument("--no-backup-check", action="store_true")
    mcgill.add_argument("--db-free-path", type=Path, default=Path("/"))
    crsp = import_subs.add_parser("crsp", help="Load CRSP parquet")
    crsp.add_argument("--file", type=Path, required=True)
    crsp.add_argument("--sums", type=Path, required=True)
    crsp_backup = crsp.add_mutually_exclusive_group(required=True)
    crsp_backup.add_argument("--backup-dir", type=Path)
    crsp_backup.add_argument("--no-backup-check", action="store_true")
    crsp.add_argument("--db-free-path", type=Path, default=Path("/"))
    crsp.add_argument("--year", type=int)
    crsp.add_argument("--row-groups", type=lambda value: [int(x) for x in value.split(",")])
    derive = import_subs.add_parser("derive-bars", help="Derive CRSP bars")
    derive.add_argument("--crsp-load", type=int, required=True)
    derive.add_argument("--calendar-file", type=Path, required=True)
    derive_backup = derive.add_mutually_exclusive_group(required=True)
    derive_backup.add_argument("--backup-dir", type=Path)
    derive_backup.add_argument("--no-backup-check", action="store_true")
    derive.add_argument("--db-free-path", type=Path, required=True)
    release = import_subs.add_parser("release", help="Build, withdraw or list data releases")
    release_subs = release.add_subparsers(dest="release_command", required=True)
    release_build = release_subs.add_parser("build", help="Build and publish a release")
    release_build.add_argument("--crsp-bars-load", type=int, required=True)
    release_build.add_argument("--mcgill-load", type=int)
    release_build.add_argument("--links", type=Path, required=True)
    release_withdraw = release_subs.add_parser("withdraw", help="Withdraw a release")
    release_withdraw.add_argument("--release-id", type=int, required=True)
    release_list = release_subs.add_parser("list", help="List releases")
    for command in (ddl, crsp, mcgill, derive, release_build, release_withdraw, release_list):
        target = command.add_mutually_exclusive_group(required=True)
        target.add_argument(
            "--db-url-env",
            type=_env_name,
            metavar="NAME",
            help="Environment variable containing the database connection",
        )
        target.add_argument("--use-configured-db", action="store_true")
        command.add_argument("--yes", action="store_true", help="Confirm write")


def resolve_import_url(args: argparse.Namespace) -> str:
    """Resolve and validate the explicitly selected import target."""
    from ggTrader.data.kaggle_import.guards import checked_pg_url

    if args.use_configured_db:
        value = get_db_connection_string()
    else:
        value = os.environ.get(args.db_url_env)
        if not value:
            sys.exit(f"Environment variable {args.db_url_env} is unset or empty")
    value = value.replace("postgresql+psycopg2://", "postgresql://", 1)
    return checked_pg_url(value)


def run_db(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915, C901 — existing command dispatcher
    """Executes database administration commands."""
    if args.db_command == "diag":
        _db_diag()
    elif args.db_command == "clean":
        if not args.yes:
            print(
                "Would delete ohlcv rows where timestamp < '2010-01-01' or "
                "symbol ~ '_[0-9]+$'; pass --yes to proceed."
            )
            sys.exit(2)
        _db_clean()
    elif args.db_command == "truncate":
        if not args.yes_wipe_ohlcv:
            print("Would delete all rows of ohlcv; pass --yes-wipe-ohlcv to proceed.")
            sys.exit(2)
        _db_truncate()
    elif args.db_command == "compression":
        if args.enable:
            _db_enable_compression()
        else:
            _db_check_compression()
    elif args.db_command == "export":
        _db_export(args)
    elif args.db_command == "load-sp500":
        _exit_unless_zero(run_load_sp500(args))
    elif args.db_command == "import":
        from ggTrader.data.kaggle_import.ddl import apply_ddl
        from ggTrader.data.universe_db import connect, to_psycopg_url

        url = resolve_import_url(args)
        if args.import_command in ("crsp", "mcgill", "derive-bars") and args.no_backup_check:
            from psycopg2.extensions import parse_dsn

            try:
                dbname = parse_dsn(to_psycopg_url(url)).get("dbname")
            except Exception:  # noqa: BLE001 — connection text can contain credentials
                sys.exit("Invalid database connection")
            if dbname == "ggtrader":
                sys.exit("Backup check cannot be disabled for ggtrader")
        if args.import_command == "release" and args.release_command != "list" and not args.yes:
            print("Would change a release; pass --yes to proceed.")
            return
        if not args.yes and args.import_command != "release":
            action = (
                "apply import schemas"
                if args.import_command == "apply-ddl"
                else (
                    f"derive bars from load {args.crsp_load}"
                    if args.import_command == "derive-bars"
                    else f"import {args.file}"
                )
            )
            print(f"Would {action}; pass --yes to proceed.")
            return
        try:
            url = to_psycopg_url(url)
            conn = connect(url)
        except Exception:  # noqa: BLE001 — connection errors can contain credentials
            sys.exit("Database connection failed")
        try:
            from ggTrader.data.kaggle_import.guards import assert_connected_database

            expected_db = unquote(urlparse(url).path.lstrip("/"))
            connected_db = assert_connected_database(conn, expected_db)
            if (
                args.import_command in ("crsp", "mcgill", "derive-bars")
                and args.no_backup_check
                and connected_db == "ggtrader"
            ):
                sys.exit("Backup check cannot be disabled for ggtrader")
            if args.import_command == "release":
                _run_release_import(conn, args)
            elif args.import_command == "apply-ddl":
                apply_ddl(conn)
            elif args.import_command == "mcgill":
                from ggTrader.data.kaggle_import.mcgill import load_mcgill

                load_id = load_mcgill(
                    conn,
                    args.file,
                    args.sums,
                    backup_dir=args.backup_dir,
                    db_free_path=args.db_free_path,
                )
                print(f"Load {load_id}")
            elif args.import_command == "derive-bars":
                from ggTrader.data.kaggle_import.crsp import compress_kaggle_chunks
                from ggTrader.data.kaggle_import.derive import derive_bars
                from ggTrader.data.kaggle_import.spells import calendar_from_file

                load_id = derive_bars(
                    conn,
                    args.crsp_load,
                    calendar_from_file(args.calendar_file),
                    backup_dir=args.backup_dir,
                    db_free_path=args.db_free_path,
                )
                print(f"Load {load_id}")
                print(f"Compressed {compress_kaggle_chunks(conn, 'market.daily_bars')} chunks")
            else:
                from ggTrader.data.kaggle_import.crsp import compress_kaggle_chunks, load_crsp

                load_id = load_crsp(
                    conn,
                    args.file,
                    args.sums,
                    backup_dir=args.backup_dir,
                    db_free_path=args.db_free_path,
                    row_groups=args.row_groups,
                    year=args.year,
                )
                print(f"Load {load_id}")
                if args.year is None and args.row_groups is None:
                    print(f"Compressed {compress_kaggle_chunks(conn)} chunks")
        finally:
            conn.close()
    elif args.db_command == "membership":
        _exit_unless_zero(run_membership(args))


def _run_release_import(conn: Any, args: argparse.Namespace) -> None:
    """Build, list, or withdraw releases at the explicitly selected target."""
    import json

    from ggTrader.data.kaggle_import.release import build_release, withdraw_release

    if args.release_command == "build":
        import pandas as pd

        from ggTrader.data.kaggle_import.links import APPROVED_LINK_SETTINGS

        rid = build_release(
            conn,
            crsp_bars_load=args.crsp_bars_load,
            mcgill_load=args.mcgill_load,
            links=pd.read_csv(args.links),
            settings=APPROVED_LINK_SETTINGS,
        )
        print(f"Release {rid}")
    elif args.release_command == "withdraw":
        withdraw_release(conn, args.release_id)
        print(f"Withdrawn release {args.release_id}")
        return
    else:
        rid = None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.release_id,r.created_at,r.status,r.yahoo_fingerprint, "
            "r.settings->'demoted_links', "
            "(SELECT jsonb_object_agg(kind,n) FROM (SELECT kind,count(*) n "
            "FROM market.stock_spans s WHERE s.release_id=r.release_id GROUP BY kind) counts) "
            "FROM ref.releases r WHERE r.status='published' AND "
            "(%s IS NULL OR r.release_id=%s) ORDER BY r.release_id",
            (rid, rid),
        )
        for release_id, created, status, fingerprint, demoted, counts in cur.fetchall():
            counts = {
                key: (counts or {}).get(key, 0)
                for key in ("linked_yahoo", "yahoo", "placeholder", "spell")
            }
            print(
                json.dumps(
                    {
                        "release_id": release_id,
                        "created_at": str(created),
                        "status": status,
                        "counts": counts,
                        "demoted_links": demoted or [],
                        "yahoo_fingerprint": fingerprint,
                    }
                )
            )


def _exit_unless_zero(code: int) -> None:
    """Turn a command's exit code into the process exit status."""
    if code:
        sys.exit(code)


def _db_diag():
    engine = create_db_engine_or_exit()
    queries = {
        "Total DB Size": "SELECT pg_size_pretty(pg_database_size('ggtrader'))",
        "Table Stats": """
            SELECT
                pg_size_pretty(pg_total_relation_size('ohlcv')) as total_size,
                pg_size_pretty(pg_relation_size('ohlcv')) as table_size,
                pg_size_pretty(pg_indexes_size('ohlcv')) as index_size,
                (SELECT count(*) FROM ohlcv) as row_count
         """,
    }
    with engine.connect() as conn:
        for name, query in queries.items():
            print(f"\n--- {name} ---")
            result = conn.execute(text(query))
            row = result.fetchone()
            if row:
                print(dict(zip(result.keys(), row, strict=False)))


def _db_clean():
    engine = create_db_engine_or_exit()
    print("\n--- Cleaning Database ---")
    with engine.begin() as conn:
        print("Deleting rows with timestamp < 2010-01-01...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE timestamp < '2010-01-01';"))
        print(f"Deleted {res.rowcount} rows.")

        print("Deleting rows with interval-suffixed symbols (e.g., _1, _4h)...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE symbol ~ '_[0-9]+$';"))
        print(f"Deleted {res.rowcount} rows.")


def _db_truncate():
    engine = create_db_engine_or_exit()
    print("\n--- Wiping OHLCV Table (TRUNCATE) ---")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE ohlcv;"))
        print("Table truncated. All data cleared.")


def _db_enable_compression():
    engine = create_db_engine_or_exit()
    with engine.begin() as conn:
        print("\n--- Enabling TimescaleDB Compression ---")
        conn.execute(
            text("""
            ALTER TABLE ohlcv SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol, interval',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
        """)
        )
        conn.execute(text("SELECT add_compression_policy('ohlcv', INTERVAL '7 days');"))
        print("ohlcv: compression policy added (7-day window).")

        # Live snapshot tables — segment by asset_class, compress after 30 days.
        for table in ("live_balance_snapshots", "live_positions_snapshot"):
            try:
                conn.execute(
                    text(f"""
                    ALTER TABLE {table} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'asset_class',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """)
                )
                conn.execute(text(f"SELECT add_compression_policy('{table}', INTERVAL '30 days');"))
                print(f"{table}: compression policy added (30-day window).")
            except Exception as e:
                print(f"{table}: skipped ({e})")


def _db_check_compression():
    engine = create_db_engine_or_exit()
    with engine.connect() as conn:
        print("\n--- Compression Stats ---")
        result = conn.execute(text("SELECT * FROM hypertable_compression_stats('ohlcv');"))
        for row in result:
            print(row)


def _db_export(args: argparse.Namespace) -> None:
    """Refactored logic for DB export via ggt CLI."""
    conn_str = get_db_connection_string()
    parsed = urlparse(conn_str)
    db_params = {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/"),
    }

    # astimezone() keeps the local wall clock in the folder name and marks the value aware (DTZ005).
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/exports") / f"ggtrader_dump_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidate = os.path.join(prog_files, "PostgreSQL", "16", "bin", "pg_dump.exe")
        pg_dump = candidate if os.path.exists(candidate) else "pg_dump"

    cmd = [
        pg_dump,
        "-h",
        db_params["host"],
        "-p",
        db_params["port"],
        "-U",
        db_params["user"],
        "-Fd",
        "-j",
        str(args.jobs),
        "-f",
        str(out_dir),
        db_params["dbname"],
    ]

    env = os.environ.copy()
    if db_params["password"]:
        env["PGPASSWORD"] = db_params["password"]

    print(f"\nExecuting parallel dump of {db_params['dbname']} to {out_dir}...")
    subprocess.run(cmd, env=env, check=True)
    print("Export complete.")
