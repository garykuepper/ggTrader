
import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import text

from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.db_engine import create_db_engine_or_exit


def register_db_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'db' subcommand."""
    parser = subparsers.add_parser("db", help="Database administration and maintenance")
    db_subs = parser.add_subparsers(dest="db_command", required=True)

    # Subcommand: ggt db diag
    db_subs.add_parser("diag", help="Check database and table sizes/stats")

    # Subcommand: ggt db clean
    db_subs.add_parser("clean", help="Remove old or malformed OHLCV data")

    # Subcommand: ggt db truncate
    db_subs.add_parser("truncate", help="WIPE ALL OHLCV DATA (Fast)")

    # Subcommand: ggt db compression
    comp = db_subs.add_parser("compression", help="Manage TimescaleDB compression")
    comp.add_argument("--enable", action="store_true", help="Enable compression policy")
    comp.add_argument("--status", action="store_true", help="Check compression stats")

    # Subcommand: ggt db export
    exp = db_subs.add_parser("export", help="Export DB via pg_dump")
    exp.add_argument("--jobs", "-j", type=int, default=4, help="Parallel jobs")
    exp.add_argument("--remote", "-r", type=str, help="Remote host for rsync")

def run_db(args: argparse.Namespace):
    """Executes database administration commands."""
    if args.db_command == "diag":
        _db_diag()
    elif args.db_command == "clean":
        _db_clean()
    elif args.db_command == "truncate":
        _db_truncate()
    elif args.db_command == "compression":
        if args.enable:
            _db_enable_compression()
        else:
            _db_check_compression()
    elif args.db_command == "export":
        _db_export(args)

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
                print(dict(zip(result.keys(), row)))

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
        conn.execute(text("""
            ALTER TABLE ohlcv SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol, interval',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
        """))
        conn.execute(text("SELECT add_compression_policy('ohlcv', INTERVAL '7 days');"))
        print("Compression policy added (7-day window).")

def _db_check_compression():
    engine = create_db_engine_or_exit()
    with engine.connect() as conn:
        print("\n--- Compression Stats ---")
        result = conn.execute(text("SELECT * FROM hypertable_compression_stats('ohlcv');"))
        for row in result:
            print(row)

def _db_export(args):
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/exports") / f"ggtrader_dump_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try common pg_dump paths for Windows
    pg_dump = shutil.which("pg_dump") or r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe"

    cmd = [
        pg_dump, "-h", db_params["host"], "-p", db_params["port"],
        "-U", db_params["user"], "-Fd", "-j", str(args.jobs), "-f", str(out_dir),
        db_params["dbname"]
    ]

    env = os.environ.copy()
    if db_params["password"]:
        env["PGPASSWORD"] = db_params["password"]

    print(f"\nExecuting parallel dump of {db_params['dbname']} to {out_dir}...")
    subprocess.run(cmd, env=env, check=True)
    print("Export complete.")
