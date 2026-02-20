import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy import create_engine, text

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ggTrader.utils.config import get_db_connection_string


def _connect():
    connection_string = get_db_connection_string()
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)


def cmd_clean(args):
    """Cleans the OHLCV table by removing old or malformed data."""
    engine = _connect()
    with engine.begin() as conn:
        print("--- Cleaning Database ---")
        print("Deleting rows with timestamp < 2010-01-01...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE timestamp < '2010-01-01';"))
        print(f"Deleted {res.rowcount} rows.")

        print("Deleting rows with interval-suffixed symbols (e.g., _1, _4h)...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE symbol ~ '_[0-9]+$';"))
        print(f"Deleted {res.rowcount} rows.")

        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows in ohlcv: {count}")

        syms = conn.execute(text("SELECT DISTINCT symbol FROM ohlcv LIMIT 20;")).fetchall()
        print("Sample symbols remaining:", [s[0] for s in syms])


def cmd_fast_clean(args):
    """Quickly clears the OHLCV table using TRUNCATE."""
    engine = _connect()
    with engine.begin() as conn:
        print("--- Clearing OHLCV Table (TRUNCATE) ---")
        conn.execute(text("TRUNCATE TABLE ohlcv;"))
        print("Table truncated. All data cleared.")
        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows in ohlcv: {count}")


def cmd_enable_compression(args):
    """Configures and enables TimescaleDB compression on the OHLCV table."""
    engine = _connect()
    with engine.begin() as conn:
        print("--- Configuring TimescaleDB Compression ---")
        print("1. Setting compression policy (Segment by: symbol, interval)...")
        try:
            conn.execute(
                text(
                    """
                ALTER TABLE ohlcv SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol, interval',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
                """
                )
            )
            print("   Success.")
        except Exception as e:
            print(f"   Note: {e}")

        print("2. Adding automatic compression policy (Chunks > 7 days)...")
        try:
            conn.execute(text("SELECT add_compression_policy('ohlcv', INTERVAL '7 days');"))
            print("   Success.")
        except Exception as e:
            print(f"   Note: {e}")
        print("\nCompression enabled. Background workers will process old data.")


def cmd_db_size_diag(args):
    """Prints DB and table sizes."""
    engine = _connect()
    queries = {
        "Total DB Size": "SELECT pg_size_pretty(pg_database_size('ggtrader'))",
        "Table Stats": """
            SELECT 
                pg_size_pretty(pg_total_relation_size('ohlcv')) as total_size,
                pg_size_pretty(pg_relation_size('ohlcv')) as table_size,
                pg_size_pretty(pg_indexes_size('ohlcv')) as index_size,
                (SELECT count(*) FROM ohlcv) as row_count
        """,
        "Check Compression Job": "SELECT * FROM timescaledb_information.job_stats WHERE job_id = 1000",
    }
    with engine.connect() as conn:
        for name, query in queries.items():
            print(f"\n--- {name} ---")
            result = conn.execute(text(query))
            row = result.fetchone()
            if row:
                print(dict(zip(result.keys(), row)))


def cmd_check_compression(args):
    """Displays TimescaleDB compression statistics for the OHLCV table."""
    engine = _connect()
    try:
        with engine.connect() as conn:
            print("\n--- TimescaleDB Hypertable Compression Stats (ohlcv) ---")
            result = conn.execute(text("SELECT * FROM hypertable_compression_stats('ohlcv');"))
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                for row in rows:
                    print("-" * 40)
                    for col, val in zip(columns, row):
                        if isinstance(val, int) and val > 1024:
                            if "bytes" in col.lower() or "size" in col.lower():
                                mb = val / (1024 * 1024)
                                print(f"{col:25}: {val} ({mb:.2f} MB)")
                                continue
                        print(f"{col:25}: {val}")
            else:
                print("No hypertable compression statistics found.")

            print("\n--- All TimescaleDB Background Job Stats ---")
            job_query = text(
                """
                SELECT 
                    j.job_id, j.proc_name, j.hypertable_name, js.last_run_started_at,
                    js.last_run_status, js.last_run_duration, js.next_start,
                    js.total_runs, js.total_failures
                FROM timescaledb_information.jobs j
                LEFT JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id
                ORDER BY js.last_run_started_at DESC NULLS LAST;
                """
            )
            result = conn.execute(job_query)
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                for row in rows:
                    print("-" * 40)
                    for col, val in zip(columns, row):
                        print(f"{col:25}: {val}")
            else:
                print("No background jobs found.")

            print("\n--- All Active Database Sessions (Non-Idle) ---")
            active_query = text(
                """
                SELECT pid, usename, now() - query_start AS duration, wait_event_type,
                    wait_event, state, backend_type, query
                FROM pg_stat_activity
                WHERE state != 'idle' AND pid != pg_backend_pid();
                """
            )
            result = conn.execute(active_query)
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                for row in rows:
                    print("-" * 40)
                    for col, val in zip(columns, row):
                        if col == "query" and val and len(str(val)) > 100:
                            val = str(val)[:100] + "..."
                        print(f"{col:25}: {val}")
            else:
                print("No active (non-idle) database sessions found.")
    except Exception as e:
        print(f"Error querying database: {e}")


def cmd_export(args):
    """Exports DB using pg_dump and syncs via rsync."""
    DEFAULT_DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exports")

    def parse_connection_string(conn_str: str) -> dict:
        parsed = urlparse(conn_str)
        return {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 5432),
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
            "dbname": parsed.path.lstrip("/"),
        }

    def find_pg_dump() -> str:
        pg_dump = shutil.which("pg_dump")
        if pg_dump:
            return pg_dump
        common_paths = [
            r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("pg_dump not found.")

    def run_parallel_pg_dump(db_params: dict, output_dir: str, jobs: int) -> str:
        pg_dump = find_pg_dump()
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
            str(jobs),
            "-Z",
            "0",
            "--verbose",
            "-f",
            output_dir,
            db_params["dbname"],
        ]
        env = os.environ.copy()
        if db_params["password"]:
            env["PGPASSWORD"] = db_params["password"]
        print(f"\n--- Starting Parallel Dump ({jobs} threads) ---")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n[ERROR] pg_dump failed.")
            sys.exit(1)
        return output_dir

    def transfer_via_rsync(local_path: str, remote_host: str, remote_path: str):
        cwd = os.getcwd()
        rel_path = os.path.relpath(local_path, cwd)
        rsync_cmd = ["rsync", "-razP", rel_path, f"{remote_host}:{remote_path}"]
        print(f"\n--- Syncing to {remote_host} ---")
        print(f"  Local path converted to: {rel_path}")
        start = time.time()
        result = subprocess.run(rsync_cmd)
        elapsed = time.time() - start
        if result.returncode != 0:
            print("\n[ERROR] Rsync failed. Falling back to SCP...")
            scp_cmd = ["scp", "-r", local_path, f"{remote_host}:{remote_path}"]
            subprocess.run(scp_cmd)
        else:
            print(f"✓ Transfer completed in {elapsed:.1f}s")

    conn_str = get_db_connection_string()
    db_params = parse_connection_string(conn_str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(os.path.join(DEFAULT_DUMP_DIR, f"ggtrader_dir_{timestamp}"))
    os.makedirs(output_path, exist_ok=True)

    run_parallel_pg_dump(db_params, output_path, jobs=args.jobs)
    if args.remote:
        transfer_via_rsync(output_path, args.remote, args.remote_path)

    dump_folder_name = os.path.basename(output_path)
    print(f"\n[DONE] Database successfully exported.")
    print(f"\n--- Remote Restore Instructions ---")
    print(f"1. docker cp {args.remote_path}{dump_folder_name} ggtrader_db:/tmp/")
    print(
        f"2. docker exec -it ggtrader_db pg_restore -U {db_params['user']} -d {db_params['dbname']} -j {args.jobs} /tmp/{dump_folder_name}"
    )


def main():
    parser = argparse.ArgumentParser(description="Manage the ggTrader database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: clean
    subparsers.add_parser("clean", help="Cleans the OHLCV table by removing old or malformed data.")

    # Subcommand: fast-clean / clear
    subparsers.add_parser(
        "fast-clean", help="Quickly clears the OHLCV table using TRUNCATE. (Same as clear)"
    )
    subparsers.add_parser("clear", help="Quickly clears the OHLCV table using TRUNCATE.")

    # Subcommand: enable-compression
    subparsers.add_parser(
        "enable-compression", help="Configures and enables TimescaleDB compression."
    )

    # Subcommand: check-compression
    subparsers.add_parser("check-compression", help="Displays TimescaleDB compression statistics.")

    # Subcommand: diag
    subparsers.add_parser("diag", help="Prints DB and table sizes.")

    # Subcommand: export
    parser_export = subparsers.add_parser(
        "export", help="Exports DB using pg_dump and syncs via rsync."
    )
    parser_export.add_argument(
        "--jobs", "-j", type=int, default=4, help="Number of jobs for pg_dump."
    )
    parser_export.add_argument("--remote", "-r", type=str, help="Remote host for rsync.")
    parser_export.add_argument(
        "--remote-path", default="/home/flynn/ggtraderdb/", help="Remote path for rsync."
    )

    args = parser.parse_args()

    try:
        if args.command == "clean":
            cmd_clean(args)
        elif args.command in ["fast-clean", "clear"]:
            cmd_fast_clean(args)
        elif args.command == "enable-compression":
            cmd_enable_compression(args)
        elif args.command == "check-compression":
            cmd_check_compression(args)
        elif args.command == "diag":
            cmd_db_size_diag(args)
        elif args.command == "export":
            cmd_export(args)
    except Exception as e:
        print(f"Error handling command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
