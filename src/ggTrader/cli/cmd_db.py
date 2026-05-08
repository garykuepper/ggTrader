
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

    # Subcommand: ggt db sync-live
    db_subs.add_parser("sync-live", help="Import live CSV data into TimescaleDB")

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

    # Subcommand: ggt db migrate-wfo-cache
    mwc = db_subs.add_parser(
        "migrate-wfo-cache",
        help="Import legacy results/wfo_cache/*.json files into the wfo_cache table",
    )
    mwc.add_argument(
        "--cache-dir",
        type=str,
        default="results/wfo_cache",
        help="Directory of legacy JSON files (default: results/wfo_cache)",
    )
    mwc.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete each JSON file after successful upsert (use with care)",
    )

    # Subcommand: ggt db purge-wfo-cache
    pwc = db_subs.add_parser(
        "purge-wfo-cache",
        help="Truncate the wfo_cache table (use after scoring config changes)",
    )
    pwc.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )

    # Subcommand: ggt db backfill-runs
    brn = db_subs.add_parser(
        "backfill-runs",
        help="Import legacy results/research/<run>/run_results.json + phase_stats.json into the runs table",
    )
    brn.add_argument(
        "--research-dir",
        type=str,
        default="results/research",
        help="Research directory (default: results/research)",
    )

def run_db(args: argparse.Namespace):
    """Executes database administration commands."""
    if args.db_command == "diag":
        _db_diag()
    elif args.db_command == "sync-live":
        _db_sync_live()
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
    elif args.db_command == "migrate-wfo-cache":
        _db_migrate_wfo_cache(args)
    elif args.db_command == "purge-wfo-cache":
        _db_purge_wfo_cache(args)
    elif args.db_command == "backfill-runs":
        _db_backfill_runs(args)

def _db_sync_live():
    from ggTrader.core.trade_tracker import TradeTracker
    from ggTrader.utils.result_db_manager import ResultDBManager

    print("\n--- Syncing Live CSV data to TimescaleDB ---")
    db_manager = ResultDBManager()
    tracker = TradeTracker(db_manager=db_manager, run_id="LIVE")
    
    counts = tracker.backfill_to_db()
    
    print(f"Sync complete:")
    print(f"  - Orders added: {counts['orders']}")
    print(f"  - Trades added: {counts['trades']}")
    print(f"  - Equity points: {counts['equity']}")

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
        print("ohlcv: compression policy added (7-day window).")

        # Live snapshot tables — segment by asset_class, compress after 30 days.
        for table in ("live_balance_snapshots", "live_positions_snapshot"):
            try:
                conn.execute(text(f"""
                    ALTER TABLE {table} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'asset_class',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                conn.execute(
                    text(f"SELECT add_compression_policy('{table}', INTERVAL '30 days');")
                )
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


def _db_migrate_wfo_cache(args):
    """One-time import of legacy results/wfo_cache/*.json into the wfo_cache table.

    Each filename stem is the cache_key (MD5 hex). Reads the JSON payload, infers
    symbol/strategy/exit/version from the payload itself, and upserts a row. Files
    that fail to parse are reported and skipped.
    """
    import json as _json

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        print(f"Cache dir not found: {cache_dir}")
        return

    files = sorted(cache_dir.glob("*.json"))
    if not files:
        print(f"No *.json files in {cache_dir} — nothing to migrate.")
        return

    engine = create_db_engine_or_exit()
    print(f"\n--- Migrating {len(files)} WFO cache files from {cache_dir} ---")

    inserted = 0
    skipped = 0
    failed = 0
    upsert_sql = text(
        """
        INSERT INTO wfo_cache (cache_key, symbol, strategy_name, exit_name,
                               cache_version, payload)
        VALUES (:k, :sym, :strat, :exit, :ver, CAST(:p AS JSONB))
        ON CONFLICT (cache_key) DO NOTHING
        """
    )

    with engine.begin() as conn:
        for fp in files:
            key = fp.stem
            try:
                payload = _json.loads(fp.read_text())
            except Exception as e:
                print(f"  SKIP {fp.name}: parse error ({e})")
                failed += 1
                continue
            try:
                res = conn.execute(
                    upsert_sql,
                    {
                        "k": key,
                        "sym": payload.get("symbol"),
                        "strat": payload.get("strategy"),
                        "exit": payload.get("exit"),
                        "ver": payload.get("version", 1),
                        "p": _json.dumps(payload),
                    },
                )
                if res.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  FAIL {fp.name}: {e}")
                failed += 1

    print(f"  inserted: {inserted}")
    print(f"  already-present (skipped): {skipped}")
    print(f"  failed: {failed}")

    if args.delete_after and failed == 0:
        print("Deleting migrated files…")
        for fp in files:
            try:
                fp.unlink()
            except OSError:
                pass
        try:
            cache_dir.rmdir()
            print(f"Removed empty {cache_dir}")
        except OSError:
            pass


def _db_purge_wfo_cache(args):
    """Truncate the wfo_cache table."""
    if not args.yes:
        ans = input("Truncate wfo_cache table? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
    engine = create_db_engine_or_exit()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE wfo_cache;"))
    print("wfo_cache truncated.")


def _db_backfill_runs(args):
    """Import historical results/research/<run>/ artifacts into the runs table.

    For each subdirectory we read run_results.json and (optionally) phase_stats.json,
    extract the asset_class / per-coin params / metrics, and upsert via add_run.
    """
    import json as _json
    from datetime import datetime

    from ggTrader.utils.result_db_manager import ResultDBManager

    research_dir = Path(args.research_dir)
    if not research_dir.is_dir():
        print(f"Research dir not found: {research_dir}")
        return

    run_dirs = sorted([d for d in research_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        print(f"No subdirectories under {research_dir}")
        return

    m = ResultDBManager()
    inserted = 0
    skipped = 0
    failed = 0

    for d in run_dirs:
        rj = d / "run_results.json"
        if not rj.exists():
            skipped += 1
            continue
        try:
            data = _json.loads(rj.read_text())
        except Exception as e:
            print(f"  SKIP {d.name}: parse error ({e})")
            failed += 1
            continue

        run_id = data.get("run_id") or d.name
        ts_str = data.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
        except Exception:
            ts = datetime.now()

        asset_class = data.get("asset_class") or data.get(
            "configuration", {}
        ).get("_raw_config", {}).get("ASSET_CLASS", "crypto")
        config = data.get("configuration", {})
        strategy_params = data.get("strategy_parameters", {})
        metrics = data.get("results", {})

        phase_stats = None
        ps_file = d / "phase_stats.json"
        if ps_file.exists():
            try:
                phase_stats = _json.loads(ps_file.read_text())
            except Exception:
                phase_stats = None

        try:
            m.add_run(
                run_id=run_id,
                run_type="research",
                script_name=data.get("script_name", "research"),
                parameters=config.get("_raw_config", config),
                metadata=config,
                metrics=metrics,
                pipeline_stage="research",
                asset_class=asset_class,
                strategy_params=strategy_params,
                phase_stats=phase_stats,
                run_dir=str(d),
                status="success",
                timestamp=ts,
            )
            inserted += 1
        except Exception as e:
            print(f"  FAIL {d.name}: {e}")
            failed += 1

    print(f"  inserted/upserted: {inserted}")
    print(f"  skipped (no run_results.json): {skipped}")
    print(f"  failed: {failed}")
