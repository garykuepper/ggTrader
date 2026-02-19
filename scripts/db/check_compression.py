"""
Queries and displays TimescaleDB compression statistics for the OHLCV table.
"""

import os
import sys
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.config import get_db_connection_string


def check_compression_status() -> None:
    """
    Connects to the database and prints compression statistics for the 'ohlcv' table.
    """
    connection_string = get_db_connection_string()

    print(f"Connecting to database at: {connection_string.split('@')[-1]}")
    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # 1. Hypertable Compression Stats
    try:
        with engine.connect() as conn:
            # 1. Hypertable Compression Stats
            print("\n--- TimescaleDB Hypertable Compression Stats (ohlcv) ---")
            result = conn.execute(
                text("SELECT * FROM hypertable_compression_stats('ohlcv');")
            )
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

            # 2. All Background Job Statistics
            print("\n--- All TimescaleDB Background Job Stats ---")
            job_query = text(
                """
                SELECT 
                    j.job_id,
                    j.proc_name,
                    j.hypertable_name,
                    js.last_run_started_at,
                    js.last_run_status,
                    js.last_run_duration,
                    js.next_start,
                    js.total_runs,
                    js.total_failures
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

            # 3. All Active Database Sessions (Non-Idle)
            print("\n--- All Active Database Sessions (Non-Idle) ---")
            active_query = text(
                """
                SELECT 
                    pid,
                    usename,
                    now() - query_start AS duration,
                    wait_event_type,
                    wait_event,
                    state,
                    backend_type,
                    query
                FROM pg_stat_activity
                WHERE state != 'idle' 
                AND pid != pg_backend_pid();
            """
            )
            result = conn.execute(active_query)
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                for row in rows:
                    print("-" * 40)
                    for col, val in zip(columns, row):
                        # Truncate long queries for readability
                        if col == "query" and val and len(str(val)) > 100:
                            val = str(val)[:100] + "..."
                        print(f"{col:25}: {val}")
            else:
                print("No active (non-idle) database sessions found.")
                print(
                    "\nNOTE: High WSL resource usage (vmmemwsl) and disk I/O are normal during"
                )
                print(
                    "heavy data operations even if Postgres isn't actively running a loop."
                )

    except Exception as e:
        print(f"Error querying database: {e}")


if __name__ == "__main__":
    check_compression_status()
