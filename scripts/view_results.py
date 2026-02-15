import sys
import os
import argparse
import pandas as pd
from tabulate import tabulate
from sqlalchemy import create_engine, text

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def get_engine(conn_str=None):
    if not conn_str:
        conn_str = os.getenv(
            "POSTGRES_CONNECTION_STRING",
            "postgresql+psycopg2://gary_admin:your_secure_password@localhost:5433/ggtrader",
        )
    return create_engine(conn_str)


def list_runs(engine):
    """Lists recent runs from the database."""
    query = """
        SELECT 
            r.run_id, 
            r.run_type, 
            r.timestamp,
            (SELECT metric_value FROM performance_metrics m WHERE m.run_id = r.run_id AND m.metric_name IN ('profit_pct', 'return_pct', 'best_value') LIMIT 1) as summary_metric
        FROM runs r
        ORDER BY r.timestamp DESC
        LIMIT 15
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error querying database: {e}")
        return

    if df.empty:
        print("No runs found in database.")
    else:
        print("\n--- RECENT RUNS ---")
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def show_run_details(engine, run_id):
    """Shows detailed information for a specific run."""
    # Run info
    run_info = pd.read_sql(
        "SELECT * FROM runs WHERE run_id = %(run_id)s",
        engine,
        params={"run_id": run_id},
    )
    if run_info.empty:
        print(f"Run ID {run_id} not found.")
        return

    print("\n--- RUN METADATA ---")
    row = run_info.iloc[0]
    for col in run_info.columns:
        print(f"{col}: {row[col]}")

    # Metrics
    metrics = pd.read_sql(
        "SELECT metric_name, metric_value FROM performance_metrics WHERE run_id = %(run_id)s",
        engine,
        params={"run_id": run_id},
    )
    if not metrics.empty:
        print("\n--- PERFORMANCE METRICS ---")
        print(tabulate(metrics, headers="keys", tablefmt="github", showindex=False))

    # WFO Windows if applicable
    windows = pd.read_sql(
        "SELECT window_id, test_start, test_end, return_pct, sharpe, sortino FROM wfo_windows WHERE run_id = %(run_id)s ORDER BY window_id",
        engine,
        params={"run_id": run_id},
    )
    if not windows.empty:
        print("\n--- WFO WINDOWS ---")
        print(tabulate(windows, headers="keys", tablefmt="github", showindex=False))


def main():
    parser = argparse.ArgumentParser(
        description="ggTrader Results Browser (PostgreSQL)"
    )
    parser.add_argument(
        "--conn",
        type=str,
        help="PostgreSQL connection string (optional)",
    )
    parser.add_argument("--list", action="store_true", help="List recent runs")
    parser.add_argument(
        "run_id", type=str, nargs="?", help="Show details for a specific run ID"
    )

    args = parser.parse_args()
    engine = get_engine(args.conn)

    if args.run_id:
        show_run_details(engine, args.run_id)
    else:
        # Default behavior is listing
        list_runs(engine)


if __name__ == "__main__":
    main()
