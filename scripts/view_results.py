import sys
import os
import argparse
import duckdb
import pandas as pd
from tabulate import tabulate
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def list_runs(db_path):
    """Lists recent runs from the database."""
    query = """
        SELECT 
            run_id, 
            run_type, 
            timestamp,
            (SELECT metric_value FROM performance_metrics m WHERE m.run_id = r.run_id AND m.metric_name IN ('profit_pct', 'return_pct', 'best_value') LIMIT 1) as summary_metric
        FROM runs r
        ORDER BY timestamp DESC
        LIMIT 15
    """
    with duckdb.connect(db_path, read_only=True) as conn:
        df = conn.execute(query).df()

    if df.empty:
        print("No runs found in database.")
    else:
        print("\n--- RECENT RUNS ---")
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def show_run_details(db_path, run_id):
    """Shows detailed information for a specific run."""
    with duckdb.connect(db_path, read_only=True) as conn:
        # Run info
        run_info = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).df()
        if run_info.empty:
            print(f"Run ID {run_id} not found.")
            return

        print("\n--- RUN METADATA ---")
        for col in run_info.columns:
            print(f"{col}: {run_info[col].iloc[0]}")

        # Metrics
        metrics = conn.execute(
            "SELECT metric_name, metric_value FROM performance_metrics WHERE run_id = ?",
            (run_id,),
        ).df()
        if not metrics.empty:
            print("\n--- PERFORMANCE METRICS ---")
            print(tabulate(metrics, headers="keys", tablefmt="github", showindex=False))

        # WFO Windows if applicable
        windows = conn.execute(
            "SELECT window_id, test_start, test_end, return_pct, sharpe, sortino FROM wfo_windows WHERE run_id = ? ORDER BY window_id",
            (run_id,),
        ).df()
        if not windows.empty:
            print("\n--- WFO WINDOWS ---")
            print(tabulate(windows, headers="keys", tablefmt="github", showindex=False))


def main():
    parser = argparse.ArgumentParser(description="ggTrader Results Browser")
    parser.add_argument(
        "--db",
        type=str,
        default="results/trading_results.db",
        help="Path to results database",
    )
    parser.add_argument("--list", action="store_true", help="List recent runs")
    parser.add_argument("--run-id", type=str, help="Show details for a specific run ID")

    args = parser.parse_args()
    db_path = str(Path(args.db).absolute())

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    if args.run_id:
        show_run_details(db_path, args.run_id)
    elif args.list or len(sys.argv) == 1:
        list_runs(db_path)


if __name__ == "__main__":
    main()
