"""Browse and inspect ggTrader results stored in the TimescaleDB database."""

from __future__ import annotations

import argparse
import json
from typing import Optional

import pandas as pd
from sqlalchemy import text
from tabulate import tabulate

from ggTrader.utils.db_engine import create_db_engine


def get_latest_run_id(engine) -> Optional[str]:
    """
    Returns the most recent run_id from the database.
    """
    query = "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"Error fetching latest run: {e}")
        return None


def list_runs(engine) -> None:
    """
    Lists recent runs from the database.
    """
    query = """
        SELECT 
            run_id, 
            run_type, 
            timestamp,
            total_profit as profit,
            sharpe,
            sortino
        FROM runs
        ORDER BY timestamp DESC
        LIMIT 20
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


def show_run_details(engine, run_id: str) -> None:
    """
    Shows detailed information for a specific run.
    """
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
    row = run_info.iloc[0].to_dict()

    # Separate flat and nested fields
    flat_data = []
    nested_fields = ["parameters", "metadata"]

    for col, val in row.items():
        if col not in nested_fields:
            flat_data.append([col, val])

    # Print flat data as a table
    print(tabulate(flat_data, tablefmt="plain"))

    # Print nested fields with pretty formatting
    for field in nested_fields:
        if field in row and row[field]:
            print(f"\n{field}:")
            try:
                # If it's already a dict, just dump it
                if isinstance(row[field], dict):
                    print(json.dumps(row[field], indent=4))
                else:
                    # Try parsing if it's a string
                    parsed = json.loads(row[field])
                    print(json.dumps(parsed, indent=4))
            except (ValueError, TypeError):
                print(row[field])

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
        "SELECT window_id, test_start, test_end, profit, return_pct, sharpe, sortino "
        "FROM wfo_windows WHERE run_id = %(run_id)s ORDER BY window_id",
        engine,
        params={"run_id": run_id},
    )
    if not windows.empty:
        print("\n--- WFO WINDOWS ---")
        # Format percentages and floats
        display_df = windows.copy()
        if "return_pct" in display_df.columns:
            display_df["return_pct"] = display_df["return_pct"].map("{:.2f}%".format)
        print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))


def main() -> None:
    """
    Main entry point for browsing results.
    """
    parser = argparse.ArgumentParser(description="ggTrader Results Browser (PostgreSQL)")
    parser.add_argument(
        "--conn",
        type=str,
        help="PostgreSQL connection string (optional)",
    )
    parser.add_argument("--list", action="store_true", help="List recent runs")
    parser.add_argument("run_id", type=str, nargs="?", help="Show details for a specific run ID")

    args = parser.parse_args()
    engine = create_db_engine(args.conn)

    if args.run_id:
        show_run_details(engine, args.run_id)
    elif args.list:
        list_runs(engine)
    else:
        # Default behavior: Show summary of recent runs + details for latest
        list_runs(engine)
        latest_id = get_latest_run_id(engine)
        if latest_id:
            print("\n" + "=" * 50)
            print(f"LATEST RUN DETAIL: {latest_id}")
            print("=" * 50)
            show_run_details(engine, latest_id)
        else:
            print("No runs found in database.")


if __name__ == "__main__":
    main()
