import duckdb
import os
import pandas as pd
from pathlib import Path
from tabulate import tabulate


def find_project_root():
    current = Path(os.getcwd()).absolute()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def analyze_consistent_movers():
    root = find_project_root()
    movers_db = root / "data" / "daily_movers.db"
    ohlcv_db = root / "data" / "ggtrader.db"

    if not movers_db.exists() or not ohlcv_db.exists():
        print("Required databases not found.")
        return

    # Connect to movers DB and attach ohlcv DB
    # Using read_only=True is essential on Windows when other processes (like backtests) are running
    try:
        conn = duckdb.connect(str(movers_db), read_only=True)
        conn.execute(f"ATTACH '{ohlcv_db}' AS ohlcv_db (READ_ONLY)")
    except Exception as e:
        print(f"\nERROR: Could not connect to database: {e}")
        print(
            "This usually happens if another process (like your running backtest) has an exclusive lock."
        )
        print("Try stopping the long-running script or waiting for it to finish.")
        return

    print("Analyzing 2023-2025 mover consistency and OHLCV coverage...")

    # We want symbols that:
    # 1. Appeared most frequently in daily_movers between 2023-01-01 and 2025-12-31
    # 2. Have data coverage in ggtrader.db for roughly the same period

    query = """
    WITH movers_stats AS (
        -- Frequency of appearance in top movers
        SELECT 
            symbol, 
            COUNT(*) as frequency,
            MIN(date) as first_mover_appearance,
            MAX(date) as last_mover_appearance
        FROM daily_movers
        WHERE date BETWEEN '2023-01-01' AND '2025-12-31'
        GROUP BY symbol
    ),
    ohlcv_coverage AS (
        -- Data density in the main OHLCV table
        -- We filter by interval='1d' to match the movers frequency
        SELECT 
            base as symbol,
            COUNT(*) as total_days,
            MIN(timestamp::DATE) as data_start,
            MAX(timestamp::DATE) as data_end
        FROM ohlcv_db.ohlcv
        WHERE interval = '1d'
        AND timestamp BETWEEN '2023-01-01' AND '2025-12-31'
        GROUP BY base
    )
    SELECT 
        m.symbol,
        m.frequency,
        c.total_days as days_of_data,
        c.data_start,
        c.data_end
    FROM movers_stats m
    JOIN ohlcv_coverage c ON m.symbol = c.symbol
    WHERE c.total_days >= 1095 -- Ensure 3 full years of data coverage (2023, 2024, 2025)
    ORDER BY m.frequency DESC
    LIMIT 50
    """

    try:
        df = conn.execute(query).df()

        if df.empty:
            print(
                "No matching symbols found. Check your date ranges and data ingestion."
            )
        else:
            print("\n--- CONSISTENT TOP MOVERS (2023-2025) ---")
            print("Sorted by appearance frequency in daily top 100 volume list.")
            print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

            # Export to JSON for reference
            export_path = root / "data" / "top_50_consistent_movers.json"
            df.to_json(export_path, orient="records", indent=4)
            print(f"\nList saved to {export_path}")

    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    analyze_consistent_movers()
