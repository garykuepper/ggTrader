import duckdb
import os
import pandas as pd
from pathlib import Path


def find_project_root():
    current = Path(os.getcwd()).absolute()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def ingest_daily_movers(
    source_parquet="data/historical_movers/historical_movers.parquet",
):
    root = find_project_root()
    source_db = root / "data" / "ggtrader.db"
    target_db = root / "data" / "daily_movers.db"
    source_parquet = root / source_parquet

    # 1. Pull data from source
    df = pd.DataFrame()

    print("Attempting to pull data from OHLCV DuckDB...")
    stables = (
        "USDT",
        "USDC",
        "DAI",
        "PYUSD",
        "EUR",
        "GBP",
        "USDG",
        "USDP",
        "TUSD",
        "AUD",
        "CAD",
        "JPY",
    )

    try:
        # Use a read-only connection
        src_conn = duckdb.connect(str(source_db), read_only=True)
        query = f"""
            WITH daily_data AS (
                SELECT 
                    timestamp::DATE as date,
                    base as symbol,
                    volume,
                    trades
                FROM ohlcv
                WHERE interval = '1d'
                AND trades > 500
                AND base NOT IN {stables}
            ),
            ranked AS (
                SELECT 
                    *,
                    row_number() OVER (PARTITION BY date ORDER BY volume DESC) as rank
                FROM daily_data
            )
            SELECT date, symbol, volume, trades
            FROM ranked
            WHERE rank <= 100
            ORDER BY date ASC, volume DESC
        """
        df = src_conn.execute(query).df()
        src_conn.close()
        print(f"Successfully pulled {len(df)} rows from OHLCV DuckDB.")
    except Exception as e:
        print(f"Locked or error reading OHLCV DuckDB: {e}")
        if source_parquet.exists():
            print(f"Falling back to legacy Parquet: {source_parquet}")
            df_full = pd.read_parquet(source_parquet)
            # Match schema: date, symbol, volume, trades
            df = df_full[["date", "symbol", "volume", "trades"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            print(f"Successfully pulled {len(df)} rows from Parquet.")
        else:
            print("Error: No source data available (DB locked and Parquet missing).")
            return

    if df.empty:
        print("No data found to migrate.")
        return

    # 2. Insert into target DB
    print(f"Initializing target DB: {target_db}")
    tgt_conn = duckdb.connect(str(target_db))
    tgt_conn.execute(
        "CREATE TABLE IF NOT EXISTS daily_movers (date DATE, symbol VARCHAR, volume DOUBLE, trades INTEGER)"
    )
    tgt_conn.execute("DELETE FROM daily_movers")

    # Register the dataframe and insert
    tgt_conn.execute("INSERT INTO daily_movers SELECT * FROM df")

    # Create index
    tgt_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_movers_date ON daily_movers (date)"
    )

    count = tgt_conn.execute("SELECT COUNT(*) FROM daily_movers").fetchone()[0]
    dates = tgt_conn.execute("SELECT MIN(date), MAX(date) FROM daily_movers").fetchone()

    print(f"Successfully migrated {count} records to {target_db}.")
    print(f"Period covered: {dates[0]} to {dates[1]}")
    tgt_conn.close()


if __name__ == "__main__":
    ingest_daily_movers()
