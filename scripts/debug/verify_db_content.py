import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)


def verify_db() -> None:
    """
    Verifies the contents and schema of the PostgreSQL database.
    """
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://gary_admin:your_secure_password@localhost:5433/ggtrader",
    )

    print(f"Connecting to: {connection_string}")
    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"Failed to create engine: {e}")
        return

    try:
        with engine.connect() as conn:
            # 1. Check Table Exists
            print("\n--- Table Check ---")
            tables = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public';"
                )
            ).fetchall()
            table_names = [t[0] for t in tables]
            print("Tables found:", table_names)

            if "ohlcv" not in table_names:
                print("ERROR: 'ohlcv' table not found!")
                return

            # 2. Check Schema
            print("\n--- Schema Check ---")
            columns = conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = 'ohlcv';"
                )
            ).fetchall()
            for col in columns:
                print(f"  {col[0]}: {col[1]}")

            # 3. Check Row Count
            print("\n--- Row Count ---")
            try:
                count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
                print(f"Total rows in ohlcv: {count}")
            except Exception as e:
                print(f"Could not count rows: {e}")
                count = 0

            if count == 0:
                print("Table is empty.")
                return

            # 4. Check Intervals
            print("\n--- Distinct Intervals ---")
            intervals = conn.execute(
                text("SELECT DISTINCT interval FROM ohlcv;")
            ).fetchall()
            print([i[0] for i in intervals])

            # 5. Sample Data
            print("\n--- Sample Data (Limit 5) ---")
            df = pd.read_sql("SELECT * FROM ohlcv LIMIT 5", conn)
            print(df.to_string())

            # 6. Check a specific symbol
            print("\n--- Specific Symbol Check (Sample) ---")
            sample_symbol = conn.execute(
                text("SELECT symbol FROM ohlcv LIMIT 1")
            ).scalar()
            if sample_symbol:
                print(f"Querying for {sample_symbol}...")
                df_sym = pd.read_sql(
                    text(f"SELECT * FROM ohlcv WHERE symbol='{sample_symbol}' LIMIT 5"),
                    conn,
                )
                print(df_sym.to_string())

    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """
    Main orchestration for DB verification.
    """
    parser = argparse.ArgumentParser(description="Verify PostgreSQL DB Content")
    parser.parse_args()  # Allow --help
    verify_db()


if __name__ == "__main__":
    main()
