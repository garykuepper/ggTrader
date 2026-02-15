import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.postgres_reader import KrakenPostgresReader


def verify_db():
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
    )

    print(f"Connecting to: {connection_string}")
    engine = create_engine(connection_string)

    try:
        with engine.connect() as conn:
            # 1. Check Table Exists
            print("\n--- Table Check ---")
            tables = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
                )
            ).fetchall()
            print("Tables found:", [t[0] for t in tables])

            if ("ohlcv",) not in tables:
                print("ERROR: 'ohlcv' table not found!")
                return

            # 2. Check Schema
            print("\n--- Schema Check ---")
            columns = conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'ohlcv';"
                )
            ).fetchall()
            for col in columns:
                print(f"  {col[0]}: {col[1]}")

            # 3. Check Row Count (Approximate if huge)
            print("\n--- Row Count (Exact) ---")
            try:
                count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
                print(f"Total rows in ohlcv: {count}")
            except Exception as e:
                print(f"Could not count rows: {e}")

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

            # 6. Check a specific symbol (e.g. BTC-USD)
            print("\n--- Specific Symbol Check (BTC-USD, 1d) ---")
            # Try to find a symbol that exists
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


if __name__ == "__main__":
    verify_db()
