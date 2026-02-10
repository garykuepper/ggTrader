import duckdb
import os
import sys


def init_db(db_path="data/ggtrader.db"):
    """
    Initializes the DuckDB database and creates the OHLCV table.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    )

    # Connect to the database (creates it if it doesn't exist)
    conn = duckdb.connect(db_path)

    print(f"Initializing database at {db_path}...")

    # Create the OHLCV table
    sql_create = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp TIMESTAMPTZ,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume DOUBLE,
            trades BIGINT,
            base VARCHAR,
            quote VARCHAR,
            pair VARCHAR,
            interval VARCHAR,
            PRIMARY KEY (pair, interval, timestamp)
        );
    """

    try:
        conn.execute(sql_create)
        print("Table 'ohlcv' created or already exists.")

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Current tables: {tables}")

        if ("ohlcv",) in tables:
            # Create indexes for faster querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_interval ON ohlcv (pair, interval);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv (timestamp);"
            )
            print("Indexes created successfully.")
        else:
            print("Error: Table 'ohlcv' was not found after creation!")

    except Exception as e:
        print(f"Error during initialization: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
