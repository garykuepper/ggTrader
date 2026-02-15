import os
import sys
from sqlalchemy import create_engine, text


def clean_db():
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
    )

    print(f"Connecting to DB...")
    engine = create_engine(connection_string)

    with engine.begin() as conn:
        print("--- Cleaning Database ---")

        # 1. Delete 1970 data
        print("Deleting rows with timestamp < 2010-01-01...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE timestamp < '2010-01-01';"))
        print(f"Deleted {res.rowcount} rows.")

        # 2. Delete bad symbols (ending with _digit)
        # e.g. 1INCHEUR_1, 1INCHEUR_1440
        print("Deleting rows with bad symbols (ending in _digit)...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE symbol ~ '_[0-9]+$';"))
        print(f"Deleted {res.rowcount} rows.")

        # 3. Delete raw pairs that were not standardized? (Optional)
        # For now, just the known bad ones.

        # 4. Analyze remaining
        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows: {count}")

        # Sample symbols
        syms = conn.execute(
            text("SELECT DISTINCT symbol FROM ohlcv LIMIT 20;")
        ).fetchall()
        print("Sample symbols remaining:", [s[0] for s in syms])


if __name__ == "__main__":
    clean_db()
