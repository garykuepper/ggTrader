import os
import sys
from sqlalchemy import create_engine, text


def fast_clean():
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
    )

    print(f"Connecting to DB...")
    engine = create_engine(connection_string)

    with engine.begin() as conn:
        print("--- FAST Cleaning Database (TRUNCATE) ---")
        conn.execute(text("TRUNCATE TABLE ohlcv;"))
        print("Table truncated. All data cleared.")

        # Verify
        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows: {count}")


if __name__ == "__main__":
    fast_clean()
