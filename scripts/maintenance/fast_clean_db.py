import os
from sqlalchemy import create_engine, text


def fast_clean() -> None:
    """
    Quickly clears the OHLCV table using TRUNCATE.
    """
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://gary_admin:your_secure_password@localhost:5433/ggtrader",
    )

    print("Connecting to database...")
    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    with engine.begin() as conn:
        print("--- Clearning OHLCV Table (TRUNCATE) ---")
        conn.execute(text("TRUNCATE TABLE ohlcv;"))
        print("Table truncated. All data cleared.")

        # Verify
        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows in ohlcv: {count}")


if __name__ == "__main__":
    fast_clean()
