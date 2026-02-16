import os
import sys
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.config import get_db_connection_string


def clean_db() -> None:
    """
    Cleans the OHLCV table by removing old or malformed data.
    """
    connection_string = get_db_connection_string()

    print("Connecting to database...")
    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    with engine.begin() as conn:
        print("--- Cleaning Database ---")

        # 1. Delete old data (pre-2010)
        print("Deleting rows with timestamp < 2010-01-01...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE timestamp < '2010-01-01';"))
        print(f"Deleted {res.rowcount} rows.")

        # 2. Delete symbols with interval suffixes (e.g., BTCUSD_1)
        print("Deleting rows with interval-suffixed symbols (e.g., _1, _4h)...")
        res = conn.execute(text("DELETE FROM ohlcv WHERE symbol ~ '_[0-9]+$';"))
        print(f"Deleted {res.rowcount} rows.")

        # 3. Analyze remaining data
        count = conn.execute(text("SELECT count(*) FROM ohlcv;")).scalar()
        print(f"Remaining rows in ohlcv: {count}")

        # Sample symbols
        syms = conn.execute(
            text("SELECT DISTINCT symbol FROM ohlcv LIMIT 20;")
        ).fetchall()
        print("Sample symbols remaining:", [s[0] for s in syms])


if __name__ == "__main__":
    clean_db()
