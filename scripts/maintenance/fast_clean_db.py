import os
import sys
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.config import get_db_connection_string


def fast_clean() -> None:
    """
    Quickly clears the OHLCV table using TRUNCATE.
    """
    connection_string = get_db_connection_string()

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
