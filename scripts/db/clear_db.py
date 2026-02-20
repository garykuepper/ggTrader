import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ggTrader.data.historical.postgres_ingestor import PostgresIngestor

# We need to find where CONSTANTS are defined.
# Based on previous context, it might be in ggTrader.core.constants or data.kraken.historical_data
# Let's try to import from historical_data where we saw it used
# from ggTrader.core.constants import CONSTANTS
from sqlalchemy import text

POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
)


def clear_db():
    print("Connecting to database...")
    ingestor = PostgresIngestor(POSTGRES_CONNECTION_STRING)
    print("Truncating ohlcv table...")
    with ingestor.engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE ohlcv;"))
        conn.commit()
    print("Database cleared.")


if __name__ == "__main__":
    clear_db()
