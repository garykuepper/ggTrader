import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv


def check_gaps():
    load_dotenv()
    conn_str = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_str:
        print("POSTGRES_CONNECTION_STRING not found in .env")
        return

    engine = create_engine(conn_str)

    print("\nChecking daily row counts for early 2024 (March to May)...")
    query = """
        SELECT 
            timestamp::DATE as date,
            interval,
            COUNT(*) as count
        FROM ohlcv
        WHERE timestamp >= '2024-03-01' AND timestamp < '2024-06-01'
        GROUP BY 1, 2
        ORDER BY 1 ASC, 2
        LIMIT 100
    """
    df = pd.read_sql(text(query), engine)
    print(df)


if __name__ == "__main__":
    check_gaps()
