import os
import sys
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from ggTrader.utils.config import get_db_connection_string

connection_string = get_db_connection_string()
engine = create_engine(connection_string)
print(connection_string)
queries = {
    "Total DB Size": "SELECT pg_size_pretty(pg_database_size('ggtrader'))",
    "Table Stats": """
        SELECT 
            pg_size_pretty(pg_total_relation_size('ohlcv')) as total_size,
            pg_size_pretty(pg_relation_size('ohlcv')) as table_size,
            pg_size_pretty(pg_indexes_size('ohlcv')) as index_size,
            (SELECT count(*) FROM ohlcv) as row_count
    """,
    "Check Compression Job": "SELECT * FROM timescaledb_information.job_stats WHERE job_id = 1000",
}

try:
    with engine.connect() as conn:
        for name, query in queries.items():
            print(f"\n--- {name} ---")
            result = conn.execute(text(query))
            row = result.fetchone()
            if row:
                print(dict(zip(result.keys(), row)))
except Exception as e:
    print(f"Error: {e}")
