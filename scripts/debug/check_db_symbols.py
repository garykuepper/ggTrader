import os
import sys
from sqlalchemy import create_engine, text
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from ggTrader.utils.config import get_db_connection_string


def main():
    conn_str = get_db_connection_string()
    engine = create_engine(conn_str)

    query = """
    SELECT symbol, SUM(volume) as total_volume, COUNT(*) as counts 
    FROM ohlcv 
    WHERE interval = '1d' 
    GROUP BY symbol 
    ORDER BY total_volume DESC 
    LIMIT 100
    """

    df = pd.read_sql(text(query), engine)
    print("Top Symbols by Volume (Base + Quote):")
    print(df)

    # Check specifically for BTC/ETH patterns
    print("\nSymbols containing XBT or BTC:")
    query_btc = "SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%XBT%' OR symbol LIKE '%BTC%'"
    print(pd.read_sql(text(query_btc), engine))

    print("\nSymbols containing ETH:")
    query_eth = "SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%ETH%'"
    print(pd.read_sql(text(query_eth), engine))


if __name__ == "__main__":
    main()
