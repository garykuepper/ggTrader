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

    print("\nSymbols containing XBT or BTC:")
    query_btc = (
        "SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE :xbt OR symbol LIKE :btc"
    )
    df_btc = pd.read_sql(
        text(query_btc), engine, params={"xbt": "%XBT%", "btc": "%BTC%"}
    )
    print(df_btc)

    print("\nUSD Quote Symbols (split part 2):")
    query_usd = (
        "SELECT DISTINCT split_part(symbol, '-', 2) as quote FROM ohlcv LIMIT 20"
    )
    print(pd.read_sql(text(query_usd), engine))


if __name__ == "__main__":
    main()
