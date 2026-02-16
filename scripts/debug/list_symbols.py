import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from ggTrader.data.kraken.postgres_reader import KrakenPostgresReader
from ggTrader.utils.config import get_db_connection_string


def main():
    conn_str = get_db_connection_string()
    reader = KrakenPostgresReader(conn_str)
    try:
        syms = reader.list_pairs()
        print(f"Total pairs: {len(syms)}")

        usd_pairs = [s for s in syms if s.endswith("-USD")]
        print(f"USD pairs (first 50): {usd_pairs[:50]}")

        btc_patterns = [s for s in syms if "XBT" in s or "BTC" in s]
        print(f"BTC patterns: {btc_patterns}")

        eth_patterns = [s for s in syms if "ETH" in s]
        print(f"ETH patterns: {eth_patterns}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
