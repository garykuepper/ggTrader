#!/usr/bin/env python3
"""
Reusable function to fetch top cryptocurrencies by market cap from CoinMarketCap,
excluding stablecoins, and return as a pandas DataFrame.

Requires:
  - requests
  - pandas
  - tabulate (optional, only if you want pretty print)
  - CMC_API_KEY in your environment
"""

import os
import requests
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv  # new: load .env files

CMC_LISTINGS = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

# Common stablecoins to exclude
STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "TUSD", "PYUSD",
    "USDP", "GUSD", "USDD", "EURS", "WBTC", "WBETH", "USDE"
}

def get_top_cmc(limit: int = 20, convert: str = "USD", print_table: bool = False) -> pd.DataFrame:
    """
    Fetch top cryptocurrencies by market cap from CoinMarketCap, excluding stablecoins.

    Args:
        limit (int): Number of non-stable coins to return (default: 20)
        convert (str): Fiat currency to convert values into (default: "USD")
        print_table (bool): If True, prints the DataFrame using tabulate

    Returns:
        pd.DataFrame: DataFrame with Rank, Symbol, Name, Price, and Market Cap
    """
    # Resolve .env from parent directory and load it, so CMC_API_KEY can be picked up
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

    api_key = os.environ.get("CMC_API_KEY")
    if not api_key:
        raise RuntimeError("Please set CMC_API_KEY in your environment variables.")

    params = {
        "start": 1,
        "limit": limit * 2,   # fetch extra to cover excluded stables
        "convert": convert,
        "sort": "market_cap",
        "sort_dir": "desc",
    }
    headers = {"X-CMC_PRO_API_KEY": api_key}
    r = requests.get(CMC_LISTINGS, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    raw = r.json()["data"]

    # Filter out stablecoins
    coins = [c for c in raw if c["symbol"].upper() not in STABLES]
    coins = coins[:limit]

    df = pd.DataFrame([{
        "Rank": c.get("cmc_rank"),
        "Symbol": c.get("symbol"),
        "Name": c.get("name"),
        f"Price ({convert})": c["quote"][convert]["price"],
        f"Market Cap ({convert})": c["quote"][convert]["market_cap"]
    } for c in coins])

    if print_table:
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".2f"))

    return df

# For standalone testing
if __name__ == "__main__":
    df = get_top_cmc(limit=20, print_table=True)
