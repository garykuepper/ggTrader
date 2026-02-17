import requests
import pandas as pd
from tabulate import tabulate
from utils.KrakenData import KrakenData


def kraken_pair_list():
    k_data = KrakenData()
    k_pairs = k_data.get_kraken_usd_ccxt()
    # strip /USD
    k_pairs = [pair.replace("/USD", "") for pair in k_pairs]
    print(k_pairs)
    k_pairs = [s.lower() for s in k_pairs]

    return k_pairs


def get_top_crypto(limit=20, currency="usd"):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": currency,
        "order": "market_cap_desc",
        "per_page": limit * 2,  # fetch more in case we filter some out
        "page": 1,
        "sparkline": "false"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    # print(data)
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["market_cap_rank", "symbol", "name", "current_price", "market_cap","total_volume","atl_date"])

    # List of known stablecoins (expand if needed)
    stablecoins = {
        "usdt", "usdc", "busd", "dai", "ust", "tusd", "gusd", "pax", "usdp", "frax",
        "usds"
    }
    special_filter = {"wbtc","usde"}
    df = df[~df["symbol"].str.lower().isin(stablecoins)]
    df = df[~df["symbol"].str.lower().isin(special_filter)]
    # filter to only those in Kraken USD pairs
    k_pairs = kraken_pair_list()
    df = df[df["symbol"].isin(k_pairs)]
    # Keep only top N after filtering
    df = df.nsmallest(limit, "market_cap_rank")

    # Format numbers
    # df["market_cap"] = df["market_cap"].apply(lambda x: f"${x:,.0f}")
    # df["current_price"] = df["current_price"].apply(lambda x: f"${x:,.2f}")
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    df = get_top_crypto(limit=20)
    k_data = KrakenData()
    k_pairs = k_data.get_kraken_usd_ccxt()
    # TODO: Get the oldest date in offline csv for each ticker,
    #  to see how long they've been in kraken
    print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
