import ccxt
import time
import pandas as pd
import requests
from tabulate import tabulate


class KrakenData:
    pass

    @staticmethod
    def fetch_ohlcv_df(exchange, symbol, timeframe='1h', limit=100, since=None):
        """
        Fetch OHLCV data via ccxt and return as a pandas DataFrame.

        Args:
            exchange: ccxt exchange instance (e.g., ccxt.kraken())
            symbol: string, e.g., 'BTC/USD'
            timeframe: string, e.g., '1h', '15m', '1d'
            limit: maximum number of candles
            since: fetch since timestamp (ms), or None

        Returns:
            Pandas DataFrame with columns: open, high, low, close, volume and datetime index
        """
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv or len(ohlcv[0]) < 6:
            raise ValueError("Returned OHLCV data is empty or unexpected format.")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def get_kraken_asset_pairs_usd():
        url = "https://api.kraken.com/0/public/AssetPairs"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken AssetPairs error: {payload['error']}")
