import ccxt
import pandas as pd
from tabulate import tabulate


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


# Public API
__all__ = ["fetch_ohlcv_df"]


# Example usage (only when run as a script)
if __name__ == "__main__":
    import ccxt  # local import to avoid import during from-imports if not needed elsewhere

    # Example usage:
    kraken = ccxt.kraken()
    kraken.load_markets()
    df = fetch_ohlcv_df(kraken, 'ETH/USD', timeframe='4h', limit=30)
    print(tabulate(df, headers="keys", tablefmt="github"))
