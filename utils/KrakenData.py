import ccxt
import time
import pandas as pd
import requests
from sqlalchemy.orm.base import state_class_str

from tabulate import tabulate


class KrakenData:
    def __init__(self):
        self.kraken = ccxt.kraken()

    @staticmethod
    def fetch_ohlcv_df(exchange, symbol, timeframe='4h', limit=100, since=None):
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
        return payload

    @staticmethod
    def get_kraken_usd_ccxt():
        kraken = ccxt.kraken()
        kraken.load_markets()
        all_markets = list(kraken.markets.keys())
        usd_markets = [s for s in all_markets if s.endswith("/USD")]
        return usd_markets

    @staticmethod
    def top_kraken_by_volume(limit=20, only_usd=True, exclude_stables: bool = True, verbose=False):
        kraken = ccxt.kraken()
        kraken.load_markets()

        # Step 1: get candidate symbols
        markets = kraken.markets  # dict: symbol -> market info
        symbols = [s for s in markets.keys() if (s.endswith('/USD') if only_usd else True)]
        # Optional: exclude stablecoins by base symbol
        if exclude_stables:
            STABLE_BASES = {"USDT", "USDC", "DAI", "USDP", "TUSD", "EUR", "GBP", "AUD", "USDG"}
            filtered = []
            for s in symbols:
                # s is like 'BASE/USD'; take the base part before '/'
                base = s.split('/', 1)[0]
                if base.upper() not in STABLE_BASES:
                    filtered.append(s)
            symbols = filtered

        # Step 2: bulk fetch tickers
        tickers = {}
        try:
            bulk = kraken.fetch_tickers(symbols or list(markets.keys()))
            if isinstance(bulk, dict):
                tickers.update(bulk)
        except Exception as e:
            if verbose:
                print(f"Bulk fetch_tickers failed: {e}. Falling back to per-symbol fetch.")
            for s in symbols:
                try:
                    tickers[s] = kraken.fetch_ticker(s)
                except Exception as ex:
                    if verbose:
                        print(f"Failed to fetch ticker for {s}: {ex}")

        # Step 3: compute a volume metric per symbol
        rows = []
        for s, t in tickers.items():
            # last price
            last = t.get('last') or t.get('close')
            # notional volume: prefer Kraken-provided fields if present
            vol_usd = 0.0
            # 3a) try quoteVolume (if available and already in USD terms)
            qv = t.get('quoteVolume')
            if qv is not None:
                vol_usd = float(qv)
            else:
                # 3b) Kraken-specific v/p notations
                v = t.get('v', [0, 0])[1] if isinstance(t.get('v'), (list, tuple)) else 0.0
                p = t.get('p', [0, 0])[1] if isinstance(t.get('p'), (list, tuple)) else 0.0
                if v is not None and p is not None:
                    vol_usd = float(v) * float(p)
                elif last is not None:
                    # fallback: approximate with baseVolume * last
                    base_vol = t.get('baseVolume') or 0.0
                    vol_usd = float(base_vol) * float(last)
            rows.append({
                'symbol': s,
                'last': float(last) if last is not None else None,
                'volume_usd': vol_usd
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values(by='volume_usd', ascending=False).reset_index(drop=True)
        if limit and limit > 0:
            df = df.head(limit)

        return df

if __name__ == "__main__":
    kData = KrakenData()
    pairs = kData.get_kraken_usd_ccxt()
    print(f"Found {len(pairs)} USD pairs on Kraken.")
    symbol = 'ETH/USD'
    ohlcv = kData.fetch_ohlcv_df(kData.kraken, symbol, timeframe='4h', limit=30)
    print(f"Fetched {len(ohlcv)} {symbol} OHLCV data points.")
    print(tabulate(ohlcv.head(), headers="keys", tablefmt="github"))
    top_volume = kData.top_kraken_by_volume(limit=20, only_usd=True, exclude_stables=True)
    print(tabulate(top_volume, headers="keys", tablefmt="github"))