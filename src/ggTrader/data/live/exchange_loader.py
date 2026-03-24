"""Live Exchange Loader wrapping CCXT."""

from typing import List, Optional

import ccxt
import pandas as pd

from ggTrader.data.core.base_loader import BaseDataLoader
from ggTrader.data.core.constants import STABLE_BASES


class LiveExchangeLoader(BaseDataLoader):
    """
    Live data loader connecting to exchanges via CCXT.
    Currently hardcoded/defaulted to Kraken, but easily extensible.
    """

    def __init__(self, exchange_id: str = "kraken"):
        self.exchange_id = exchange_id

        # Initialize the CCXT exchange dynamically
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class()

        # Cache for markets to avoid redundant network calls
        self._markets_cached = False

    def _ensure_markets_loaded(self):
        """Load markets once and cache."""
        if not self._markets_cached:
            self.exchange.load_markets()
            self._markets_cached = True

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = 1000,
        quote: str = "USD",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data via ccxt and return as a pandas DataFrame.
        Outputs exactly the same MultiIndex structure as TimescaleDBLoader.
        """
        self._ensure_markets_loaded()

        since = int(start_date.timestamp() * 1000) if start_date else None

        all_dfs = []

        for symbol in symbols:
            # ccxt expects 'BTC/USD'
            # CCXT expects 'BTC/USD', project uses 'BTC-USD'
            has_explicit_sep = "-" in symbol or "/" in symbol
            pair = symbol.replace("-", "/") if has_explicit_sep else f"{symbol}/{quote}"
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    pair, timeframe=interval, since=since, limit=limit
                )
            except Exception as e:
                print(f"Failed to fetch live OHLCV for {pair}: {e}")
                continue

            if not ohlcv or len(ohlcv[0]) < 6:
                continue

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC")
            df.set_index("datetime", inplace=True)
            df.drop(columns=["timestamp"], inplace=True)

            # Create MultiIndex columns for just this symbol
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Concatenate horizontally (aligning on datetime index)
        combined_df = pd.concat(all_dfs, axis=1)

        # Slice appropriately if end_date was provided
        if end_date:
            combined_df = combined_df[combined_df.index <= end_date]

        return combined_df

    def list_symbols(self, quote: str = "USD") -> List[str]:
        """Returns a list of available symbols from this source."""
        self._ensure_markets_loaded()
        all_markets = list(self.exchange.markets.keys())
        if quote:
            return [s for s in all_markets if s.endswith(f"/{quote}")]
        return all_markets

    def get_top_by_volume(
        self,
        limit: int = 20,
        quote: str = "USD",
        exclude_stables: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch the top symbols by 24h volume.
        Refactored implementation without manual iteration fallbacks.
        """
        self._ensure_markets_loaded()

        symbols = self.list_symbols(quote=quote)

        if exclude_stables:
            symbols = [s for s in symbols if s.split("/")[0].upper() not in STABLE_BASES]

        # Bulk fetch
        tickers = {}
        try:
            bulk = self.exchange.fetch_tickers(symbols)
            if isinstance(bulk, dict):
                tickers.update(bulk)
        except ccxt.NetworkError as e:
            if verbose:
                print(f"Network error fetching tickers: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            if verbose:
                print(f"Exchange error fetching tickers: {e}")
            return pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"Unexpected error fetching tickers: {e}")
            return pd.DataFrame()

        rows = []
        for s, t in tickers.items():
            parsed = self._parse_ticker_volume(s, t)
            if parsed:
                rows.append(parsed)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values(by="volume_usd", ascending=False).reset_index(drop=True)
        if limit and limit > 0:
            df = df.head(limit)

        return df

    def _parse_ticker_volume(self, symbol: str, ticker_data: dict) -> Optional[dict]:
        """Helper to parse raw ccxt ticker data into a standardized volume row."""
        try:
            if "/" in symbol:
                base, quote = symbol.split("/")
            elif "-" in symbol:
                base, quote = symbol.split("-")
            else:
                base = symbol
                quote = "USD"  # Default
            last = ticker_data.get("last") or ticker_data.get("close")

            vol_usd = 0.0
            qv = ticker_data.get("quoteVolume")

            if qv is not None:
                vol_usd = float(qv)
            else:
                # CCXT provider-specific fallbacks
                v = (
                    ticker_data.get("v", [0, 0])[1]
                    if isinstance(ticker_data.get("v"), (list, tuple))
                    else 0.0
                )
                p = (
                    ticker_data.get("p", [0, 0])[1]
                    if isinstance(ticker_data.get("p"), (list, tuple))
                    else 0.0
                )
                if v is not None and p is not None:
                    vol_usd = float(v) * float(p)
                elif last is not None:
                    base_vol = ticker_data.get("baseVolume") or 0.0
                    vol_usd = float(base_vol) * float(last)

            if vol_usd <= 0:
                return None

            return {
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "volume_usd": vol_usd,
                "last": float(last) if last is not None else None,
                "percentage": round(ticker_data.get("percentage", 0) or 0, 2),
            }
        except Exception:
            return None
