"""Stock data loader using yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf

from ggTrader.data.core.base_loader import BaseDataLoader
from ggTrader.data.core.stock_constants import YFINANCE_INTERVAL_MAP, SP500_SYMBOLS


class YFinanceDataLoader(BaseDataLoader):
    """Stock data loader using yfinance. Free, no API key, daily bars back to 1980+."""

    def __init__(self):
        self.logger = logging.getLogger("ggTraderLive")

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV via yfinance and return MultiIndex DataFrame.

        Args:
            symbols: List of stock tickers (e.g. ['AAPL', 'MSFT']).
            interval: Project canonical interval (e.g. '1d', '1h').
            start_date: Optional start timestamp.
            end_date: Optional end timestamp (defaults to now).
            limit: Ignored for yfinance (uses start/end).

        Returns:
            A MultiIndex Pandas DataFrame compatible with VectorBT.
            Columns: (symbol, metric) where metric is lowercase 'open', 'high', etc.
            Index: DatetimeIndex in UTC.
        """
        yf_interval = YFINANCE_INTERVAL_MAP.get(interval)
        if not yf_interval:
            raise ValueError(f"Interval '{interval}' not supported by yfinance")

        # Map start/end dates
        start = start_date.strftime("%Y-%m-%d") if start_date else None
        # Use end_date or now
        end_ts = end_date or pd.Timestamp.now(tz=timezone.utc)
        end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.logger.info(f"Fetching {len(symbols)} stocks from yfinance ({yf_interval}): {start} -> {end}")

        try:
            # group_by="ticker" ensures we always get a MultiIndex (symbol, metric)
            # even for a single symbol request.
            df = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
            )
        except Exception as e:
            self.logger.error(f"yfinance download failed: {e!r}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Reshape and clean
        # yfinance returns columns like (AAPL, Close)
        # We need (AAPL, close)
        df.columns = df.columns.set_levels(df.columns.levels[1].str.lower(), level=1)

        # Ensure index is localized to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Sort columns to ensure consistent order
        df = df.reindex(columns=sorted(df.columns), level=0)

        # Drop any rows that are all NaN for all symbols
        df.dropna(how="all", inplace=True)

        return df

    def list_symbols(self) -> List[str]:
        """Return the pre-configured S&P 500 list."""
        return SP500_SYMBOLS

    def get_top_by_volume(
        self,
        limit: int = 50,
        window: str = "30d",
    ) -> List[dict]:
        """Rank S&P 500 stocks by average daily dollar volume.

        Returns list of dicts with 'symbol', 'rank', 'average_notional_volume'.
        """
        # This will be implemented more efficiently in update_universe_stocks.py
        # but providing a stub here for interface completeness.
        raise NotImplementedError("Use scripts/update_universe_stocks.py for volume ranking")
