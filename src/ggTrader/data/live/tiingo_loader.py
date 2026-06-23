"""Stock data loader using the Tiingo REST API.

Free tier: 500 unique symbols/day, 20+ year history, covers delisted tickers.
API docs: https://www.tiingo.com/documentation/end-of-day
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import pandas as pd
import requests

from ggTrader.data.core.base_loader import BaseDataLoader
from ggTrader.data.core.stock_constants import SP500_SYMBOLS

TIINGO_EOD_URL = "https://api.tiingo.com/tiingo/daily"

TIINGO_INTERVAL_MAP = {
    "1d": "daily",
    "1wk": "weekly",
    "1mo": "monthly",
}

# Free tier: 50 requests/hour, 1000/day, 1GB/month
REQUESTS_PER_HOUR = 50
MIN_REQUEST_INTERVAL = 3600.0 / REQUESTS_PER_HOUR  # 72s between requests


def _get_api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "")
    if not key:
        raise ValueError(
            "TIINGO_API_KEY not set. Get a free key at https://www.tiingo.com"
        )
    return key


class TiingoDataLoader(BaseDataLoader):
    """Stock data loader using Tiingo. Free tier, 20yr+ history, delisted coverage."""

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        self.logger = logging.getLogger("ggTraderLive")
        self.api_key = api_key or _get_api_key()
        self.batch_size = batch_size
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_key}",
            }
        )

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV via Tiingo and return MultiIndex DataFrame.

        Returns:
            MultiIndex (symbol, field) DataFrame with UTC DatetimeIndex,
            matching the yfinance loader output format.
        """
        tiingo_freq = TIINGO_INTERVAL_MAP.get(interval)
        if not tiingo_freq:
            raise ValueError(
                f"Interval '{interval}' not supported by Tiingo "
                f"(supported: {list(TIINGO_INTERVAL_MAP)})"
            )

        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None

        self.logger.info(
            f"Fetching {len(symbols)} stocks from Tiingo"
            f" ({tiingo_freq}): {start_str} -> {end_str}"
        )

        frames: list[pd.DataFrame] = []
        failed: list[str] = []

        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i : i + self.batch_size]
            for sym in batch:
                df = self._fetch_one(sym, tiingo_freq, start_str, end_str)
                if df is not None and not df.empty:
                    frames.append(df)
                else:
                    failed.append(sym)

        if failed:
            self.logger.warning(
                f"Tiingo: {len(failed)} symbols returned no data "
                f"(first 10: {failed[:10]})"
            )

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined.sort_index(axis=1, inplace=True)
        combined.dropna(how="all", inplace=True)

        return combined.tail(limit) if limit else combined

    def _fetch_one(
        self,
        symbol: str,
        freq: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.DataFrame]:
        url = f"{TIINGO_EOD_URL}/{symbol}/prices"
        params: dict = {"resampleFreq": freq}
        if start:
            params["startDate"] = start
        if end:
            params["endDate"] = end

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            self.logger.debug(f"Tiingo fetch failed for {symbol}: {exc!r}")
            return None

        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)

        keep = ["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
        df = df[[c for c in keep if c in df.columns]]
        df.rename(
            columns={
                "adjOpen": "open",
                "adjHigh": "high",
                "adjLow": "low",
                "adjClose": "close",
                "adjVolume": "volume",
            },
            inplace=True,
        )

        result = pd.concat({symbol: df}, axis=1)
        result.columns.names = [None, None]
        return result

    def list_symbols(self) -> List[str]:
        """Return the pre-configured S&P 500 list."""
        return SP500_SYMBOLS

    def get_top_by_volume(
        self,
        limit: int = 50,
        window: str = "30d",
    ) -> List[dict]:
        raise NotImplementedError("Use scripts/update_universe_stocks.py for volume ranking")
