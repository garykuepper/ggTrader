"""Stock data loader with TimescaleDB caching."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.data.live.yfinance_loader import YFinanceDataLoader


class CachedYFinanceLoader(YFinanceDataLoader):
    """Stock data loader with TimescaleDB caching. All yfinance fetches are persisted to DB."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the loader."""
        super().__init__()
        self.db_loader = TimescaleDBLoader(connection_string=connection_string)
        self.connection_string = self.db_loader.connection_string

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with a 'DB-first' cache strategy."""
        # 1. Fetch from DB
        db_df = self.db_loader.fetch_ohlcv(
            symbols=symbols,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        # Determine if we need to fetch more from yfinance
        now_utc = pd.Timestamp.now(tz="UTC")

        needs_fetch = False
        if db_df.empty:
            needs_fetch = True
        else:
            last_ts = db_df.index.max()
            delta = self._interval_to_timedelta(interval)
            # Fetch if the last candle is older than 1.5 * interval
            if (now_utc - last_ts) > (delta * 1.5):
                needs_fetch = True

        if not needs_fetch:
            if limit and len(db_df) >= limit:
                return db_df

        # 2. Fetch from yfinance
        fetch_start = start_date
        if not db_df.empty:
            # yfinance start is inclusive, so we add 1 unit to avoid duplication
            delta = self._interval_to_timedelta(interval)
            fetch_start = db_df.index.max() + delta

        live_df = super().fetch_ohlcv(
            symbols=symbols,
            interval=interval,
            start_date=fetch_start,
            end_date=end_date,
            limit=limit,
        )

        if live_df.empty:
            return db_df

        # 3. Store new data in DB
        self._cache_to_db(live_df, interval)

        # 4. Combine and return
        if db_df.empty:
            return live_df

        combined = pd.concat([db_df, live_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        if limit:
            combined = combined.tail(limit)

        return combined

    def _interval_to_timedelta(self, interval: str) -> pd.Timedelta:
        """Convert interval string to Timedelta."""
        unit = interval[-1]
        try:
            value = int(interval[:-1])
        except ValueError:
            return pd.Timedelta(days=1)

        if unit == "m":
            return pd.Timedelta(minutes=value)
        if unit == "h":
            return pd.Timedelta(hours=value)
        if unit == "d":
            return pd.Timedelta(days=value)
        return pd.Timedelta(days=1)

    def _cache_to_db(self, df: pd.DataFrame, interval: str) -> None:
        """Save yfinance OHLCV DataFrame to TimescaleDB."""
        if df.empty:
            return

        records = []
        symbols = df.columns.levels[0]

        for symbol in symbols:
            symbol_data = df[symbol]
            for ts, row in symbol_data.iterrows():
                if pd.isna(row["close"]) or pd.isna(ts):
                    continue

                records.append(
                    (
                        ts.to_pydatetime(),
                        symbol,  # Stocks don't need replace("-", "/")
                        interval,
                        float(row["open"]) if not pd.isna(row["open"]) else None,
                        float(row["high"]) if not pd.isna(row["high"]) else None,
                        float(row["low"]) if not pd.isna(row["low"]) else None,
                        float(row["close"]) if not pd.isna(row["close"]) else None,
                        float(row["volume"]) if not pd.isna(row["volume"]) else None,
                        0,  # yfinance doesn't provide trade counts
                    )
                )

        if not records:
            return

        query = """
            INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
            VALUES %s
            ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades;
        """

        try:
            conn = psycopg2.connect(
                self.connection_string.replace("postgresql+psycopg2://", "postgresql://")
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                execute_values(cur, query, records)
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to cache stock data to DB: {e}")
