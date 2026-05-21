"""Cached data loader combining CCXT and TimescaleDB."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.data.live.exchange_loader import LiveExchangeLoader


class CachedExchangeLoader(LiveExchangeLoader):
    """
    Live data loader with TimescaleDB caching.

    If data is available in the database, it returns it.
    If data is missing (notably the most recent candles), it fetches from
    the exchange and stores it in the database before returning.
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        connection_string: Optional[str] = None,
    ):
        """
        Initialize the loader.

        Args:
            exchange_id: CCXT exchange ID.
            connection_string: Optional DB connection string.
        """
        super().__init__(exchange_id=exchange_id)
        self.db_loader = TimescaleDBLoader(connection_string=connection_string)
        self.connection_string = self.db_loader.connection_string

    @property
    def venue(self) -> str:
        """Map exchange ID to DB venue string."""
        if self.exchange_id == "binanceus":
            return "binanceus_spot"
        return "kraken_spot"

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
        Fetch OHLCV data with a 'DB-first' cache strategy.

        1. Try to fetch from DB.
        2. If DB is missing the most recent data, fetch from exchange.
        3. Save new data to DB.
        4. Return combined result.
        """
        # 1. Fetch from DB
        db_df = self.db_loader.fetch_ohlcv(
            symbols=symbols,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            quote=quote,
            venue=self.venue,
        )

        # Determine if we need to fetch more from live exchange
        now_utc = pd.Timestamp.now(tz="UTC")

        # If DB is empty, or if the last timestamp in DB is too old
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
            # Check if we have enough bars to satisfy the limit
            if limit and len(db_df) >= limit:
                return db_df
            # If not enough bars even if up to date, we might still want to fetch
            # But usually it means there isn't more data available or start_date restricted it.

        # 2. Fetch from Exchange
        # If we have some data in DB, fetch from the last timestamp + 1s
        fetch_start = start_date
        if not db_df.empty:
            fetch_start = db_df.index.max() + pd.Timedelta(seconds=1)

        live_df = super().fetch_ohlcv(
            symbols=symbols,
            interval=interval,
            start_date=fetch_start,
            end_date=end_date,
            limit=limit,
            quote=quote,
        )

        if live_df.empty:
            return db_df

        # 3. Store new data in DB
        self._cache_to_db(live_df, interval)

        # 4. Combine and return
        if db_df.empty:
            return live_df

        # Concatenate and handle overlaps/duplicates
        combined = pd.concat([db_df, live_df])
        # Drop duplicates based on index
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Trim to limit if needed
        if limit:
            combined = combined.tail(limit)

        return combined

    def _interval_to_timedelta(self, interval: str) -> pd.Timedelta:
        """Convert interval string (e.g., '1h', '4h', '1d') to Timedelta."""
        unit = interval[-1]
        try:
            value = int(interval[:-1])
        except ValueError:
            return pd.Timedelta(hours=4)

        if unit == "m":
            return pd.Timedelta(minutes=value)
        if unit == "h":
            return pd.Timedelta(hours=value)
        if unit == "d":
            return pd.Timedelta(days=value)
        if unit == "w":
            return pd.Timedelta(weeks=value)
        return pd.Timedelta(hours=4)

    def _cache_to_db(self, df: pd.DataFrame, interval: str) -> None:
        """Save OHLCV DataFrame to TimescaleDB."""
        if df.empty:
            return

        records = []
        # MultiIndex columns: (symbol, field)
        symbols = df.columns.levels[0]

        for symbol in symbols:
            # Normalize symbol for DB (e.g. BTC/USD -> BTC-USD)
            db_symbol = symbol.replace("/", "-")

            symbol_data = df[symbol]
            for ts, row in symbol_data.iterrows():
                # Skip rows with NaNs in close or missing timestamp
                if pd.isna(row["close"]) or pd.isna(ts):
                    continue

                # `ohlcv.timestamp` is `timestamp without time zone`. Passing a
                # tz-aware datetime makes psycopg2 convert to the connection's
                # local TZ before stripping, shifting bars by 7-8h depending
                # on DST. Convert to UTC and drop the tz so the stored label
                # is the actual UTC wall-clock.
                if hasattr(ts, "tz_convert") and ts.tzinfo is not None:
                    ts_naive_utc = ts.tz_convert("UTC").tz_localize(None)
                else:
                    ts_naive_utc = ts
                records.append(
                    (
                        ts_naive_utc.to_pydatetime(),
                        db_symbol,
                        interval,
                        float(row["open"]) if not pd.isna(row["open"]) else None,
                        float(row["high"]) if not pd.isna(row["high"]) else None,
                        float(row["low"]) if not pd.isna(row["low"]) else None,
                        float(row["close"]) if not pd.isna(row["close"]) else None,
                        float(row["volume"]) if not pd.isna(row["volume"]) else None,
                        int(row["trades"]) if "trades" in row and not pd.isna(row["trades"]) else 0,
                        self.venue,
                    )
                )

        if not records:
            return

        query = """
            INSERT INTO ohlcv (
                timestamp, symbol, interval, open, high, low, close, volume, trades, venue
            )
            VALUES %s
            ON CONFLICT (timestamp, symbol, interval, venue) DO UPDATE SET
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
            # Log error but don't crash the fetching process
            print(f"Failed to cache live data to DB: {e}")
