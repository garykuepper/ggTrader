"""TimescaleDB historical data loader."""

from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from ggTrader.data.core.base_loader import BaseDataLoader
from ggTrader.utils.config import get_db_connection_string


class TimescaleDBLoader(BaseDataLoader):
    """
    Historical data loader connecting to TimescaleDB (PostgreSQL).
    Reads historical OHLCV data for backtesting.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the DB connection lazily or explicitly."""
        self.connection_string = connection_string or get_db_connection_string()
        # Keep engine None until needed to prevent connection drops in long processes
        self._engine = None

    @property
    def engine(self):
        """Lazy engine initialization."""
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
        quote: str = "USD",
        venue: str = "kraken_spot",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from TimescaleDB and format it as a MultiIndex DataFrame
        suitable for VectorBT (columns: symbol -> metric).
        """
        if not symbols:
            return pd.DataFrame()

        formatted_symbols = [s if "-" in s or "/" in s else f"{s}-{quote}" for s in symbols]

        # Build query constraints
        conditions = ["symbol = ANY(:symbols)", "interval = :interval", "venue = :venue"]
        params = {
            "symbols": formatted_symbols,
            "interval": interval,
            "quote": quote,
            "venue": venue,
        }

        if start_date is not None:
            conditions.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date is not None:
            conditions.append("timestamp <= :end_date")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions)
        limit_clause = f" LIMIT {limit}" if limit else ""

        query = f"""
            SELECT timestamp as datetime, symbol, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE {where_clause}
            ORDER BY timestamp ASC
            {limit_clause};
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query),
                    conn,
                    params=params,
                    parse_dates=["datetime"],
                )
        except Exception as e:
            print(f"Failed to read OHLCV from Postgres: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Ensure timezone UTC
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        elif str(df["datetime"].dt.tz) != "UTC":
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")

        # Pivot the dataframe to MultiIndex
        # Expected shape: index=datetime, columns=[symbol][open, high, low, close, volume, trades]
        df_pivoted = df.pivot(index="datetime", columns="symbol")

        # Swap levels to be [symbol][metric]
        df_pivoted = df_pivoted.swaplevel(1, 0, axis=1)
        # Sort index for performance
        df_pivoted.sort_index(axis=1, inplace=True)
        # Reorder expected metrics to standard OHLCV
        metrics = ["open", "high", "low", "close", "volume"]
        if "trades" in df_pivoted.columns.get_level_values(1):
            metrics.append("trades")

        # Keep only the metrics we need and handle missing gracefully
        # df_pivoted is sorted, so we can slice it

        # Sometimes a column might be fully missing for a metric, reindex level 1
        return df_pivoted.reindex(columns=metrics, level=1)

    def list_symbols(self, quote: str = "USD", venue: str = "kraken_spot") -> List[str]:
        """Returns a list of available symbols from this source (filtered by quote and venue)."""
        query = "SELECT DISTINCT symbol FROM ohlcv WHERE venue = :venue;"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"venue": venue})
                symbols = [row[0] for row in result]

            if quote:
                return [s for s in symbols if s.endswith(f"-{quote}")]
            return symbols
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
