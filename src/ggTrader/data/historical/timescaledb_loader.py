"""TimescaleDB historical data loader."""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import List, Optional, Tuple

from ggTrader.data.core.base_loader import BaseDataLoader
from ggTrader.utils.config import get_db_connection_string


class TimescaleDBLoader(BaseDataLoader):
    """
    Historical data loader connecting to TimescaleDB (PostgreSQL).
    Consolidates reading OHLCV data and generating mover masks for backtesting.
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
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from TimescaleDB and format it as a MultiIndex DataFrame
        suitable for VectorBT (columns: symbol -> metric).
        """
        if not symbols:
            return pd.DataFrame()

        # Ensure symbols are correctly formatted with quote (e.g., BTC-USD)
        formatted_symbols = [s if "-" in s or "/" in s else f"{s}-{quote}" for s in symbols]

        # Build query constraints
        conditions = ["symbol = ANY(:symbols)", "interval = :interval"]
        params = {
            "symbols": formatted_symbols,
            "interval": interval,
            "quote": quote,
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
        idx = pd.IndexSlice
        # df_pivoted is sorted, so we can slice it

        # Sometimes a column might be fully missing for a metric, reindex level 1
        return df_pivoted.reindex(columns=metrics, level=1)

    def list_symbols(self, quote: str = "USD") -> List[str]:
        """Returns a list of available symbols from this source (filtered by quote)."""
        query = "SELECT DISTINCT symbol FROM ohlcv;"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                symbols = [row[0] for row in result]

            if quote:
                return [s for s in symbols if s.endswith(f"-{quote}")]
            return symbols
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []

    def get_daily_mover_mask(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        top_n: int = 20,
        quote: str = "USD",
        trades_threshold: int = 500,
    ) -> pd.DataFrame:
        """
        Build a (date x symbol) boolean mask of daily top-N movers.
        """
        # Note: The ohlcv table should ideally have a 1d interval available
        query = """
            WITH DailyStats AS (
                SELECT
                    DATE_TRUNC('day', timestamp) AS daily_date,
                    symbol,
                    close,
                    volume,
                    (volume * close) AS notional_volume
                FROM ohlcv
                WHERE interval = '1d'
                  AND timestamp >= :start AND timestamp <= :end
                  AND symbol LIKE :quote_pattern
                  AND COALESCE(trades, 0) >= :trades_threshold
            ),
            RankedStats AS (
                SELECT
                    daily_date,
                    symbol,
                    notional_volume,
                    RANK() OVER (PARTITION BY daily_date ORDER BY notional_volume DESC) as rnk
                FROM DailyStats
            )
            SELECT daily_date, symbol, rnk
            FROM RankedStats
            WHERE rnk <= :top_n
            ORDER BY daily_date ASC, rnk ASC;
        """

        params = {
            "start": start,
            "end": end,
            "top_n": top_n,
            "quote_pattern": f"%-{quote}",
            "trades_threshold": trades_threshold,
        }

        try:
            with self.engine.connect() as conn:
                top_df = pd.read_sql(text(query), conn, params=params)

            if top_df.empty:
                return pd.DataFrame()

            top_df["daily_date"] = pd.to_datetime(top_df["daily_date"]).dt.tz_localize("UTC")

            # Pivot to wide format: Index=daily_date, Columns=symbol, Values=True where ranked
            top_df["is_top"] = True
            mask_df = top_df.pivot(index="daily_date", columns="symbol", values="is_top")
            mask_df = mask_df.fillna(False).astype(bool)

            # Ensure a full date range without gaps
            full_date_range = pd.date_range(
                start=start.floor("D"), end=end.ceil("D"), freq="D", tz="UTC"
            )
            mask_df = mask_df.reindex(full_date_range, fill_value=False)
            mask_df.index.name = "datetime"

            return mask_df

        except Exception as e:
            print(f"Error building mover mask: {e}")
            return pd.DataFrame()

    def get_consistent_movers(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
        daily_top_n: int = 50,
        output_n: int = 50,
        trades_threshold: int = 500,
        stables: bool = False,
        quote: str = "USD",
    ) -> pd.DataFrame:
        """
        Calculates consistent movers over a lookback period.
        Finds assets that appeared most frequently in the top N by notional volume each day.
        """
        excluded = []
        if not stables:
            excluded = ["USDT", "USDC", "DAI", "PYUSD", "EUR", "GBP", "AUD", "USDG"]

        end_ts = pd.Timestamp(end_date)
        start_ts = pd.Timestamp(start_date)

        query = f"""
            WITH daily_tops AS (
                SELECT 
                    DATE_TRUNC('day', timestamp) as date,
                    symbol as asset,
                    volume * close as notional_volume,
                    ROW_NUMBER() OVER (
                        PARTITION BY DATE_TRUNC('day', timestamp)
                        ORDER BY volume * close DESC
                    ) as rank
                FROM ohlcv
                WHERE interval = '1d'
                  AND COALESCE(trades, 0) >= :threshold
                  AND timestamp >= :start
                  AND symbol LIKE :quote_pattern
            )
            SELECT 
                asset as symbol,
                COUNT(*) as frequency,
                AVG(notional_volume) as average_notional_volume
            FROM daily_tops
            WHERE rank <= :daily_limit
            GROUP BY asset
            ORDER BY frequency DESC, average_notional_volume DESC
            LIMIT :limit;
        """

        params = {
            "threshold": trades_threshold,
            "start": start_ts,
            "quote_pattern": f"%-{quote}",
            "daily_limit": daily_top_n,
            "limit": output_n,
        }

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            if not stables:
                df = df[~df["symbol"].str.split("-").str[0].isin(excluded)]

            return df
        except Exception as e:
            print(f"Error getting consistent movers: {e}")
            return pd.DataFrame()

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
