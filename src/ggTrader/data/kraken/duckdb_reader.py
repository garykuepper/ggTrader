import os
import duckdb
import pandas as pd
import numpy as np
from .utils import (
    align_to_datetime_index,
    fill_after_first_non_nan_multilevel_safe,
    fill_symbol_metadata,
    ensure_utc_timestamp,
    filter_out_stables,
)


class KrakenDuckDBReader:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def _get_conn(self):
        # We use a persistent connection but handle potential locking
        if self.conn is None:
            # read_only=True allows multiple processes to read
            self.conn = duckdb.connect(self.db_path, read_only=True)
        return self.conn

    def read_ohlcv(
        self, pair=None, interval=None, start=None, end=None, symbols=None, quote="USD"
    ):
        """
        Read OHLCV data from DuckDB.
        Can filter by pair/interval or multiple symbols.
        """
        conn = self._get_conn()

        conditions = []
        params = []

        if pair:
            conditions.append("pair = ?")
            params.append(pair)
        elif symbols:
            pairs = [f"{s}-{quote}" for s in symbols]
            conditions.append(f"pair IN ({','.join(['?'] * len(pairs))})")
            params.extend(pairs)

        if interval:
            conditions.append("interval = ?")
            params.append(interval)

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)

        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"SELECT * FROM ohlcv{where_clause} ORDER BY timestamp ASC"

        # DuckDB can return a result as a Pandas DataFrame directly
        df = conn.execute(sql, params).df()

        if df.empty:
            return df

        # Set index and ensure datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

        return df

    def get_ohlcv_df(
        self, symbols: list, interval="1d", quote="USD", start=None, end=None
    ):
        """
        Retrieve aligned OHLCV data for multiple symbols, matching the KrakenParquetReader API.
        """
        # Convert start/end to consistent format
        if start:
            start = ensure_utc_timestamp(start)
        if end:
            end = ensure_utc_timestamp(end)

        df = self.read_ohlcv(
            symbols=symbols, interval=interval, start=start, end=end, quote=quote
        )

        if df.empty:
            return pd.DataFrame()

        # Reshape to multi-index (columns: symbol -> metric)
        # DuckDB returned a flat df with a 'pair' column.
        # We need to pivot to match the expected format:
        # index: timestamp, columns: MultiIndex (level 0: symbol, level 1: ohlcv metrics)

        # Map pair back to symbol
        df["symbol"] = df["pair"].apply(lambda x: x.split("-")[0])

        # Pivot
        pivoted = df.pivot(
            columns="symbol",
            values=["open", "high", "low", "close", "volume", "trades"],
        )

        # The result of pivot has metrics at level 0 and symbol at level 1.
        # We want symbol at level 0 and metrics at level 1.
        ohlcv_df = pivoted.swaplevel(0, 1, axis=1).sort_index(axis=1)

        # Final processing matching ParquetReader
        ohlcv_df = align_to_datetime_index(ohlcv_df, interval=interval)
        ohlcv_df = fill_after_first_non_nan_multilevel_safe(ohlcv_df, symbols=symbols)
        ohlcv_df = fill_symbol_metadata(ohlcv_df, symbols)

        return ohlcv_df

    def list_pairs(self) -> list[str]:
        conn = self._get_conn()
        res = conn.execute("SELECT DISTINCT pair FROM ohlcv ORDER BY pair").fetchall()
        return [r[0] for r in res]

    def list_symbols(self, quote="USD"):
        pairs = self.list_pairs()
        symbols = [p.split("-")[0] for p in pairs if p.endswith(f"-{quote}")]
        return symbols

    def get_random_symbols(self, n=10, quote="USD"):
        symbols = self.list_symbols(quote=quote)
        if not symbols:
            return []
        return np.random.choice(
            symbols, size=min(n, len(symbols)), replace=False
        ).tolist()

    def get_daily_historical_movers(
        self, top_n=20, trades_threshold=500, stables=False
    ):
        """
        Identify top movers using DuckDB SQL for much better performance.
        """
        conn = self._get_conn()

        # Filter stablecoins if requested (using the utility or a simple list)
        excluded_symbols = []
        if not stables:
            # Hardcoded common stables to avoid excessive imports/calls in SQL
            excluded_symbols = ["USDT", "USDC", "DAI", "PYUSD", "EUR", "GBP"]

        in_clause = ""
        if excluded_symbols:
            in_clause = f"AND base NOT IN ({','.join(['?']*len(excluded_symbols))})"

        sql = f"""
            WITH daily_data AS (
                SELECT 
                    timestamp::DATE as date,
                    pair,
                    base as symbol,
                    volume,
                    trades
                FROM ohlcv
                WHERE interval = '1d'
                AND trades > ?
                {in_clause}
            ),
            ranked AS (
                SELECT 
                    *,
                    row_number() OVER (PARTITION BY date ORDER BY volume DESC) as rank
                FROM daily_data
            )
            SELECT date, symbol, volume, trades
            FROM ranked
            WHERE rank <= ?
            ORDER BY date ASC, volume DESC
        """

        params = [trades_threshold] + excluded_symbols + [top_n]
        return conn.execute(sql, params).df()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class KrakenDailyMoversReader:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def _get_conn(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path, read_only=True)
        return self.conn

    def get_historical_movers_by_day(self, date, top_n=20):
        """
        Get top movers for a specific historical date from the daily_movers database.
        """
        conn = self._get_conn()

        # Ensure date is just a date string or object
        if hasattr(date, "date"):
            date = date.date()

        sql = "SELECT * FROM daily_movers WHERE date = ? ORDER BY volume DESC LIMIT ?"
        return conn.execute(sql, [date, top_n]).df()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
