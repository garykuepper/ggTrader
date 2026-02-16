import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from .utils import (
    align_to_datetime_index,
    fill_after_first_non_nan_multilevel_safe,
    fill_symbol_metadata,
    ensure_utc_timestamp,
    filter_out_stables,
)


class KrakenPostgresReader:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def read_ohlcv(
        self,
        symbol=None,
        interval=None,
        start=None,
        end=None,
        symbols=None,
        quote="USD",
    ):
        """
        Read OHLCV data from PostgreSQL.
        """
        where_clauses = []
        params = {}

        if symbol:
            # Ensure symbol has quote if missing
            if "-" not in symbol:
                symbol = f"{symbol}-{quote}"
            where_clauses.append("symbol = :symbol")
            params["symbol"] = symbol
        elif symbols:
            # Ensure symbols have quote if missing
            formatted_symbols = [f"{s}-{quote}" if "-" not in s else s for s in symbols]
            where_clauses.append("symbol IN :symbols")
            params["symbols"] = tuple(formatted_symbols)

        if interval:
            where_clauses.append("interval = :interval")
            params["interval"] = interval

        if start:
            where_clauses.append("timestamp >= :start")
            params["start"] = start

        if end:
            where_clauses.append("timestamp <= :end")
            params["end"] = end

        query = "SELECT * FROM ohlcv"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY timestamp ASC"

        # Using pandas read_sql with text() for named parameter binding
        df = pd.read_sql(text(query), self.engine, params=params)

        if df.empty:
            return df

        # Strip quote suffix from symbol column for consistency with base symbols
        if quote:
            df["symbol"] = df["symbol"].str.replace(f"-{quote}", "", regex=False)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

        return df

    def get_ohlcv_df(
        self, symbols: list, interval="1d", quote="USD", start=None, end=None
    ):
        """
        Retrieve aligned OHLCV data for multiple symbols.
        Returns MultiIndex DataFrame (columns: Symbol -> Metric).
        """
        if start:
            start = ensure_utc_timestamp(start)
        if end:
            end = ensure_utc_timestamp(end)

        # Ensure symbols are formatted for the query
        formatted_symbols = [f"{s}-{quote}" if "-" not in s else s for s in symbols]

        df = self.read_ohlcv(
            symbols=formatted_symbols,
            interval=interval,
            start=start,
            end=end,
            quote=quote,
        )

        if df.empty:
            return pd.DataFrame()

        # Reshape to multi-index (columns: symbol -> metric)
        # Postgres returns flat: timestamp, symbol, interval, open, high...

        # Pivot
        pivoted = df.pivot(
            columns="symbol",
            values=["open", "high", "low", "close", "volume", "trades"],
        )

        # Result: level 0 = metrics, level 1 = symbol
        # We want: level 0 = symbol, level 1 = metrics
        ohlcv_df = pivoted.swaplevel(0, 1, axis=1).sort_index(axis=1)

        # Standard post-processing
        ohlcv_df = align_to_datetime_index(ohlcv_df, interval=interval)
        ohlcv_df = fill_after_first_non_nan_multilevel_safe(ohlcv_df, symbols=symbols)
        ohlcv_df = fill_symbol_metadata(ohlcv_df, symbols)

        return ohlcv_df

    def list_symbols(self, quote="USD"):
        """List available symbols (optionally filtering by quote currency if stored)."""
        # Our postgres ingestor stores 'symbol' like 'XBT-USD' or 'ETH-USD'.
        # So checking endsWith quote is valid.
        query = "SELECT DISTINCT symbol FROM ohlcv"
        with self.engine.connect() as conn:
            res = conn.execute(text(query)).fetchall()

        all_syms = [r[0] for r in res]
        if quote:
            # assume symbols are BASE-QUOTE
            return [s.split("-")[0] for s in all_syms if s.endswith(f"-{quote}")]
        return [s.split("-")[0] for s in all_syms]

    def list_pairs(self):
        query = "SELECT DISTINCT symbol FROM ohlcv"
        with self.engine.connect() as conn:
            res = conn.execute(text(query)).fetchall()
        return [r[0] for r in res]

    def get_random_symbols(self, n=10, quote="USD"):
        symbols = self.list_symbols(quote=quote)
        if not symbols:
            return []
        return np.random.choice(
            symbols, size=min(n, len(symbols)), replace=False
        ).tolist()

    def get_daily_historical_movers(
        self, date, top_n=20, trades_threshold=500, stables=False
    ):
        """
        Identify top movers for a specific date.
        Queries the ohlcv table where interval='1d'.
        """
        # Exclude stables logic
        excluded = []
        if not stables:
            excluded = ["USDT", "USDC", "DAI", "PYUSD", "EUR", "GBP"]

        where_parts = [
            "interval = '1d'",
            "trades > :threshold",
            "timestamp::DATE = :date",
        ]

        if excluded:
            # We need to filter by base symbol.
            # symbol is 'BASE-QUOTE'. 'USDT-USD'.
            # split_part(symbol, '-', 1) NOT IN ...
            where_parts.append(f"split_part(symbol, '-', 1) NOT IN :excluded")

        query = f"""
            SELECT 
                timestamp::DATE as date,
                symbol,
                volume,
                trades
            FROM ohlcv
            WHERE {" AND ".join(where_parts)}
            ORDER BY volume DESC
            LIMIT :limit
        """

        params = {
            "threshold": trades_threshold,
            "date": date,
            "limit": top_n,
            "excluded": tuple(excluded) if excluded else None,
        }

        # If tuple is empty, sqlalchemy might complain if we use IN.
        # But we controlled that.

        # Note: 'symbol' here is the PAIR (e.g. XBT-USD).
        # The original code returned 'symbol' as pure base?
        # DuckDB query: `base as symbol`.
        # Here we verify what we return.
        # Let's return the Pair as symbol or split it?
        # Standardize on Base symbol for analysis?
        # In trading.py, we use symbols like 'XBT', 'ETH'.
        # So we should return BASE.

        # Modify query
        query = f"""
            SELECT 
                timestamp::DATE as date,
                split_part(symbol, '-', 1) as symbol,
                volume,
                trades
            FROM ohlcv
            WHERE {" AND ".join(where_parts)}
            ORDER BY volume DESC
            LIMIT :limit
        """

        # Use text() for pandas read_sql
        df = pd.read_sql(text(query), self.engine, params=params)
        return df

    def close(self):
        self.engine.dispose()
