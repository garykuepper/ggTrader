"""Stock data loader with TimescaleDB caching (venue='yfinance').

Self-contained for equities: the crypto TimescaleDBLoader appends '-USD' to
bare symbols and filters venue='kraken_spot', both wrong for stocks. This
loader reads/writes the shared ``ohlcv`` hypertable directly with exact stock
tickers and venue='yfinance' (PK: timestamp, symbol, interval, venue).

Freshness is evaluated PER SYMBOL: symbols absent from (or barely covered in)
the cache get a full-range fetch; stale symbols get an incremental fetch.
A single joint incremental window would silently truncate history for every
ticker newly added to an existing cache.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

STOCK_VENUE = "yfinance"


def _default_connection_string() -> str:
    return os.environ.get(
        "GGTRADER_DB_URL",
        "postgresql://ggtrader:ggtrader@localhost:5433/ggtrader",
    )


class CachedYFinanceLoader(YFinanceDataLoader):
    """Stock data loader with TimescaleDB caching. All yfinance fetches are persisted."""

    def __init__(self, connection_string: Optional[str] = None):
        super().__init__()
        conn = connection_string
        if conn is None:
            try:
                from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader

                conn = TimescaleDBLoader().connection_string
            except Exception:
                conn = _default_connection_string()
        self.connection_string = conn.replace("postgresql+psycopg2://", "postgresql://")

    @staticmethod
    def _interval_to_timedelta(interval: str) -> pd.Timedelta:
        _MAP = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "60m": pd.Timedelta(hours=1),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
            "5d": pd.Timedelta(days=5),
            "1wk": pd.Timedelta(weeks=1),
            "1mo": pd.Timedelta(days=30),
            "3mo": pd.Timedelta(days=90),
        }
        if interval not in _MAP:
            raise ValueError(f"Unknown interval {interval!r}")
        return _MAP[interval]

    # ------------------------------------------------------------------
    # DB access
    # ------------------------------------------------------------------

    def _connect(self):
        return psycopg2.connect(self.connection_string)

    def _db_coverage(self, symbols: List[str], interval: str) -> Dict[str, Tuple]:
        """Per-symbol (first, last) cached timestamps for venue='yfinance'."""
        query = """
            SELECT symbol, min(timestamp), max(timestamp)
            FROM ohlcv
            WHERE symbol = ANY(%s) AND interval = %s AND venue = %s
            GROUP BY symbol;
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (list(symbols), interval, STOCK_VENUE))
            rows = cur.fetchall()
        return {
            sym: (pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
            for sym, lo, hi in rows
        }

    def _db_fetch(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        conditions = ["symbol = ANY(%s)", "interval = %s", "venue = %s"]
        params: list = [list(symbols), interval, STOCK_VENUE]
        if start_date is not None:
            conditions.append("timestamp >= %s")
            params.append(start_date.to_pydatetime())
        if end_date is not None:
            conditions.append("timestamp <= %s")
            params.append(end_date.to_pydatetime())
        query = f"""
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM ohlcv WHERE {" AND ".join(conditions)}
            ORDER BY timestamp ASC;
        """
        with self._connect() as conn:
            flat = pd.read_sql(query, conn, params=params, parse_dates=["timestamp"])
        if flat.empty:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(flat["timestamp"])
        flat["timestamp"] = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        wide = flat.pivot(
            index="timestamp", columns="symbol", values=["open", "high", "low", "close", "volume"]
        )
        wide.columns = pd.MultiIndex.from_tuples(
            [(sym, metric) for metric, sym in wide.columns]
        )
        wide.sort_index(axis=1, inplace=True)
        return wide

    def _cache_to_db(self, df: pd.DataFrame, interval: str) -> None:
        """Persist a yfinance OHLCV frame (PK includes venue)."""
        if df.empty:
            return
        records = []
        for symbol in df.columns.get_level_values(0).unique():
            symbol_data = df[symbol]
            for ts, row in symbol_data.iterrows():
                if pd.isna(row.get("close")) or pd.isna(ts):
                    continue
                records.append(
                    (
                        ts.to_pydatetime(),
                        symbol,
                        interval,
                        STOCK_VENUE,
                        float(row["open"]) if not pd.isna(row["open"]) else None,
                        float(row["high"]) if not pd.isna(row["high"]) else None,
                        float(row["low"]) if not pd.isna(row["low"]) else None,
                        float(row["close"]),
                        float(row["volume"]) if not pd.isna(row["volume"]) else None,
                        0,
                    )
                )
        if not records:
            return
        query = """
            INSERT INTO ohlcv
                (timestamp, symbol, interval, venue, open, high, low, close, volume, trades)
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
            conn = self._connect()
            conn.autocommit = True
            with conn.cursor() as cur:
                execute_values(cur, query, records, page_size=5000)
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to cache stock data to DB: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """DB-first fetch with per-symbol freshness.

        full-range fetch: symbols with no cache, or whose cached history starts
        more than ~5 trading days after the requested start.
        incremental fetch: cached symbols whose last bar is older than
        1.5x the interval relative to ``end_date`` (or now).
        """
        now_utc = pd.Timestamp.now(tz="UTC")
        effective_end = end_date or now_utc
        delta = self._interval_to_timedelta(interval)

        coverage = {}
        try:
            coverage = self._db_coverage(symbols, interval)
        except Exception as e:
            self.logger.error(f"DB coverage query failed ({e}); fetching all from yfinance")

        full_fetch: List[str] = []
        incr_fetch: List[str] = []
        incr_start: Optional[pd.Timestamp] = None
        for sym in symbols:
            cov = coverage.get(sym)
            if cov is None:
                full_fetch.append(sym)
                continue
            first, last = cov
            if start_date is not None and first > start_date + pd.Timedelta(days=7):
                full_fetch.append(sym)
                continue
            # End freshness: don't demand bars newer than the data can be
            # (e.g. end_date in the future or on a weekend).
            target = min(effective_end, now_utc)
            if (target - last) > delta * 1.5:
                incr_fetch.append(sym)
                incr_start = last + delta if incr_start is None else min(incr_start, last + delta)

        fetched_frames: List[pd.DataFrame] = []
        if full_fetch:
            self.logger.info(f"Full-range yfinance fetch for {len(full_fetch)} symbols")
            live = super().fetch_ohlcv(full_fetch, interval, start_date, end_date)
            if not live.empty:
                self._cache_to_db(live, interval)
                fetched_frames.append(live)
        if incr_fetch:
            self.logger.info(
                f"Incremental yfinance fetch for {len(incr_fetch)} symbols from {incr_start}"
            )
            live = super().fetch_ohlcv(incr_fetch, interval, incr_start, end_date)
            if not live.empty:
                self._cache_to_db(live, interval)
                fetched_frames.append(live)

        try:
            db_df = self._db_fetch(symbols, interval, start_date, end_date)
        except Exception as e:
            self.logger.error(f"DB read failed ({e}); using live data only")
            db_df = pd.DataFrame()

        if db_df.empty:
            frames = [f for f in fetched_frames if not f.empty]
            if not frames:
                return pd.DataFrame()
            combined = pd.concat(frames, axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            return combined.tail(limit) if limit else combined

        # DB now holds everything we just persisted; it is the source of truth.
        return db_df.tail(limit) if limit else db_df
