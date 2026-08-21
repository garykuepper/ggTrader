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

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ggTrader.data.live.yfinance_loader import YFinanceDataLoader
from ggTrader.utils.config import get_db_connection_string

STOCK_VENUE = "yfinance"


def _default_connection_string() -> str:
    if "GGTRADER_DB_URL" in os.environ:
        return os.environ["GGTRADER_DB_URL"]
    return get_db_connection_string()


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
            sym: (pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")) for sym, lo, hi in rows
        }

    # ------------------------------------------------------------------
    # Per-symbol inception tracking
    # ------------------------------------------------------------------
    #
    # A symbol whose true first bar postdates a fixed historical
    # ``start_date`` by more than 7 days (recent IPOs, 2020+ index
    # additions, leveraged ETFs vs. a 2010 data_start) would otherwise
    # trip the "first > start_date + 7d" full-refetch trigger on EVERY
    # run forever, since the cached first bar never moves closer to
    # start_date no matter how many times it's fetched. We record the
    # earliest bar a full fetch has ever actually returned for a symbol;
    # once that recorded inception matches what's currently cached, a
    # start-date gap is known-genuine (the listing/data really starts
    # there) rather than a sign the cache is missing fetchable history,
    # and the full refetch is skipped. If the cached first bar ever
    # regresses (e.g. table wiped and partially repopulated), the
    # mismatch against the recorded inception forces a full refetch again.

    _INCEPTION_SCHEMA = """
        CREATE TABLE IF NOT EXISTS symbol_inception (
            symbol      text        NOT NULL,
            interval    text        NOT NULL,
            venue       text        NOT NULL,
            first_date  timestamptz NOT NULL,
            recorded_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, interval, venue)
        );
    """

    def _ensure_inception_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(self._INCEPTION_SCHEMA)
            conn.commit()

    def _get_known_inceptions(self, symbols: List[str], interval: str) -> Dict[str, pd.Timestamp]:
        try:
            self._ensure_inception_schema()
            query = """
                SELECT symbol, first_date FROM symbol_inception
                WHERE symbol = ANY(%s) AND interval = %s AND venue = %s;
            """
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(query, (list(symbols), interval, STOCK_VENUE))
                rows = cur.fetchall()
            return {sym: pd.Timestamp(dt, tz="UTC") for sym, dt in rows}
        except Exception as e:
            self.logger.error(f"Inception lookup failed ({e}); treating as unknown")
            return {}

    def _record_inceptions(self, inceptions: Dict[str, pd.Timestamp], interval: str) -> None:
        if not inceptions:
            return
        try:
            self._ensure_inception_schema()
            query = """
                INSERT INTO symbol_inception (symbol, interval, venue, first_date)
                VALUES %s
                ON CONFLICT (symbol, interval, venue) DO UPDATE SET
                    first_date = EXCLUDED.first_date,
                    recorded_at = now();
            """
            records = [
                (sym, interval, STOCK_VENUE, ts.to_pydatetime()) for sym, ts in inceptions.items()
            ]
            with self._connect() as conn, conn.cursor() as cur:
                execute_values(cur, query, records)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record symbol inception ({e})")

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
        wide.columns = pd.MultiIndex.from_tuples([(sym, metric) for metric, sym in wide.columns])
        wide.sort_index(axis=1, inplace=True)
        return wide

    def _cache_to_db(self, df: pd.DataFrame, interval: str) -> None:
        """Persist a yfinance OHLCV frame (PK includes venue).

        Vectorized per-symbol: builds each symbol's row block from numpy
        arrays (to_numpy over the [open, high, low, close, volume] columns)
        instead of iterrows()'s per-cell Series construction, then extends
        the shared ``records`` list via zip(). NaN handling is identical to
        the row-by-row version: a row is dropped entirely if close or the
        timestamp is NaN/NaT; any other NaN OHLCV field becomes SQL NULL.
        """
        if df.empty:
            return
        records = []
        for symbol in df.columns.get_level_values(0).unique():
            sub = df[symbol].reindex(columns=["open", "high", "low", "close", "volume"])
            valid = sub["close"].notna().to_numpy() & ~sub.index.isna()
            if not valid.any():
                continue
            sub = sub.loc[valid]
            n = len(sub)
            # ``ohlcv.timestamp`` is `timestamp WITHOUT time zone`. Handing
            # psycopg2 a tz-AWARE datetime lets Postgres rebase the offset
            # into the session timezone on the way in -- on this box
            # (America/Los_Angeles) that turned 2026-08-03 00:00+00:00 into
            # 2026-08-02 17:00, stamping every equity bar one calendar day
            # EARLY (the whole store ran Sun-Thu: 1.07M Sunday bars against
            # 168 Friday ones). Strip the tz so what we hand psycopg2 is
            # already naive UTC and Postgres has nothing to rebase.
            idx = sub.index
            idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
            ts_list = idx.to_pydatetime().tolist()
            arr = sub.to_numpy(dtype=float)
            obj = arr.astype(object)
            obj[np.isnan(arr)] = None
            open_c, high_c, low_c, close_c, vol_c = obj.T
            records.extend(
                zip(
                    ts_list,
                    [symbol] * n,
                    [interval] * n,
                    [STOCK_VENUE] * n,
                    open_c.tolist(),
                    high_c.tolist(),
                    low_c.tolist(),
                    close_c.tolist(),
                    vol_c.tolist(),
                    [0] * n,
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
        conn = None
        try:
            conn = self._connect()
            conn.autocommit = True
            with conn.cursor() as cur:
                execute_values(cur, query, records, page_size=5000)
        except Exception as e:
            self.logger.error(f"Failed to cache stock data to DB: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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

        full-range fetch: symbols with no cache; or whose cached history
        starts more than ~5 trading days after the requested start AND
        whose cached first bar does not match a previously-recorded
        confirmed inception (see the "Per-symbol inception tracking" note
        above) -- avoids re-fetching a stable, known-late-listing symbol
        (e.g. a 2021 IPO vs. a 2010 data_start) on every single run.
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

        # Only symbols with a real start-date gap need the inception lookup.
        gap_candidates = [
            sym
            for sym, (first, _last) in coverage.items()
            if start_date is not None and first > start_date + pd.Timedelta(days=7)
        ]
        known_inceptions = (
            self._get_known_inceptions(gap_candidates, interval) if gap_candidates else {}
        )

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
                known = known_inceptions.get(sym)
                if known is None or known != first:
                    full_fetch.append(sym)
                    continue
                # Confirmed known inception: the cached first bar matches a
                # prior full fetch's earliest returned bar, so this is a
                # genuine listing-start gap, not missing fetchable history.
                # Fall through to the (unaffected) end-freshness check below.
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
                if start_date is not None:
                    live_syms = set(live.columns.get_level_values(0).unique())
                    new_inceptions: Dict[str, pd.Timestamp] = {}
                    for sym in full_fetch:
                        if sym not in live_syms:
                            continue
                        sym_close = live[sym]["close"].dropna()
                        if sym_close.empty:
                            continue
                        sym_first = sym_close.index.min()
                        if sym_first > start_date + pd.Timedelta(days=7):
                            new_inceptions[sym] = sym_first
                    self._record_inceptions(new_inceptions, interval)
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
