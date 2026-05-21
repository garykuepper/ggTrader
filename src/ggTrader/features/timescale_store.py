"""TimescaleDB-backed FeatureStore — production implementation of the
widened Phase 3.5 ``FeatureStore`` Protocol.

Resolves feature requests against the matviews populated by
``data.sources.kraken_futures`` and the existing spot ``ohlcv`` table.

Supported features (Phase 4):
  - ``mid_price``                arity="single"
  - ``funding_apr`` (+ _24h/_7d/_30d)             arity="single"
  - ``basis_premium_apr`` (+ _24h/_7d/_30d)       arity="pair"

A simple in-memory range cache keyed by (feature, instrument-tuple) avoids
re-querying when the backtest engine asks for the same series across many
``get_at`` calls. The cache is invalidated only by explicit ``clear()``.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from ggTrader.core.instrument import Instrument
from ggTrader.features.base import (
    InstrumentArg,
    _normalize_instruments,
)
from ggTrader.features.derivatives import (
    BASIS_FEATURE_NAMES,
    FUNDING_FEATURE_NAMES,
    fetch_basis_premium_apr,
    fetch_funding_apr,
)
from ggTrader.features.price import fetch_mid_price


def _connect_from_env() -> psycopg2.extensions.connection:
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ggtrader"),
        password=os.getenv("DB_PASS", "ggtrader"),
        dbname=os.getenv("DB_NAME", "ggtrader"),
    )


class TimescaleFeatureStore:
    """Production FeatureStore. Satisfies the Phase 3.5 Protocol shape."""

    def __init__(self, conn: psycopg2.extensions.connection | None = None) -> None:
        self._conn = conn if conn is not None else _connect_from_env()
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[datetime, datetime, pd.DataFrame]] = {}

    # ---------------------------------------------------------------- public

    def get(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end must be tz-aware UTC")
        insts = _normalize_instruments(instruments, start)
        key = (feature_name, tuple(i.symbol for i in insts))
        cached = self._cache.get(key)
        if cached is not None:
            c_start, c_end, c_df = cached
            if c_start <= start and end <= c_end:
                return c_df.loc[(c_df.index >= start) & (c_df.index <= end)]
        df = self._fetch(feature_name, insts, start, end)
        self._cache[key] = (start, end, df)
        return df

    def get_at(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        ts: datetime,
    ) -> pd.Series:
        if ts.tzinfo is None:
            raise ValueError("ts must be tz-aware UTC")
        df = self.get(feature_name, instruments, ts, ts)
        # asof-style lookup: most recent value at or before ts
        eligible = df.loc[df.index <= ts]
        if eligible.empty:
            raise LookupError(f"no value for {feature_name} at {ts.isoformat()}")
        row = eligible.iloc[-1]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- dispatch

    def _fetch(
        self,
        feature_name: str,
        instruments: list[Instrument],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if feature_name == "mid_price":
            return fetch_mid_price(self._conn, instruments, start, end)
        if feature_name in FUNDING_FEATURE_NAMES:
            if len(instruments) != 1:
                raise KeyError(
                    f"{feature_name} is arity='single'; got {len(instruments)} instruments"
                )
            return fetch_funding_apr(self._conn, feature_name, instruments[0], start, end)
        if feature_name in BASIS_FEATURE_NAMES:
            return fetch_basis_premium_apr(self._conn, feature_name, instruments, start, end)
        raise KeyError(f"unknown feature: {feature_name}")


__all__ = ["TimescaleFeatureStore"]
