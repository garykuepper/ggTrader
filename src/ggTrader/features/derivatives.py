"""Production derivatives features: funding_apr and basis_premium_apr.

Phase 4. Pull from the TimescaleDB ``funding_apr`` and ``basis_series``
matviews populated by ``data.sources.kraken_futures``.

Two strategy-facing features:
- ``funding_apr``: arity="single" — annualized realized funding for one
  perp. Available only over the ~1y window Kraken's API exposes.
- ``basis_premium_apr``: arity="pair" — annualized (perp - spot) / spot,
  the funding proxy. Available wherever both spot and perp OHLCV exist.

The smoothing windows (24h / 7d / 30d) are stored as separate columns in
the matviews and exposed as named features
(``funding_apr_30d``, ``basis_premium_apr_30d``, etc.) so strategies can
pick the right horizon without computing rolling means at request time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
import psycopg2

from ggTrader.core.instrument import Instrument
from ggTrader.features.base import pair_column_label

# Available smoothing columns per feature family
_FUNDING_COLUMNS = {
    "funding_apr": "funding_apr",
    "funding_apr_24h": "funding_apr_24h",
    "funding_apr_7d": "funding_apr_7d",
    "funding_apr_30d": "funding_apr_30d",
}
_BASIS_COLUMNS = {
    "basis_premium_apr": "premium_apr",
    "basis_premium_apr_24h": "premium_apr_24h",
    "basis_premium_apr_7d": "premium_apr_7d",
    "basis_premium_apr_30d": "premium_apr_30d",
}


def _to_naive_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def fetch_funding_apr(
    conn: psycopg2.extensions.connection,
    feature_name: str,
    instrument: Instrument,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    col = _FUNDING_COLUMNS[feature_name]
    venue_sym = instrument.venue_specific_id or instrument.symbol
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT "timestamp", {col}
                FROM funding_apr
                WHERE symbol = %s AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" """,
            (venue_sym, _to_naive_utc(start), _to_naive_utc(end)),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=[instrument.symbol])
    idx = pd.DatetimeIndex([ts.replace(tzinfo=timezone.utc) for ts, _ in rows])
    return pd.DataFrame({instrument.symbol: [v for _, v in rows]}, index=idx)


def fetch_basis_premium_apr(
    conn: psycopg2.extensions.connection,
    feature_name: str,
    instruments: Sequence[Instrument],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if len(instruments) != 2:
        raise KeyError(
            f"basis_premium_apr is arity='pair'; expected 2 instruments, got {len(instruments)}"
        )
    _, perp_inst = instruments
    venue_sym = perp_inst.venue_specific_id or perp_inst.symbol
    col = _BASIS_COLUMNS[feature_name]
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT "timestamp", {col}
                FROM basis_series
                WHERE perp_symbol = %s AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" """,
            (venue_sym, _to_naive_utc(start), _to_naive_utc(end)),
        )
        rows = cur.fetchall()
    label = pair_column_label(instruments)
    if not rows:
        return pd.DataFrame(columns=[label])
    idx = pd.DatetimeIndex([ts.replace(tzinfo=timezone.utc) for ts, _ in rows])
    return pd.DataFrame({label: [v for _, v in rows]}, index=idx)


FUNDING_FEATURE_NAMES = tuple(_FUNDING_COLUMNS.keys())
BASIS_FEATURE_NAMES = tuple(_BASIS_COLUMNS.keys())

__all__ = [
    "BASIS_FEATURE_NAMES",
    "FUNDING_FEATURE_NAMES",
    "fetch_basis_premium_apr",
    "fetch_funding_apr",
]
