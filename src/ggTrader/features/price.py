"""mid_price feature — production version pulling from TimescaleDB.

Resolves the right table per instrument: spot OHLCV (``ohlcv``) for spot
instruments, perp OHLCV (``perp_ohlcv``) for crypto perpetuals. Returns
close prices as the mid (sufficient for backtests; live trading would
swap in bid/ask average from a ticker).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
import psycopg2

from ggTrader.core.instrument import AssetClass, Instrument


def _to_naive_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def _table_and_symbol_for(instrument: Instrument) -> tuple[str, str]:
    if instrument.asset_class is AssetClass.CRYPTO_PERP:
        return "perp_ohlcv", instrument.venue_specific_id or instrument.symbol
    return "ohlcv", instrument.symbol


def fetch_mid_price(
    conn: psycopg2.extensions.connection,
    instruments: Sequence[Instrument],
    start: datetime,
    end: datetime,
    interval: str = "1h",
) -> pd.DataFrame:
    """One column per instrument, labeled ``instrument.symbol``."""
    cols: dict[str, pd.Series] = {}
    start_naive = _to_naive_utc(start)
    end_naive = _to_naive_utc(end)
    for inst in instruments:
        table, db_symbol = _table_and_symbol_for(inst)
        with conn.cursor() as cur:
            if table == "ohlcv":
                cur.execute(
                    f"""SELECT "timestamp", close
                        FROM {table}
                        WHERE symbol = %s AND "interval" = %s
                          AND venue = %s
                          AND "timestamp" BETWEEN %s AND %s
                        ORDER BY "timestamp" """,
                    (db_symbol, interval, inst.venue.value, start_naive, end_naive),
                )
            else:
                cur.execute(
                    f"""SELECT "timestamp", close
                        FROM {table}
                        WHERE symbol = %s AND "interval" = %s
                          AND "timestamp" BETWEEN %s AND %s
                        ORDER BY "timestamp" """,
                    (db_symbol, interval, start_naive, end_naive),
                )
            rows = cur.fetchall()
        if not rows:
            cols[inst.symbol] = pd.Series(dtype=float)
            continue
        idx = pd.DatetimeIndex([ts.replace(tzinfo=timezone.utc) for ts, _ in rows])
        cols[inst.symbol] = pd.Series([float(v) for _, v in rows], index=idx, name=inst.symbol)
    return pd.DataFrame(cols)


__all__ = ["fetch_mid_price"]
