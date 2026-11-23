"""Shared Yahoo and CRSP inputs for return matching."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ggTrader.data.bars import connect_readonly
from ggTrader.data.kaggle_import.guards import checked_pg_url


def yahoo_data(url: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple:
    """Read Yahoo prices and whole-history lifetimes in a read-only session."""
    from urllib.parse import unquote, urlsplit

    from ggTrader.data.kaggle_import.guards import assert_connected_database

    url = checked_pg_url(url.replace("postgresql+psycopg2://", "postgresql://", 1))
    conn = connect_readonly(url)
    try:
        assert_connected_database(conn, unquote(urlsplit(url).path.lstrip("/")))
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, timestamp, close FROM ohlcv WHERE venue = 'yfinance' "
                "AND interval = '1d' AND timestamp >= %s AND timestamp < %s",
                (start - pd.Timedelta(days=30), end + pd.Timedelta(days=1)),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT symbol, min(timestamp)::date, max(timestamp)::date FROM ohlcv "
                "WHERE venue = 'yfinance' AND interval = '1d' GROUP BY symbol"
            )
            lives = cur.fetchall()
    finally:
        conn.close()
    prices = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    sessions = prices.index[prices["SPY"].notna()]
    prices = prices.reindex(sessions)
    returns = prices.pct_change(fill_method=None).loc[start:end]
    life = pd.DataFrame(lives, columns=["symbol", "first", "last"]).set_index("symbol")
    return returns, life["first"], life["last"]


def crsp_data(
    path: Path, sessions: pd.DatetimeIndex, sample: int | None, min_sessions: int = 250
) -> tuple:
    """Stream twice: count observations and lifetimes, then fill the return matrix."""
    parquet = pq.ParquetFile(path)
    counts, first, last = {}, {}, {}
    for i in range(parquet.num_row_groups):
        rows = (
            parquet.read_row_group(i, columns=["permno", "date", "ret"])
            .to_pandas()
            .dropna(subset=["permno", "date"])
        )
        rows["date"] = pd.to_datetime(rows["date"])
        for permno, group in rows.groupby("permno"):
            permno = int(permno)
            first[permno] = min(first.get(permno, group.date.min()), group.date.min())
            last[permno] = max(last.get(permno, group.date.max()), group.date.max())
            counts[permno] = counts.get(permno, 0) + int(
                (group.date.isin(sessions) & np.isfinite(group.ret)).sum()
            )
    ids = sorted(p for p, count in counts.items() if count >= min_sessions)
    if sample is not None:
        ids = ids[:sample]
    matrix = np.full((len(sessions), len(ids)), np.nan, dtype=np.float32)
    positions = pd.Index(ids)
    for i in range(parquet.num_row_groups):
        rows = (
            parquet.read_row_group(i, columns=["permno", "date", "ret"])
            .to_pandas()
            .dropna(subset=["permno", "date"])
        )
        dates = sessions.get_indexer(pd.to_datetime(rows.date))
        columns = positions.get_indexer(rows.permno)
        mask = (dates >= 0) & (columns >= 0)
        matrix[dates[mask], columns[mask]] = rows.ret.to_numpy()[mask]
    return pd.DataFrame(matrix, index=sessions, columns=ids), pd.Series(first), pd.Series(last)
