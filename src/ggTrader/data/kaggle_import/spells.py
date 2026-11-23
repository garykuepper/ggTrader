"""Assign continuous return spells to CRSP observations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def assign_spells(
    rows: pd.DataFrame, calendar: pd.DatetimeIndex, *, tolerance: float = 0.02, max_gap: int = 10
) -> pd.DataFrame:
    """Assign independent spells for sorted stocks on the supplied calendar."""
    result = rows.copy()
    result["session"] = calendar.get_indexer(pd.to_datetime(result["date"]))
    if (result["session"] < 0).any():
        raise ValueError("date absent from trading calendar")
    grouped = result.groupby("permno", sort=False)
    previous_session = grouped["session"].shift()
    first = previous_session.isna()
    gap = (result["session"] - previous_session - 1).fillna(0).astype(int)
    result["gap_sessions"] = gap
    previous_shares = grouped["shrout"].shift()
    shares_ok = (
        result["shrout"].notna()
        & result["shrout"].ne(0)
        & previous_shares.notna()
        & previous_shares.ne(0)
    )
    mv = result["prc"].abs() * result["shrout"]
    previous_mv = mv.groupby(result["permno"], sort=False).shift()
    change = mv / previous_mv - 1
    valid = result["prc"].notna() & result["ret"].notna()
    bridge = (
        ~first
        & gap.gt(0)
        & gap.le(max_gap)
        & shares_ok
        & valid
        & (change - result["ret"]).abs().le(tolerance)
    )
    cut = ~first & (~valid | (gap.gt(0) & ~bridge))
    result["spell"] = cut.groupby(result["permno"], sort=False).cumsum().astype(int)
    result["chained_return"] = result["ret"].where(~first & ~cut, np.nan).where(~bridge, change)
    result["gap_return_approx"] = bridge.astype(bool)
    result.attrs.update(tolerance=tolerance, max_gap=max_gap)
    return result


def calendar_from_file(parquet_path: Path) -> pd.DatetimeIndex:
    """Read the distinct dates using only one row group's date column at a time."""
    parquet = pq.ParquetFile(parquet_path)
    dates: set = set()
    for index in range(parquet.num_row_groups):
        dates.update(parquet.read_row_group(index, columns=["date"]).column("date").to_pylist())
    return pd.DatetimeIndex(sorted(dates))


def gap_row_counts(spelled: pd.DataFrame) -> dict[str, int]:
    """Count gaps and the reasons for cutting short gaps."""
    max_gap = spelled.attrs.get("max_gap", 10)
    tolerance = spelled.attrs.get("tolerance", 0.02)
    gap = spelled["gap_sessions"]
    short = gap.between(1, max_gap)
    prev_shares = spelled.groupby("permno", sort=False)["shrout"].shift()
    no_shares = (
        spelled["shrout"].isna() | spelled["shrout"].eq(0) | prev_shares.isna() | prev_shares.eq(0)
    )
    mv = spelled["prc"].abs() * spelled["shrout"]
    change = mv / mv.groupby(spelled["permno"], sort=False).shift() - 1
    disagree = (change - spelled["ret"]).abs().gt(tolerance)
    return {
        "gap_rows": int(gap.ge(1).sum()),
        "short_rows": int(short.sum()),
        "short_cut_disagree": int((short & ~no_shares & disagree).sum()),
        "short_cut_no_shares": int((short & no_shares).sum()),
        "long_rows": int(gap.gt(max_gap).sum()),
    }
