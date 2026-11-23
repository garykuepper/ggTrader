#!/usr/bin/env python3
"""Cut a small fixture of real CRSP rows (selected stocks, windows around events) for the tests."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_FILE = Path(
    "/media/thesix/ggTrader-data/market-data/crsp-kaggle/cleaned/crsp_1968_2024_daily_cleaned.parquet"
)
GAP_STOCKS = (10066, 10124, 10128, 10669)
SPLIT_STOCKS = (12490, 14593, 10107)  # IBM, AAPL, MSFT
COLUMNS = ["permno", "date", "prc", "ret", "shrout"]


def stock_rows(parquet: pq.ParquetFile, permnos: tuple[int, ...]) -> pd.DataFrame:
    """Rows of the given stocks, read one row group at a time (bounded memory)."""
    parts = []
    for i in range(parquet.num_row_groups):
        table = parquet.read_row_group(i, columns=COLUMNS)
        mask = pc.is_in(table["permno"], value_set=pc.cast(list(permnos), table["permno"].type))
        hit = table.filter(mask)
        if hit.num_rows:
            parts.append(hit.to_pandas())
    return pd.concat(parts).sort_values(["permno", "date"]).reset_index(drop=True)


def calendar(parquet: pq.ParquetFile) -> pd.DatetimeIndex:
    """The file's own trading calendar (every distinct date)."""
    dates = set()
    for i in range(parquet.num_row_groups):
        column = parquet.read_row_group(i, columns=["date"]).column("date")
        dates.update(column.unique().to_pylist())
    return pd.DatetimeIndex(sorted(dates))


def window(rows: pd.DataFrame, centers: list[int], half: int) -> pd.DataFrame:
    """Rows within `half` positions of any centre position."""
    keep = np.zeros(len(rows), dtype=bool)
    for c in centers:
        keep[max(c - half, 0) : c + half + 1] = True
    return rows[keep]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    parquet = pq.ParquetFile(args.file)
    cal = calendar(parquet)
    rows = stock_rows(parquet, GAP_STOCKS + SPLIT_STOCKS)
    pieces = []
    session = pd.Series(np.arange(len(cal)), index=cal)
    for permno, frame in rows.groupby("permno"):
        frame = frame.reset_index(drop=True)
        s = frame["date"].map(session).to_numpy()
        if permno in GAP_STOCKS:
            gaps = np.where(np.diff(s) > 1)[0]  # position before each missing-session gap
            centers = [int(g) for g in gaps] + [int(g) + 1 for g in gaps]
            pieces.append(window(frame, centers, 6))
        else:
            ap = frame["prc"].abs()
            disagree = ((ap / ap.shift() - 1) - frame["ret"]).abs() > 0.10
            pieces.append(window(frame, [int(i) for i in np.where(disagree)[0]], 8))
    out = pd.concat(pieces)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "crsp_cases.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame({"date": cal}).to_csv(
        args.out_dir / "crsp_calendar.csv", index=False, date_format="%Y-%m-%d"
    )
    print(f"rows {len(out)}, stocks {out['permno'].nunique()}, sessions {len(cal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
