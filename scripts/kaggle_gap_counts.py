"""Count gap decisions in a sorted CRSP Parquet file, one row group at a time."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ggTrader.data.kaggle_import.spells import assign_spells, calendar_from_file, gap_row_counts


def count_file(path: Path, calendar: pd.DatetimeIndex) -> dict[str, int]:
    """Count decisions while retaining the preceding stock row at each boundary."""
    parquet = pq.ParquetFile(path)
    totals = dict.fromkeys(
        ("gap_rows", "short_rows", "short_cut_disagree", "short_cut_no_shares", "long_rows"), 0
    )
    carry = None
    for index in range(parquet.num_row_groups):
        rows = parquet.read_row_group(
            index, columns=["permno", "date", "prc", "ret", "shrout"]
        ).to_pandas()
        if rows.empty:
            continue
        if carry is not None:
            rows = pd.concat([carry, rows], ignore_index=True)
        counts = gap_row_counts(assign_spells(rows, calendar))
        for key, value in counts.items():
            totals[key] += value
        carry = rows.tail(1)
    return totals


def main() -> None:
    """Write gap counts as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = count_file(args.file, calendar_from_file(args.file))
    args.out.write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
