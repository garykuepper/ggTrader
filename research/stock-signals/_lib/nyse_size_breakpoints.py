"""Parse French NYSE market-equity breakpoints without inferring availability."""

import csv
import io

import numpy as np
import pandas as pd


def parse(text: str) -> pd.DataFrame:
    """Return month-end median market equity in USD millions and stock counts.

    This parses supplied text only. The caller verifies the archive, source vintage,
    universe comparability and when any historical value became available.
    """
    records = []
    footer = False
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(field.strip() for field in row):
            continue
        label = row[0].strip()
        if records and label.startswith("Copyright ") and len(row) == 1:
            footer = True
            continue
        if not label.isdigit() or len(label) != 6:
            if records or label[:1].isdigit():
                raise ValueError("Invalid breakpoint row")
            continue
        if footer:
            raise ValueError("Breakpoint row after footer")
        try:
            if len(row) != 22:
                raise ValueError("Invalid breakpoint row")
            month = pd.Period(row[0].strip(), freq="M")
            count = int(row[1])
            percentiles = np.asarray(row[2:], dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid breakpoint row") from exc
        if (
            count <= 0
            or not np.isfinite(percentiles).all()
            or (percentiles <= 0).any()
            or (np.diff(percentiles) < 0).any()
        ):
            raise ValueError("Invalid breakpoint values")
        records.append((month, count, percentiles[9]))
    if not records:
        raise ValueError("No breakpoint rows")
    months = pd.PeriodIndex([r[0] for r in records], freq="M")
    if months.has_duplicates or not months.equals(pd.period_range(months[0], months[-1], freq="M")):
        raise ValueError("Breakpoint months are not consecutive")
    return pd.DataFrame(
        {"count": [r[1] for r in records], "median_usd_millions": [r[2] for r in records]},
        index=months.to_timestamp("M"),
    )
