"""Audit membership and daily-bar presence without reading prices or returns."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.data.core.index_constituents import coverage_stats, load_sp500_history
from ggTrader.utils.manifest import sha256_file


def coverage(bars: pd.DataFrame, history: pd.DataFrame, dates: pd.DatetimeIndex) -> list[dict]:
    """Compare normalized members with bars on each requested decision date."""
    if bars[["timestamp", "symbol"]].isna().any().any():
        raise ValueError("Null bar timestamps or symbols")
    if bars.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Found duplicate symbol-date bars")
    timestamps = pd.DatetimeIndex(bars["timestamp"])
    if timestamps.tz is not None or not (timestamps == timestamps.normalize()).all():
        raise ValueError("Tape timestamps must be naive midnight dates")
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("Membership snapshots must be unique and ordered")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Decision dates must be unique and ordered")
    lookup = {date: set(group["symbol"]) for date, group in bars.groupby("timestamp")}
    rows = []
    for date in dates:
        stamp = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
        index = history.index.searchsorted(stamp, side="right") - 1
        if index < 0:
            raise ValueError(f"Decision date {date.date()} predates membership history")
        members = history["tickers"].iloc[index]
        if not members:
            raise ValueError(f"Empty membership on {date.date()}")
        day = stamp.tz_localize(None).normalize()
        rows.append(
            {
                "date": str(day.date()),
                "membership_snapshot": str(history.index[index].date()),
                **coverage_stats(members, lookup.get(day, set())),
            }
        )
    return rows


def main() -> None:
    """Read pinned local inputs and write a new month-end coverage record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--tape-sha256", required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--start", required=True, help="First month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for path, expected in (
        (args.tape, args.tape_sha256),
        (args.membership, args.membership_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    start = pd.Period(args.start, freq="M")
    end = pd.Period(args.end, freq="M")
    if end < start:
        raise ValueError("End month precedes start month")
    sessions = (
        mcal.get_calendar("NYSE").valid_days(start.start_time, end.end_time).tz_localize(None)
    )
    dates = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).last()
    if len(dates) != end.ordinal - start.ordinal + 1:
        raise ValueError("Could not determine every month-end session")
    bars = pd.read_parquet(args.tape, columns=["timestamp", "symbol"])
    rows = coverage(bars, load_sp500_history(args.membership), pd.DatetimeIndex(dates))
    result = {
        "purpose": "Research data availability only; no price, return or identity validation",
        "tape_sha256": args.tape_sha256,
        "membership_sha256": args.membership_sha256,
        "audit_code_sha256": sha256_file(Path(__file__)),
        "calendar": "NYSE, pandas-market-calendars",
        "calendar_version": version("pandas-market-calendars"),
        "start": str(start),
        "end": str(end),
        "month_ends": rows,
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
