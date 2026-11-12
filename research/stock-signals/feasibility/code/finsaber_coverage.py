"""Measure direct-ticker lookback coverage; never infer a security identity."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from coverage_alias_candidates import ALIASES  # noqa: E402

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]


def measure(
    rows: list[dict],
    bars: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    excluded: set[str],
    window: int,
) -> dict:
    """Count date completeness and basic price/volume validity, not identity or returns."""
    if bars.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate symbol/date rows")
    if window < 1 or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("invalid trading calendar or window")
    dates = pd.to_datetime(bars["date"], errors="raise")
    if dates.isna().any() or bars["symbol"].isna().any():
        raise ValueError("missing symbol or date")
    values = bars[["open", "high", "low", "close", "adjusted_close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (values["high"] >= values[["open", "close", "low"]].max(axis=1)) & (
        values["low"] <= values[["open", "close", "high"]].min(axis=1)
    )
    work = pd.DataFrame({"symbol": bars.symbol, "date": dates, "valid": valid})
    all_dates = {s: set(g.date) for s, g in work.groupby("symbol")}
    good_dates = {s: set(g.loc[g.valid, "date"]) for s, g in work.groupby("symbol")}
    symbols = sorted({s for row in rows for s in row["missing"]} - excluded)
    results = {}
    for symbol in symbols:
        eligible = [pd.Timestamp(row["date"]) for row in rows if symbol in row["missing"]]
        same_day = complete = complete_valid = 0
        for date in eligible:
            index = sessions.get_indexer([date])[0]
            if index < window - 1:
                raise ValueError(f"insufficient calendar history for {date.date()}")
            required = set(sessions[index - window + 1 : index + 1])
            available = all_dates.get(symbol, set())
            same_day += date in available
            complete += required <= available
            complete_valid += required <= good_dates.get(symbol, set())
        results[symbol] = {
            "missing_months": len(eligible),
            "same_day": same_day,
            "complete_dates": complete,
            "complete_valid_bars": complete_valid,
        }
    return results


def main() -> None:
    """Read local pinned inputs and write a new availability-only result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--prices-sha256", required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--coverage-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for path, expected in [
        (args.prices, args.prices_sha256),
        (args.coverage, args.coverage_sha256),
    ]:
        if sha256_file(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    rows = json.loads(args.coverage.read_text(encoding="utf-8"))["month_ends"]
    symbols = {s for row in rows for s in row["missing"]} - set(ALIASES)
    parts = []
    for chunk in pd.read_csv(args.prices, usecols=COLUMNS, chunksize=500_000):
        parts.append(chunk[chunk.symbol.isin(symbols)])
    bars = pd.concat(parts, ignore_index=True)
    start = pd.Timestamp(rows[0]["date"]) - pd.Timedelta(days=550)
    end = pd.Timestamp(rows[-1]["date"])
    sessions = mcal.get_calendar("NYSE").valid_days(start, end).tz_localize(None)
    result = {
        "purpose": "Coverage and elementary bar checks only; not identity or return validation",
        "prices_sha256": args.prices_sha256,
        "coverage_sha256": args.coverage_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "alias_code_sha256": sha256_file(Path(__file__).with_name("coverage_alias_candidates.py")),
        "calendar_version": version("pandas-market-calendars"),
        "window_sessions": 252,
        "candidates": measure(rows, bars, sessions, excluded=set(ALIASES), window=252),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
