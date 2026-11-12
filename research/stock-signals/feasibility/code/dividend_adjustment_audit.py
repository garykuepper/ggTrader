"""Flag price-adjustment factors unchanged across independently reported dividends."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.utils.manifest import sha256_file

# Four-decimal source prices can create small rounding differences in the ratio.
FACTOR_TOLERANCE = 1e-6


def assess(source: pd.DataFrame, reference: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    """Compare adjustment factors around reported dividend dates, not portfolio returns."""
    source = source.copy()
    reference = reference.copy()
    for frame in (source, reference):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame[["symbol", "date"]].isna().any().any():
            raise ValueError("missing symbol or date")
        if frame.duplicated(["symbol", "date"]).any():
            raise ValueError("duplicate symbol/date")
    if not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("invalid session calendar")
    source = source.set_index(["symbol", "date"])
    reference["dividend"] = pd.to_numeric(reference.dividend, errors="coerce")
    if not np.isfinite(reference.dividend).all() or (reference.dividend < 0).any():
        raise ValueError("invalid reference dividend")
    results = {}
    for symbol, group in reference[reference.dividend > 0].groupby("symbol"):
        unchanged, uncheckable = [], []
        checked = 0
        for row in group.sort_values("date").itertuples():
            date = row.date
            position = sessions.get_indexer([date])[0]
            previous = sessions[position - 1] if position > 0 else None
            keys = [(symbol, previous), (symbol, date)]
            if previous is None or any(key not in source.index for key in keys):
                uncheckable.append(str(date.date()))
                continue
            prices = source.loc[keys, ["close", "adjusted_close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            if not np.isfinite(prices).all().all() or not (prices > 0).all().all():
                uncheckable.append(str(date.date()))
                continue
            factors = prices.adjusted_close / prices.close
            checked += 1
            if abs(factors.iloc[1] / factors.iloc[0] - 1) <= FACTOR_TOLERANCE:
                unchanged.append(str(date.date()))
        results[symbol] = {
            "reference_events": len(group),
            "checked": checked,
            "unchanged_factor_dates": unchanged,
            "uncheckable_dates": uncheckable,
        }
    return results


def main() -> None:
    """Read hash-pinned files and write an adjustment-risk inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prices", "reference"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for name in ("prices", "reference"):
        if sha256_file(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"Input hash mismatch: {name}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("end precedes start")
    parts = []
    for chunk in pd.read_csv(
        args.prices, usecols=["date", "symbol", "close", "adjusted_close"], chunksize=500_000
    ):
        parts.append(chunk[chunk.symbol.isin(args.symbols)])
    source = pd.concat(parts, ignore_index=True)
    reference = pd.read_parquet(
        args.reference,
        filters=[("ticker", "in", args.symbols)],
        columns=["date", "ticker", "dividends"],
    ).rename(columns={"ticker": "symbol", "dividends": "dividend"})
    reference = reference[reference.date.between(start, end)]
    sessions = (
        mcal.get_calendar("NYSE").valid_days(start - pd.Timedelta(days=10), end).tz_localize(None)
    )
    result = {
        "purpose": (
            "Flag unchanged adjustment factors at reference dividend dates, not returns or proof "
            "of errors"
        ),
        "prices_sha256": args.prices_sha256,
        "reference_sha256": args.reference_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "calendar_version": version("pandas-market-calendars"),
        "start": args.start,
        "end": args.end,
        "factor_tolerance": FACTOR_TOLERANCE,
        "requested_symbols": args.symbols,
        "candidates": assess(source, reference, sessions),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
