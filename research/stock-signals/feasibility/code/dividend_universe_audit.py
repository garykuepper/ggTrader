"""Apply the dividend adjustment diagnostic to exact ticker overlaps."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from dividend_adjustment_audit import FACTOR_TOLERANCE, assess  # noqa: E402


def summarize(results: dict) -> dict:
    """Count reports, checks and flags without treating missing checks as passes."""
    return {
        "symbols_with_events": len(results),
        "events": sum(x["reference_events"] for x in results.values()),
        "checked_events": sum(x["checked"] for x in results.values()),
        "flagged_events": sum(len(x["unchanged_factor_dates"]) for x in results.values()),
        "uncheckable_events": sum(len(x["uncheckable_dates"]) for x in results.values()),
        "symbols_flagged": sum(bool(x["unchanged_factor_dates"]) for x in results.values()),
    }


def main() -> None:
    """Inspect exact-label overlaps in a bounded interval from pinned local files."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prices", "reference"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
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
    before = start - pd.Timedelta(days=10)
    source_parts = []
    for c in pd.read_csv(
        args.prices, usecols=["date", "symbol", "close", "adjusted_close"], chunksize=500_000
    ):
        source_parts.append(c[c.date.between(str(before.date()), str(end.date()))])
    source = pd.concat(source_parts, ignore_index=True)
    source["date"] = pd.to_datetime(source.date, errors="raise")
    reference = pd.read_parquet(
        args.reference,
        filters=[("date", ">=", start), ("date", "<=", end)],
        columns=["date", "ticker", "dividends"],
    ).rename(columns={"ticker": "symbol", "dividends": "dividend"})
    overlap = set(source.symbol.dropna()) & set(reference.symbol.dropna())
    source = source[source.symbol.isin(overlap)]
    reference = reference[reference.symbol.isin(overlap)]
    sessions = mcal.get_calendar("NYSE").valid_days(before, end).tz_localize(None)
    candidates = assess(source, reference, sessions)
    result = {
        "purpose": "Exact ticker overlap; flags do not establish identity or price errors",
        "prices_sha256": args.prices_sha256,
        "reference_sha256": args.reference_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "diagnostic_sha256": sha256_file(Path(__file__).with_name("dividend_adjustment_audit.py")),
        "calendar_version": version("pandas-market-calendars"),
        "start": args.start,
        "end": args.end,
        "factor_tolerance": FACTOR_TOLERANCE,
        "overlapping_labels": len(overlap),
        "summary": summarize(candidates),
        "candidates": candidates,
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
