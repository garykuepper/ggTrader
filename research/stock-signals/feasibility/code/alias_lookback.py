"""Check session continuity for candidate historical ticker links."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from coverage_alias_candidates import ALIASES


def count_lookback(
    rows: list[dict], bars: pd.DataFrame, sessions: pd.DatetimeIndex, aliases: dict, window: int
) -> dict:
    """Count missing old tickers with a successor and a complete session lookback."""
    if window < 1 or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("Invalid session calendar or window")
    if bars[["timestamp", "symbol"]].isna().any().any():
        raise ValueError("Null bar dates or symbols")
    if bars.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Duplicate bar dates")
    dates = pd.DatetimeIndex(bars.timestamp)
    if dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("Bar dates must be naive midnight timestamps")
    by_symbol = {name: set(group.timestamp) for name, group in bars.groupby("symbol")}
    result = {}
    for old, (new, effective) in aliases.items():
        selected = [
            row["date"] for row in rows if old in row["missing"] and row["date"] < effective
        ]
        same_day = complete = 0
        for value in selected:
            day = pd.Timestamp(value)
            idx = sessions.get_indexer([day])[0]
            if idx < window - 1:
                raise ValueError(f"Insufficient calendar history for {value}")
            available = by_symbol.get(new, set())
            same_day += day in available
            complete += all(date in available for date in sessions[idx - window + 1 : idx + 1])
        result[old] = {
            "eligible": len(selected),
            "same_day": same_day,
            "complete_lookback": complete,
        }
    return result


def main() -> None:
    """Write a hash-linked coverage record without evaluating prices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    tape_hash = sha256_file(args.tape)
    if tape_hash != coverage["tape_sha256"]:
        raise ValueError("Tape hash does not match coverage record")
    rows = coverage["month_ends"]
    start = pd.Timestamp(rows[0]["date"]) - pd.Timedelta(days=550)
    end = pd.Timestamp(rows[-1]["date"])
    sessions = mcal.get_calendar("NYSE").valid_days(start, end).tz_localize(None)
    bars = pd.read_parquet(args.tape, columns=["timestamp", "symbol"])
    result = {
        "purpose": "Session presence only; not a price, identity or strategy validation",
        "tape_sha256": tape_hash,
        "coverage_sha256": sha256_file(args.coverage),
        "code_sha256": sha256_file(Path(__file__)),
        "alias_code_sha256": sha256_file(Path(__file__).with_name("coverage_alias_candidates.py")),
        "calendar": "NYSE, pandas-market-calendars",
        "calendar_version": version("pandas-market-calendars"),
        "window_sessions_including_decision_day": 252,
        "candidates": count_lookback(rows, bars, sessions, ALIASES, 252),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
