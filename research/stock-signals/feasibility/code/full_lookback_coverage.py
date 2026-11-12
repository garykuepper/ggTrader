"""Measure monthly member lookback availability without inspecting prices."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.data.core.index_constituents import load_sp500_history, normalize_yf_ticker
from ggTrader.utils.manifest import sha256_file


def measure(
    rows: list[dict],
    bars: pd.DataFrame,
    history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    window: int,
) -> list[dict]:
    """Count available and complete member histories on each decision session."""
    if bars[["timestamp", "symbol"]].isna().any().any():
        raise ValueError("missing symbol or timestamp")
    if bars.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate symbol/date bars")
    dates = pd.DatetimeIndex(bars.timestamp)
    if dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("bar timestamps must be naive dates")
    if window < 1 or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("invalid session calendar or window")
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("invalid membership history")
    available = {s: set(g.timestamp) for s, g in bars.groupby("symbol")}
    result = []
    for row in rows:
        date = pd.Timestamp(row["date"])
        index = sessions.get_indexer([date])[0]
        if index < window - 1:
            raise ValueError(f"date {date.date()} is missing from calendar or lacks history")
        snap = history.index.searchsorted(date.tz_localize("UTC"), side="right") - 1
        if snap < 0:
            raise ValueError(f"no membership snapshot on {date.date()}")
        members = {normalize_yf_ticker(s) for s in history.tickers.iloc[snap]}
        if len(members) != row["n_members"]:
            raise ValueError(f"membership count mismatch on {date.date()}")
        required = set(sessions[index - window + 1 : index + 1])
        same_day = {s for s in members if date in available.get(s, set())}
        complete = {s for s in same_day if required <= available[s]}
        result.append(
            {
                "date": str(date.date()),
                "membership_snapshot": str(history.index[snap].date()),
                "members": len(members),
                "same_day": len(same_day),
                "complete_lookback": len(complete),
                "missing_same_day": sorted(members - same_day),
                "insufficient_lookback": sorted(same_day - complete),
            }
        )
    return result


def main() -> None:
    """Write a fresh result against three explicitly hash-pinned inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tape", "membership", "coverage"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for name in ("tape", "membership", "coverage"):
        if sha256_file(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"Input hash mismatch: {name}")
    rows = json.loads(args.coverage.read_text(encoding="utf-8"))["month_ends"]
    sessions = (
        mcal.get_calendar("NYSE")
        .valid_days(pd.Timestamp(rows[0]["date"]) - pd.Timedelta(days=550), rows[-1]["date"])
        .tz_localize(None)
    )
    bars = pd.read_parquet(args.tape, columns=["timestamp", "symbol"])
    result = {
        "purpose": "Session dates only; not prices, identity or point-in-time verification",
        "tape_sha256": args.tape_sha256,
        "membership_sha256": args.membership_sha256,
        "coverage_sha256": args.coverage_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "calendar_version": version("pandas-market-calendars"),
        "window_sessions": 252,
        "month_ends": measure(
            rows, bars, load_sp500_history(args.membership), sessions, window=252
        ),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
