"""Compare portfolios over the SPY worst-drawdown episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def matched_episode(curves: dict[str, pd.Series]) -> dict:
    """Use SPY's worst peak and trough and track recovery to each starting value."""
    spy = curves["spy"]
    trough = int(np.argmin((spy / spy.cummax() - 1).to_numpy()))
    peak = int(np.argmax(spy.iloc[: trough + 1].to_numpy()))
    result = {}
    for name, curve in curves.items():
        start = float(curve.iloc[peak])
        low = peak + int(np.argmin(curve.iloc[peak : trough + 1].to_numpy()))
        hits = np.flatnonzero(curve.iloc[low:].to_numpy() >= start)
        end = low + int(hits[0]) if len(hits) else len(curve) - 1
        result[name] = {
            "reference_peak": str(spy.index[peak].date()),
            "reference_trough": str(spy.index[trough].date()),
            "return_at_spy_trough": float(curve.iloc[trough] / start - 1),
            "lowest_return_during_episode": float(curve.iloc[low] / start - 1),
            "recovered": bool(len(hits)),
            "recovery_or_observed_end": str(curve.index[end].date()),
            "sessions": end - peak,
        }
    return result


def main() -> None:
    import descriptive_protection as original

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("original_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pinned = json.loads(args.original_result.read_text())
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != pinned["driver_sha256"]:
        raise ValueError("Original driver does not match prior result")
    series = {}
    for symbol in original.SYMBOLS:
        path = args.input_dir / f"{symbol}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != pinned["input_sha256"][symbol]:
            raise ValueError(f"Input hash mismatch: {symbol}")
        frame = pd.read_csv(path)
        dates = (
            pd.to_datetime(frame["date"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        series[symbol] = pd.Series(frame["Adj Close"].to_numpy(), index=dates)
    prices = pd.DataFrame(series).loc["2021-04-30":"2026-04-30"]
    if prices.isna().any().any() or (prices <= 0).any().any() or prices.index.has_duplicates:
        raise ValueError("Missing, invalid or duplicate adjusted prices")
    curves = {name: original.portfolio(prices, w, 0.0001) for name, w in original.WEIGHTS.items()}
    result = {
        "status": "descriptive; SPY episode identified after seeing historical data",
        "original_result_sha256": hashlib.sha256(args.original_result.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episode": matched_episode(curves),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
