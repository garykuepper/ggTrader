"""Compare engine.stats.window_stats with lab.metrics.curve_stats on real tape price paths.

Review MAJOR-1 follow-up. Each symbol's close is treated as an equity curve and scored on four
legacy-style windows both ways; the script reports the largest absolute difference per
statistic and how many windows are bit-identical. Writes JSON next to this file by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.data.bars import load_bars
from ggTrader.engine import stats
from ggTrader.engine.provenance import provenance
from ggTrader.lab.metrics import curve_stats

WINDOWS = [
    ("2022-01-31", "2023-01-31"),
    ("2023-01-31", "2023-04-30"),
    ("2024-02-01", "2025-02-01"),
    ("2021-01-31", "2026-04-29"),
]
STATS = ("sharpe", "volatility", "max_drawdown", "sortino", "cagr")


def reference(ref: dict[str, float]) -> dict[str, float]:
    """curve_stats output in the engine's units (fractions, not percent)."""
    return {
        "sharpe": ref["sharpe"],
        "volatility": ref["ann_vol_pct"] / 100,
        "max_drawdown": ref["max_drawdown_pct"] / 100,
        "sortino": ref["sortino"],
        "cagr": ref["cagr_pct"] / 100,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("window_check.json"))
    parser.add_argument("--min-bars", type=int, default=30, help="skip shorter windows")
    args = parser.parse_args(argv)

    bars = load_bars(
        universe="sp500",
        universe_span=("2021-01-31", "2026-04-30"),
        start="2019-03-01",
        end="2026-04-29",
    )
    close = bars.close.ffill()
    worst = dict.fromkeys(STATS, 0.0)
    identical = dict.fromkeys(STATS, 0)
    windows = 0
    for symbol in close.columns:
        equity = close[symbol].dropna()
        returns = equity.pct_change(fill_method=None).fillna(0.0)
        for lo, hi in WINDOWS:
            a = equity.index.searchsorted(pd.Timestamp(lo, tz="UTC"))
            b = equity.index.searchsorted(pd.Timestamp(hi, tz="UTC"), side="right")
            if b - a < args.min_bars or b > len(equity):
                continue
            ref = reference(curve_stats(equity.iloc[a:b]))
            got = stats.window_stats(returns, a, b).iloc[:, 0]
            windows += 1
            for key in STATS:
                diff = 0.0 if np.isnan(ref[key]) and np.isnan(got[key]) else abs(got[key] - ref[key])
                worst[key] = max(worst[key], float(diff))
                identical[key] += int(diff == 0.0)
    payload = {
        "symbols": close.shape[1],
        "windows_compared": windows,
        "max_abs_difference": worst,
        "windows_bit_identical": identical,
        "provenance": provenance(tape_sha256=bars.provenance["snapshot"]["sha256"]),
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "provenance"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
