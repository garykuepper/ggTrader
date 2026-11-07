"""P3 evidence: vectorised PIT mask versus ``lab.data.eligible_at`` on the frozen tape.

Samples bars (every constituent-change date in the tape with its neighbours, plus random
bars), compares the symbol set of ``eligible_at(d, ohlcv.loc[:d])`` with ``pit_mask`` row ``d``
for four configurations, and records mismatch counts, sampled dates, build time and memory in
JSON. Also records the OD6/OD7 policy statistics, per-year member-day coverage and the
constituent-file age. Read-only: the snapshot is the only input; no database, no network.

Usage:
    scripts/p3_pit_parity.py [--out PATH] [--n-random 160] [--seed 7] [--snapshot PATH]
Exit status is 1 if any mismatch is found (the examples are in the JSON and printed).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Scratch by default; pass --out research/programme/vectorbt-engine/p3-evidence/... to commit.
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ggtrader_evidence"

EVAL_SPAN = ("2021-01-31", "2026-04-30")
BAR_WINDOW = ("2019-03-01", "2026-10-30")
EXTRAS = ("SPY", "TLT", "GLD", "PDBC", "BIL")
CONFIGS: dict[str, dict[str, Any]] = {
    "min_history_400": {"min_history_bars": 400},
    "min_history_60": {"min_history_bars": 60},
    "min_history_60_cap_50": {"min_history_bars": 60, "max_stocks": 50},
    "min_history_60_cap_off": {"min_history_bars": 60, "max_stocks": None},
}
BUILD_REPEATS = 5


def sample_dates(close: pd.DataFrame, history: pd.DataFrame, n_random: int, seed: int) -> list:
    """Bars at -1/0/+1 around every constituent change inside the tape, plus random bars."""
    changes = [d for d in history.index if close.index[0] <= d <= close.index[-1]]
    edge = {
        close.index[min(max(close.index.searchsorted(d) + k, 0), len(close) - 1)]
        for d in changes
        for k in (-1, 0, 1)
    }
    rng = np.random.default_rng(seed)
    rand = set(pd.DatetimeIndex(rng.choice(close.index[1:], size=n_random, replace=False)))
    return sorted(edge | rand)


def measure_build(close: pd.DataFrame, history: pd.DataFrame, kw: dict[str, Any]) -> dict[str, Any]:
    """Best-of-N wall time and tracemalloc peak for one mask build."""
    from ggTrader.lab.universe import pit_mask

    times = []
    for _ in range(BUILD_REPEATS):
        t0 = time.perf_counter()
        mask = pit_mask(close, history, on_stale="ignore", **kw)
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    mask = pit_mask(close, history, on_stale="ignore", **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "build_seconds_best": round(min(times), 4),
        "build_seconds_all": [round(t, 4) for t in times],
        "tracemalloc_peak_mb": round(peak / 1e6, 2),
        "mask_mb": round(mask.to_numpy().nbytes / 1e6, 3),
        "mask_shape": list(mask.shape),
    }


def run_config(
    name: str,
    kw: dict[str, Any],
    close: pd.DataFrame,
    ohlcv: pd.DataFrame,
    history: pd.DataFrame,
    dates: list,
) -> dict[str, Any]:
    """Compare one config on every sampled date; collects every mismatch."""
    from ggTrader.lab.data import eligible_at
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.universe import pit_mask

    mask = pit_mask(close, history, on_stale="ignore", **kw)
    cfg = LabConfig(min_history_bars=kw["min_history_bars"], max_stocks=kw.get("max_stocks"))
    mismatches: list[dict[str, Any]] = []
    total_eligible = 0
    t0 = time.perf_counter()
    for d in dates:
        expected, _ = eligible_at(d, ohlcv.loc[:d], cfg)
        got = set(mask.columns[mask.loc[d].to_numpy()])
        diff = got ^ set(expected)
        total_eligible += len(expected)
        if diff:
            mismatches.append({"date": str(d.date()), "symbols": sorted(diff)[:10], "n": len(diff)})
    loop_seconds = time.perf_counter() - t0
    out = {
        "config": kw,
        "dates_checked": len(dates),
        "symbol_mismatches": sum(m["n"] for m in mismatches),
        "dates_with_mismatch": len(mismatches),
        "first_mismatches": mismatches[:5],
        "mean_eligible": round(total_eligible / len(dates), 1),
        "eligible_at_loop_seconds": round(loop_seconds, 1),
        "eligible_at_seconds_per_date": round(loop_seconds / len(dates), 4),
        **measure_build(close, history, kw),
    }
    print(
        f"{name}: {len(dates)} dates, mismatches={out['symbol_mismatches']}, "
        f"build {out['build_seconds_best']}s vs loop {out['eligible_at_loop_seconds']}s",
        flush=True,
    )
    return out


def policy_stats(bars: Any, history: pd.DataFrame) -> dict[str, Any]:
    """OD6/OD7 event counts, coverage by year and constituent-file age on the real tape."""
    from ggTrader.lab import universe as uni

    close = bars.close
    member = uni.membership_mask(close.index, list(close.columns), history)
    removal = uni.removal_exits(member)
    terminal = uni.terminal_exits(close)
    full_cols = list(bars.requested)
    member_full = uni.membership_mask(close.index, full_cols, history)
    return {
        "removal_events": int(removal.to_numpy().sum()),
        "symbols_that_left": int(removal.any().sum()),
        "terminal_exit": {k: v for k, v in terminal.meta.items() if k != "dead_symbols"},
        "terminal_dead_symbols": terminal.meta["dead_symbols"],
        "coverage_by_year": uni.coverage_by_year(member_full, close),
        "constituent_age": uni.check_constituent_age(close.index[-1], history, on_stale="ignore"),
        "bars_summary": bars.summary(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the parity study and write the evidence JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "p3_pit_parity.json")
    ap.add_argument("--n-random", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--snapshot", type=Path, default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ggTrader.data.bars import load_bars
    from ggTrader.data.core.index_constituents import load_sp500_history

    bars = load_bars(
        universe="sp500",
        universe_span=EVAL_SPAN,
        start=BAR_WINDOW[0],
        end=BAR_WINDOW[1],
        extra_symbols=EXTRAS,
        snapshot_path=args.snapshot,
        deep_verify=True,
    )
    close = bars.close
    history = load_sp500_history()
    ohlcv = pd.concat({s: close[[s]].rename(columns={s: "close"}) for s in close.columns}, axis=1)
    dates = sample_dates(close, history, args.n_random, args.seed)
    print(f"tape {close.shape}; sampled {len(dates)} dates", flush=True)

    result: dict[str, Any] = {
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "snapshot": bars.provenance["snapshot"],
        "tape_shape": list(close.shape),
        "sampled_dates": [str(d.date()) for d in dates],
        "n_sampled": len(dates),
        "seed": args.seed,
        "configs": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, kw in CONFIGS.items():
        try:
            result["configs"][name] = run_config(name, kw, close, ohlcv, history, dates)
        except Exception as exc:  # keep going; the JSON records the failure
            result["configs"][name] = {"error": repr(exc)}
            print(f"{name}: ERROR {exc!r}", flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    result["policy_stats"] = policy_stats(bars, history)
    result["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    total = sum(c.get("symbol_mismatches", 1) for c in result["configs"].values())
    result["total_symbol_mismatches"] = total
    args.out.write_text(json.dumps(result, indent=1))
    print(f"total mismatches: {total}; wrote {args.out}", flush=True)
    if total:
        for name, c in result["configs"].items():
            print(name, c.get("first_mismatches"), flush=True)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
