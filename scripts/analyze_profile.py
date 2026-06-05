"""Summarize a cProfile .prof file for the WFO backtest profiling report.

Prints the top functions by cumulative and total (self) time, then pulls out the
specific functions of interest for the vectorization review (entry/exit signal
generation, the fixed_sl_tp Python loop, the numba stop kernels, the vbt portfolio
simulation, and data loading). Optionally emits a markdown table for the report.

Usage:
    python scripts/analyze_profile.py results/profiling/wfo_<ts>.prof [--top 30] \
        [--markdown results/profiling/wfo_<ts>.md]
"""

from __future__ import annotations

import argparse
import io
import pstats

# Substrings identifying the functions we care about for the vectorization review.
# Keyed label -> substring matched against "filename:lineno(funcname)".
FUNCTIONS_OF_INTEREST = {
    # --- the real bottleneck (per-fold train-metric accessors); see
    #     docs/profiling_report_2026-06-05.md. Watch these to verify any metric-path fix. ---
    "metric: get_returns_acc": "get_returns_acc",
    "metric: returns": "base.py:4601(returns)",
    "metric: value": "base.py:4514(value)",
    "metric: _train_metric_series": "_train_metric_series",
    "vbt config.__init__": "config.py:332(__init__)",
    "vbt array_wrapper.__init__": "array_wrapper.py:122(__init__)",
    # --- simulation + signal generation (the original prompt's targets) ---
    "compute_exits": "compute_exits",
    "compute_entries": "compute_entries",
    "fixed_sl_tp loop": "strategies.py:1183",  # FixedSLTPExit.compute_exits (Python loop)
    "atr_trailing numba": "_atr_trailing_stop_long_ohlc_touch_2d_numba",
    "trailing numba": "_trailing_stop_long_ohlc_touch_2d_numba",
    "Portfolio.from_signals": "from_signals",
    "data load": "load_data_and_setup",
    "ccxt fetch_ohlcv": "fetch_ohlcv",
}


def _rows_from_stats(stats: pstats.Stats):
    """Yield (cumtime, tottime, ncalls, label) for every profiled function."""
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        filename, lineno, funcname = func
        label = f"{filename.split('/')[-1]}:{lineno}({funcname})"
        yield ct, tt, nc, label


def _print_top(stats: pstats.Stats, sort_key: str, top: int) -> None:
    buf = io.StringIO()
    stats.stream = buf  # type: ignore[attr-defined]
    stats.sort_stats(sort_key).print_stats(top)
    print(buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", help="Path to a cProfile .prof file")
    ap.add_argument("--top", type=int, default=30, help="Top-N functions to show (default 30)")
    ap.add_argument("--markdown", default=None, help="Optional path to write a markdown summary")
    args = ap.parse_args()

    stats = pstats.Stats(args.profile)
    total_time = stats.total_tt  # type: ignore[attr-defined]
    print(f"\n=== Profile: {args.profile} ===")
    print(f"Total runtime (profiled): {total_time:.2f}s\n")

    print(f"--- Top {args.top} by CUMULATIVE time ---")
    _print_top(stats, "cumulative", args.top)

    print(f"--- Top {args.top} by TOTAL (self) time ---")
    _print_top(stats, "tottime", args.top)

    # Functions of interest: aggregate by substring match.
    rows = list(_rows_from_stats(stats))
    print("--- Functions of interest (by cumulative time, % of total) ---")
    md_lines = [
        "| Function | cum (s) | % total | self (s) | calls |",
        "|---|---|---|---|---|",
    ]
    for label_key, needle in FUNCTIONS_OF_INTEREST.items():
        matched = [r for r in rows if needle in r[3]]
        if not matched:
            continue
        cum = sum(r[0] for r in matched)
        self_t = sum(r[1] for r in matched)
        calls = sum(r[2] for r in matched)
        pct = (cum / total_time * 100.0) if total_time else 0.0
        line = f"  {label_key:22s} cum={cum:8.2f}s ({pct:5.1f}%) self={self_t:8.2f}s calls={calls}"
        print(line)
        md_lines.append(f"| {label_key} | {cum:.2f} | {pct:.1f}% | {self_t:.2f} | {calls} |")

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(f"# Profile summary: {args.profile}\n\n")
            f.write(f"Total profiled runtime: **{total_time:.2f}s**\n\n")
            f.write("## Functions of interest\n\n")
            f.write("\n".join(md_lines) + "\n")
        print(f"\nWrote markdown summary to {args.markdown}")


if __name__ == "__main__":
    main()
