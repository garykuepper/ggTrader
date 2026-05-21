#!/usr/bin/env python3
"""Compute WFO research scorecard for a given run directory.

Used to compare research outcomes step-to-step across the WFO overfitting
experiments. Reads per-worker JSON (Phase 1 selections) and phase_stats.json
(Phase 2/3 portfolio results).

Output: JSON with B-criterion primary metrics, secondary metrics, and
Phase 3 sanity numbers. Print to stdout; pipe to a file or jq as needed.

Usage:
    python scripts/scorecard_step1.py results/research/research_20260508_135840
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_per_coin_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load and merge per_coin_results from all worker JSONs in run_dir."""
    merged: dict[str, dict[str, Any]] = {}
    for worker_path in sorted(run_dir.glob("worker_*_results.json")):
        try:
            data = json.loads(worker_path.read_text())
        except Exception as exc:
            print(f"WARN: failed to read {worker_path.name}: {exc!r}", file=sys.stderr)
            continue
        per_coin = data.get("strategy_parameters", {}).get("per_coin", {}) or {}
        merged.update(per_coin)
    return merged


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def scorecard(run_dir: Path) -> dict[str, Any]:
    per_coin = load_per_coin_results(run_dir)
    rows = []
    for symbol, r in per_coin.items():
        is_score = safe_float(r.get("is_robustness_score"))
        oos_score = safe_float(r.get("oos_robustness_score"))
        cons = safe_float(r.get("fold_consistency"))
        gate = safe_float(r.get("robustness_score"))
        rows.append(
            {
                "symbol": symbol,
                "is": is_score,
                "oos": oos_score,
                "consistency": cons,
                "gate": gate,
            }
        )

    # B-criterion primary metrics
    oos_finite = [row["oos"] for row in rows if row["oos"] is not None]
    cons_finite = [row["consistency"] for row in rows if row["consistency"] is not None]
    n_oos_gt_30 = sum(1 for v in oos_finite if v > 0.30)
    n_oos_gt_0 = sum(1 for v in oos_finite if v > 0.0)
    median_cons = statistics.median(cons_finite) if cons_finite else None

    # Secondary metrics
    n_survivors = len(rows)
    gaps = [
        row["is"] - row["oos"] for row in rows if row["is"] is not None and row["oos"] is not None
    ]
    median_gap = statistics.median(gaps) if gaps else None

    # Phase 3 sanity from phase_stats.json (best-effort)
    phase_stats_path = run_dir / "phase_stats.json"
    phase3 = {}
    if phase_stats_path.exists():
        try:
            ps = json.loads(phase_stats_path.read_text())
            p3 = ps.get("phase_3_stats") or {}
            phase3 = {
                "total_return_pct": safe_float(
                    p3.get("total_return_pct") or p3.get("total_return") or p3.get("profit_pct")
                ),
                "cagr_pct": safe_float(p3.get("cagr_pct") or p3.get("cagr")),
                "sharpe": safe_float(p3.get("sharpe")),
                "max_drawdown_pct": safe_float(
                    p3.get("max_drawdown_pct") or p3.get("max_drawdown")
                ),
                "btc_cagr_pct": safe_float(p3.get("btc_cagr_pct") or p3.get("benchmark_cagr_pct")),
            }
        except Exception as exc:
            print(f"WARN: could not parse phase_stats.json: {exc!r}", file=sys.stderr)

    return {
        "run_dir": str(run_dir),
        "primary": {
            "n_oos_gt_0_30": n_oos_gt_30,
            "median_fold_consistency": median_cons,
            "n_oos_gt_0": n_oos_gt_0,
        },
        "secondary": {
            "n_survivors": n_survivors,
            "median_is_minus_oos_gap": median_gap,
        },
        "phase3_sanity": phase3,
        "per_coin": sorted(rows, key=lambda r: -(r["gate"] or 0)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    if not args.run_dir.is_dir():
        print(f"ERROR: not a directory: {args.run_dir}", file=sys.stderr)
        return 2
    print(json.dumps(scorecard(args.run_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
