#!/usr/bin/env python3
"""S&P 500 rolling monthly walk-forward — the honest out-of-sample backtest.

Each month-end: re-select the top-N stocks (full entry x exit strategy
tournament, point-in-time universe, trailing 2-year window, data <= T only),
then trade the next month with the frozen book. Checkpointed and resumable.

Usage:
    source .venv/bin/activate

    # smoke test (~minutes): 20 stocks, 6 months, 2x1 strategies
    python -u scripts/sp500_monthly_walkforward.py --quick

    # verify no lookahead in the selection layer
    python -u scripts/sp500_monthly_walkforward.py --quick --leak-check

    # full run (unattended; resumable via checkpoints)
    nohup python -u scripts/sp500_monthly_walkforward.py --jobs 8 \
        > results/monthly_wf/full_run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY  # noqa: E402
from ggTrader.research.monthly_walkforward import (  # noqa: E402
    MonthlyHarnessConfig,
    leak_check,
    run_monthly_walkforward,
)


def _parse_strategies(value: str, registry: dict, kind: str) -> list[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    if names == ["all"]:
        return list(registry.keys())
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Unknown {kind} strategies: {unknown}. Available: {list(registry)}")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-start", default="2021-01-31")
    p.add_argument("--eval-end", default=None)
    p.add_argument("--lookback-bars", type=int, default=504)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-ratio", type=float, default=3.0)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--max-position-pct", type=float, default=0.02)
    p.add_argument("--entries", default="all")
    p.add_argument("--exits", default="all")
    p.add_argument("--grid", choices=["coarse", "detailed"], default="detailed")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--refit-every", type=int, default=1, help="Months between re-selections")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--quick", action="store_true", help="20 stocks, ~6 months, 2x1 strategies")
    p.add_argument("--leak-check", action="store_true", help="Verify selection has no lookahead")
    args = p.parse_args()

    entries = _parse_strategies(args.entries, ENTRY_REGISTRY, "entry")
    exits = _parse_strategies(args.exits, EXIT_REGISTRY, "exit")

    cfg = MonthlyHarnessConfig(
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        lookback_bars=args.lookback_bars,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        top_n=args.top_n,
        max_position_pct=args.max_position_pct,
        entries=entries,
        exits=exits,
        grid_book=args.grid,
        n_jobs=args.jobs,
        refit_every_n_months=args.refit_every,
        max_stocks=args.max_stocks,
        run_id=args.run_id or ("sp500_quick" if args.quick else "sp500_monthly"),
    )

    if args.quick:
        cfg.max_stocks = cfg.max_stocks or 20
        cfg.top_n = min(cfg.top_n, 10)
        cfg.n_splits = 4
        cfg.eval_start = args.eval_start if args.eval_start != "2021-01-31" else "2025-11-30"
        if args.entries == "all":
            cfg.entries = ["psar_adx", "ema_cross"]
        if args.exits == "all":
            cfg.exits = ["atr_trailing"]

    if args.leak_check:
        ok = leak_check(cfg)
        raise SystemExit(0 if ok else 1)

    summary = run_monthly_walkforward(cfg)
    print("\n" + "=" * 78)
    print("MONTHLY WALK-FORWARD SUMMARY (out-of-sample by construction)")
    print("=" * 78)
    print(json.dumps(summary["report"], indent=2))
    print(f"holding days: {summary['holding_days']}")
    print(f"avg monthly turnover: {summary['avg_monthly_turnover']}")
    print("combo selection counts (top 10):")
    for combo, n in list(summary["combo_selection_counts"].items())[:10]:
        print(f"  {combo:<40} {n}")


if __name__ == "__main__":
    main()
