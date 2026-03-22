"""Re-run recent-window frozen-params combined backtest from a saved WFO ``run_results.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ggTrader.core.orchestrator import run_frozen_params_combined_backtest
from ggTrader.indicators.strategies import EXIT_REGISTRY
from ggTrader.pipeline.exit_tournament import parse_exit_tournament
from ggTrader.utils.setup import load_hybrid_validation_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load per_coin_results from run_wfo_per_coin_multi_strategy/run_results.json "
            "and replay the combined portfolio on a recent OHLCV window (no WFO)."
        )
    )
    parser.add_argument(
        "--run-results",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to run_results.json (under results/run_wfo_per_coin_multi_strategy_*/)",
    )
    parser.add_argument(
        "--validation-start",
        type=str,
        required=True,
        metavar="DATE",
        help="YYYY-MM-DD (UTC)",
    )
    parser.add_argument(
        "--validation-end",
        type=str,
        default=None,
        metavar="DATE",
        help="YYYY-MM-DD (UTC); default: now",
    )
    parser.add_argument(
        "--ccxt-tail",
        action="store_true",
        help="Append CCXT OHLCV after the last TimescaleDB bar",
    )
    args = parser.parse_args()

    path = Path(args.run_results)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    meta = doc.get("configuration", {}).get("_raw_config") or doc.get("metadata")
    if not meta or "per_coin_results" not in meta:
        print(
            "run_results.json must contain configuration._raw_config.per_coin_results",
            file=sys.stderr,
        )
        sys.exit(1)

    per_coin = meta["per_coin_results"]
    config = {
        k: v
        for k, v in meta.items()
        if k not in ("per_coin_results", "exit_tournament")
    }
    if "SYMBOLS" not in config and doc.get("configuration", {}).get("symbols"):
        config["SYMBOLS"] = doc["configuration"]["symbols"]

    start = pd.to_datetime(args.validation_start).tz_localize("UTC")
    if args.validation_end:
        end = pd.to_datetime(args.validation_end).tz_localize("UTC")
    else:
        end = pd.Timestamp.now(tz="UTC")

    ohlcv = load_hybrid_validation_ohlcv(
        config,
        start,
        end,
        use_ccxt_tail=args.ccxt_tail,
    )
    exit_tournament = parse_exit_tournament(
        config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY,
    )
    out = run_frozen_params_combined_backtest(
        ohlcv,
        per_coin,
        config,
        exit_tournament=exit_tournament,
        save_results=False,
        phase_title="STANDALONE RECENT VALIDATION",
        combined_portfolio_label="Recent window - combined portfolio (frozen params)",
    )
    fs = out["final_stats"]
    print("\nSummary:")
    print(f"  Period: {fs.get('backtest_start')} -> {fs.get('backtest_end')}")
    print(f"  Return %: {fs.get('profit_pct', 0):.2f}")
    print(f"  CAGR %: {fs.get('cagr_pct')}")
    print(f"  Max DD %: {fs.get('max_drawdown', 0):.2f}")
    print(f"  Benchmark CAGR %: {fs.get('benchmark_cagr_pct')}")


if __name__ == "__main__":
    main()
