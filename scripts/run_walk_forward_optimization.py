"""Run Walk-Forward Optimization (WFO) using VectorBT time-series CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ggTrader.pipeline.param_grids import (
    DETAILED_ENTRY_PARAM_GRIDS,
    DETAILED_EXIT_AXIS_GRIDS,
    build_wfo_superset_grids,
)
from ggTrader.utils.pipeline_phases import (
    phase_1_per_coin_multi_strategy_wfo,
    phase_2_full_data_validation,
    phase_3_recent_performance,
)
from ggTrader.utils.pipeline_status_logger import StatusLogger
from ggTrader.utils.run_config import merge_run_config


def main() -> None:
    """Run Walk-Forward Optimization using the pipeline's phase 1 orchestrator."""
    parser = argparse.ArgumentParser(description="Run Isolated WFO (Phase 1)")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Override SYMBOLS_FILE JSON",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols",
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        default=None,
        choices=("sharpe", "sortino", "calmar", "composite"),
        help="Override TRAIN_METRIC",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1 (WFO)",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run Phase 2 (Validation)",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Run Phase 3 (Recent Data)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to process (overrides file)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Training start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    # Default to all if none specified
    if not (args.phase1 or args.phase2 or args.phase3):
        args.phase1 = args.phase2 = args.phase3 = True

    # We use the full pipeline config to ensure we mimic its splits and metrics
    from ggTrader.utils.pipeline_phases import prepare_config_and_symbols
    from ggTrader.utils.run_config import full_pipeline_config

    config_overrides = {
        "SYMBOLS_FILE": args.symbols_file,
        "MAX_SYMBOLS": args.max_symbols,
        "TRAIN_METRIC": args.train_metric,
        "START_DATE": args.start_date,
        "END_DATE": args.end_date,
    }
    if args.symbols:
        config_overrides["SYMBOLS"] = args.symbols.split(",")

    config = merge_run_config(full_pipeline_config(), **config_overrides)

    # Resolve symbols list from JSON file (if SYMBOLS not already set)
    if not config.get("SYMBOLS"):
        config = prepare_config_and_symbols(config)

    show_progress = not args.no_progress and sys.stdout.isatty()

    logger = StatusLogger(Path("results/wfo_isolated_status.txt"))
    wfo_results = None

    if args.phase1:
        logger.update("Starting isolated multi-strategy WFO (Phase 1)...")
        # Build the exact same grids as the full pipeline does
        narrowed_grids = build_wfo_superset_grids(
            DETAILED_ENTRY_PARAM_GRIDS,
            DETAILED_EXIT_AXIS_GRIDS,
            list(DETAILED_EXIT_AXIS_GRIDS.keys()),
            dry_run=False,
        )
        wfo_results = phase_1_per_coin_multi_strategy_wfo(
            config=config,
            narrowed_grids=narrowed_grids,
            show_progress=show_progress,
            logger=logger,
            save_results=True,
        )

    if args.phase2:
        logger.update("Starting full data validation (Phase 2)...")

        # If we skipped Phase 1, but provided a symbols-file with parameters,
        # we construct a dummy wfo_results for Phase 2.
        if wfo_results is None and args.symbols_file:
            print(f"Loading pre-optimized parameters from {args.symbols_file} for Phase 2...")
            with open(args.symbols_file, "r") as f:
                import json

                params = json.load(f)
                # Phase 2 expects a dict where 'results' or the dict itself contains 'per_coin_results'
                # Actually, orchestrator's run_frozen_params expects a dict with symbol keys.
                wfo_results = {"per_coin_results": params}

        phase_2_full_data_validation(
            config=config,
            wfo_results=wfo_results,
            logger=logger,
        )

    if args.phase3:
        logger.update("Starting recent data validation (Phase 3)...")
        # Load params if needed (similar to Phase 2)
        if wfo_results is None and args.symbols_file:
            print(f"Loading pre-optimized parameters from {args.symbols_file} for Phase 3...")
            with open(args.symbols_file, "r") as f:
                import json

                params = json.load(f)
                wfo_results = {"per_coin_results": params}

        phase_3_recent_performance(
            config=config,
            wfo_results=wfo_results,
            logger=logger,
        )

    print("\nRequested WFO phases completed successfully.")

    if wfo_results and "final_portfolio" in wfo_results:
        pf = wfo_results["final_portfolio"]
        print(
            f"\nFinal Dynamic Walk-Forward Portfolio Return (Across concatenated folds): {pf.total_return().mean() * 100:.2f}%\n"
        )


if __name__ == "__main__":
    main()
