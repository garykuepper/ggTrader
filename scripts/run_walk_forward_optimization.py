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
)
from ggTrader.utils.pipeline_status_logger import StatusLogger
from ggTrader.utils.run_config import merge_run_config, wfo_script_config


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
    args = parser.parse_args()

    # We use the full pipeline config to ensure we mimic its splits and metrics
    from ggTrader.utils.run_config import full_pipeline_config
    config = merge_run_config(
        full_pipeline_config(), 
        SYMBOLS_FILE=args.symbols_file,
        MAX_SYMBOLS=args.max_symbols,
        TRAIN_METRIC=args.train_metric
    )
    show_progress = not args.no_progress and sys.stdout.isatty()

    # Build the exact same grids as the full pipeline does (Iteration 6: Expanded Discovery)
    narrowed_grids = build_wfo_superset_grids(
        DETAILED_ENTRY_PARAM_GRIDS,
        DETAILED_EXIT_AXIS_GRIDS,
        list(DETAILED_EXIT_AXIS_GRIDS.keys()),
        dry_run=False
    )

    logger = StatusLogger(Path("results/wfo_isolated_status.txt"))
    logger.update("Starting isolated multi-strategy WFO (Phase 1)...")

    # Delegate to the single source of truth for Phase 1 WFO
    wfo_results = phase_1_per_coin_multi_strategy_wfo(
        config=config,
        narrowed_grids=narrowed_grids,
        show_progress=show_progress,
        logger=logger,
        save_results=True,
    )
    
    print("\nIsolated WFO execution completed successfully.")
    
    if "final_portfolio" in wfo_results:
        pf = wfo_results["final_portfolio"]
        print(f"\nFinal Dynamic Walk-Forward Portfolio Return (Across concatenated folds): {pf.total_return().mean() * 100:.2f}%\n")

    # Run the user-requested static backtest on the single set of top robust parameters over the entire time range (Phase 2)
    phase_2_full_data_validation(config, wfo_results, logger=logger)

if __name__ == "__main__":
    main()
