"""Full pipeline: Per-Coin Multi-Strategy WFO → Final Validation → Report (optional Phase 1 sensitivity)."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY
from ggTrader.pipeline.param_grids import (
    COARSE_ENTRY_PARAM_GRIDS,
    DETAILED_ENTRY_PARAM_GRIDS,
    DETAILED_EXIT_AXIS_GRIDS,
    EXIT_AXIS_GRIDS,
    build_wfo_superset_grids,
)
from ggTrader.utils.pipeline_phases import (
    phase_1_sensitivity_analysis,
    phase_2_per_coin_multi_strategy_wfo,
    prepare_config_and_symbols,
    run_recent_validation_window,
)
from ggTrader.utils.pipeline_status_logger import StatusLogger
from ggTrader.utils.pipeline_run_history import append_automated_run_section
from ggTrader.utils.report_generator import generate_pipeline_report
from ggTrader.utils.run_config import full_pipeline_config

CONSTANTS = full_pipeline_config()


def _cli_summary(args: argparse.Namespace) -> str:
    """Compact flag string for run history logging."""
    parts: list[str] = []
    if args.no_progress:
        parts.append("--no-progress")
    if args.no_save:
        parts.append("--no-save")
    if args.dry_run:
        parts.append("--dry-run")
    if args.max_symbols is not None:
        parts.append(f"--max-symbols {args.max_symbols}")
    if args.symbols_file:
        parts.append(f"--symbols-file {args.symbols_file}")
    if args.detailed_sensitivity:
        parts.append("--detailed-sensitivity")
    if args.sensitivity:
        parts.append("--sensitivity")
    if getattr(args, "wfo_debug_metrics", False):
        parts.append("--wfo-debug-metrics")
    if getattr(args, "train_metric", None):
        parts.append(f"--train-metric {args.train_metric}")
    if getattr(args, "exits", None):
        parts.append(f"--exits {args.exits}")
    if getattr(args, "dual_exits", False):
        parts.append("--dual-exits")
    if getattr(args, "recent_validation_start", None):
        parts.append(f"--recent-validation-start {args.recent_validation_start}")
    if getattr(args, "recent_validation_end", None):
        parts.append(f"--recent-validation-end {args.recent_validation_end}")
    if getattr(args, "recent_validation_ccxt_tail", False):
        parts.append("--recent-validation-ccxt-tail")
    return " ".join(parts) if parts else "(defaults)"


def main() -> None:
    """Orchestrate the full 4-phase pipeline."""
    parser = argparse.ArgumentParser(description="Run Full Trading Strategy Pipeline")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (recommended for background runs)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 3 coins, 1 chunk, 2 folds")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        metavar="N",
        help="Override MAX_SYMBOLS (default: use CONSTANTS, currently 5 for debug)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override SYMBOLS_FILE JSON (rank list). Use e.g. data/top_50_*.json when "
            "--max-symbols exceeds rows in the default top_25 file."
        ),
    )
    parser.add_argument(
        "--detailed-sensitivity",
        action="store_true",
        help=(
            "Use DETAILED_SENSITIVITY_PARAM_GRIDS for WFO (and for Phase 1 if --sensitivity); "
            "default is coarse grids"
        ),
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help=(
            "Run Phase 1 sensitivity screen + narrow grids for WFO; default skips Phase 1 "
            "and WFO uses the active book unchanged"
        ),
    )
    parser.add_argument(
        "--wfo-debug-metrics",
        action="store_true",
        help=(
            "Set WFO_DEBUG_METRICS in pipeline config: log per-fold train-metric len/finite "
            "counts and combined robustness stats during multi-strategy WFO (verbose)"
        ),
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        default=None,
        choices=("sharpe", "sortino", "calmar", "composite"),
        metavar="NAME",
        help=(
            "Override TRAIN_METRIC for WFO (and Phase 1 if --sensitivity): "
            "sharpe | sortino | calmar | composite (default: from full_pipeline_config)"
        ),
    )
    parser.add_argument(
        "--exits",
        type=str,
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated EXIT_TOURNAMENT override, e.g. atr_trailing or fixed_sl_tp or "
            "atr_trailing,fixed_sl_tp (default: from full_pipeline_config)"
        ),
    )
    parser.add_argument(
        "--dual-exits",
        action="store_true",
        help="Use both atr_trailing and fixed_sl_tp in EXIT_TOURNAMENT (ignored if --exits is set)",
    )
    parser.add_argument(
        "--recent-validation-start",
        type=str,
        default=None,
        metavar="DATE",
        help="YYYY-MM-DD: after WFO, run frozen-params combined backtest on this window (Phase 3B)",
    )
    parser.add_argument(
        "--recent-validation-end",
        type=str,
        default=None,
        metavar="DATE",
        help="YYYY-MM-DD end for recent validation (default: now UTC)",
    )
    parser.add_argument(
        "--recent-validation-ccxt-tail",
        action="store_true",
        help="After last TimescaleDB bar, append Kraken OHLCV via CCXT through validation end",
    )
    args = parser.parse_args()

    entry_book = (
        DETAILED_ENTRY_PARAM_GRIDS if args.detailed_sensitivity else COARSE_ENTRY_PARAM_GRIDS
    )
    exit_book = DETAILED_EXIT_AXIS_GRIDS if args.detailed_sensitivity else EXIT_AXIS_GRIDS

    pipeline_config = dict(CONSTANTS)
    if args.max_symbols is not None:
        pipeline_config["MAX_SYMBOLS"] = args.max_symbols
    if args.symbols_file is not None:
        pipeline_config["SYMBOLS_FILE"] = args.symbols_file
    if args.wfo_debug_metrics:
        pipeline_config["WFO_DEBUG_METRICS"] = True
    if args.train_metric is not None:
        pipeline_config["TRAIN_METRIC"] = args.train_metric
    if args.recent_validation_start is not None:
        pipeline_config["RECENT_VALIDATION_START_DATE"] = args.recent_validation_start.strip()
    if args.recent_validation_end is not None:
        pipeline_config["RECENT_VALIDATION_END_DATE"] = args.recent_validation_end.strip()
    if args.recent_validation_ccxt_tail:
        pipeline_config["RECENT_VALIDATION_USE_CCXT_TAIL"] = True

    config = prepare_config_and_symbols(pipeline_config)
    print(f"TRAIN_METRIC={config.get('TRAIN_METRIC', 'sharpe')!r}")

    if args.dual_exits and args.exits is None:
        config["EXIT_TOURNAMENT"] = ["atr_trailing", "fixed_sl_tp"]

    if args.exits is not None:
        names = [x.strip() for x in args.exits.split(",") if x.strip()]
        invalid = [x for x in names if x not in EXIT_REGISTRY]
        if invalid:
            parser.error(
                f"Unknown exit name(s) {invalid!r}. Valid: {list(EXIT_REGISTRY.keys())}"
            )
        if not names:
            parser.error("--exits must list at least one exit name.")
        config["EXIT_TOURNAMENT"] = names

    exit_tournament: list[str] = config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys()))

    if args.dry_run:
        print("Running in DRY-RUN mode: 3 coins, 1 chunk, 2 folds")
        config["SYMBOLS"] = config["SYMBOLS"][:3]
        config["MAX_SYMBOLS"] = 3
        config["CHUNK_SIZE"] = 50
        config["N_SPLITS"] = 2

    save_results = not args.no_save

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_results_dir = Path(f"results/pipeline_{timestamp}")
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline results will be saved to: {pipeline_results_dir}")

    show_progress = not args.no_progress and sys.stdout.isatty()

    status_path = pipeline_results_dir / "status.txt"
    logger = StatusLogger(status_path)
    try:
        status_rel = status_path.relative_to(Path.cwd())
    except ValueError:
        status_rel = status_path
    print("Monitor progress (second terminal, repo root):")
    print("  ggtrader-pipeline-status --watch --interval 10")
    print("  python scripts/pipeline_status.py --watch --interval 10")
    print(f"  PowerShell tail: Get-Content -Path '{status_rel}' -Wait -Tail 40")
    print()

    n_coins = len(config["SYMBOLS"])
    n_strategies = len(ENTRY_REGISTRY)
    n_folds = config.get("N_SPLITS", 4)
    logger.update(
        f"Starting pipeline: {n_coins} coins, {n_strategies} strategies, {n_folds} WFO folds, "
        f"exits={exit_tournament}"
    )

    narrowed_grids: dict = {}
    sensitivity_results: dict = {}
    wfo_results: dict = {}

    try:
        if args.sensitivity:
            sensitivity_results = phase_1_sensitivity_analysis(
                config,
                narrowed_grids,
                show_progress,
                args.dry_run,
                logger,
                entry_book=entry_book,
                exit_book=exit_book,
                exit_tournament=exit_tournament,
                save_results=save_results,
            )
        else:
            narrowed_grids.update(
                build_wfo_superset_grids(entry_book, exit_book, exit_tournament, args.dry_run)
            )
            sensitivity_results = {}
            print("\n" + "=" * 100)
            print("PHASE 1: OMITTED (default — pass --sensitivity to run coarse/detailed screen)")
            print("=" * 100)
            print(
                "WFO will search the active parameter book as-is (no analyze_sensitivity_results "
                "narrowing). Edit grids in src/ggTrader/pipeline/param_grids.py to change ranges."
            )
            logger.update(
                "Phase 1 omitted (default): WFO grids = active book "
                f"(coarse or --detailed-sensitivity) | exits={exit_tournament}"
            )
            logger.phase_done("Phase 1 (sensitivity omitted)")

        wfo_results = phase_2_per_coin_multi_strategy_wfo(
            config, narrowed_grids, show_progress, logger, save_results=save_results
        )

        run_recent_validation_window(config, wfo_results, logger)

        final_backtest_results = {
            "final_portfolio": wfo_results.get("final_portfolio"),
            "per_coin_results": wfo_results.get("per_coin_results", {}),
            "per_coin_final_stats": wfo_results.get("per_coin_final_stats", {}),
            "final_stats": wfo_results.get("final_stats", {}),
            "recent_validation_stats": wfo_results.get("recent_validation_stats"),
            "recent_validation_per_coin_final_stats": wfo_results.get(
                "recent_validation_per_coin_final_stats", {}
            ),
        }

        print("\n" + "=" * 100)
        print("PHASE 4: REPORT GENERATION")
        print("=" * 100)

        generate_pipeline_report(
            sensitivity_results=sensitivity_results,
            wfo_results=wfo_results,
            final_backtest_results=final_backtest_results,
            output_dir=str(pipeline_results_dir),
        )

        append_automated_run_section(
            run_folder_name=pipeline_results_dir.name,
            final_stats=final_backtest_results.get("final_stats", {}),
            config=config,
            cli_summary=_cli_summary(args),
        )

        total_elapsed = time.time() - logger.pipeline_start
        logger.update(
            f"Pipeline COMPLETE in {str(timedelta(seconds=int(total_elapsed)))} "
            f"| Results: {pipeline_results_dir}"
        )
        print(f"\n>>> Pipeline complete! Results saved to: {pipeline_results_dir}")
        print("  - pipeline_report.md: Comprehensive analysis report")
        print("  - status.txt: Full timestamped execution log")
        if args.sensitivity:
            print("  - sensitivity_*.csv: Parameter sensitivity analysis (Phase 1)")
        print("  - wfo_*.csv: Walk-forward optimization results")

    except Exception as e:
        tb = traceback.format_exc()
        logger.update(f"FAILED: {e}")
        with open(status_path, "a", encoding="utf-8") as f:
            f.write("\n--- Traceback ---\n")
            f.write(tb)
            f.write("\n")
        print(f"\nX Pipeline failed: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
