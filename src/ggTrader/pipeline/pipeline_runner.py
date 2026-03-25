"""Orchestrator to sequence the 4 phases of the automated trading pipeline."""

import sys
import time
import traceback
from argparse import Namespace
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
    phase_0_sensitivity_analysis,
    phase_1_per_coin_multi_strategy_wfo,
    phase_2_full_data_validation,
    phase_3_recent_performance,
    prepare_config_and_symbols,
)
from ggTrader.utils.pipeline_run_history import append_automated_run_section
from ggTrader.utils.pipeline_status_logger import StatusLogger
from ggTrader.utils.report_generator import generate_pipeline_report


def execute_full_pipeline(args: Namespace, base_constants: dict) -> None:
    """Executes the full sequence of automated pipeline phases based on CLI arguments."""
    entry_book = (
        DETAILED_ENTRY_PARAM_GRIDS if args.detailed_sensitivity else COARSE_ENTRY_PARAM_GRIDS
    )
    exit_book = DETAILED_EXIT_AXIS_GRIDS if args.detailed_sensitivity else EXIT_AXIS_GRIDS

    pipeline_config = dict(base_constants)
    if getattr(args, "max_symbols", None) is not None:
        pipeline_config["MAX_SYMBOLS"] = args.max_symbols
    if getattr(args, "symbols_file", None) is not None:
        pipeline_config["SYMBOLS_FILE"] = args.symbols_file
    if getattr(args, "wfo_debug_metrics", False):
        pipeline_config["WFO_DEBUG_METRICS"] = True
    if getattr(args, "train_metric", None) is not None:
        pipeline_config["TRAIN_METRIC"] = args.train_metric
    if getattr(args, "recent_validation_start", None) is not None:
        pipeline_config["RECENT_VALIDATION_START_DATE"] = args.recent_validation_start.strip()
    if getattr(args, "recent_validation_end", None) is not None:
        pipeline_config["RECENT_VALIDATION_END_DATE"] = args.recent_validation_end.strip()
    if getattr(args, "recent_validation_ccxt_tail", False):
        pipeline_config["RECENT_VALIDATION_USE_CCXT_TAIL"] = True

    config = prepare_config_and_symbols(pipeline_config)
    print(f"TRAIN_METRIC={config.get('TRAIN_METRIC', 'sharpe')!r}")

    if getattr(args, "dual_exits", False) and getattr(args, "exits", None) is None:
        config["EXIT_TOURNAMENT"] = ["atr_trailing", "fixed_sl_tp"]

    if getattr(args, "exits", None) is not None:
        names = [x.strip() for x in args.exits.split(",") if x.strip()]
        invalid = [x for x in names if x not in EXIT_REGISTRY]
        if invalid:
            raise ValueError(
                f"Unknown exit name(s) {invalid!r}. Valid: {list(EXIT_REGISTRY.keys())}"
            )
        if not names:
            raise ValueError("--exits must list at least one exit name.")
        config["EXIT_TOURNAMENT"] = names

    exit_tournament: list[str] = config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys()))

    if getattr(args, "dry_run", False):
        print("Running in DRY-RUN mode: 3 coins, 1 chunk, 2 folds")
        config["SYMBOLS"] = config["SYMBOLS"][:3]
        config["MAX_SYMBOLS"] = 3
        config["CHUNK_SIZE"] = 50
        config["N_SPLITS"] = 2
        # Disable robustness gate in dry-run — 2 artificial folds always fail the gate.
        config["MIN_ROBUSTNESS_SCORE"] = None

    save_results = not getattr(args, "no_save", False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_results_dir = Path(f"results/pipeline_{timestamp}")
    if save_results:
        pipeline_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Pipeline results will be saved to: {pipeline_results_dir}")

    show_progress = not getattr(args, "no_progress", False) and sys.stdout.isatty()

    status_path = pipeline_results_dir / "status.txt"
    logger = StatusLogger(status_path)
    try:
        status_rel = status_path.relative_to(Path.cwd())
    except ValueError:
        status_rel = status_path

    if save_results:
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
        if getattr(args, "sensitivity", False):
            sensitivity_results = phase_0_sensitivity_analysis(
                config,
                narrowed_grids,
                show_progress,
                getattr(args, "dry_run", False),
                logger,
                entry_book=entry_book,
                exit_book=exit_book,
                exit_tournament=exit_tournament,
                save_results=save_results,
            )
        else:
            narrowed_grids.update(
                build_wfo_superset_grids(
                    entry_book, exit_book, exit_tournament, getattr(args, "dry_run", False)
                )
            )
            sensitivity_results = {}
            print("\n" + "=" * 100)
            print("PHASE 0: OMITTED (default — pass --sensitivity to run coarse/detailed screen)")
            print("=" * 100)
            logger.update("Phase 0 omitted (default): WFO grids = active book.")
            logger.phase_done("Phase 0 (sensitivity omitted)")

        wfo_results = phase_1_per_coin_multi_strategy_wfo(
            config, narrowed_grids, show_progress, logger, save_results=save_results
        )

        phase_2_full_data_validation(config, wfo_results, logger)
        phase_3_recent_performance(config, wfo_results, logger)

        if save_results:
            final_backtest_results = {
                "final_portfolio": wfo_results.get("final_portfolio"),
                "per_coin_results": wfo_results.get("per_coin_results", {}),
                "per_coin_final_stats": wfo_results.get("per_coin_final_stats", {}),
                "final_stats": wfo_results.get("final_stats", {}),
                "phase_2_stats": wfo_results.get("phase_2_stats"),
                "phase_2_per_coin_final_stats": wfo_results.get("phase_2_per_coin_final_stats", {}),
                "phase_3_stats": wfo_results.get("phase_3_stats"),
                "phase_3_per_coin_final_stats": wfo_results.get("phase_3_per_coin_final_stats", {}),
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

            # Build a simple summary representation for the CLI arg string logger:
            args_dict = vars(args)
            cli_summary_arr = [f"--{k.replace('_', '-')} {v}" for k, v in args_dict.items() if v]
            cli_summary = " ".join(cli_summary_arr)

            append_automated_run_section(
                run_folder_name=pipeline_results_dir.name,
                final_stats=final_backtest_results.get("final_stats", {}),
                config=config,
                cli_summary=cli_summary,
            )

        total_elapsed = time.time() - logger.pipeline_start
        logger.update(
            f"Pipeline COMPLETE in {str(timedelta(seconds=int(total_elapsed)))} "
            f"| Results: {pipeline_results_dir}"
        )
        print(f"\n>>> Pipeline complete! Results saved to: {pipeline_results_dir}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.update(f"FAILED: {e}")
        if save_results:
            with open(status_path, "a", encoding="utf-8") as f:
                f.write("\n--- Traceback ---\n")
                f.write(tb)
                f.write("\n")
        print(f"\nX Pipeline failed: {e}")
        traceback.print_exc()
        raise
