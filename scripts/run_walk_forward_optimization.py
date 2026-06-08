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
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit directory to save results",
    )
    parser.add_argument(
        "--pipeline-stage",
        type=str,
        default="research",
        help="Pipeline stage string for folder routing",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Identifier for the worker process",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="Override EXCHANGE data venue (e.g. kraken, binanceus)",
    )
    args = parser.parse_args()

    # Default to all if none specified
    if not (args.phase1 or args.phase2 or args.phase3):
        args.phase1 = args.phase2 = args.phase3 = True

    from ggTrader.utils.pipeline_phases import prepare_config_and_symbols
    from ggTrader.utils.run_config import full_pipeline_config

    config_overrides = {
        "SYMBOLS_FILE": args.symbols_file,
        "MAX_SYMBOLS": args.max_symbols,
        "TRAIN_METRIC": args.train_metric,
        "START_DATE": args.start_date,
        "END_DATE": args.end_date,
        "EXPLICIT_RUN_DIR": args.run_dir,
        "PIPELINE_STAGE": args.pipeline_stage,
        "WORKER_ID": args.worker_id,
        "EXCHANGE": args.exchange,
    }
    if args.symbols:
        config_overrides["SYMBOLS"] = args.symbols.split(",")

    config = merge_run_config(full_pipeline_config(), **config_overrides)

    # Resolve symbols list from JSON file (if SYMBOLS not already set)
    if not config.get("SYMBOLS"):
        config = prepare_config_and_symbols(config)

    show_progress = not args.no_progress and sys.stdout.isatty()

    if args.run_dir:
        logger_path = Path(args.run_dir) / "wfo_isolated_status.txt"
    else:
        logger_path = Path("results/wfo_isolated_status.txt")

    logger = StatusLogger(logger_path)
    wfo_results = None
    effective_run_dir = args.run_dir  # may be updated after Phase 1

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
        # Pick up the auto-created run directory from ResultsManager when --run-dir not given
        rm = wfo_results.get("results_manager")
        if rm is not None and effective_run_dir is None:
            effective_run_dir = str(rm.run_dir)

    # Automatic Result Discovery for Phase 2/3 if Phase 1 was skipped
    if wfo_results is None and args.run_dir:
        res_json = Path(args.run_dir) / "run_results.json"
        if res_json.exists():
            print(f"Auto-discovered results in {res_json}. Loading for validation phases...")
            with open(res_json, "r") as f:
                import json

                raw_data = json.load(f)
                # Check different nested versions of per_coin_results
                sp = raw_data.get("strategy_parameters", {})
                if "per_coin" in sp:
                    wfo_results = {"per_coin_results": sp["per_coin"]}
                else:
                    # Fallback to metadata or raw if strategy_parameters is flat
                    wfo_results = {
                        "per_coin_results": raw_data.get("metadata", {}).get("per_coin_results", sp)
                    }
            print(f"  Loaded results for symbols: {list(wfo_results['per_coin_results'].keys())}")
            # Inject a ResultsManager so Phase 3 can save the YTD dashboard plot
            from ggTrader.utils.results_manager import ResultsManager

            wfo_results["results_manager"] = ResultsManager(
                script_name="run_wfo", explicit_run_dir=args.run_dir
            )

    if args.phase2:
        logger.update("Starting full data validation (Phase 2)...")

        # If we skipped Phase 1, but provided a symbols-file with parameters (legacy behavior)
        if wfo_results is None and args.symbols_file:
            print(f"Loading pre-optimized parameters from {args.symbols_file} for Phase 2...")
            with open(args.symbols_file, "r") as f:
                import json

                params = json.load(f)
                # Phase 2 expects a dict where 'results' or the dict itself
                # contains 'per_coin_results'
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

    # Automatic Report Generation (Integration)
    if effective_run_dir and wfo_results:
        from ggTrader.utils.report_generator import generate_pipeline_report

        print(f"Generating research report in {effective_run_dir}...")
        # Construct final_backtest_results for the reporter
        # The reporter expects certain keys like 'phase_2_stats' and 'phase_3_stats'
        # which are already added to wfo_results by the pipeline_phases functions.
        final_backtest_results = {
            "final_stats": wfo_results.get("phase_2_stats", {}),
            "per_coin_final_stats": wfo_results.get("phase_2_per_coin_final_stats", {}),
            "phase_2_stats": wfo_results.get("phase_2_stats"),
            "phase_3_stats": wfo_results.get("phase_3_stats"),
        }

        # Handle sensitivity results if they were loaded/run (Ph0 is usually separate)
        sensitivity_results = wfo_results.get("sensitivity_results", {})

        try:
            # Persist phase stats so the report can be regenerated without re-running backtests
            import json as _json

            def _serializable(obj):
                """Recursively convert non-JSON-serializable types."""
                import numpy as np

                if isinstance(obj, dict):
                    return {k: _serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_serializable(v) for v in obj]
                if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                    return None
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return obj

            phase_stats_path = Path(effective_run_dir) / "phase_stats.json"
            with open(phase_stats_path, "w") as _f:
                _json.dump(_serializable(final_backtest_results), _f, indent=2, default=str)

            generate_pipeline_report(
                sensitivity_results=sensitivity_results,
                wfo_results=wfo_results,
                final_backtest_results=final_backtest_results,
                output_dir=effective_run_dir,
            )
            # Standardize filename to research_report.md for this script
            report_src = Path(effective_run_dir) / "pipeline_report.md"
            report_dst = Path(effective_run_dir) / "research_report.md"
            if report_src.exists():
                if report_dst.exists():
                    report_dst.unlink()
                report_src.rename(report_dst)
                print(f"  >>> Research report: {report_dst}")
            # Print dashboard link if available
            plots_dir = Path(effective_run_dir) / "plots"
            if plots_dir.exists():
                for png in sorted(plots_dir.glob("*.png")):
                    print(f"  >>> Plot: {png}")
        except Exception as e:
            print(f"Warning: Failed to generate report: {e}")

    if wfo_results and wfo_results.get("final_portfolio") is not None:
        pf = wfo_results["final_portfolio"]
        print(
            f"\nFinal Dynamic Walk-Forward Portfolio Return "
            f"(Across concatenated folds): {pf.total_return().mean() * 100:.2f}%\n"
        )
    elif wfo_results is not None:
        print(
            "\nNo combined portfolio produced — every coin in this slice was "
            "filtered out by selection gates. Worker exiting cleanly with empty results.\n"
        )


if __name__ == "__main__":
    main()
