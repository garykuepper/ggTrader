"""CLI Command: ggt report — regenerate research_report.md from saved run data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def register_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report", help="Regenerate research_report.md from a saved research run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to research run directory (default: auto-detect latest)",
    )


def _find_latest_research_dir() -> Optional[Path]:
    from ggTrader.utils.paths import find_project_root

    base = find_project_root() / "results" / "research"
    if not base.exists():
        return None
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("research_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_report(args: argparse.Namespace) -> None:
    from ggTrader.utils.report_generator import generate_pipeline_report

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_research_dir()
    if run_dir is None:
        print("No research run directory found. Run `ggt research` first.")
        return

    # Load per_coin_results from run_results.json
    run_results_path = run_dir / "run_results.json"
    if not run_results_path.exists():
        print(f"No run_results.json found in {run_dir}")
        return

    with open(run_results_path) as f:
        run_data = json.load(f)

    per_coin_results = run_data.get("strategy_parameters", {}).get("per_coin", {})
    if not per_coin_results:
        print("run_results.json has no per_coin results.")
        return

    # Load phase stats (phase_2_stats, phase_3_stats) from phase_stats.json
    phase_stats_path = run_dir / "phase_stats.json"
    if not phase_stats_path.exists():
        print(
            f"No phase_stats.json found in {run_dir}.\n"
            "This file is written on runs after the report fix. "
            "Re-run with: ggt research --top N --workers N"
        )
        return

    with open(phase_stats_path) as f:
        final_backtest_results = json.load(f)

    from ggTrader.utils.results_manager import ResultsManager
    rm = ResultsManager(script_name="ggt_report", explicit_run_dir=str(run_dir))
    wfo_results = {"per_coin_results": per_coin_results, "results_manager": rm}

    # Re-run Phase 3 if the YTD plot is missing
    ytd_plot = run_dir / "plots" / "combined_portfolio_ytd_dashboard.png"
    if not ytd_plot.exists():
        print("YTD plot missing — re-running Phase 3 to generate it...")
        from ggTrader.utils.pipeline_phases import phase_3_recent_performance
        from ggTrader.utils.run_config import full_pipeline_config
        symbols_file = run_dir / "top_ccxt_volume.json"
        config = {
            **full_pipeline_config(),
            "EXPLICIT_RUN_DIR": str(run_dir),
            **({"SYMBOLS_FILE": str(symbols_file)} if symbols_file.exists() else {}),
        }
        phase_3_recent_performance(config=config, wfo_results=wfo_results)
        final_backtest_results["phase_3_stats"] = wfo_results.get("phase_3_stats", final_backtest_results.get("phase_3_stats"))

    print(f"Regenerating report for {run_dir.name} ...")
    generate_pipeline_report(
        sensitivity_results={},
        wfo_results=wfo_results,
        final_backtest_results=final_backtest_results,
        output_dir=str(run_dir),
    )

    report_src = run_dir / "pipeline_report.md"
    report_dst = run_dir / "research_report.md"
    if report_src.exists():
        if report_dst.exists():
            report_dst.unlink()
        report_src.rename(report_dst)

    print(f"  >>> {report_dst}")
