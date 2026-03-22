"""Thin console-script targets; each delegates to ``scripts/<name>.py``."""

from __future__ import annotations

from ggTrader.cli.script_entry import run_script


def backtest() -> None:
    run_script("run_backtest.py")


def wfo() -> None:
    run_script("run_walk_forward_optimization.py")


def sensitivity() -> None:
    run_script("run_sensitivity_analysis.py")


def pipeline() -> None:
    run_script("run_full_pipeline.py")


def compare_strategies() -> None:
    run_script("run_strategy_comparison.py")


def view_results() -> None:
    run_script("view_results.py")


def pipeline_status() -> None:
    run_script("pipeline_status.py")


def manage_data() -> None:
    run_script("manage_data.py")


def manage_db() -> None:
    run_script("manage_db.py")
