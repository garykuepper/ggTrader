"""Sensitivity driver parses its inputs without starting a research run."""

import subprocess
import sys
from pathlib import Path

from scripts.lab_dead_stock_sensitivity import build_parser

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/lab_dead_stock_sensitivity.py"


def test_argument_parsing():
    args = build_parser().parse_args(
        [
            "--strategy",
            "ensemble",
            "--universe",
            "sp500",
            "--membership",
            "db:2",
            "--eval-start",
            "2020-01-01",
            "--eval-end",
            "2024-01-01",
            "--release-id",
            "3",
        ]
    )
    assert args.strategy == "ensemble" and args.release_id == 3 and args.membership == "db:2"
    assert str(args.result_dir) == "results/kaggle_dead_stock"


def test_help():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--release-id" in result.stdout
