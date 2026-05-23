"""Unit tests for utility managers."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ggTrader.utils.results_manager import ResultsManager


@pytest.fixture
def results_dir():
    test_dir = Path("test_results_dir").absolute()
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_results_manager_init(results_dir):
    # Patch ResultDBManager and Path operations
    with patch("ggTrader.utils.results_manager.ResultDBManager"), patch(
        "ggTrader.utils.results_manager.find_project_root", return_value=results_dir
    ):
        rm = ResultsManager("test_run", results_dir="test_sub")
        assert "test_run" in str(rm.run_dir)


@patch("ggTrader.utils.results_manager.ResultDBManager")
def test_results_manager_save_metrics(mock_db_cls, results_dir):
    with patch("ggTrader.utils.results_manager.find_project_root", return_value=results_dir):
        rm = ResultsManager("test_save")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rm.save_metrics(df, "metrics.csv", save_csv=True)
        assert (rm.run_dir / "metrics.csv").exists()


@patch("ggTrader.utils.result_db_manager.create_engine")
def test_result_db_manager_add_run(mock_engine, results_dir):
    from ggTrader.utils.result_db_manager import ResultDBManager

    with patch.object(ResultDBManager, "_init_db"):
        db_manager = ResultDBManager(
            connection_string="sqlite://", log_path=str(results_dir / "log.csv")
        )
        metrics = {"sharpe": 1.5, "Start": "2023-01-01", "End": "2023-01-02"}
        params = {"mode": "test"}

        with patch.object(db_manager.engine, "begin") as mock_begin:
            db_manager.add_run("id1", "backtest", "script.py", params, metrics=metrics)
            mock_begin.assert_called_once()
