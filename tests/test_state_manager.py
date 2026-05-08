"""Unit tests for the state manager's auto-discovery logic."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ggTrader.utils.state_manager import (
    get_latest_production_weights,
    get_latest_research_run,
)


def _no_db():
    """Force the disk-fallback path by making the DB lookup raise."""
    return patch(
        "ggTrader.utils.result_db_manager.ResultDBManager",
        side_effect=RuntimeError("DB disabled in test"),
    )


class TestStateManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    @_no_db()
    def test_get_latest_research_run(self, _):
        dir1 = self.results_dir / "pipeline_20230101_100000"
        dir2 = self.results_dir / "pipeline_20240101_100000"
        dir3 = self.results_dir / "recalibration_20240101_100000"

        dir1.mkdir()
        dir2.mkdir()
        dir3.mkdir()

        (dir1 / "run_results.json").touch()
        time.sleep(0.05)
        (dir2 / "run_results.json").touch()
        time.sleep(0.05)
        (dir3 / "run_results.json").touch()

        latest = get_latest_research_run(str(self.results_dir))

        self.assertIsNotNone(latest)
        # Recalibration directories are excluded from research discovery.
        self.assertEqual(latest.run_dir.name, "pipeline_20240101_100000")

    def test_get_latest_production_weights(self):
        dir1 = self.results_dir / "recalibration_111"
        dir2 = self.results_dir / "recalibration_222"

        (dir1 / "portfolio_analysis").mkdir(parents=True)
        (dir2 / "portfolio_analysis").mkdir(parents=True)

        (dir1 / "portfolio_analysis" / "portfolio_weights.json").touch()
        time.sleep(0.05)
        (dir2 / "portfolio_analysis" / "portfolio_weights.json").touch()

        latest = get_latest_production_weights(str(self.results_dir))

        self.assertIsNotNone(latest)
        self.assertEqual(latest.parent.parent.name, "recalibration_222")

    @_no_db()
    def test_empty_directory(self, _):
        self.assertIsNone(get_latest_research_run(str(self.results_dir)))
        self.assertIsNone(get_latest_production_weights(str(self.results_dir)))


class TestResearchRunDiscovery(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.test_dir.name)
        (self.results_dir / "research").mkdir()

    def tearDown(self):
        self.test_dir.cleanup()

    def _make_run(self, name: str) -> Path:
        run_dir = self.results_dir / "research" / name
        rj = run_dir / "run_results.json"
        rj.parent.mkdir(parents=True, exist_ok=True)
        with open(rj, "w") as f:
            json.dump({"run_id": name, "results": {}}, f)
        return rj

    @_no_db()
    def test_picks_newest(self, _):
        self._make_run("research_a")
        time.sleep(0.05)
        self._make_run("research_b")
        latest = get_latest_research_run(str(self.results_dir))
        self.assertIsNotNone(latest)
        self.assertEqual(latest.run_dir.name, "research_b")


if __name__ == "__main__":
    unittest.main()
