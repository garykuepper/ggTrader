"""Unit tests for the state manager's auto-discovery logic."""

import os
import tempfile
import time
import sys
from pathlib import Path
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ggTrader.utils.state_manager import get_latest_research_run, get_latest_production_weights

class TestStateManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.test_dir.name)
        
    def tearDown(self):
        self.test_dir.cleanup()

    def test_get_latest_research_run(self):
        # Setup: Create some fake pipeline directories
        dir1 = self.results_dir / "pipeline_20230101_100000"
        dir2 = self.results_dir / "pipeline_20240101_100000"
        dir3 = self.results_dir / "recalibration_20240101_100000"
        
        dir1.mkdir()
        dir2.mkdir()
        dir3.mkdir()
        
        # Add run_results.json
        (dir1 / "run_results.json").touch()
        time.sleep(0.05)
        (dir2 / "run_results.json").touch()
        time.sleep(0.05)
        # Recalibration might look exactly like a research directory inside
        (dir3 / "run_results.json").touch()

        # Execute
        latest = get_latest_research_run(str(self.results_dir))
        
        # Verify
        self.assertIsNotNone(latest)
        self.assertEqual(latest.parent.name, "pipeline_20240101_100000")
        
    def test_get_latest_production_weights(self):
        # Setup
        dir1 = self.results_dir / "recalibration_111"
        dir2 = self.results_dir / "recalibration_222"
        
        (dir1 / "portfolio_analysis").mkdir(parents=True)
        (dir2 / "portfolio_analysis").mkdir(parents=True)
        
        (dir1 / "portfolio_analysis" / "portfolio_weights.json").touch()
        time.sleep(0.05)
        (dir2 / "portfolio_analysis" / "portfolio_weights.json").touch()
        
        # Execute
        latest = get_latest_production_weights(str(self.results_dir))
        
        # Verify
        self.assertIsNotNone(latest)
        self.assertEqual(latest.parent.parent.name, "recalibration_222")

    def test_empty_directory(self):
        self.assertIsNone(get_latest_research_run(str(self.results_dir)))
        self.assertIsNone(get_latest_production_weights(str(self.results_dir)))

if __name__ == "__main__":
    unittest.main()
