"""Unit tests for the state manager's auto-discovery logic."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ggTrader.utils.state_manager import (
    get_latest_production_weights,
    get_latest_research_run,
    validate_results_asset_class,
)


def _write_run_results(path: Path, asset_class=None, raw_asset_class=None) -> None:
    """Write a minimal run_results.json with optional asset_class fields."""
    data: dict = {"run_id": path.parent.name, "results": {}}
    if asset_class is not None:
        data["asset_class"] = asset_class
    if raw_asset_class is not None:
        data["configuration"] = {"_raw_config": {"ASSET_CLASS": raw_asset_class}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


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


class TestAssetClassFiltering(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.test_dir.name)
        (self.results_dir / "research").mkdir()

    def tearDown(self):
        self.test_dir.cleanup()

    def _make_run(self, name: str, asset_class=None, raw_asset_class=None) -> Path:
        run_dir = self.results_dir / "research" / name
        rj = run_dir / "run_results.json"
        _write_run_results(rj, asset_class=asset_class, raw_asset_class=raw_asset_class)
        return rj

    def test_asset_class_filter_picks_matching_class(self):
        # Newer stocks run, older crypto run — asking for crypto must skip the newer stocks
        crypto_path = self._make_run("research_old_crypto", asset_class="crypto")
        time.sleep(0.05)
        self._make_run("research_new_stocks", asset_class="stocks")

        latest = get_latest_research_run(str(self.results_dir), asset_class="crypto")
        self.assertEqual(latest, crypto_path)

    def test_asset_class_filter_returns_none_when_no_match(self):
        self._make_run("research_only_stocks", asset_class="stocks")
        self.assertIsNone(
            get_latest_research_run(str(self.results_dir), asset_class="crypto")
        )

    def test_legacy_raw_config_fallback(self):
        # No top-level asset_class, only configuration._raw_config.ASSET_CLASS
        legacy = self._make_run("research_legacy", raw_asset_class="stocks")
        latest = get_latest_research_run(str(self.results_dir), asset_class="stocks")
        self.assertEqual(latest, legacy)

    def test_truly_legacy_defaults_to_crypto(self):
        # JSON without either asset_class or _raw_config.ASSET_CLASS — old run
        path = self.results_dir / "research" / "research_ancient" / "run_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"run_id": "ancient"}, f)

        # Asking for crypto should match (default fallback)
        self.assertEqual(
            get_latest_research_run(str(self.results_dir), asset_class="crypto"),
            path,
        )
        # Asking for stocks should NOT match
        self.assertIsNone(
            get_latest_research_run(str(self.results_dir), asset_class="stocks")
        )

    def test_no_filter_returns_newest_regardless_of_class(self):
        self._make_run("research_a", asset_class="crypto")
        time.sleep(0.05)
        newest = self._make_run("research_b", asset_class="stocks")
        self.assertEqual(get_latest_research_run(str(self.results_dir)), newest)

    def test_validate_passes_on_match(self):
        path = self._make_run("research_match", asset_class="crypto")
        # No exception
        validate_results_asset_class(path, expected="crypto")

    def test_validate_raises_on_mismatch(self):
        path = self._make_run("research_mismatch", asset_class="stocks")
        with self.assertRaises(SystemExit):
            validate_results_asset_class(path, expected="crypto")

    def test_validate_raises_on_missing_path(self):
        with self.assertRaises(SystemExit):
            validate_results_asset_class(
                self.results_dir / "does_not_exist.json", expected="crypto"
            )


class TestProductionWeightsAssetClass(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.test_dir.name)
        (self.results_dir / "production").mkdir()

    def tearDown(self):
        self.test_dir.cleanup()

    def _make_run(self, name: str, asset_class: str) -> Path:
        run_dir = self.results_dir / "production" / name
        (run_dir / "portfolio_analysis").mkdir(parents=True)
        weights = run_dir / "portfolio_analysis" / "portfolio_weights.json"
        weights.touch()
        _write_run_results(run_dir / "run_results.json", asset_class=asset_class)
        return weights

    def test_filter_skips_other_asset_class(self):
        crypto = self._make_run("production_old", asset_class="crypto")
        time.sleep(0.05)
        self._make_run("production_new_stocks", asset_class="stocks")

        latest = get_latest_production_weights(str(self.results_dir), asset_class="crypto")
        self.assertEqual(latest, crypto)


if __name__ == "__main__":
    unittest.main()
