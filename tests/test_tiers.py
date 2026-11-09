"""The test tiers: markers are registered, tiers select what they say, one source for all."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import tiers

ROOT = Path(__file__).resolve().parents[1]


def test_every_tier_marker_is_registered_in_pyproject() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    registered = {line.split(":")[0] for line in config["markers"]}
    assert registered == set(tiers.FAST_EXCLUDED)


def test_the_tiers_name_the_markers_they_leave_out() -> None:
    assert tiers.TIER_MARKEXPR["fast"] == (
        "not integration and not db and not slow and not numba and not tripwire "
        "and not study and not tooling"
    )
    assert tiers.TIER_MARKEXPR["full"] == "not integration"
    assert tiers.TIER_MARKEXPR["studies"] == "study and not integration"


@pytest.mark.parametrize("tier", ["fast", "full", "studies"])
def test_apply_tier_replaces_the_default_marker_expression(tier: str) -> None:
    option = SimpleNamespace(markexpr="not integration")
    tiers.apply_tier(option, tier)
    assert option.markexpr == tiers.TIER_MARKEXPR[tier]


def test_no_tier_leaves_the_marker_expression_alone() -> None:
    option = SimpleNamespace(markexpr="not integration")
    tiers.apply_tier(option, None)
    assert option.markexpr == "not integration"


def test_an_unknown_tier_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown test tier"):
        tiers.apply_tier(SimpleNamespace(markexpr=""), "nightly")


def _collect(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.stdout


def test_the_fast_tier_drops_the_live_golden_and_the_full_tier_keeps_it() -> None:
    golden = "tests/paper/test_live_golden.py"
    assert "test_rebuilt_signal_runner_reproduces_the_live_golden" not in _collect(
        golden, "--tier", "fast"
    )
    assert "test_rebuilt_signal_runner_reproduces_the_live_golden" in _collect(
        golden, "--tier", "full"
    )


def test_the_study_half_promises_a_non_negative_minimum_and_two_folder_patterns() -> None:
    assert isinstance(tiers.STUDIES_MIN_COLLECTED, int) and tiers.STUDIES_MIN_COLLECTED >= 0
    assert tiers.STUDY_TEST_GLOBS == ("research/*/*/code/tests", "research/*/_lib/tests")
    # Never the bare research folder: it holds frozen test_*.py files that import deleted modules.
    assert all(
        glob != "research" and glob.startswith("research/") for glob in tiers.STUDY_TEST_GLOBS
    )


def test_study_test_dirs_lists_only_existing_folders(tmp_path: Path) -> None:
    (tmp_path / "research/a/s/code/tests").mkdir(parents=True)
    (tmp_path / "research/a/_lib/tests").mkdir(parents=True)
    (tmp_path / "research/a/s2/code").mkdir(parents=True)  # no tests folder
    (tmp_path / "research/b/t/code").mkdir(parents=True)
    (tmp_path / "research/b/t/code/tests").write_text("a file, not a folder")
    assert tiers.study_test_dirs(tmp_path) == ["research/a/_lib/tests", "research/a/s/code/tests"]
    assert tiers.study_test_dirs(tmp_path / "missing") == []
