"""The test-running rules in CLAUDE.md and the changelog must state what the scripts do."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = (ROOT / "CLAUDE.md").read_text()  # a symlink to AGENTS.md
RUN_TESTS = (ROOT / "scripts" / "run_tests.sh").read_text()
CAPPED = (ROOT / "scripts" / "run_capped.sh").read_text()
CONFTEST = (ROOT / "conftest.py").read_text()


def _default(script: str, name: str) -> str:
    match = re.search(rf"{name}:-([0-9A-Za-z]+)\}}", script)
    assert match, f"{name} has no default in the script"
    return match.group(1)


def test_the_rules_state_the_runner_defaults() -> None:
    assert f"MemoryMax={_default(RUN_TESTS, 'GGT_TEST_MEM_MAX')}" in RULES
    test_limit = _default(RUN_TESTS, "GGT_TEST_MAX_SECONDS")
    run_limit = _default(CAPPED, "GGT_RUN_MAX_SECONDS")
    assert f"`GGT_TEST_MAX_SECONDS` (default {test_limit})" in RULES
    assert f"`GGT_RUN_MAX_SECONDS` (default {run_limit})" in RULES
    timeout = re.search(r"DEFAULT_TEST_TIMEOUT_SECONDS = (\d+)", CONFTEST)
    assert timeout and f"default {timeout.group(1)}" in RULES


def test_the_rules_do_not_keep_the_stale_memory_figure() -> None:
    assert "peaks at ~650 MB" not in RULES


def test_the_rules_say_the_per_test_timeout_still_applies_in_ci() -> None:
    assert "per-test timeout still applies" in RULES


def test_the_changelog_does_not_call_a_floor_the_peak() -> None:
    entry = (ROOT / "docs" / "changelog.md").read_text().split("\n## ")[1]
    assert "peaked at about 1.8 GiB" not in entry
    assert "no time limit" not in entry or "per-test" in entry
