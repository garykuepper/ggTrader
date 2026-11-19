"""Static checks of the disposable-guest harness: every check must be able to fail.

The harness itself only runs inside a throwaway guest (it installs units and reboots), so these
tests read it; they do not run it.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[2] / "deploy" / "whole_job" / "guest" / "harness.sh"
CHECK = re.compile(r"^\s*check\s+(\S+)\s+(.*)$")


def _checks() -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for line in HARNESS.read_text().splitlines()
        if (match := CHECK.match(line))
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_harness_parses() -> None:
    subprocess.run(["bash", "-n", str(HARNESS)], check=True)


def test_no_check_is_unconditionally_true() -> None:
    vacuous = re.compile(r"(^|[;&|]\s*|\")\s*(true|:)\s*\"?\s*$")
    for name, command in _checks().items():
        assert not vacuous.search(command), f"{name} cannot fail: {command}"
        assert "op() { :; }" not in command, f"{name} stubs out the operator call"


def test_normal_completion_asserts_the_reported_outcome() -> None:
    command = _checks()["s1-outcome"]
    assert "field outcome" in command
    assert "job ended; driver reported status 0" in command
