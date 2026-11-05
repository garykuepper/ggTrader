"""Tests for the memory-capped runner scripts/run_capped.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("systemd-run") is None, reason="systemd-run not available"
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_capped.sh"
KILL_MESSAGE = "KILLED AT MEMORY CAP ({cap}): do not retry with a higher cap; report it"
# Multiplying bytes writes every page, so the allocation is resident; a bare
# bytearray(n) is lazily zeroed and may never reach the cap.
ALLOC_512M = "x = b'\\x01' * (512*1024*1024); print(len(x))"


def _run(*args: str, cap: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("GGT_RUN_MEM_MAX", None)
    if cap is not None:
        env["GGT_RUN_MEM_MAX"] = cap
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=120
    )


def test_runs_command_and_returns_its_exit_code() -> None:
    assert _run("true").returncode == 0
    assert _run("false").returncode == 1


def test_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "GGT_RUN_MEM_MAX" in result.stdout


def test_allocation_over_cap_exits_137_and_prints_message() -> None:
    result = _run("python", "-c", ALLOC_512M, cap="64M")
    assert result.returncode == 137
    assert KILL_MESSAGE.format(cap="64M") in result.stderr


def test_same_allocation_under_a_larger_cap_succeeds() -> None:
    # Guards the over-cap test: the command must only fail because of the cap.
    result = _run("python", "-c", ALLOC_512M, cap="2G")
    assert result.returncode == 0
    assert str(512 * 1024 * 1024) in result.stdout
