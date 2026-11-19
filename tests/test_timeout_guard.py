"""Tests for the per-test wall-clock timeout (tests/timeout_guard.py and the root conftest)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.timeout_guard import UNTIMED_TESTS, TestTimedOut, time_limit

ROOT = Path(__file__).resolve().parents[1]
PROBE = "tests/timeout_probe/hang_probe.py"


def _child(*args: str, timeout_env: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GGT_TEST_TIMEOUT": timeout_env}
    return subprocess.run(
        [sys.executable, "-m", "pytest", PROBE, "-p", "no:cacheprovider", "-q", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_spin_loop_is_interrupted() -> None:
    started = time.monotonic()
    with pytest.raises(TestTimedOut, match="exceeded 0.5s"), time_limit(0.5):
        while True:
            pass
    assert time.monotonic() - started < 3


def test_previous_handler_and_timer_restored() -> None:
    # The autouse fixture already runs a limit around this test: the inner one must hand back
    # the same handler and an outer timer that has only counted down.
    handler = signal.getsignal(signal.SIGALRM)
    before, _ = signal.getitimer(signal.ITIMER_REAL)
    with time_limit(5):
        assert signal.getitimer(signal.ITIMER_REAL)[0] <= 5
    after, _ = signal.getitimer(signal.ITIMER_REAL)
    assert signal.getsignal(signal.SIGALRM) is handler
    assert 0 < after <= before


def test_a_running_outer_timer_survives_the_inner_limit() -> None:
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        with time_limit(5):
            pass
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining <= 30
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_zero_disables_the_limit() -> None:
    handler = signal.getsignal(signal.SIGALRM)
    before, _ = signal.getitimer(signal.ITIMER_REAL)
    with time_limit(0):
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] <= before


def test_fixture_fails_a_hanging_test() -> None:
    result = _child("-k", "test_hangs", timeout_env="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "exceeded 1s" in result.stdout


def test_marker_overrides_the_default() -> None:
    result = _child("-k", "test_marked_hangs", timeout_env="600")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "exceeded 1s" in result.stdout


def test_every_untimed_test_still_exists() -> None:
    """An exemption for a test that was renamed or removed would silently exempt nothing."""
    assert UNTIMED_TESTS, "the exemption list is not meant to be empty while the probe test exists"
    for nodeid in UNTIMED_TESTS:
        path, _, name = nodeid.partition("::")
        source = (ROOT / path).read_text()
        assert f"def {name}(" in source, f"{nodeid} no longer exists: drop it from UNTIMED_TESTS"


def test_a_broad_except_cannot_swallow_the_timeout() -> None:
    """The order-polling loop of the incident wraps its work in ``except Exception``."""
    started = time.monotonic()
    deadline = started + 3  # an inner bound, so a broken guard fails this test instead of hanging
    with pytest.raises(TestTimedOut), time_limit(0.5):
        try:
            while time.monotonic() < deadline:
                pass
        except Exception:
            pass
        while time.monotonic() < deadline:
            pass
    assert time.monotonic() - started < 2


def test_a_swallowed_alarm_fires_again() -> None:
    hits = 0
    started = time.monotonic()
    deadline = started + 4
    with pytest.raises(TestTimedOut), time_limit(0.3):
        while time.monotonic() < deadline:
            try:
                while time.monotonic() < deadline:
                    pass
            except BaseException:
                hits += 1
                if hits >= 2:
                    raise
    assert hits >= 2
