"""Tests for the heavy-job guard shared by scripts/run_capped.sh and scripts/run_tests.sh.

The guard (scripts/_heavy_job.sh) keeps two heavy ggTrader jobs from running at once, refuses a job
when free memory is short, and stops a hung job at a wall-clock limit. It exists because two
parallel ``tests/paper/`` runs once exhausted Encom's memory (2026-11-18).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAPPED = ROOT / "scripts" / "run_capped.sh"
TESTS = ROOT / "scripts" / "run_tests.sh"
FLOCK = shutil.which("flock")


def _systemd_scope_available() -> bool:
    """Return True only if a user-session transient scope can actually be started."""
    if shutil.which("systemd-run") is None:
        return False
    try:
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "--collect", "true"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


needs_systemd_scope = pytest.mark.skipif(
    not _systemd_scope_available(), reason="systemd-run --user --scope not usable here"
)
needs_flock = pytest.mark.skipif(FLOCK is None, reason="flock not installed")


def _python() -> str:
    """The interpreter the dry-run tests hand to run_tests.sh (it only prints the command)."""
    return os.environ.get("GGT_TEST_PYTHON") or shutil.which("python3") or "python3"


#: Variables the guard reads. A test starts from none of them: run_tests.sh exports
#: GGT_HEAVY_LOCK_HELD to its children, which would otherwise make every test below a no-op.
_GUARD_VARS = (
    "GGT_TEST_NO_CGROUP",
    "GGT_TEST_DRY_RUN",
    "GGT_RUN_MEM_MAX",
    "GGT_RUN_MAX_SECONDS",
    "GGT_TEST_MEM_MAX",
    "GGT_TEST_MAX_SECONDS",
    "GGT_MEMINFO",
)


def _env(lock: Path, **extra: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _GUARD_VARS and not key.startswith("GGT_HEAVY")
    }
    env["GGT_HEAVY_LOCK"] = str(lock)
    # A fake, roomy meminfo: the real preflight would refuse these tests on a busy host (media or
    # game servers running). Tests of the preflight itself pass their own GGT_MEMINFO.
    plenty = lock.parent / "meminfo-plenty"
    plenty.write_text("MemTotal:       134217728 kB\nMemAvailable:   100663296 kB\n")
    env["GGT_MEMINFO"] = str(plenty)
    env.update(extra)
    return env


class _Holder:
    """A process that holds the lock file's flock, in its own process group."""

    def __init__(self, lock: Path) -> None:
        self.lock = lock
        self.proc = subprocess.Popen(
            [FLOCK or "flock", str(lock), "sleep", "60"], start_new_session=True
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            probe = subprocess.run([FLOCK or "flock", "-n", str(lock), "true"], check=False)
            if probe.returncode != 0:
                return
            time.sleep(0.05)
        self.kill()
        raise AssertionError("the holder never took the lock")

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()


@pytest.fixture
def holder(tmp_path: Path):
    held = _Holder(tmp_path / "lock")
    yield held
    held.kill()


@needs_flock
def test_second_job_waits_then_refuses(tmp_path: Path, holder: _Holder) -> None:
    started = time.monotonic()
    result = subprocess.run(
        [str(CAPPED), "true"],
        capture_output=True,
        text=True,
        env=_env(holder.lock, GGT_HEAVY_WAIT="1"),
        timeout=30,
    )
    assert result.returncode == 75, result.stderr
    assert "waiting for another heavy ggTrader job" in result.stderr
    assert "still running after 1s" in result.stderr
    assert time.monotonic() - started < 10


@needs_flock
@needs_systemd_scope
def test_killed_holder_releases_the_lock(tmp_path: Path, holder: _Holder) -> None:
    holder.kill()
    # A stale holder note from the killed job must not block the next one.
    Path(f"{holder.lock}.holder").write_text("pid=1 label=stale\n")
    result = subprocess.run(
        [str(CAPPED), "true"],
        capture_output=True,
        text=True,
        env=_env(holder.lock, GGT_HEAVY_WAIT="5"),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@needs_flock
@needs_systemd_scope
def test_nested_call_does_not_wait(tmp_path: Path, holder: _Holder) -> None:
    # A script run from inside a locked job (its own tests do this) must not wait on that lock.
    started = time.monotonic()
    result = subprocess.run(
        [str(CAPPED), "true"],
        capture_output=True,
        text=True,
        env=_env(holder.lock, GGT_HEAVY_LOCK_HELD="1", GGT_HEAVY_WAIT="1"),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 20


def test_low_memory_refuses(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemAvailable:    1048576 kB\n")
    result = subprocess.run(
        [str(CAPPED), "true"],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_MEMINFO=str(meminfo), GGT_RUN_MEM_MAX="6G"),
        timeout=30,
    )
    assert result.returncode == 75, result.stderr
    assert "not enough free memory" in result.stderr


@needs_flock
@needs_systemd_scope
def test_two_jobs_never_overlap(tmp_path: Path) -> None:
    log = tmp_path / "log"
    body = f"echo start >> {log}; sleep 2; echo end >> {log}"
    env = _env(tmp_path / "lock", GGT_HEAVY_WAIT="60")
    jobs = [
        subprocess.Popen([str(CAPPED), "bash", "-c", body], env=env, stderr=subprocess.DEVNULL)
        for _ in range(2)
    ]
    assert [job.wait(timeout=90) for job in jobs] == [0, 0]
    assert log.read_text().split() == ["start", "end", "start", "end"]


@needs_flock
@needs_systemd_scope
def test_runtime_limit_stops_a_hang(tmp_path: Path) -> None:
    started = time.monotonic()
    result = subprocess.run(
        [str(CAPPED), "sleep", "30"],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_RUN_MAX_SECONDS="2"),
        timeout=60,
    )
    assert result.returncode == 143, result.stderr
    assert "time limit or another signal" in result.stderr
    assert time.monotonic() - started < 15


def _holder_note(lock: Path) -> str:
    note = Path(f"{lock}.holder")
    return note.read_text() if note.exists() else ""


def test_dry_run_takes_no_lock(tmp_path: Path, holder: _Holder) -> None:
    # tests/test_root_conftest.py runs run_tests.sh in dry-run mode from inside a locked test run.
    result = subprocess.run(
        [str(TESTS)],
        capture_output=True,
        text=True,
        env=_env(holder.lock, GGT_TEST_DRY_RUN="1", GGT_HEAVY_WAIT="1", GGT_PYTHON=_python()),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "waiting for another heavy ggTrader job" not in result.stderr


def test_run_tests_default_cap_is_4g(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(TESTS)],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_TEST_DRY_RUN="1", GGT_PYTHON=_python()),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "MemoryMax=4G" in result.stderr


@needs_flock
def test_run_tests_refuses_when_busy(tmp_path: Path, holder: _Holder) -> None:
    result = subprocess.run(
        [str(TESTS), "tests/utils/test_import_rules.py", "-q"],
        capture_output=True,
        text=True,
        env=_env(holder.lock, GGT_HEAVY_WAIT="1"),
        timeout=60,
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert "another heavy ggTrader job is still running after 1s" in result.stderr
    assert "passed" not in result.stdout


@needs_flock
@needs_systemd_scope
def test_run_tests_lock_covers_both_halves(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    proc = subprocess.Popen(
        [str(TESTS), "--studies", "--collect-only", "-q"],
        env=_env(lock, GGT_HEAVY_WAIT="60"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    polls: list[bool] = []
    try:
        deadline = time.monotonic() + 240
        while proc.poll() is None and time.monotonic() < deadline:
            busy = subprocess.run([FLOCK or "flock", "-n", str(lock), "true"], check=False)
            if proc.poll() is None:
                polls.append(busy.returncode != 0)
            time.sleep(0.2)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert any(polls), "the lock was never seen held"
    first = polls.index(True)
    assert all(polls[first:]), "the lock was free between the two pytest runs"


@needs_systemd_scope
def test_run_tests_time_limit_stops_a_hang(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nsleep 30\n")
    fake_python.chmod(0o755)
    started = time.monotonic()
    result = subprocess.run(
        [str(TESTS), "tests/utils/test_import_rules.py"],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_PYTHON=str(fake_python), GGT_TEST_MAX_SECONDS="2"),
        timeout=60,
    )
    assert result.returncode == 143, result.stderr
    assert "time limit or another signal" in result.stderr
    assert time.monotonic() - started < 15


def test_guard_tests_do_not_depend_on_the_hosts_free_memory(tmp_path: Path) -> None:
    """On a busy host (media or game servers running) the real preflight would refuse these."""
    env = _env(tmp_path / "lock")
    meminfo = Path(env["GGT_MEMINFO"])
    available_kb = int(
        next(
            line for line in meminfo.read_text().splitlines() if line.startswith("MemAvailable:")
        ).split()[1]
    )
    assert available_kb >= 64 * 1024 * 1024


@pytest.mark.parametrize("bad", ["1.5G", "50%", "4GiB", "4g", "abc"])
def test_an_unreadable_cap_is_refused_before_anything_runs(tmp_path: Path, bad: str) -> None:
    ran = tmp_path / "ran"
    result = subprocess.run(
        [str(CAPPED), "bash", "-c", f"touch {ran}"],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_RUN_MEM_MAX=bad),
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "cannot read the size" in result.stderr
    assert not ran.exists()


def test_an_unreadable_margin_is_refused(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(CAPPED), "true"],
        capture_output=True,
        text=True,
        env=_env(tmp_path / "lock", GGT_HEAVY_MARGIN="lots"),
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "cannot read the size" in result.stderr
