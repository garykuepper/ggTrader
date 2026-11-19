"""Launcher arguments and failures, using a fake systemd-run only."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_env(tmp_path: Path) -> dict[str, str]:
    launcher = tmp_path / "systemd-run"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['LAUNCH_LOG'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "if os.environ.get('LAUNCH_FAIL'): sys.exit(int(os.environ['LAUNCH_FAIL']))\n"
        "if os.environ.get('LAUNCH_SIGNAL'): os.kill(os.getppid(), signal.SIGTERM); sys.exit(0)\n"
        "i = 0\n"
        "while args[i].startswith('-'):\n"
        "    i += 2 if args[i] == '-p' else 1\n"
        "sys.exit(subprocess.run(args[i:], check=False).returncode)\n"
    )
    launcher.chmod(0o755)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 100663296 kB\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GGT_")}
    env.update(
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        LAUNCH_LOG=str(tmp_path / "launches"),
        GGT_HEAVY_LOCK=str(tmp_path / "lock"),
        GGT_MEMINFO=str(meminfo),
    )
    return env


def _calls(env: dict[str, str]) -> list[list[str]]:
    return [json.loads(line) for line in Path(env["LAUNCH_LOG"]).read_text().splitlines()]


def test_capped_nested_launches_inherit_slice(fake_env: dict[str, str]) -> None:
    capped = str(ROOT / "scripts/run_capped.sh")
    result = subprocess.run(
        [capped, capped, "true"], env=fake_env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(fake_env)
    assert len(calls) == 2
    for args in calls:
        assert args[:5] == ["--user", "--scope", "-q", "--collect", "--slice-inherit"]
        assert args[5:11] == [
            "-p",
            "MemoryMax=6G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "RuntimeMaxSec=7200",
        ]


def test_capped_scope_failure_does_not_run_payload(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    fake_env["LAUNCH_FAIL"] = "1"
    marker = tmp_path / "ran"
    result = subprocess.run(
        [str(ROOT / "scripts/run_capped.sh"), "touch", str(marker)],
        env=fake_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert len(_calls(fake_env)) == 1


@pytest.mark.parametrize(
    ("failure", "expected", "calls_expected", "cleanup_fails"),
    [
        (False, 1, 2, False),
        (True, 125, 1, False),
        (137, 137, 1, False),
        (False, 125, 1, True),
        (137, 137, 1, True),
    ],
)
def test_scope_failure_aborts_later_phase(
    tmp_path: Path,
    fake_env: dict[str, str],
    failure: bool | int,
    expected: int,
    calls_expected: int,
    cleanup_fails: bool,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "research/x/y/code/tests").mkdir(parents=True)
    for name in ("run_tests.sh", "_heavy_job.sh"):
        shutil.copy(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    shutil.copy(ROOT / "tests/tiers.py", tmp_path / "tests/tiers.py")
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1] == '-c': exec(sys.argv[2])\n"
        "else: sys.exit(1 if '--tier' in sys.argv else 0)\n"
    )
    python.chmod(0o755)
    fake_env["GGT_PYTHON"] = str(python)
    if failure:
        fake_env["LAUNCH_FAIL"] = str(int(failure))
    if cleanup_fails:
        rmdir = tmp_path / "rmdir"
        rmdir.write_text('#!/bin/sh\n/usr/bin/rmdir "$@"\nexit 1\n')
        rmdir.chmod(0o755)
    result = subprocess.run(
        [str(tmp_path / "scripts/run_tests.sh"), "--full"],
        env=fake_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    calls = _calls(fake_env)
    assert len(calls) == calls_expected
    assert all(
        args[:5] == ["--user", "--scope", "-q", "--collect", "--slice-inherit"] for args in calls
    )
    assert all(
        args[5:11] == ["-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "-p", "RuntimeMaxSec=1800"]
        for args in calls
    )


def test_ci_explicitly_skips_launcher(fake_env: dict[str, str]) -> None:
    fake_env["GGT_TEST_NO_CGROUP"] = "1"
    fake_env["GGT_PYTHON"] = "/bin/true"
    result = subprocess.run(
        [str(ROOT / "scripts/run_tests.sh"), "tests/test_runner_docs.py"],
        env=fake_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(fake_env["LAUNCH_LOG"]).exists()


def test_capped_help_does_not_claim_signal_cause() -> None:
    result = subprocess.run(
        [str(ROOT / "scripts/run_capped.sh"), "--help"], capture_output=True, text=True, check=True
    )
    assert "alone does not establish the cause" in result.stdout
    assert "killed at the cap" not in result.stdout


def test_interrupted_scope_cleans_marker(fake_env: dict[str, str]) -> None:
    fake_env["LAUNCH_SIGNAL"] = "1"
    fake_env["GGT_PYTHON"] = "/bin/true"
    result = subprocess.run(
        [str(ROOT / "scripts/run_tests.sh"), "tests/test_runner_docs.py"],
        env=fake_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 143, result.stderr
    calls = _calls(fake_env)
    assert len(calls) == 1
    marker = next(arg for arg in calls[0] if arg.startswith("/tmp/ggtrader-test."))
    assert not Path(marker).parent.exists()
