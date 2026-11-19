"""Offline tests for the two-phase test runner's exit handling."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("tier", "first_status", "second_status", "expected_status", "phases"),
    [
        ("--full", 137, 0, 137, 1),
        ("--full", 143, 0, 143, 1),
        ("--full", 130, 0, 130, 1),
        ("--full", 139, 0, 139, 1),
        ("--full", 1, 0, 1, 2),
        ("--studies", 5, 0, 0, 2),
        ("--full", 1, 137, 137, 2),
    ],
)
def test_terminal_status_stops_later_phases(
    tmp_path: Path,
    tier: str,
    first_status: int,
    second_status: int,
    expected_status: int,
    phases: int,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "research/x/y/code/tests").mkdir(parents=True)
    for name in ("run_tests.sh", "_heavy_job.sh"):
        shutil.copy(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    shutil.copy(ROOT / "tests/tiers.py", tmp_path / "tests/tiers.py")
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    exec(sys.argv[2])\n"
        "else:\n"
        "    with open(os.environ['PHASE_LOG'], 'a') as log:\n"
        "        log.write('phase\\n')\n"
        "    key = 'FIRST_STATUS' if '--tier' in sys.argv else 'SECOND_STATUS'\n"
        "    sys.exit(int(os.environ[key]))\n"
    )
    fake.chmod(0o755)
    log = tmp_path / "phases"
    env = {
        **os.environ,
        "GGT_TEST_NO_CGROUP": "1",
        "GGT_TEST_DRY_RUN": "",
        "GGT_PYTHON": str(fake),
        "PHASE_LOG": str(log),
        "FIRST_STATUS": str(first_status),
        "SECOND_STATUS": str(second_status),
    }
    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/run_tests.sh"), tier],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == expected_status, result.stderr
    assert len(log.read_text().splitlines()) == phases
    if first_status == 137:
        assert "cap was hit" not in result.stderr
        assert "killed" in result.stderr.lower()
    if first_status == 143:
        assert "time limit or another signal" in result.stderr
