"""Tests for the disposable restore drill."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.kaggle_restore_drill import TABLES, compare_counts, parse_expected, restore_command

SCRIPT = Path("scripts/kaggle_restore_drill.py")


def test_cli_help_and_required_url():
    assert subprocess.run([sys.executable, str(SCRIPT), "--help"], check=False).returncode == 0
    assert subprocess.run([sys.executable, str(SCRIPT)], check=False).returncode == 2


def test_expected_counts(tmp_path):
    path = tmp_path / "counts"
    path.write_text("295270566 1293 311 205 783982")
    assert list(parse_expected(path).values()) == [295270566, 1293, 311, 205, 783982]
    path.write_text("1 2")
    with pytest.raises(ValueError):
        parse_expected(path)


def test_compare_counts():
    expected = dict.fromkeys(TABLES, 1)
    assert all(item["ok"] for item in compare_counts(expected, expected).values())
    actual = dict(expected, **{TABLES[0]: 2})
    assert not compare_counts(expected, actual)[TABLES[0]]["ok"]


def test_command_has_no_credentials():
    command = restore_command("ggt_restore_drill_12345678")
    assert all("://" not in item and "password" not in item.lower() for item in command)


def test_missing_scratch_env(monkeypatch, tmp_path):
    monkeypatch.delenv("GGT_SCRATCH_URL", raising=False)
    result = subprocess.run(
        [sys.executable, "scripts/kaggle_restore_drill.py", "--out", str(tmp_path / "out")],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"GGT_SCRATCH_URL" in result.stderr
