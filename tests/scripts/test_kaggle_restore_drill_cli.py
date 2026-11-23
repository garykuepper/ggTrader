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


def test_insufficient_database_volume_space_refuses_before_restore(monkeypatch, tmp_path):
    import argparse

    from scripts import kaggle_restore_drill as drill

    dump = tmp_path / "backup.dump"
    dump.write_bytes(b"1234")
    monkeypatch.setattr(drill, "verify_sha256", lambda *args: None)
    monkeypatch.setattr(drill, "parse_expected", lambda *args: {})
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "port", "ggtrader_db"]:
            return subprocess.CompletedProcess(command, 0, stdout="0.0.0.0:5433\n")
        if command[:3] == ["docker", "exec", "ggtrader_db"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="Filesystem 1024-blocks Used Available Capacity Mounted\n"
                "/dev/x 100 90 10 90% /data\n",
            )
        raise AssertionError("restore must not run")

    monkeypatch.setattr(drill.subprocess, "run", fake_run)
    monkeypatch.setattr(drill.psycopg2, "connect", lambda *args: pytest.fail("must not connect"))
    args = argparse.Namespace(
        scratch_url="postgresql://localhost:5433/postgres",
        dump=dump,
        sums=tmp_path / "sums",
        expected=tmp_path / "expected",
        keep=False,
    )
    report = {}
    with pytest.raises(ValueError, match="free space"):
        drill.run(args, report)
    assert report["database_volume"]["required_bytes"] == 3 * 4 + 40_000_000_000
    assert not any("pg_restore" in command for command in calls)


def test_restore_refuses_unmatched_container_binding(monkeypatch, tmp_path):
    from scripts import kaggle_restore_drill as drill

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="127.0.0.2:5433\n")

    monkeypatch.setattr(drill.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="match"):
        drill.check_database_volume("postgresql://127.0.0.1:5433/postgres", tmp_path / "dump", {})
