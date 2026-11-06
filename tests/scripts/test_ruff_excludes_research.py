"""Check that ruff leaves frozen research/ Python artifacts alone, even by explicit path."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUFF = ROOT / ".venv/bin/ruff"
RUFF_TOML = ROOT / "ruff.toml"
BAD_SOURCE = "import os\nx=1\n"  # unused import (F401) and unformatted (E225 / format)


def _tracked_research_python() -> list[str]:
    """List tracked research/**/*.py files relative to the repo root."""
    out = subprocess.run(
        ["git", "ls-files", "research/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def _ruff(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the project's ruff and capture its output."""
    return subprocess.run([str(RUFF), *args], cwd=cwd, capture_output=True, text=True)


def test_ruff_force_excludes_research_python_files() -> None:
    """Explicitly passed research/*.py files are skipped by both ruff check and format."""
    files = _tracked_research_python()
    assert len(files) >= 4, f"expected at least 4 tracked research .py files, got {files}"
    for rel in files:
        check = _ruff(["check", "--force-exclude", rel], ROOT)
        assert check.returncode == 0, (rel, check.stdout, check.stderr)
        fmt = _ruff(["format", "--check", "--force-exclude", rel], ROOT)
        assert fmt.returncode == 0, (rel, fmt.stdout, fmt.stderr)
        assert "would reformat" not in fmt.stdout + fmt.stderr, rel


def test_ruff_toml_declares_the_exclusion() -> None:
    """ruff.toml force-excludes research so hooks that pass explicit paths honor it."""
    config = tomllib.loads(RUFF_TOML.read_text())
    assert config.get("force-exclude") is True
    assert "research" in config.get("extend-exclude", [])


def test_repo_ruff_config_skips_bad_research_file_but_lints_others(tmp_path: Path) -> None:
    """Prove the mechanism with the real ruff.toml on a throwaway tree (never touch research/)."""
    shutil.copy(RUFF_TOML, tmp_path / "ruff.toml")
    (tmp_path / "research").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "research/bad.py").write_text(BAD_SOURCE)
    (tmp_path / "scripts/bad.py").write_text(BAD_SOURCE)

    skipped = _ruff(["check", "--force-exclude", "research/bad.py"], tmp_path)
    assert skipped.returncode == 0, (skipped.stdout, skipped.stderr)
    skipped_fmt = _ruff(["format", "--check", "--force-exclude", "research/bad.py"], tmp_path)
    assert skipped_fmt.returncode == 0, (skipped_fmt.stdout, skipped_fmt.stderr)

    linted = _ruff(["check", "--force-exclude", "scripts/bad.py"], tmp_path)
    assert linted.returncode != 0 and "F401" in linted.stdout
    formatted = _ruff(["format", "--check", "--force-exclude", "scripts/bad.py"], tmp_path)
    assert formatted.returncode != 0
