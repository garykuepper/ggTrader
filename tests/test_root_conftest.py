"""The repo-root conftest.py reaches tests that live outside tests/, and run_tests.sh parses.

Study tests will live under research/<category>/<study>/code/tests. They must get the same
`_no_dotenv` safety fixture (a test must never read the real `.env`), so the fixture lives in the
repo-root conftest.py. These tests prove it with a throwaway test file collected from outside
tests/, and prove that `scripts/run_tests.sh --studies` builds a command pytest accepts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_TESTS = ROOT / "scripts" / "run_tests.sh"

#: A test that points the project root at a temporary directory holding a `.env` with a key,
#: loads the environment the way the code does, and asserts the key did not arrive.
INNER_TEST = textwrap.dedent(
    """
    import os

    from ggTrader.utils import config


    def test_a_tmp_dotenv_is_never_loaded(tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("GGT_SENTINEL_KEY=leaked\\n")
        monkeypatch.setattr(config, "find_project_root", lambda: tmp_path)
        try:
            config._load_env()
            assert "GGT_SENTINEL_KEY" not in os.environ
        finally:
            os.environ.pop("GGT_SENTINEL_KEY", None)
    """
)


def _pytest(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_the_shared_conftest_is_at_the_repo_root_only() -> None:
    assert (ROOT / "conftest.py").is_file()
    assert not (ROOT / "tests" / "conftest.py").exists()
    text = (ROOT / "conftest.py").read_text()
    for needle in ("_no_dotenv", "--tier", "--update-golden", "--golden-reason", "Agg"):
        assert needle in text, needle


def test_a_test_collected_from_outside_tests_cannot_see_a_tmp_dotenv() -> None:
    scratch = Path(tempfile.mkdtemp(prefix=".sentinel_", dir=ROOT))
    try:
        (scratch / "test_inner.py").write_text(INNER_TEST)
        result = _pytest(str(scratch / "test_inner.py"), "-q")
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
        assert "1 passed" in result.stdout
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    assert "GGT_SENTINEL_KEY" not in os.environ


def test_the_sentinel_has_teeth_without_the_fixture_the_key_would_load() -> None:
    """Outside the repo root the shared conftest does not apply, so the key does leak there."""
    with tempfile.TemporaryDirectory() as raw:
        (Path(raw) / "test_inner.py").write_text(INNER_TEST)
        result = _pytest("-q", "--rootdir", raw, "-c", os.devnull, raw, cwd=Path(raw))
    assert result.returncode != 0 and "GGT_SENTINEL_KEY" in result.stdout


def _dry_run(repo: Path, *args: str) -> str:
    env = {**os.environ, "GGT_TEST_DRY_RUN": "1", "GGT_PYTHON": sys.executable}
    done = subprocess.run(
        ["bash", str(repo / "scripts" / "run_tests.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return done.stdout.strip()


def test_run_tests_picks_the_tier_from_the_first_argument() -> None:
    assert "--tier fast" in _dry_run(ROOT) and "--tier fast" in _dry_run(ROOT, "--fast", "-q")
    assert "--tier full" in _dry_run(ROOT, "--full", "-q")
    assert "--tier" not in _dry_run(ROOT, "tests/paper", "-q")  # unchanged pass-through
    assert "--tier" not in _dry_run(ROOT, "-q")


def test_run_tests_studies_names_explicit_research_paths_never_the_bare_folder() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw)
        (repo / "scripts").mkdir()
        shutil.copy(RUN_TESTS, repo / "scripts" / "run_tests.sh")
        (repo / "research" / "alloc" / "static" / "code" / "tests").mkdir(parents=True)
        (repo / "research" / "stock" / "_lib" / "tests").mkdir(parents=True)
        out = _dry_run(repo, "--studies", "-q").split()
    assert "--import-mode=importlib" in out and "studies" in out
    assert "research/alloc/static/code/tests" in out and "research/stock/_lib/tests" in out
    assert "research" not in out and "tests" in out


def test_run_tests_studies_without_study_folders_is_just_the_tests_folder() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw)
        (repo / "scripts").mkdir()
        shutil.copy(RUN_TESTS, repo / "scripts" / "run_tests.sh")
        out = _dry_run(repo, "--studies").split()
    assert "--import-mode=importlib" not in out and out[-3:] == ["studies", "-rs", "tests"]


def test_pytest_accepts_the_studies_command_line() -> None:
    result = _pytest(
        "--collect-only",
        "-q",
        "--import-mode=importlib",
        "--tier",
        "studies",
        "-rs",
        "tests/tiers.py",
    )
    assert "unrecognized arguments" not in result.stderr, result.stderr
