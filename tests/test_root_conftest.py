"""The repo-root conftest.py reaches tests that live outside tests/, and run_tests.sh parses.

Study tests will live under research/<category>/<study>/code/tests. They must get the same
`_no_dotenv` safety fixture (a test must never read the real `.env`), so the fixture lives in the
repo-root conftest.py. These tests prove it with a throwaway test file collected from outside
tests/, and prove that `scripts/run_tests.sh --studies` builds a command pytest accepts.
"""

from __future__ import annotations

import os
import re
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


#: A test collected from outside tests/ that asserts the membership fixture is selected for it.
MEMBERSHIP_INNER_TEST = textwrap.dedent(
    """
    import os

    from ggTrader.data.membership import selected_source


    def test_membership_is_the_frozen_fixture():
        assert os.environ["GGT_MEMBERSHIP"] == "fixture:sp500_v1_2026-06-02"
        assert selected_source().vintage().kind == "fixture"
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
    for needle in (
        "_no_dotenv",
        "_membership_is_a_fixture",
        "--tier",
        "--update-golden",
        "--golden-reason",
        "Agg",
    ):
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


def test_membership_is_the_frozen_fixture_in_this_test() -> None:
    from ggTrader.data.membership import selected_source

    assert os.environ["GGT_MEMBERSHIP"] == "fixture:sp500_v1_2026-06-02"
    assert selected_source().vintage().kind == "fixture"


def test_membership_is_the_frozen_fixture_in_a_test_outside_tests() -> None:
    scratch = Path(tempfile.mkdtemp(prefix=".sentinel_", dir=ROOT))
    try:
        (scratch / "test_inner.py").write_text(MEMBERSHIP_INNER_TEST)
        result = _pytest(str(scratch / "test_inner.py"), "-q")
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_the_sentinel_has_teeth_without_the_fixture_the_key_would_load() -> None:
    """Outside the repo root the shared conftest does not apply, so the key does leak there."""
    with tempfile.TemporaryDirectory() as raw:
        (Path(raw) / "test_inner.py").write_text(INNER_TEST)
        result = _pytest("-q", "--rootdir", raw, "-c", os.devnull, raw, cwd=Path(raw))
    assert result.returncode != 0 and "GGT_SENTINEL_KEY" in result.stdout


def _dry_run(repo: Path, *args: str) -> str:
    """The pytest command line(s) run_tests.sh would run in ``repo``, one per line."""
    return _dry_run_result(repo, *args).stdout.strip()


def _dry_run_result(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GGT_TEST_DRY_RUN": "1", "GGT_PYTHON": sys.executable}
    return subprocess.run(
        ["bash", str(repo / "scripts" / "run_tests.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _scratch_repo(root: Path, study_dirs: tuple[str, ...] = (), minimum: int | None = None) -> Path:
    """A throwaway repo holding run_tests.sh, tests/tiers.py and some study test folders."""
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    shutil.copy(RUN_TESTS, root / "scripts" / "run_tests.sh")
    tiers_text = (ROOT / "tests" / "tiers.py").read_text()
    # The scratch repo never inherits the real floor (it grows with every study move).
    tiers_text = re.sub(
        r"STUDIES_MIN_COLLECTED = \d+", f"STUDIES_MIN_COLLECTED = {minimum or 0}", tiers_text
    )
    (root / "tests" / "tiers.py").write_text(tiers_text)
    for folder in study_dirs:
        (root / folder).mkdir(parents=True)
    return root


def test_run_tests_picks_the_tier_from_the_first_argument() -> None:
    assert "--tier fast" in _dry_run(ROOT) and "--tier fast" in _dry_run(ROOT, "--fast", "-q")
    assert "--tier full" in _dry_run(ROOT, "--full", "-q")
    assert "--tier" not in _dry_run(ROOT, "tests/paper", "-q")  # unchanged pass-through
    assert "--tier" not in _dry_run(ROOT, "-q")


FOLDERS = ("research/alloc/static/code/tests", "research/stock/_lib/tests")


def test_run_tests_studies_runs_tests_then_the_study_half_on_explicit_paths() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS)
        first, second = _dry_run(repo, "--studies", "-q").splitlines()
    one, two = first.split(), second.split()
    assert one[-5:] == ["--tier", "studies", "-rs", "tests", "-q"]
    assert "--import-mode=importlib" not in one
    assert "--import-mode=importlib" in two and "not integration" in second
    assert two[two.index("--min-collected") + 1] == "0"
    assert FOLDERS[0] in two and FOLDERS[1] in two
    assert "research" not in two and "tests" not in two and "--tier" not in two


def test_run_tests_full_runs_tests_then_the_same_study_half() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS)
        first, second = _dry_run(repo, "--full").splitlines()
    assert first.split()[-4:] == ["--tier", "full", "-rs", "tests"]
    assert "--import-mode=importlib" in second and FOLDERS[0] in second.split()


def test_run_tests_without_study_folders_skips_the_study_half_with_a_note() -> None:
    with tempfile.TemporaryDirectory() as raw:
        done = _dry_run_result(_scratch_repo(Path(raw)), "--studies")
    assert done.returncode == 0
    assert (
        done.stdout.split()[-3:] == ["studies", "-rs", "tests"]
        and len(done.stdout.splitlines()) == 1
    )
    assert "skipping the study half" in done.stderr


def test_run_tests_fails_when_a_minimum_is_promised_but_no_study_folder_exists() -> None:
    with tempfile.TemporaryDirectory() as raw:
        done = _dry_run_result(_scratch_repo(Path(raw), minimum=7), "--studies")
    assert done.returncode != 0 and "STUDIES_MIN_COLLECTED = 7" in done.stderr


def test_run_tests_passes_the_promised_minimum_to_the_study_half() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS, minimum=7)
        second = _dry_run(repo, "--studies").splitlines()[1].split()
    assert second[second.index("--min-collected") + 1] == "7"


def test_run_tests_with_a_path_is_one_targeted_run_without_the_study_half() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS)
        (repo / "tests" / "test_one.py").write_text("")
        out = _dry_run(repo, "--full", "tests/test_one.py::test_x", "-q").splitlines()
    assert len(out) == 1 and "--tier full" in out[0] and "importlib" not in out[0]
    assert "tests/test_one.py::test_x" in out[0]


def test_run_tests_option_values_are_not_taken_for_target_paths() -> None:
    """`-k research`, `-m tests` and `--ignore tests/x` name a filter, not a path to run."""
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS)
        (repo / "tests" / "x").mkdir()
        for extra in (["-k", "research"], ["-m", "tests"], ["--ignore", "tests/x"]):
            lines = _dry_run(repo, "--studies", *extra).splitlines()
            assert len(lines) == 2, (extra, lines)  # still the full two-part studies run
            assert "importlib" in lines[1], (extra, lines)
        # an attached value (`-krequires`, `--ignore=tests/x`) is one option and never a path
        lines = _dry_run(repo, "--studies", "--ignore=tests/x", "-krequires").splitlines()
        assert len(lines) == 2, lines


def test_run_tests_never_passes_the_bare_research_folder() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), FOLDERS)
        for tier in ("--full", "--studies"):
            for line in _dry_run(repo, tier).splitlines():
                assert "research" not in line.split(), line


def test_the_study_folders_run_tests_scans_are_the_ones_in_tiers() -> None:
    from tests import tiers

    assert tiers.STUDY_TEST_GLOBS == ("research/*/*/code/tests", "research/*/_lib/tests")
    with tempfile.TemporaryDirectory() as raw:
        repo = _scratch_repo(Path(raw), (*FOLDERS, "research/alloc/static/code/other"))
        assert tiers.study_test_dirs(repo) == sorted(FOLDERS)


def test_min_collected_fails_a_run_that_collects_too_few_tests() -> None:
    scratch = Path(tempfile.mkdtemp(prefix=".mincollected_", dir=ROOT))
    try:
        (scratch / "test_two.py").write_text(
            "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"
        )
        enough = _pytest(str(scratch / "test_two.py"), "-q", "--min-collected", "2")
        too_few = _pytest(str(scratch / "test_two.py"), "-q", "--min-collected", "3")
        default = _pytest(str(scratch / "test_two.py"), "-q")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    assert enough.returncode == 0 and "2 passed" in enough.stdout
    assert default.returncode == 0
    assert too_few.returncode != 0 and "--min-collected is 3" in too_few.stderr + too_few.stdout


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
