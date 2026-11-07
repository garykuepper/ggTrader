"""Study plumbing: repo root, sys.path bootstrap, study folder and driver loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ggTrader.utils import study


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repository with one study under research/<category>/<study>."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / ".git").mkdir()
    code = tmp_path / "research" / "cat" / "stud" / "code"
    (code / "tests").mkdir(parents=True)
    (tmp_path / "research" / "cat" / "_lib" / "tests").mkdir(parents=True)
    (code / "run.py").write_text("VALUE = 7\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return tmp_path


def test_repo_root_walks_up_to_pyproject_and_git(repo: Path) -> None:
    assert study.repo_root(repo / "research/cat/stud/code/run.py") == repo.resolve()


def test_repo_root_accepts_a_git_file_as_in_a_linked_worktree(repo: Path) -> None:
    (repo / ".git").rmdir()
    (repo / ".git").write_text("gitdir: /elsewhere\n")
    assert study.repo_root(repo / "research/cat/stud/code/run.py") == repo.resolve()


def test_repo_root_needs_both_markers(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "a").mkdir()
    with pytest.raises(FileNotFoundError):
        study.repo_root(tmp_path / "a" / "f.py")


def test_bootstrap_from_code_adds_study_code_then_category_lib(repo: Path) -> None:
    root = study.bootstrap(repo / "research/cat/stud/code/run.py")
    assert root == repo.resolve()
    code = str((repo / "research/cat/stud/code").resolve())
    lib = str((repo / "research/cat/_lib").resolve())
    assert sys.path[:2] == [code, lib]


def test_bootstrap_from_a_code_test_folder_finds_the_code_folder(repo: Path) -> None:
    study.bootstrap(repo / "research/cat/stud/code/tests/conftest.py")
    assert sys.path[0] == str((repo / "research/cat/stud/code").resolve())


def test_bootstrap_from_lib_adds_only_lib(repo: Path) -> None:
    before = list(sys.path)
    study.bootstrap(repo / "research/cat/_lib/tests/conftest.py")
    assert sys.path[0] == str((repo / "research/cat/_lib").resolve())
    assert sys.path[1:] == before


def test_bootstrap_is_idempotent(repo: Path) -> None:
    file = repo / "research/cat/stud/code/run.py"
    study.bootstrap(file)
    once = list(sys.path)
    study.bootstrap(file)
    assert sys.path == once


def test_bootstrap_without_a_lib_folder_adds_only_code(repo: Path) -> None:
    (repo / "research/cat/_lib/tests").rmdir()
    (repo / "research/cat/_lib").rmdir()
    before = len(sys.path)
    study.bootstrap(repo / "research/cat/stud/code/run.py")
    assert len(sys.path) == before + 1


def test_study_dir_is_the_folder_above_code(repo: Path) -> None:
    assert (
        study.study_dir(repo / "research/cat/stud/code/tests/conftest.py")
        == (repo / "research/cat/stud").resolve()
    )
    with pytest.raises(ValueError):
        study.study_dir(repo / "pyproject.toml")


def test_load_driver_names_the_module_and_runs_it(repo: Path) -> None:
    module = study.load_driver(repo / "research/cat/stud/code/run.py")
    assert module.VALUE == 7
    assert module.__name__ == "study_cat_stud_run"
    assert sys.modules["study_cat_stud_run"] is module


def test_load_driver_builds_a_fresh_module_each_call(repo: Path) -> None:
    path = repo / "research/cat/stud/code/run.py"
    assert study.load_driver(path) is not study.load_driver(path)


def test_load_driver_forgets_a_module_that_fails_to_import(repo: Path) -> None:
    path = repo / "research/cat/stud/code/bad.py"
    path.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        study.load_driver(path)
    assert "study_cat_stud_bad" not in sys.modules
