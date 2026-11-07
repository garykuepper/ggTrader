"""Shared path and window constants."""

from __future__ import annotations

from ggTrader.utils import paths


def test_repo_root_holds_pyproject_and_git() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / ".git").exists()
    assert paths.REPO_ROOT == paths.find_project_root()


def test_the_17_fold_window_is_pinned() -> None:
    assert (paths.EVAL_START_17, paths.EVAL_END_17) == ("2021-01-31", "2026-04-30")
