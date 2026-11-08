"""Check that every saved result the fold-attribution driver reads exists in the research tree."""

from pathlib import Path

import pytest

from ggTrader.utils.study import load_driver, repo_root, study_dir

pytestmark = pytest.mark.study

CODE_DIR = study_dir(__file__) / "code"


def test_fold_attribution_sources_exist() -> None:
    attribution = load_driver(CODE_DIR / "run_fold_attribution.py")
    root = repo_root(__file__)
    for filename, _ in attribution.RUN_SOURCES:
        path = attribution.ARTIFACT_PATHS[filename]
        assert Path(path).name == filename
        assert (root / path).is_file(), path
    assert attribution.DEFAULT_OUT.parent.is_dir()
    assert attribution.PROJECT_ROOT == root
