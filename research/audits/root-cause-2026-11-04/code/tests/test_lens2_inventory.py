"""The Lens 2 build script finds its inputs: every inventory artifact resolves to a real file."""

from __future__ import annotations

from pathlib import Path

import pytest

from ggTrader.utils.study import load_driver, repo_root, study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"
ROOT = repo_root(__file__)
lens2 = load_driver(CODE / "lens2_build.py")


def test_lens2_root_is_the_repository_root() -> None:
    # A root inside research/ would write lens output into the study folder.
    assert lens2.ROOT == ROOT
    assert (lens2.ROOT / "pyproject.toml").is_file()
    assert "research" not in lens2.ROOT.relative_to(ROOT).parts


def test_lens2_resolves_every_inventory_artifact() -> None:
    inventory = lens2.load(lens2.INVENTORY_JSON)
    listed = {r["raw_artifact"] for r in inventory if r.get("raw_artifact")}
    for art in listed:
        loc = lens2.ARTIFACT_LOCATIONS.get(Path(art).name, art)
        if loc.startswith("results/"):
            continue  # gitignored run output; absent on a fresh clone by design
        assert (ROOT / loc).is_file(), f"{art} -> {loc} missing"
    for key in lens2.HANDLERS:
        if not key.startswith("results/"):
            assert key in lens2.ARTIFACT_LOCATIONS.values(), key
