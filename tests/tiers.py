"""The test tiers: which markers each tier leaves out (one source for conftest, CI, scripts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Markers whose tests stay out of the FAST CORE (the developer loop). The same names are
#: registered in pyproject.toml; tests/test_tiers.py fails if the two lists differ.
FAST_EXCLUDED: tuple[str, ...] = ("integration", "slow", "numba", "tripwire", "study", "tooling")

#: tier -> pytest marker expression. FULL is the merge, deploy and vectorbt-upgrade gate: it is
#: also what a bare ``pytest`` runs (the pyproject default), so nothing is skipped by accident.
TIER_MARKEXPR: dict[str, str] = {
    "fast": " and ".join(f"not {name}" for name in FAST_EXCLUDED),
    "full": "not integration",
    "studies": "study and not integration",
}

#: Fewest tests the study half of ``--full`` / ``--studies`` must collect (the root conftest's
#: ``--min-collected``). Every slice that moves study tests raises it to the count it measured, so
#: a path typo or a collection failure cannot turn the study half into a silent zero.
STUDIES_MIN_COLLECTED = 207

#: Where study tests live: next to the study code, or in a category's shared ``_lib``. These are
#: passed to pytest explicitly (never the bare ``research`` folder: it holds frozen test_*.py
#: files that import deleted modules) and run with ``--import-mode=importlib`` because study test
#: file names may repeat across studies.
STUDY_TEST_GLOBS: tuple[str, ...] = ("research/*/*/code/tests", "research/*/_lib/tests")


def study_test_dirs(root: Path) -> list[str]:
    """The study test folders that exist under ``root``, repo-relative and sorted."""
    found = {p.relative_to(root).as_posix() for glob in STUDY_TEST_GLOBS for p in root.glob(glob)}
    return sorted(p for p in found if (root / p).is_dir())


def apply_tier(option: Any, tier: str | None) -> None:
    """Point ``option.markexpr`` at a tier's marker expression (``None`` changes nothing)."""
    if tier is None:
        return
    if tier not in TIER_MARKEXPR:
        raise ValueError(f"unknown test tier {tier!r}; choose from {sorted(TIER_MARKEXPR)}")
    option.markexpr = TIER_MARKEXPR[tier]
