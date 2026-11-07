"""The test tiers: which markers each tier leaves out (one source for conftest, CI, scripts)."""

from __future__ import annotations

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


def apply_tier(option: Any, tier: str | None) -> None:
    """Point ``option.markexpr`` at a tier's marker expression (``None`` changes nothing)."""
    if tier is None:
        return
    if tier not in TIER_MARKEXPR:
        raise ValueError(f"unknown test tier {tier!r}; choose from {sorted(TIER_MARKEXPR)}")
    option.markexpr = TIER_MARKEXPR[tier]
