"""Frozen golden files for the engine tests: stored values, provenance and the update gate.

Goldens are regenerated only with ``pytest tests/engine --update-golden --golden-reason "..."``.
The reason is written into the file's provenance block and must also appear in
``docs/changelog.md`` (checked by ``test_goldens_are_changelogged``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden(name: str) -> dict[str, Any]:
    """The stored ``data`` of a golden file."""
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(f"golden {path.name} is missing; generate it with --update-golden")
    return json.loads(path.read_text())["data"]


def save_golden(name: str, data: dict[str, Any], reason: str) -> None:
    """Write a golden file stamped with the provenance of the stack that produced it."""
    from ggTrader.engine.provenance import provenance

    block = provenance()
    block["reason"] = reason
    GOLDEN_DIR.mkdir(exist_ok=True)
    text = json.dumps({"provenance": block, "data": data}, indent=1, sort_keys=True)
    (GOLDEN_DIR / f"{name}.json").write_text(text + "\n")


def check_or_update(
    request: pytest.FixtureRequest, name: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Return the stored golden, or rewrite it when ``--update-golden`` is given."""
    if request.config.getoption("--update-golden"):
        reason = request.config.getoption("--golden-reason")
        if not reason:
            pytest.fail(
                "--update-golden requires --golden-reason '<why>' (also put it in the changelog)"
            )
        save_golden(name, data, reason)
    return load_golden(name)
