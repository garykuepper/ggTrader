"""Result JSON helpers: every cited number gets a raw artifact stamped with its provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ggTrader.utils.io import atomic_write_text, json_safe

PROVENANCE_KEY = "provenance"

#: Kept for older call sites; the function now lives in ``ggTrader.utils.io``.
_clean = json_safe


def write_result_json(
    path: str | Path, payload: dict[str, Any], *, provenance: dict[str, Any] | None = None
) -> Path:
    """Write ``payload`` plus a top-level ``provenance`` block, atomically.

    Args:
        path: output file; parent directories are created.
        payload: the result; it must not already contain ``provenance``.
        provenance: explicit block; default is ``ggTrader.engine.provenance.provenance()``.

    Returns:
        The written path.
    """
    if PROVENANCE_KEY in payload:
        raise ValueError("payload already has a 'provenance' key; it is stamped here")
    if provenance is None:
        from ggTrader.engine.provenance import provenance as build_provenance

        provenance = build_provenance()
    text = json.dumps({**json_safe(payload), PROVENANCE_KEY: json_safe(provenance)}, indent=2)
    return atomic_write_text(path, text + "\n")


def read_result_json(path: str | Path) -> dict[str, Any]:
    """Read a result JSON; reject one with no provenance or no recorded ``year_freq``."""
    data = json.loads(Path(path).read_text())
    block = data.get(PROVENANCE_KEY)
    if not isinstance(block, dict):
        raise ValueError(f"{path} has no provenance block (a legacy-engine result?)")
    if not block.get("definitions", {}).get("year_freq"):
        raise ValueError(f"{path} provenance does not record year_freq")
    return data
