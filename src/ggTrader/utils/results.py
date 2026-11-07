"""Result JSON helpers: every cited number gets a raw artifact stamped with its provenance."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

PROVENANCE_KEY = "provenance"


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN/inf to None, numpy/pandas scalars to Python, timestamps to ISO."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        return _clean(obj.item())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


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
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({**_clean(payload), PROVENANCE_KEY: _clean(provenance)}, indent=2)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text + "\n")
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def read_result_json(path: str | Path) -> dict[str, Any]:
    """Read a result JSON; reject one with no provenance or no recorded ``year_freq``."""
    data = json.loads(Path(path).read_text())
    block = data.get(PROVENANCE_KEY)
    if not isinstance(block, dict):
        raise ValueError(f"{path} has no provenance block (a legacy-engine result?)")
    if not block.get("definitions", {}).get("year_freq"):
        raise ValueError(f"{path} provenance does not record year_freq")
    return data
