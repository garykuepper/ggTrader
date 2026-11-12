"""Atomic file writes and JSON-safe result writing shared by the lab and by studies."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ggTrader.utils.json_safe import json_safe


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` so a reader never sees a half-written file.

    The bytes go to a temporary file in the same folder, then ``os.replace`` swaps it in. Parent
    folders are created.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically (see ``atomic_write_bytes``)."""
    return atomic_write_bytes(path, text.encode(encoding))


def write_json(path: str | Path, payload: Any, *, strict: bool = False, indent: int = 2) -> Path:
    """Write ``payload`` as JSON, atomically, with a trailing newline.

    Args:
        path: output file; parent folders are created.
        payload: the value to write.
        strict: ``False`` (default) makes the payload JSON-safe first (NaN/inf become null,
            numpy scalars and timestamps are converted). ``True`` writes it as given and raises
            ``ValueError`` on NaN/inf and ``TypeError`` on a value JSON cannot hold, so a bad
            number fails loudly instead of being silently nulled.
        indent: JSON indent.

    Returns:
        The written path.
    """
    body = payload if strict else json_safe(payload)
    text = json.dumps(body, indent=indent, allow_nan=not strict)
    return atomic_write_text(path, text + "\n")
