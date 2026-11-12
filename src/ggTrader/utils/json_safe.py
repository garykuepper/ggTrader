"""Convert common values to a JSON-safe representation."""

import hashlib
import json
import math
from typing import Any


def json_safe(obj: Any) -> Any:
    """JSON-safe copy: NaN/inf to None, numpy/pandas scalars to Python, timestamps to ISO."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        return json_safe(obj.item())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def canonical_hash(value: Any) -> str:
    """Return SHA-256 of JSON-safe, sorted and compact JSON."""
    text = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
