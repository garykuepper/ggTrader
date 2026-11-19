"""Encode and compare ordered, type-preserving characterization records."""

import difflib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RTOL = 1e-9


def encode(value: object) -> object:
    """Convert a supported value to a JSON-compatible tagged tree."""
    if value is pd.NaT:
        return {"$type": "pandas.NaT"}
    if isinstance(value, pd.Timestamp):
        return {"$type": "pandas.Timestamp", "v": value.isoformat()}
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            raise TypeError(f"Unsupported type: {type(value).__name__}")
        return {"$type": f"numpy.{type(value).__name__}", "v": encode(item)}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isnan(value):
            return {"$float": "NaN"}
        if math.isinf(value):
            return {"$float": "Infinity" if value > 0 else "-Infinity"}
        return value
    if type(value) is dict:
        return {"$dict": [[encode(k), encode(v)] for k, v in value.items()]}
    if type(value) is list:
        return [encode(item) for item in value]
    if type(value) is tuple:
        return {"$tuple": [encode(item) for item in value]}
    if type(value) in (set, frozenset):
        return {"$set": sorted((encode(item) for item in value), key=dumps)}
    if isinstance(value, pd.Series):
        return {
            "$series": {
                "name": encode(value.name),
                "dtype": str(value.dtype),
                "index": [encode(x) for x in value.index],
                "values": [encode(x) for x in value.tolist()],
            }
        }
    if isinstance(value, pd.DataFrame):
        return {
            "$frame": {
                "columns": [encode(x) for x in value.columns],
                "dtypes": [str(x) for x in value.dtypes],
                "index": [encode(x) for x in value.index],
                "rows": [
                    [encode(x) for x in row] for row in value.itertuples(index=False, name=None)
                ],
            }
        }
    if isinstance(value, np.ndarray):
        return {"$array": {"dtype": str(value.dtype), "values": encode(value.tolist())}}
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def dumps(encoded: object) -> str:
    """Render an encoded tree as canonical text."""
    return json.dumps(encoded, indent=1, ensure_ascii=False) + "\n"


def compare(actual: object, golden: object, *, rtol: float, exact: bool) -> list[str]:
    """Return paths of differences between encoded trees."""
    differences: list[str] = []

    def walk(a: object, b: object, path: str) -> None:
        if type(a) is not type(b):
            differences.append(path or "$")
        elif isinstance(a, dict):
            if list(a) != list(b):
                differences.append(path or "$")
            elif list(a) == ["$dict"]:
                left, right = a["$dict"], b["$dict"]
                if len(left) != len(right):
                    differences.append(path or "$")
                for i, (lp, rp) in enumerate(zip(left, right)):
                    if lp[0] != rp[0] or type(lp[0]) is not type(rp[0]):
                        differences.append(f"{path}.$dict[{i}]".lstrip("."))
                    else:
                        key = lp[0]
                        child = (
                            f"{path}.{key}".lstrip(".")
                            if isinstance(key, str)
                            else f"{path}.$dict[{i}]".lstrip(".")
                        )
                        walk(lp[1], rp[1], child)
            else:
                for key in a:
                    walk(a[key], b[key], f"{path}.{key}" if path else key)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(path or "$")
            for i, (left, right) in enumerate(zip(a, b)):
                walk(left, right, f"{path}[{i}]")
        elif isinstance(a, float) and not exact:
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
                differences.append(path or "$")
        elif isinstance(a, float) and exact:
            if a != b or math.copysign(1.0, a) != math.copysign(1.0, b):
                differences.append(path or "$")
        elif a != b:
            differences.append(path or "$")

    walk(actual, golden, "")
    return differences


def check_golden(path: Path, actual: object) -> None:
    """Compare a value with a golden file, or regenerate it on request."""
    encoded = encode(actual)
    actual_text = dumps(encoded)
    if os.environ.get("GGT_REGEN_CHARACTERIZATION") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_text, encoding="utf-8")
        pytest.fail(f"regenerated {path}; review the diff and rerun")
    if not path.exists():
        pytest.fail(
            f"missing {path}; regenerate with "
            "GGT_REGEN_CHARACTERIZATION=1 scripts/run_tests.sh tests/"
        )
    golden_text = path.read_text(encoding="utf-8")
    golden = json.loads(golden_text)
    paths = compare(
        encoded, golden, rtol=RTOL, exact=os.environ.get("GGT_CHARACTERIZATION_EXACT") == "1"
    )
    if paths:
        diff = difflib.unified_diff(
            dumps(golden).splitlines(),
            actual_text.splitlines(),
            fromfile="golden",
            tofile="actual",
            lineterm="",
        )
        pytest.fail("differing paths: " + ", ".join(paths) + "\n" + "\n".join(list(diff)[:60]))
