"""Compare the P1 17-fold new-stack runs with the P0 legacy references, exactly (P1 exit criterion).

Compares, for each run (``core17_ensemble_sp500`` and its DSR-threshold-0 variant):
P0 legacy reference vs new stack numba, P0 vs new stack Rust, and new numba vs new Rust.
Every JSON leaf that differs is listed with both values and the absolute difference; the OOS
equity curves are compared as raw CSV text, as parsed float64 arrays (``np.array_equal``) and by
sha256 of the array bytes. Fields that legitimately differ between runs (versions, timings, paths,
engine proof) are excluded and reported separately.

Example:
    python scripts/new_stack_parity_compare.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
VBT = REPO_ROOT / "research" / "programme" / "vectorbt-engine"
P0 = VBT / "legacy-references"
P1 = VBT / "p1-parity" / "runs17fold"
RUNS = ("core17_ensemble_sp500", "core17_ensemble_sp500_dsr0")
#: Keys that describe the run environment, not its results.
EXCLUDE = {
    "engine",
    "started",
    "versions",
    "load_seconds",
    "wall_seconds",
    "wall_seconds_total",
    "peak_rss_mb",
    "rss_trace",
    "engine_requested",
    "vbt_engine_setting",
    "rust_available",
    "engine_proof",
    "ggtrader_file",
    "oos_equity_sha256",
    "legacy_tag",
    "legacy_commit",
    "run",
    "memory_hygiene",
}
#: Input descriptions: compared and reported separately from the result fields.
INPUT_KEYS = ("config", "spec", "tape")


def diff(
    a: Any, b: Any, path: str, out: list[dict[str, Any]], skip: frozenset = frozenset()
) -> None:
    """Collect every leaf difference between two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not path and k in skip:
                continue
            if k not in a or k not in b:
                out.append(
                    {
                        "path": f"{path}/{k}",
                        "a": a.get(k, "<missing>"),
                        "b": b.get(k, "<missing>"),
                        "abs_diff": None,
                    }
                )
            else:
                diff(a[k], b[k], f"{path}/{k}", out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "a": f"len {len(a)}", "b": f"len {len(b)}", "abs_diff": None})
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, skip)
    elif a != b:
        both_num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
        out.append({"path": path, "a": a, "b": b, "abs_diff": abs(a - b) if both_num else None})


def equity(path: Path) -> tuple[pd.Series, str]:
    """Return the OOS equity series and the raw CSV text hash."""
    raw = path.read_bytes()
    ser = pd.read_csv(path, index_col=0, parse_dates=True)["oos_equity"]
    return ser, hashlib.sha256(raw).hexdigest()


def compare_equity(pa: Path, pb: Path) -> dict[str, Any]:
    """Exact comparison of two OOS equity CSVs."""
    sa, ha = equity(pa)
    sb, hb = equity(pb)
    same_idx = bool(sa.index.equals(sb.index))
    res: dict[str, Any] = {
        "bars": [int(len(sa)), int(len(sb))],
        "csv_bytes_identical": ha == hb,
        "index_identical": same_idx,
    }
    if len(sa) == len(sb):
        va, vb = sa.to_numpy("float64"), sb.to_numpy("float64")
        res["array_equal"] = bool(np.array_equal(va, vb, equal_nan=True))
        res["max_abs_diff"] = float(np.nanmax(np.abs(va - vb))) if len(va) else 0.0
        res["n_differing_bars"] = int((va != vb).sum())
        res["sha256_a"] = hashlib.sha256(np.ascontiguousarray(va).tobytes()).hexdigest()
        res["sha256_b"] = hashlib.sha256(np.ascontiguousarray(vb).tobytes()).hexdigest()
    return res


def classify(diffs: list[dict[str, Any]], eq: dict[str, Any]) -> str:
    """Spec 7.1 class: bit-identical, documented 1-ulp dsr_value only, or real difference."""
    if not diffs and eq.get("array_equal") and eq.get("index_identical"):
        return "bit-identical"
    if (
        diffs
        and all(d["path"].endswith("/dsr_value") and (d["abs_diff"] or 1) < 1e-15 for d in diffs)
        and eq.get("array_equal")
        and eq.get("index_identical")
    ):
        return "bit-identical except documented 1-ulp dsr_value"
    return "REAL DIFFERENCE (stop and report)"


def main(argv: Optional[list[str]] = None) -> int:
    """Write the parity JSON and print a one-line verdict per pair."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(VBT / "p1-parity" / "p1-17fold-parity.json"))
    args = ap.parse_args(argv)
    result: dict[str, Any] = {"pairs": {}}
    for run in RUNS:
        refs = {
            "p0_legacy": (P0 / f"_legacy_reference_{run}.json", P0 / f"{run}_oos_equity.csv"),
            "new_numba": (
                P1 / "numba" / f"_newstack_numba_{run}.json",
                P1 / "numba" / f"{run}_oos_equity.csv",
            ),
            "new_rust": (
                P1 / "rust" / f"_newstack_rust_{run}.json",
                P1 / "rust" / f"{run}_oos_equity.csv",
            ),
        }
        data = {k: json.loads(v[0].read_text()) for k, v in refs.items() if v[0].exists()}
        for pair in (
            ("p0_legacy", "new_numba"),
            ("p0_legacy", "new_rust"),
            ("new_numba", "new_rust"),
        ):
            if pair[0] not in data or pair[1] not in data:
                continue
            a, b = data[pair[0]], data[pair[1]]
            out: list[dict[str, Any]] = []
            diff(a, b, "", out, frozenset(EXCLUDE) | frozenset(INPUT_KEYS))
            inputs: list[dict[str, Any]] = []
            diff({k: a.get(k) for k in INPUT_KEYS}, {k: b.get(k) for k in INPUT_KEYS}, "", inputs)
            eq = compare_equity(refs[pair[0]][1], refs[pair[1]][1])
            num = [d["abs_diff"] for d in out if d["abs_diff"] is not None]
            key = f"{run}: {pair[0]} vs {pair[1]}"
            result["pairs"][key] = {
                "classification": classify(out, eq),
                "n_json_leaf_differences": len(out),
                "max_abs_json_difference": max(num) if num else 0.0,
                "differences": out,
                "input_description_differences": inputs,
                "equity": eq,
                "ok_flags": [bool(a.get("ok")), bool(b.get("ok"))],
                "n_folds": [a.get("n_folds"), b.get("n_folds")],
                "gates": [a.get("gates"), b.get("gates")],
                "anchor_folds": [a.get("anchor_folds"), b.get("anchor_folds")],
                "oos": [a.get("oos"), b.get("oos")],
            }
            print(
                f"{key}: {result['pairs'][key]['classification']} "
                f"({len(out)} json diffs, equity max abs {eq.get('max_abs_diff')})"
            )
    result["runs"] = {}
    for eng in ("numba", "rust"):
        for run in RUNS:
            f = P1 / eng / f"_newstack_{eng}_{run}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            result["runs"][f"{eng}:{run}"] = {
                k: r.get(k)
                for k in (
                    "ok",
                    "versions",
                    "vbt_engine_setting",
                    "rust_available",
                    "engine_proof",
                    "wall_seconds",
                    "wall_seconds_total",
                    "load_seconds",
                    "peak_rss_mb",
                    "oos_equity_sha256",
                    "ggtrader_file",
                    "tape",
                    "gates",
                    "anchor_folds",
                )
            }
    ulps = []
    for key, pair in result["pairs"].items():
        for d in pair["differences"]:
            if d["path"].endswith("/dsr_value"):
                ulps.append(
                    {
                        "pair": key,
                        "path": d["path"],
                        "abs_diff": d["abs_diff"],
                        "ulps": d["abs_diff"] / float(np.spacing(abs(d["a"]))),
                    }
                )
    result["dsr_value_ulps"] = ulps
    Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
