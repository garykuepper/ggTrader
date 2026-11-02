"""Classify saved WFO fold rows by gate outcome and fallback usage."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

#: Mutually exclusive fold classes, in reporting order.
FOLD_CLASSES = ("gate_fail", "gates_pass_anchor", "gates_pass_winner", "anomaly")


def classify_fold(row: Dict[str, Any]) -> str:
    """Return the fold's class from its saved ``gates_passed`` / ``used_anchor`` flags.

    ``gates_pass_anchor`` implies the system was halted entering the fold, because
    ``run_wfo`` deploys the anchor only when gates fail or a halt is active.
    A row where neither flag is true, or a flag is absent, is an ``anomaly``:
    the harness cannot produce it, so it must not be silently counted.
    """
    gates = row.get("gates_passed")
    anchor = row.get("used_anchor")
    if not isinstance(gates, bool) or not isinstance(anchor, bool):
        return "anomaly"  # absent, null or non-boolean flags are never guessed at
    if not gates:
        return "gate_fail" if anchor else "anomaly"
    return "gates_pass_anchor" if anchor else "gates_pass_winner"


def summarize_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count fold classes, NaN/None WFE values, and gaps in ``fold_num`` for one run."""
    out: Dict[str, Any] = {"n_folds": len(rows)}
    for cls in FOLD_CLASSES:
        out[cls] = 0
    nan_wfe = none_wfe = wfe_absent = 0
    for row in rows:
        out[classify_fold(row)] += 1
        if "wfe" not in row:
            wfe_absent += 1
            continue
        wfe = row["wfe"]
        if wfe is None:
            none_wfe += 1
        elif isinstance(wfe, float) and math.isnan(wfe):
            nan_wfe += 1
    nums = sorted(int(r["fold_num"]) for r in rows if r.get("fold_num") is not None)
    missing = [n for n in range(1, nums[-1] + 1) if n not in set(nums)] if nums else []
    out["nan_wfe"] = nan_wfe
    out["none_wfe"] = none_wfe
    out["wfe_absent"] = wfe_absent
    out["fold_numbers_missing"] = missing
    return out


def regime_from_ancestry(first_after_fix: Optional[bool], last_after_fix: Optional[bool]) -> str:
    """Label an artifact's DSR regime from where its commits sit relative to the fix commit.

    ``first_after_fix`` / ``last_after_fix`` say whether the artifact's first / last commit
    descends from the fix commit (``None`` = the Git query failed). Only "committed before the
    fix" proves the old gate; a later commit only bounds when the file entered Git, so it is
    reported as ``committed_after_fix`` and never as proof of the corrected gate.
    """
    if first_after_fix is None or last_after_fix is None:
        return "unknown"
    if not last_after_fix:
        return "pre_dsr_fix"
    if first_after_fix:
        return "committed_after_fix"
    return "unknown"
