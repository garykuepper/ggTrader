"""Attribute saved WFO fold records to gate failure, anchor fallback or winner deployment.

Reads only existing result JSON under research/ (per-study folders, see ARTIFACT_PATHS);
reruns nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ggTrader.lab.wfo_attribution import regime_from_ancestry, summarize_run  # noqa: E402

#: Repo-relative location of each source artifact, keyed by the file name used in RUN_SOURCES
#: and in the output's ``file`` field. Each study keeps its results in its own folder.
ARTIFACT_PATHS: Dict[str, str] = {
    "_anchor_fix_reproduction_results.json": (
        "research/audits/rebaseline/_anchor_fix_reproduction_results.json"
    ),
    "_audit_reruns_20261019.json": "research/audits/production-replica/_audit_reruns_20261019.json",
    "_cross_asset_sleeve_results.json": (
        "research/allocation/cross-asset-sleeve/_cross_asset_sleeve_results.json"
    ),
    "_ic_kelly_rebaseline_20260923.json": (
        "research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json"
    ),
    "_month_end_treasury_results.json": (
        "research/allocation/month-end-treasury/_month_end_treasury_results.json"
    ),
    "_pit_rebaseline_20260923.json": "research/audits/rebaseline/_pit_rebaseline_20260923.json",
    "_rebaseline_corrected_tape_20260822.json": (
        "research/audits/rebaseline/_rebaseline_corrected_tape_20260822.json"
    ),
}
DEFAULT_OUT = PROJECT_ROOT / "research/programme/wfo-methods/_wfo_fold_attribution_results.json"
#: Commit that corrected the DSR gate (committed 2026-10-19T16:04:44-07:00).
DSR_FIX_COMMIT = "09f9f40"

#: (artifact file, key path to the ``fold_rows`` list). Keys are dict keys.
RUN_SOURCES: List[Tuple[str, Tuple[str, ...]]] = [
    ("_anchor_fix_reproduction_results.json", ("core", "fold_rows")),
    ("_audit_reruns_20261019.json", ("fomc_drift", "wfo", "fold_rows")),
    ("_audit_reruns_20261019.json", ("xs_momentum", "fold_rows")),
    ("_audit_reruns_20261019.json", ("dual_momentum", "fold_rows")),
    ("_cross_asset_sleeve_results.json", ("wfo", "fold_rows")),
    ("_ic_kelly_rebaseline_20260923.json", ("ensemble", "fold_rows")),
    ("_ic_kelly_rebaseline_20260923.json", ("ensemble_ic", "fold_rows")),
    ("_ic_kelly_rebaseline_20260923.json", ("ensemble_kelly", "fold_rows")),
    ("_month_end_treasury_results.json", ("wfo", "fold_rows")),
    ("_pit_rebaseline_20260923.json", ("ensemble", "fold_rows")),
    ("_pit_rebaseline_20260923.json", ("ensemble_ic", "fold_rows")),
    ("_pit_rebaseline_20260923.json", ("ensemble_kelly", "fold_rows")),
    ("_rebaseline_corrected_tape_20260822.json", ("core", "fold_rows")),
]


def _git(*args: str) -> Optional[str]:
    """Run a read-only git command in the repo; return stdout or None on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _is_ancestor(ancestor: str, descendant: str) -> Optional[bool]:
    """Return True/False for ancestry, or None when Git cannot answer (error, bad ref)."""
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return True
    return False if proc.returncode == 1 else None


#: Gate versions stated in repo documents (not inferable from Git), keyed by (file, run key).
DOCUMENTED_GATE_VERSION: Dict[Tuple[str, str], Dict[str, str]] = {
    ("_audit_reruns_20261019.json", key): {
        "gate_version": "corrected harness (DSR fix 09f9f40)",
        "source": "research/audits/production-replica/2026-10-19-production-replica-audit-nogo.md "
        "('Momentum and FOMC corrected-harness reruns')",
    }
    for key in ("xs_momentum", "dual_momentum", "fomc_drift")
}


def artifact_regime(rel_path: str) -> Dict[str, Any]:
    """Classify an artifact's DSR regime from Git history; a bound, not a proof.

    Only "last commit predates the fix" shows the old gate. A commit after the fix says when the
    file entered Git, not which gate generated the run, so it is reported as
    ``committed_after_fix`` (see ``regime_from_ancestry``).
    """
    # --diff-filter=r drops pure-rename commits (the 2026-11-05 move into research/), so
    # moving a file does not count as a new commit and the regime label stays as it was.
    log = _git("log", "--format=%H", "--follow", "--diff-filter=r", "--", rel_path)
    commits = log.splitlines() if log else []
    if not commits:
        return {"regime": "unknown", "basis": "artifact has no git history", "commits": []}
    first, last = commits[-1], commits[0]
    regime = regime_from_ancestry(
        _is_ancestor(DSR_FIX_COMMIT, first), _is_ancestor(DSR_FIX_COMMIT, last)
    )
    basis = {
        "pre_dsr_fix": "last artifact commit predates the DSR fix",
        "committed_after_fix": "artifact first committed after the DSR fix; the gate version "
        "that generated the run is not proven by Git",
        "unknown": "ancestry unclear (straddles the fix, or a Git query failed)",
    }[regime]
    return {
        "regime": regime,
        "basis": basis,
        "first_commit": first[:7],
        "last_commit": last[:7],
        "n_commits": len(commits),
    }


def _dig(obj: Any, path: Tuple[str, ...]) -> Any:
    """Follow a dict key path; raise KeyError naming the failing step."""
    for key in path:
        obj = obj[key]
    return obj


def build_results(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Build the attribution results object for every configured run."""
    runs: List[Dict[str, Any]] = []
    for filename, path in RUN_SOURCES:
        run: Dict[str, Any] = {"file": filename, "json_path": "/".join(path)}
        try:
            rows = _dig(json.loads((project_root / ARTIFACT_PATHS[filename]).read_text()), path)
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            run["error"] = f"{type(exc).__name__}: {exc}"
            runs.append(run)
            continue
        run["summary"] = summarize_run(rows)
        run["has_halted_field"] = bool(rows) and all("halted" in r for r in rows)
        run.update(artifact_regime(ARTIFACT_PATHS[filename]))
        documented = DOCUMENTED_GATE_VERSION.get((filename, path[0]))
        if documented:
            run["documented_gate_version"] = documented
        runs.append(run)
    return {
        "schema_version": 1,
        "dsr_fix_commit": DSR_FIX_COMMIT,
        "note": (
            "Counts describe saved folds only. The undeployed winner's performance, the "
            "halt state entering a fold, and folds skipped for missing training metrics "
            "are not recoverable from these rows."
        ),
        "runs": runs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Write the attribution JSON and print a one-line summary per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results JSON path")
    args = parser.parse_args(argv)
    results = build_results()
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for run in results["runs"]:
        s = run.get("summary")
        label = f"{run['file']}:{run['json_path']}"
        if s is None:
            print(f"{label}  ERROR {run['error']}")
            continue
        print(
            f"{label}  n={s['n_folds']} fail={s['gate_fail']} "
            f"pass_anchor={s['gates_pass_anchor']} pass_winner={s['gates_pass_winner']} "
            f"anomaly={s['anomaly']} [{run['regime']}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
