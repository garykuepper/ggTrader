"""Compare a fresh legacy 17-fold reference JSON with an older results JSON, fold by fold.

Used in P0 to document how the fresh frozen-tape legacy run differs from the historical
``research/audits/rebaseline/_pit_rebaseline_20260923.json`` figures (Sharpe 0.592 and so on).

Example:
    python scripts/legacy_reference_compare.py --help
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "research/programme/vectorbt-engine/legacy-references"
DEFAULT_NEW = REFS / "_legacy_reference_core17_ensemble_sp500.json"
DEFAULT_OLD = REPO_ROOT / "research/audits/rebaseline/_pit_rebaseline_20260923.json"
# Scratch by default; the committed copy sits in REFS and is rewritten only by an explicit --out.
DEFAULT_OUT = (
    Path(tempfile.gettempdir()) / "ggtrader_evidence" / "_legacy_reference_core17_vs_20260923.json"
)
HEADLINE = ("sharpe", "cagr_pct", "max_drawdown_pct")


def compare(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """Headline and per-fold differences between the fresh run and the older record."""
    head = {
        "new": {k: new["oos"][k] for k in HEADLINE},
        "old": {k: old[k] for k in HEADLINE},
        "spy_sharpe_new": new["spy_oos_index"]["sharpe"],
        "spy_sharpe_old": old["spy_sharpe"],
        "gates_new": new["gates"],
        "gates_old": old["gate_pass_str"],
        "elapsed_seconds_new": new["wall_seconds"],
        "elapsed_seconds_old": old["elapsed_sec"],
    }
    head["delta"] = {k: head["new"][k] - head["old"][k] for k in HEADLINE}
    folds = []
    for fn, fo in zip(new["fold_table"], old["fold_rows"]):
        folds.append(
            {
                "fold": fo["fold_num"],
                "test_start": fo["test_start"],
                "winner_same": fn["winner_combo"] == fo["winner_combo"],
                "gates_new": fn["gates_passed"],
                "gates_old": fo["gates_passed"],
                "ndh_new": fn["ndh_passed"],
                "ndh_old": fo["ndh_passed"],
                "dsr_new": fn["dsr_passed"],
                "dsr_old": fo["dsr_passed"],
                "anchor_new": fn["used_anchor"],
                "anchor_old": fo["used_anchor"],
                "oos_sharpe_new": fn["oos_sharpe"],
                "oos_sharpe_old": fo["oos_sharpe"],
            }
        )
    head["folds"] = folds
    head["winners_same"] = f"{sum(f['winner_same'] for f in folds)}/{len(folds)}"
    head["ndh_same"] = f"{sum(f['ndh_new'] == f['ndh_old'] for f in folds)}/{len(folds)}"
    head["dsr_same"] = f"{sum(f['dsr_new'] == f['dsr_old'] for f in folds)}/{len(folds)}"
    return head


def _rel(path: str) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def main(argv: Optional[list[str]] = None) -> int:
    """Write the comparison JSON and print a short summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", default=str(DEFAULT_NEW), help="fresh reference JSON")
    ap.add_argument("--old", default=str(DEFAULT_OLD), help="older results JSON")
    ap.add_argument("--old-key", default="ensemble", help="section of the old JSON to read")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="comparison JSON to write")
    args = ap.parse_args(argv)
    new = json.loads(Path(args.new).read_text())
    old = json.loads(Path(args.old).read_text())[args.old_key]
    result = compare(new, old)
    result["sources"] = {
        "new": _rel(args.new),
        "old": _rel(args.old),
        "old_key": args.old_key,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
