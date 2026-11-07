"""P5b: before/after of the selection-window guards on the 17-fold SP500 ensemble runs.

Reads the P5 evidence (unguarded: ``p5-evidence/_p5_<run>.json``) and the guarded runs written
by ``scripts/p5_wfo_validation.py --out-dir p5-evidence/guarded`` (and, for attribution, the
zero-entry-rule-only runs in ``p5-evidence/guarded_zero_entry_only``). Writes
``p5-evidence/guarded/_p5b_before_after.json`` with, per run: winners per fold (before and after,
which folds changed), gates, anchor use, OOS Sharpe/CAGR/MaxDD and the SPY figures; prints the
same as markdown tables.

Example:
    .venv/bin/python scripts/p5b_guard_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "p5-evidence"
RUNS = ("core17_ensemble_sp500", "core17_ensemble_sp500_dsr0")
METRICS = ("sharpe", "cagr_pct", "max_drawdown_pct")


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def summarise(run: dict[str, Any]) -> dict[str, Any]:
    """Headline numbers of one run record."""
    rows = run["fold_table"]
    oos = run["diagnostics"]["oos_metrics"]
    spy = run["diagnostics"]["spy_metrics"]
    return {
        "gates": run["gates"],
        "gate_pass_folds": [r["fold_num"] for r in rows if r["gates_passed"]],
        "anchor_folds": run["anchor_folds"],
        "winners": run["winners"],
        "anchors": [r.get("anchor_combo") for r in rows],
        "oos": {k: oos[k] for k in METRICS},
        "spy": {k: spy[k] for k in METRICS},
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Fold-level differences between two run records."""
    b, a = summarise(before), summarise(after)
    return {
        "before": b,
        "after": a,
        "winner_changed_folds": [
            i + 1 for i, (x, y) in enumerate(zip(b["winners"], a["winners"], strict=True)) if x != y
        ],
        "gate_changed_folds": sorted(
            set(b["gate_pass_folds"]).symmetric_difference(a["gate_pass_folds"])
        ),
    }


def _fmt(oos: dict[str, float]) -> str:
    return f"{oos['sharpe']:.3f} / {oos['cagr_pct']:.2f}% / {oos['max_drawdown_pct']:.2f}%"


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and print the tables."""
    p = argparse.ArgumentParser(description="Before/after of the P5b selection guards.")
    p.add_argument("--evidence", default=str(EVIDENCE), help="p5-evidence directory")
    args = p.parse_args(argv)
    root = Path(args.evidence)
    out: dict[str, Any] = {}
    print(
        "| Run | Variant | Gates | Anchor folds | OOS Sharpe / CAGR / MaxDD "
        "| Winners changed vs unguarded |"
    )
    print("|---|---|---|---|---|---|")
    for name in RUNS:
        before = _load(root / f"_p5_{name}.json")
        if before is None:
            print(f"missing unguarded evidence for {name}", file=sys.stderr)
            return 2
        out[name] = {}
        print(
            f"| {name} | unguarded (P5) | {before['gates']} | {before['anchor_folds']} | "
            f"{_fmt(summarise(before)['oos'])} | - |"
        )
        for label, sub in (
            ("zero-entry rule only", "guarded_zero_entry_only"),
            ("guarded", "guarded"),
        ):
            after = _load(root / sub / f"_p5_{name}.json")
            if after is None:
                continue
            cmp = compare(before, after)
            out[name][label] = cmp
            print(
                f"| {name} | {label} | {after['gates']} | {after['anchor_folds']} | "
                f"{_fmt(cmp['after']['oos'])} | {cmp['winner_changed_folds']} |"
            )
        spy = summarise(before)["spy"]
        print(f"| {name} | SPY, same days | - | - | {_fmt(spy)} | - |")
    target = root / "guarded" / "_p5b_before_after.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
