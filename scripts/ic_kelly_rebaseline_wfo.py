"""Re-run `ensemble_ic` and `ensemble_kelly` against the corrected-tape core.

Both were rejected only against the phantom 1.12 baseline (real: 0.99,
docs/research/_rebaseline_corrected_tape_20260822.json). This runs each
through the identical pinned setup as that re-baseline (`run_core` from
`anchor_fix_reproduction_wfo.py`: sp500, 17 folds, 2021-01-31 -> 2026-04-30,
full grid, gated WFO), plus `ensemble` itself as a same-night control so a
tape change since 2026-08-22 can't masquerade as an edge.

Pre-registered pass criteria (2026-09-10 audit, section 5.3), fixed before
any result was seen:
  * OOS Sharpe > 1.05 AND > control Sharpe
  * OOS MaxDD no worse than -10%
  * winner stability: the most common fold winner wins >= 8 of 17 folds

Results are checkpointed to JSON after each strategy; re-running skips any
strategy already recorded ok.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from anchor_fix_reproduction_wfo import EVAL_END, EVAL_START, run_core  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs/research/_ic_kelly_rebaseline_20260923.json"
STRATEGIES = ["ensemble", "ensemble_ic", "ensemble_kelly"]
MIN_SHARPE = 1.05
MIN_MAX_DD_PCT = -10.0
MIN_STABLE_FOLDS = 8


def verdict(res: dict, control: dict | None) -> dict:
    """Apply the pre-registered criteria to one strategy's result."""
    winners = Counter(r["winner_combo"] for r in res["fold_rows"])
    top_combo, top_count = winners.most_common(1)[0]
    checks = {
        "sharpe_gt_1.05": res["sharpe"] > MIN_SHARPE,
        "sharpe_gt_control": control is not None and res["sharpe"] > control["sharpe"],
        "maxdd_ge_-10": res["max_drawdown_pct"] >= MIN_MAX_DD_PCT,
        "stable_ge_8_of_17": top_count >= MIN_STABLE_FOLDS,
    }
    return {
        "checks": checks,
        "go": all(checks.values()),
        "top_winner": top_combo,
        "top_winner_folds": top_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    args = ap.parse_args()

    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    out["meta"] = {"eval_start": EVAL_START, "eval_end": EVAL_END}

    for name in args.strategies:
        if out.get(name, {}).get("ok"):
            print(f"[skip] {name} already recorded", flush=True)
            continue
        print(f"[run] {name}", flush=True)
        out[name] = run_core(EVAL_START, EVAL_END, strategy=name)
        if not out[name]["ok"]:
            print(f"[fail] {name}: {out[name]['error']}", flush=True)
        args.out.write_text(json.dumps(out, indent=2, default=str))

    control = out.get("ensemble") if out.get("ensemble", {}).get("ok") else None
    for name in args.strategies:
        if name != "ensemble" and out.get(name, {}).get("ok"):
            out[name]["verdict"] = verdict(out[name], control)
    args.out.write_text(json.dumps(out, indent=2, default=str))

    for name in args.strategies:
        r = out.get(name, {})
        if r.get("ok"):
            v = r.get("verdict", {})
            print(
                f"{name:15s} Sharpe {r['sharpe']:.2f} CAGR {r['cagr_pct']:.1f}% "
                f"MaxDD {r['max_drawdown_pct']:.1f}% gates {r['gate_pass_str']} "
                f"SPY {r['spy_sharpe']:.2f} {'GO' if v.get('go') else ''}"
                f"{'NO-GO' if v and not v.get('go') else ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
