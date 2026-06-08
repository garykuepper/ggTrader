"""Offline replay of the WFO aggregate gates on a wfo_stats_snapshot.json.

Lets us sweep gate thresholds in seconds without re-running the full WFO.
Imports the SAME gate functions the orchestrator uses, so verdicts match.

Usage:
  python scratch/gate_replay.py <snapshot.json> [wfe_min prof_min cv_max dd_max] [--verbose]
Defaults: 0.5 0.6 0.3 2.0  (the production gates)
"""

import sys
from collections import defaultdict

sys.path.insert(0, "src")
import json  # noqa: E402

from ggTrader.core.wfo_aggregate import (  # noqa: E402
    apply_gates,
    compute_wfe,
    dd_ratio,
    fraction_profitable_folds,
    parameter_cv,
)

path = sys.argv[1]
nums = [a for a in sys.argv[2:] if not a.startswith("--")]
verbose = "--verbose" in sys.argv
wfe_min = float(nums[0]) if len(nums) > 0 else 0.5
prof_min = float(nums[1]) if len(nums) > 1 else 0.6
cv_max = float(nums[2]) if len(nums) > 2 else 0.3
dd_max = float(nums[3]) if len(nums) > 3 else 2.0
th = {"wfe_min": wfe_min, "profitable_min": prof_min, "cv_max": cv_max, "dd_max": dd_max}

snap = json.load(open(path))

rows = []
for rec in snap:
    folds = rec["payload"]["wfo_stats"]
    tr = [float(f.get("train_annualized_return", float("nan"))) for f in folds]
    te = [float(f.get("oos_annualized_return", float("nan"))) for f in folds]
    trd = [float(f.get("train_max_dd", float("nan"))) for f in folds]
    ted = [float(f.get("oos_max_dd", float("nan"))) for f in folds]
    fp = [f.get("params", {}) for f in folds]
    wfe = compute_wfe(tr, te)
    prof = fraction_profitable_folds(te)
    pcv = parameter_cv(fp)
    ddr = dd_ratio(trd, ted)
    gr = apply_gates(wfe, prof, pcv, ddr, th)
    rows.append((rec["symbol"], rec["strategy"], rec["exit"], wfe, prof, pcv, ddr, gr))

passers = defaultdict(list)
for sym, strat, ex, wfe, prof, pcv, ddr, gr in rows:
    if gr["passed"]:
        passers[sym].append((strat, ex, wfe, prof, pcv, ddr))

print(f"Thresholds: WFE>={wfe_min}  prof>={prof_min}  paramCV<={cv_max}  DDratio<={dd_max}")
print(f"COINS PASSING: {len(passers)}  -> {sorted(passers)}")
for sym in sorted(passers):
    for strat, ex, wfe, prof, pcv, ddr in passers[sym]:
        print(
            f"   {sym:<9} {strat}/{ex:<13} WFE={wfe:.2f} prof={prof:.0%} CV={pcv:.2f} DD={ddr:.2f}"
        )

if verbose:
    print("\nPer-coin CLOSEST combo (fewest gate failures) + which gates fail:")
    bycoin = defaultdict(list)
    for sym, strat, ex, wfe, prof, pcv, ddr, gr in rows:
        bycoin[sym].append((len(gr["failures"]), strat, ex, wfe, prof, pcv, ddr, gr["failures"]))
    hdr = f"{'coin':<9}{'WFE':>7}{'prof':>7}{'CV':>7}{'DD':>7}  fails (best combo)"
    print(hdr)
    for sym in sorted(bycoin):
        best = min(bycoin[sym], key=lambda x: x[0])
        _, strat, ex, wfe, prof, pcv, ddr, fails = best
        print(
            f"{sym:<9}{wfe:>7.2f}{prof:>7.0%}{pcv:>7.2f}{ddr:>7.2f}  "
            f"{','.join(fails) if fails else 'PASS'}  [{strat}/{ex}]"
        )
