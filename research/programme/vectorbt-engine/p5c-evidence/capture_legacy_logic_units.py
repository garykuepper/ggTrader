"""Freeze the legacy ``lab.wfo`` / ``lab.sweep`` pure-function outputs the P5 tests compared against.

Run once, BEFORE the legacy modules were deleted (P5c), with the legacy code still importable. The
inputs replay the seeded random cases of ``tests/lab/wfo_v2/test_logic_units.py`` and
``test_report_inputs_weights.py``; the outputs go to ``tests/lab/wfo_v2/golden/legacy_logic_units.json``
so those tests keep an independent oracle after the legacy modules are gone. After P5c this script
can only run at git tag legacy-engine-vbt0285 (scripts/legacy_env.sh).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ggTrader.lab import wfo as legacy  # noqa: E402


def num(x):
    if x is None:
        return None
    x = float(x)
    return "nan" if math.isnan(x) else x


def random_metrics(rng, n):
    rows = []
    for _ in range(n):
        rows.append(
            {
                "sharpe": float(rng.normal(0.5, 1.0)) if rng.random() > 0.2 else math.nan,
                "sortino": float(rng.normal(0.7, 1.2)) if rng.random() > 0.2 else math.nan,
                "max_drawdown_pct": -float(rng.uniform(0, 30)),
            }
        )
    return rows


def state_tuple(s):
    return [[num(x) for x in s.wfe_history], [num(x) for x in s.oos_sharpes], bool(s.halted),
            s.halt_reason, list(s.shadow_window)]


def main() -> None:
    out = {"generated_by": "research/programme/vectorbt-engine/p5c-evidence/capture_legacy_logic_units.py",
           "source": "legacy lab.wfo before P5c deletion (tag legacy-engine-vbt0285 equivalent)"}
    # folds
    folds = {}
    for start, end in [("2021-01-31", "2026-04-30"), ("2021-01-31", "2022-07-31"),
                       ("2020-02-29", "2023-05-31"), ("2021-01-31", "2021-12-31")]:
        es, ee = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        folds[f"{start}|{end}"] = [[str(x) for x in f] for f in legacy.generate_folds(es, ee)]
    out["folds"] = folds
    # composite
    rng = np.random.default_rng(3)
    comp = {}
    for n in (1, 2, 5, 48):
        rows = random_metrics(rng, n)
        comp[str(n)] = [num(x) for x in legacy.composite_score(rows)]
    out["composite"] = comp
    # pick live winner
    rows = [{"combo": c} for c in "abcd"]
    scores = [0.2, 0.9, 0.4, 0.8]
    pl = []
    for counts in ({}, {"a": 2}, {"b": 0, "c": 3}, {"d": 1, "c": 1}):
        pl.append(legacy._pick_live_winner(rows, scores, counts))
    out["pick_live_winner"] = pl
    # breaker
    rng = np.random.default_rng(11)
    br = []
    for _ in range(300):
        n = int(rng.integers(0, 9))
        wfe = [None if rng.random() < 0.25 else float(rng.normal(0.4, 0.8)) for _ in range(n)]
        if rng.random() < 0.2 and wfe:
            wfe[int(rng.integers(0, len(wfe)))] = math.nan
        oos = [float(rng.normal(0, 1)) for _ in range(int(rng.integers(0, 6)))]
        case = {}
        for policy in ("legacy", "exclude"):
            st = legacy.WfoState(list(wfe), list(oos), False, None, [])
            case[policy] = state_tuple(legacy.check_circuit_breaker(st, nan_policy=policy))
        br.append(case)
    out["breaker"] = br
    # shadow reentry
    rng = np.random.default_rng(5)
    sh = []
    for _ in range(200):
        st = legacy.WfoState([], [], True, None,
                             [bool(x) for x in rng.integers(0, 2, int(rng.integers(0, 3)))])
        args = (bool(rng.integers(0, 2)), bool(rng.integers(0, 2)),
                [None, 0.2, 0.9][int(rng.integers(0, 3))])
        sh.append(state_tuple(legacy.check_shadow_reentry(st, *args)))
    out["shadow"] = sh
    out["wfe_0.5_0.2"] = num(legacy.compute_wfe(0.5, 0.2))
    # extract_grid_arrays
    grid = [{"a": a, "b": b} for a in (1, 2, 3) for b in (10, 20)]
    rng = np.random.default_rng(2)
    rows = [{"params": g, "sharpe": float(rng.normal()), "total_return_pct": float(rng.normal(5, 3)),
             "n_trades": int(rng.integers(0, 9))} for g in grid]
    old = legacy._extract_grid_arrays(rows, grid, "s")
    out["grid_arrays"] = {"arrays": [np.asarray(a, dtype=float).tolist() for a in old[:3]],
                          "fourth": repr(old[3])}
    # table text
    sys.path.insert(0, str(ROOT / "tests"))
    from tests.lab.wfo_v2.test_report_inputs_weights import _fold  # noqa: E402

    folds_ = [_fold(1), _fold(2), _fold(3)]
    oos = {"sharpe": 0.62, "cagr_pct": 3.3, "max_drawdown_pct": -5.3}
    spy = {"sharpe": 0.78, "cagr_pct": 13.0, "max_drawdown_pct": -22.1}
    live = {"combo": "ensemble__a", "stability": 2, "train_start": folds_[0]["train_start"],
            "train_end": folds_[0]["train_end"],
            "train_metrics": {"sharpe": 1.3, "cagr_pct": 20.2, "max_drawdown_pct": -3.0}}
    args = (folds_, oos, spy, live, "ensemble", 48, 3)
    out["table_text"] = legacy.format_wfo_table(
        *args, halted=True, anchor=legacy.AnchorSet("ensemble__b", {}, -7.7, 4.2, 0.69))
    dest = ROOT / "tests/lab/wfo_v2/golden/legacy_logic_units.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
