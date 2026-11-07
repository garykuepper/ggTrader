"""One-combination grids: DSR gate before (legacy dsr_check on the scorer's Sharpe) vs after.

Runs the 17-fold sp500 ensemble WFO with a ONE-combination grid (the live defaults) twice on the
frozen tape: once with the old run_dsr logic (restored here from the legacy oracle) and once with
the new engine.stats-only run_dsr. Prints per-fold DSR values and the differences.
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/flynn/ggTrader-vbt12")
import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default="/home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet")
    ap.add_argument("--eval-start", default="2021-01-31")
    ap.add_argument("--eval-end", default="2026-04-30")
    ap.add_argument("--max-stocks", type=int, default=0)
    args = ap.parse_args(argv)

    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategies.ensemble import Ensemble
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo import gates as wgates
    from ggTrader.lab.wfo import run as wrun
    from ggTrader.lab.wfo.cli_support import load_run_bars, sleeve_inputs
    from tests.oracles.legacy_dsr import dsr_check

    es = pd.Timestamp(args.eval_start, tz="UTC")
    ee = pd.Timestamp(args.eval_end, tz="UTC")
    cfg = LabConfig(max_stocks=args.max_stocks or None) if args.max_stocks else LabConfig()
    bars = load_run_bars(["sp500"], cfg, es, ee, args.snapshot)
    spy = bars.close["SPY"].dropna()
    sleeve, mask = sleeve_inputs(bars, "sp500", cfg, es, ee)
    grid = Ensemble.live_grid()
    assert len(grid) == 1

    old_vals: list[tuple[float, bool]] = []
    new_vals: list[tuple[float, bool]] = []

    def old_run_dsr(rets, n_trials, thr):
        # the pre-P10 logic for n_trials == 1, with the scorer's Sharpe computed like the scorer
        rets = rets.dropna()
        if len(rets) < wgates.MIN_DSR_BARS:
            return False, 0.0
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
        skew = float(rets.skew()) if len(rets) >= 3 else 0.0
        kurt = float(rets.kurtosis()) if len(rets) >= 3 else 0.0
        r = dsr_check(sharpe, len(rets), n_trials, skew, kurt, thr)
        return r.passed, r.dsr_value

    real = wgates.run_dsr

    def spy_run(label, fn):
        def inner(rets, n_trials, thr):
            out = fn(rets, n_trials, thr)
            (old_vals if label == "old" else new_vals).append(out)
            return out
        return inner

    results = {}
    for label, fn in (("new", real), ("old", old_run_dsr)):
        wrun.run_dsr = spy_run(label, fn)
        res = wrun.run_wfo(
            "ensemble", Ensemble, cfg, sleeve, spy, eval_start=args.eval_start, eval_end=args.eval_end,
            market="equity", base_config=dict(STOCK_BASE_CONFIG), grid=grid, mask=mask, verbose=False,
        )
        results[label] = {
            "gates_passed": [bool(r["gates_passed"]) for r in res.fold_results],
            "dsr_value": [float(r["dsr_value"]) for r in res.fold_results],
            "oos_sharpe": float(res.aggregate["oos_sharpe"]) if hasattr(res, "aggregate") else None,
        }
    diffs = [abs(a - b) for a, b in zip(results["old"]["dsr_value"], results["new"]["dsr_value"])]
    out = {"folds": len(diffs), "max_abs_dsr_diff": max(diffs) if diffs else None, **results,
           "gate_flips": int(sum(a != b for a, b in zip(results["old"]["gates_passed"], results["new"]["gates_passed"])))}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("folds", "max_abs_dsr_diff", "gate_flips")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
