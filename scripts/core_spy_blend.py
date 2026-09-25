"""Does the live construction (SP500 core + idle cash swept into SPY) beat SPY?

The PIT core scores Sharpe 0.59 / CAGR 3.1% vs SPY 0.78 / 13.0%
(docs/research/2026-09-23-pit-rebaseline-core-nogo.md), but that curve holds
idle cash at 0%. Live, idle cash above a 5% reserve is swept into SPY, so the
deployed book is core positions + SPY. This measures that book directly.

Method: re-run the identical pinned PIT core WFO (run_core's setup), recording
each OOS fold's daily cash share from the vbt portfolio. Then, per day t:

    r_live(k) = k * r_core(t) + max(k * idle(t-1) + (1 - k) - RESERVE, 0) * r_spy(t)

where idle is the core's cash share and k scales the sleeve (k=1 is live,
k=0 is ~all-SPY). Scaling is linear in the core's own returns: an
approximation of changing position size, not a re-simulation.

Pre-registered decision (fixed 2026-09-24 before any run):
  * KEEP the sleeve at current size if k=1 has Sharpe > SPY AND MaxDD no worse.
  * else HALVE position size if k=0.5 passes the same test.
  * else SHRINK: recommend cutting the sleeve to a small forward-test slice.
A result is a recommendation only; any live change needs a separate ask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs/research/_core_spy_blend_20260924.json"
CURVES = REPO_ROOT / "docs/research/_core_spy_blend_curves_20260924.csv"
EVAL_START = "2021-01-31"
EVAL_END = "2026-04-30"
RESERVE = 0.05
KS = [0.0, 0.5, 1.0, 1.5]


def run_core_with_cash() -> pd.DataFrame:
    """Pinned PIT core WFO; returns OOS equity, idle-cash share and SPY close."""
    import ggTrader.lab.simulate as sim
    from ggTrader.lab.data import (
        STOCK_BASE_CONFIG,
        eligible_at,
        equity_universe_between,
        load_ohlcv,
    )
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import run_wfo

    # ponytail: records cash via a patched simulate_signals. Only the per-fold
    # OOS test sims run a single combo (train/anchor/live sweep the full grid),
    # so single-target calls are exactly the 17 test folds, in order.
    orig = sim.simulate_signals
    test_idle: list[pd.Series] = []

    def recording(targets, prices, cfg, ohlcv=None, return_pf=False):
        out = orig(targets, prices, cfg, ohlcv=ohlcv, return_pf=True)
        if len(targets) == 1:
            pf = out[3]
            cash, value = pf.cash(), pf.value()
            squeeze = lambda x: x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x  # noqa: E731
            test_idle.append(squeeze(cash) / squeeze(value))
        return out if return_pf else out[:3]

    sim.simulate_signals = recording

    cfg = LabConfig()
    es, ee = pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC")
    warmup = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    universe = equity_universe_between(es, ee, universe="sp500")
    ohlcv = load_ohlcv(
        universe + ["SPY"], str((es - pd.Timedelta(days=warmup)).date()), EVAL_END, True
    )
    spy = ohlcv["SPY"]["close"].dropna()
    ohlcv = ohlcv[[s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]]
    res = run_wfo(
        "ensemble",
        STRATEGY_REGISTRY["ensemble"],
        cfg,
        ohlcv,
        spy,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market="stock",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=build_grid(STRATEGY_REGISTRY["ensemble"]),
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0],
    )
    sim.simulate_signals = orig
    if len(test_idle) != len(res.fold_results):
        raise RuntimeError(f"recorded {len(test_idle)} test sims for {len(res.fold_results)} folds")

    idle_parts = [
        s.loc[f["test_start"] : f["test_end"]] for s, f in zip(test_idle, res.fold_results)
    ]
    idle = pd.concat(idle_parts)
    idle = idle[~idle.index.duplicated(keep="last")]
    eq = res.oos_equity
    return pd.DataFrame(
        {"core_equity": eq, "idle": idle.reindex(eq.index), "spy": spy.reindex(eq.index).ffill()}
    )


def blend(curves: pd.DataFrame) -> dict:
    from ggTrader.lab.metrics import curve_stats

    r_core = curves["core_equity"].pct_change().fillna(0.0)
    r_spy = curves["spy"].pct_change().fillna(0.0)
    idle_prev = curves["idle"].shift(1).bfill()
    out = {
        "spy": curve_stats(curves["spy"]),
        "core_alone": curve_stats(curves["core_equity"]),
        "core_idle_mean_pct": float(curves["idle"].mean() * 100),
        "corr_core_vs_spy_daily": float(r_core.corr(r_spy)),
    }
    for k in KS:
        spy_w = (k * idle_prev + (1 - k) - RESERVE).clip(lower=0.0)
        r = k * r_core + spy_w * r_spy
        out[f"live_k{k}"] = {
            **curve_stats((1 + r).cumprod()),
            "mean_spy_weight": float(spy_w.mean()),
        }

    def passes(key: str) -> bool:
        return (
            out[key]["sharpe"] > out["spy"]["sharpe"]
            and out[key]["max_drawdown_pct"] >= out["spy"]["max_drawdown_pct"]
        )

    out["decision"] = (
        "KEEP" if passes("live_k1.0") else "HALVE" if passes("live_k0.5") else "SHRINK"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rerun", action="store_true", help="re-run the WFO even if curves exist")
    args = ap.parse_args()

    if CURVES.exists() and not args.rerun:
        curves = pd.read_csv(CURVES, index_col=0, parse_dates=True)
    else:
        curves = run_core_with_cash()
        curves.to_csv(CURVES)
    result = blend(curves)
    result["meta"] = {"eval_start": EVAL_START, "eval_end": EVAL_END, "reserve": RESERVE}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
