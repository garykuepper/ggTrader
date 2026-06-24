#!/usr/bin/env python
"""One-off voter ablation: re-run the 3-vs-6 voter ablation under the FIXED
gates (axis-aligned NDH, regime-param exclusion) and ATR-corrected exits.

The 2026-06-23 ablation ran on a broken NDH gate (rejecting ~every fold, so
results were anchor-driven) and pre-ATR-fix exits. This re-tests every roadmap
claim (signal dilution, MTF-harmful, RSI-critical) on a clean footing.

Targeted matrix (11 configs):
  - core {bb,rsi,ema}              (3-voter baseline)
  - core + each 2^3 optional combo (macd / vbb / mtf on/off)  -> 8 configs
  - core leave-one-out             (drop bb / rsi / ema)       -> 3 configs

Per-config the grid is clamped so min_agree <= n_voters (min_agree=4 cannot
fire with 3 voters; leaving it in pollutes the sweep and inflates DSR n_trials).

Configs run 3-at-a-time: each WFO is single-core-bound (<1 core, ~5 GB), so on a
4-core/20 GB box 3 concurrent runs (~14 GB) give ~3x speedup. Workers inherit the
loaded OHLCV copy-on-write via fork, so the big frame is never pickled.

Run:
    source .venv/bin/activate
    python scripts/ablation_voters.py 2>&1 | tee ablation_voters.log
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pandas as pd

from ggTrader.lab.data import (
    DEFAULT_UNIVERSE,
    STOCK_BASE_CONFIG,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo

# ── Configuration ──────────────────────────────────────────────────────
EVAL_START = "2021-01-31"
EVAL_END = None  # -> now
UNIVERSE = DEFAULT_UNIVERSE  # sp500, full universe (no max_stocks)
WORKERS = 3
LOG_DIR = "ablation_logs"

# Module globals populated in main() before the pool forks; children inherit
# them read-only via copy-on-write.
_OHLCV: pd.DataFrame | None = None
_SPY: pd.Series | None = None
_CFG: LabConfig | None = None
_BASE_GRID: List[Dict[str, Any]] | None = None
_EVAL_START: str = ""
_EVAL_END: str = ""

CORE = ("bb", "rsi", "ema")
OPTIONAL = ("macd", "vbb", "mtf")


def _optional_subsets() -> List[Tuple[str, ...]]:
    """All 2^3 subsets of the optional voters (incl. empty = pure 3-voter)."""
    subs: List[Tuple[str, ...]] = []
    for mask in range(2 ** len(OPTIONAL)):
        subs.append(tuple(v for i, v in enumerate(OPTIONAL) if mask & (1 << i)))
    return subs


def build_matrix() -> List[Tuple[str, Tuple[str, ...]]]:
    """(label, voters) for the 11-config targeted ablation."""
    configs: List[Tuple[str, Tuple[str, ...]]] = []
    for opt in _optional_subsets():
        voters = CORE + opt
        label = "core" if not opt else "core+" + "+".join(opt)
        configs.append((label, voters))
    # core leave-one-out
    for drop in CORE:
        voters = tuple(v for v in CORE if v != drop)
        configs.append((f"core-{drop}", voters))
    return configs


def make_ensemble_cls(voters: Tuple[str, ...]):
    """Factory: EnsembleSignal subclass that binds a fixed voter set.

    run_wfo instantiates strategy_cls(cfg) with no voter hook, so we bind
    voters at the class level.
    """
    bound = voters

    class _BoundEnsemble(EnsembleSignal):
        def __init__(self, cfg: LabConfig) -> None:
            super().__init__(cfg, voters=bound)

    _BoundEnsemble.__name__ = "Ensemble_" + "_".join(voters)
    return _BoundEnsemble


def clamp_grid(grid: List[Dict[str, Any]], n_voters: int) -> List[Dict[str, Any]]:
    """Drop combos whose min_agree exceeds the number of voters."""
    clamped = [c for c in grid if c.get("min_agree", 2) <= n_voters]
    # De-dup: clamping can leave identical combos if min_agree was the only
    # differing axis above the cap — keep unique param dicts.
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in clamped:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ── Parse run_wfo's table output ───────────────────────────────────────
def parse_table(table: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(
        r"OOS Aggregate: Sharpe ([\-\d.nan]+) \| CAGR ([\-\d.nan]+)% \| MaxDD ([\-\d.nan]+)%",
        table,
    )
    if m:
        out["oos_sharpe"], out["oos_cagr"], out["oos_maxdd"] = (
            _f(m.group(1)),
            _f(m.group(2)),
            _f(m.group(3)),
        )
    m = re.search(
        r"SPY baseline:\s+Sharpe ([\-\d.nan]+) \| CAGR ([\-\d.nan]+)% \| MaxDD ([\-\d.nan]+)%",
        table,
    )
    if m:
        out["spy_sharpe"], out["spy_cagr"], out["spy_maxdd"] = (
            _f(m.group(1)),
            _f(m.group(2)),
            _f(m.group(3)),
        )
    m = re.search(r"Aggregate WFE: ([\-\d.nan]+)", table)
    if m:
        out["wfe"] = _f(m.group(1))
    # Per-fold gate / anchor accounting from the fold rows.
    fold_rows = [ln for ln in table.splitlines() if re.match(r"^\d+\s", ln)]
    out["n_folds"] = len(fold_rows)
    out["gate_pass"] = sum(1 for ln in fold_rows if " PASS " in ln)
    out["anchor_used"] = sum(1 for ln in fold_rows if "[A]" in ln)
    return out


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ── Worker (runs in a forked child) ────────────────────────────────────
def _run_one(label: str, voters: Tuple[str, ...]) -> Dict[str, Any]:
    """Run one config's WFO. Reads inherited globals; logs to its own file."""
    assert _OHLCV is not None and _SPY is not None
    assert _CFG is not None and _BASE_GRID is not None
    grid = clamp_grid(_BASE_GRID, len(voters))
    cls = make_ensemble_cls(voters)
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    t0 = time.time()
    with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
        print(
            f"=== {label}  voters={voters}  ({len(voters)}v, {len(grid)} combos) ===",
            flush=True,
        )
        table = run_wfo(
            "ensemble",
            cls,
            _CFG,
            _OHLCV,
            _SPY,
            eval_start=_EVAL_START,
            eval_end=_EVAL_END,
            market="equity",
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )
    elapsed = time.time() - t0
    parsed = parse_table(table)
    parsed.update(
        {
            "label": label,
            "voters": "+".join(voters),
            "n_voters": len(voters),
            "n_combos": len(grid),
            "elapsed_s": round(elapsed, 1),
        }
    )
    return parsed


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    cfg = LabConfig()  # defaults: top_n=50, lookback=252, skip=21, max_stocks=None
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = (
        pd.Timestamp(EVAL_END, tz="UTC") if EVAL_END else pd.Timestamp.now(tz="UTC").normalize()
    )

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end, universe=UNIVERSE)
    print(f"[universe] {UNIVERSE}: {len(universe)} symbols", flush=True)

    print("[data] loading OHLCV (one-time)...", flush=True)
    ohlcv = load_ohlcv(
        universe + ["SPY"],
        str(data_start.date()),
        str(eval_end.date()),
        use_negative_cache=True,
    )
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    # Publish to module globals so forked workers inherit them copy-on-write.
    global _OHLCV, _SPY, _CFG, _BASE_GRID, _EVAL_START, _EVAL_END
    _OHLCV = ohlcv
    _SPY = spy_close
    _CFG = cfg
    _BASE_GRID = build_grid(EnsembleSignal)
    _EVAL_START = str(eval_start.date())
    _EVAL_END = str(eval_end.date())

    matrix = build_matrix()
    os.makedirs(LOG_DIR, exist_ok=True)
    print(
        f"[ablation] {len(matrix)} configs, {WORKERS} workers (per-config logs in {LOG_DIR}/)\n",
        flush=True,
    )

    # fork so children inherit the loaded OHLCV without pickling it.
    ctx = mp.get_context("fork")
    by_label: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, label, voters): label for label, voters in matrix}
        done = 0
        for fut in as_completed(futures):
            parsed = fut.result()
            by_label[parsed["label"]] = parsed
            done += 1
            print(
                f"[{done}/{len(matrix)}] {parsed['label']}: "
                f"OOS Sharpe {parsed.get('oos_sharpe')} "
                f"CAGR {parsed.get('oos_cagr')}% "
                f"gate {parsed.get('gate_pass')}/{parsed.get('n_folds')} "
                f"({parsed.get('elapsed_s')}s)",
                flush=True,
            )

    # Re-order results to match the matrix order for a stable summary.
    results = [by_label[label] for label, _ in matrix if label in by_label]
    _print_summary(results)


def _print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("VOTER ABLATION SUMMARY (fixed gates + ATR exits)")
    print("=" * 100)
    hdr = (
        f"{'config':<16}{'v':>2}{'cmb':>5}{'OOS Sh':>8}{'CAGR%':>8}"
        f"{'MaxDD%':>8}{'WFE':>6}{'gate':>7}{'anchor':>8}"
    )
    print(hdr)
    print("-" * 100)
    for r in results:
        print(
            f"{r['label']:<16}{r['n_voters']:>2}{r['n_combos']:>5}"
            f"{r.get('oos_sharpe', float('nan')):>8.2f}"
            f"{r.get('oos_cagr', float('nan')):>8.1f}"
            f"{r.get('oos_maxdd', float('nan')):>8.1f}"
            f"{r.get('wfe', float('nan')):>6.2f}"
            f"{str(r.get('gate_pass')) + '/' + str(r.get('n_folds')):>7}"
            f"{r.get('anchor_used', 0):>8}"
        )
    if results:
        spy = results[0]
        print("-" * 100)
        print(
            f"{'SPY baseline':<16}{'':>2}{'':>5}"
            f"{spy.get('spy_sharpe', float('nan')):>8.2f}"
            f"{spy.get('spy_cagr', float('nan')):>8.1f}"
            f"{spy.get('spy_maxdd', float('nan')):>8.1f}"
        )
    print("=" * 100, flush=True)


if __name__ == "__main__":
    sys.exit(main())
