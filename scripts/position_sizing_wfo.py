"""Research script: Regime P vs Regime F sizing, through the GATED WFO harness.

Companion to `scripts/position_sizing_regimes.py` (single-window comparison,
see docs/research/2026-08-19-position-sizing-regimes.md). That report could
not reproduce the validated headline (Sharpe 1.14 / MaxDD -5.39%), because
the headline comes from the full gated walk-forward
(`ggTrader.lab.wfo.run_wfo` / `ggTrader.lab.blend.run_blend`): per-fold combo
sweep + NDH/DSR gates + per-fold expanding-window defensive anchor
(anchor-leakage fix landed 2026-08-18, commit 2eafab1). This script runs the
SAME regime comparison through that SAME harness so the result is directly
comparable to the validated baseline -- reusing `run_blend`/`run_wfo`
unmodified, plus the Regime F admission-filter kernel imported unchanged
from `position_sizing_regimes.py`. Nothing in `src/ggTrader/lab` or
`src/ggTrader/paper` is modified; Regime F is injected purely by swapping
the `ggTrader.lab.simulate.simulate_signals` module attribute at runtime
(picked up by `sweep.sweep_signal_group`'s function-local import) inside
each config's own worker process, so the validated Regime P code path is
untouched when running Regime P.

COMPUTE BUDGET CUTS (task requires <=4h wall clock; the 2026-08-18 anchor
report measured ~15 min/fold for ONE sleeve under the corrected code -- a
naive 3-sleeve x 17-fold x 3-config grid would run ~12+ hours for a single
config alone). This script intentionally does NOT reproduce the full
validated study. What is cut, explicitly:
  - SLEEVES: one sleeve only (sp500 -- the same sleeve the anchor report
    itself measured, so the 15 min/fold estimate applies directly), not the
    validated 3-sleeve blend (sp500 + midcap400 + nasdaq100).
  - FOLDS: 6 of the validated 17 rolling folds (most recent 30 months,
    2023-10-30 -> 2026-04-30, ending on the validated study's own end date
    for recency comparability), not the full 2021-01-31 -> 2026-04-30 span.
  - CONFIGS: 3, not 6 -- Regime P (re-baseline) + Regime F cap=16 (best
    blend-level MaxDD/leverage match per the single-window report's Q1) +
    Regime F cap=30 (best-scoring cap in the single-window report's Q2).
    cap=8/12/20/25 dropped.
  - GRID: unchanged (full 48-combo EnsembleSignal grid per fold -- this is
    the one dimension NOT cut, so gate behavior (NDH/DSR) is faithful).
The single-sleeve Sharpe/CAGR/MaxDD numbers here are NOT the validated
3-sleeve blend headline and must not be reported as such -- report them as
"sp500 sleeve only, 6-fold reduced-budget WFO" and separately flag that the
Regime P run here is the new, honest, anchor-fixed re-baseline for THIS
reduced construction, not a claim that the 1.14 headline itself changed
(that requires the full 3-sleeve/17-fold re-run this script does not do).

The 3 configs run as separate processes via joblib (n_jobs=3, leaving one
of the box's 4 cores free) so wall-clock time is ~max(per-config time), not
the sum -- this is what makes n_jobs=3 meaningful here despite folds within
one config being strictly sequential (the circuit-breaker/shadow-reentry
state in wfo.py carries across folds and cannot itself be parallelized).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SLEEVE_UNIVERSE = "sp500"
SLEEVE_STRATEGY = "ensemble"
POSITION_PCT = 0.033  # matches live RiskGuard.position_pct (risk.py)

# Production overlay params (src/ggTrader/paper/overlay.py,
# docs/research/RESEARCH_SNAPSHOT.md:21-29). Passed EXPLICITLY to every
# run_blend() call below -- do NOT rely on run_blend's own defaults, which
# are research-only (target_vol=0.068, window=60, but max_leverage=2.0, see
# ggTrader.lab.blend.run_blend). That 2.0 default silently doubled the
# effective leverage in a 2026-08-18 run of this script (docs/roadmap.md:115
# records this exact trap being hit once before, 2026-07-13). The default
# stays 2.0 because other callers (the idealized/research-mode blend
# comparisons) intentionally want the unconstrained number -- it's this
# script specifically that must always pin the deployable 1.0x cap.
PRODUCTION_TARGET_VOL = 0.068
PRODUCTION_BLEND_WINDOW = 60
PRODUCTION_MAX_LEVERAGE = 1.0

#: (name, regime, slot_cap) -- slot_cap only used for regime_f.
CONFIGS: tuple[tuple[str, str, int | None], ...] = (
    ("regime_p", "regime_p", None),
    ("regime_f_cap16", "regime_f", 16),
    ("regime_f_cap30", "regime_f", 30),
)

# Reduced-budget fold window: 6 of the validated 17 folds, ending on the
# validated study's own end date. See module docstring for why.
DEFAULT_EVAL_START = "2023-10-30"
DEFAULT_EVAL_END = "2026-04-30"

VALIDATED_HEADLINE_SHARPE = 1.14
VALIDATED_HEADLINE_MAXDD = -5.39


def _build_regime_f_simulate(slot_cap: int, position_pct: float = POSITION_PCT):
    """Build a `simulate_signals`-compatible function implementing Regime F
    (fixed-notional, slot-capped) sizing, for injection into the WFO harness.

    The admission-filter algorithm (sequential per-bar concurrent-position
    cap) is imported UNCHANGED from position_sizing_regimes.py -- not
    reimplemented. What differs from that script's `simulate_targetpercent`
    is only the batching: the WFO fold sweep calls `simulate_signals` once
    per fold with UP TO 48 combos packed into one `targets_by_strategy`
    dict (grouped by stop-config in sweep.py), so this loops the admission
    filter + order-size construction over every combo group present in a
    single call, then issues one batched `vbt.Portfolio.from_orders` call
    across all of them (mirroring how the real `simulate_signals` batches
    all combos into one `from_signals` call).
    """
    import vectorbt as vbt
    from position_sizing_regimes import admission_filter

    def _simulate_signals_regime_f(
        targets_by_strategy, prices, base_config, ohlcv=None, return_pf=False
    ):
        names = list(targets_by_strategy)
        size_blocks, close_blocks, groups = [], [], []
        for name in names:
            st = targets_by_strategy[name]
            cols = pd.MultiIndex.from_product(
                [[name], st.entries.columns], names=["strategy", "symbol"]
            )
            entries_np = st.entries.to_numpy()
            exits_np = st.exits.to_numpy()
            admitted = admission_filter(entries_np, exits_np, slot_cap)
            size_np = np.full(entries_np.shape, np.nan)
            size_np[admitted] = position_pct
            size_np[exits_np] = 0.0
            size = pd.DataFrame(
                size_np, index=st.entries.index, columns=st.entries.columns
            ).set_axis(cols, axis=1)
            close = (
                prices[st.entries.columns].reindex(st.entries.index).ffill().set_axis(cols, axis=1)
            )
            size_blocks.append(size)
            close_blocks.append(close)
            groups.extend([name] * st.entries.shape[1])

        size_df = pd.concat(size_blocks, axis=1)
        close_df = pd.concat(close_blocks, axis=1)
        pf = vbt.Portfolio.from_orders(
            close=close_df,
            size=size_df,
            size_type="targetpercent",
            init_cash=float(base_config["START_CASH"]),
            fees=float(base_config["FEES"]),
            slippage=float(base_config["SLIPPAGE"]),
            freq=base_config["FREQ"],
            cash_sharing=True,
            group_by=pd.Index(groups, name="strategy"),
            call_seq="auto",
        ).copy()

        value = pf.value()
        if isinstance(value, pd.Series):
            value = value.to_frame(names[0])
        value = value[names]
        returns = value.pct_change(fill_method=None).fillna(0.0)
        diags = {
            n: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[n].entries.shape[1])}
            for n in names
        }
        if return_pf:
            return returns, value, diags, pf
        return returns, value, diags

    return _simulate_signals_regime_f


def run_one_config(
    name: str,
    regime: str,
    slot_cap: int | None,
    eval_start: str,
    eval_end: str,
    sleeve_universe: str,
) -> dict:
    """Run one (regime, slot_cap) config through run_blend for one sleeve.

    Executed inside its own joblib worker process -- the simulate_signals
    monkeypatch (for regime_f) is applied to THIS process's module state
    only, so it never affects the regime_p worker or the caller.
    """
    t0 = time.time()
    result: dict = {"name": name, "regime": regime, "slot_cap": slot_cap}
    try:
        import ggTrader.lab.simulate as simulate_mod
        from ggTrader.lab.blend import run_blend
        from ggTrader.lab.data import STOCK_BASE_CONFIG
        from ggTrader.lab.metrics import curve_stats
        from ggTrader.lab.strategy import LabConfig

        base_config = dict(STOCK_BASE_CONFIG)
        if regime == "regime_f":
            if slot_cap is None:
                raise ValueError("regime_f requires slot_cap")
            simulate_mod.simulate_signals = _build_regime_f_simulate(slot_cap)
            base_config["REGIME_F_SLOT_CAP"] = slot_cap

        cfg = LabConfig()
        blend_result = run_blend(
            [(SLEEVE_STRATEGY, sleeve_universe)],
            cfg,
            eval_start,
            eval_end,
            market="stock",
            base_config=base_config,
            target_vol=PRODUCTION_TARGET_VOL,
            window=PRODUCTION_BLEND_WINDOW,
            max_leverage=PRODUCTION_MAX_LEVERAGE,
        )
        stats = curve_stats(blend_result.blended_equity)
        result.update(
            ok=True,
            sharpe=stats["sharpe"],
            cagr_pct=stats["cagr_pct"],
            max_drawdown_pct=stats["max_drawdown_pct"],
            sortino=stats.get("sortino"),
            ann_vol_pct=stats.get("ann_vol_pct"),
            table=blend_result.table,
            run_id=blend_result.run_id,
            elapsed_sec=time.time() - t0,
            overlay_params={
                "target_vol": PRODUCTION_TARGET_VOL,
                "window": PRODUCTION_BLEND_WINDOW,
                "max_leverage": PRODUCTION_MAX_LEVERAGE,
            },
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced in JSON + Telegram, not swallowed
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_sec=time.time() - t0,
        )
    return result


def run_smoke(eval_start: str, eval_end: str, out_path: str) -> dict:
    """Trivial-scale end-to-end pipeline check: 1 fold, 2-combo grid, both
    regimes, run via run_wfo directly (bypassing run_blend's full 48-combo
    build_grid so this finishes in ~1-2 minutes, not ~15).
    """
    import ggTrader.lab.simulate as simulate_mod
    from ggTrader.lab.data import STOCK_BASE_CONFIG, equity_universe_between, load_ohlcv
    from ggTrader.lab.strategies.ensemble import EnsembleSignal
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo import run_wfo

    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    cfg = LabConfig()
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    symbols = equity_universe_between(es, ee, universe=SLEEVE_UNIVERSE)[:40]
    all_symbols = sorted(set(symbols) | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    spy_close = ohlcv["SPY"]["close"].dropna()
    syms = [s for s in symbols if s in available]

    def universe_fn(asof, past):
        from ggTrader.lab.data import eligible_at

        return eligible_at(asof, past, cfg, universe=SLEEVE_UNIVERSE)[0]

    tiny_grid = [
        {"min_agree": 2, "min_agree_exit": 1},
        {"min_agree": 3, "min_agree_exit": 1},
    ]

    out = {}
    for name, regime, slot_cap in (("smoke_p", "regime_p", None), ("smoke_f", "regime_f", 16)):
        t0 = time.time()
        base_config = dict(STOCK_BASE_CONFIG)
        if regime == "regime_f":
            simulate_mod.simulate_signals = _build_regime_f_simulate(slot_cap)
        else:
            import importlib

            importlib.reload(simulate_mod)  # restore the real simulate_signals
        table = run_wfo(
            SLEEVE_STRATEGY,
            EnsembleSignal,
            cfg,
            ohlcv[syms],
            spy_close,
            eval_start=eval_start,
            eval_end=eval_end,
            market="stock",
            base_config=base_config,
            grid=tiny_grid,
            universe_fn=universe_fn,
        )
        out[name] = {"regime": regime, "table": table, "elapsed_sec": time.time() - t0}
        print(f"[smoke:{name}] {time.time() - t0:.1f}s\n{table}\n", flush=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _fmt_row(r: dict) -> str:
    if not r.get("ok"):
        return f"| {r['name']} | ERROR | ERROR | ERROR |"
    return (
        f"| {r['name']} | {r['sharpe']:.2f} | {r['cagr_pct']:.2f}% | {r['max_drawdown_pct']:.2f}% |"
    )


def send_summary(results: list[dict], meta: dict) -> None:
    sys.path.insert(0, "/home/flynn/scripts")
    from notify import load_telegram_credentials, send_telegram

    lines = [
        "**Position-Sizing Regimes: Gated WFO (reduced budget)**",
        "",
        f"Sleeve: `{SLEEVE_STRATEGY}@{SLEEVE_UNIVERSE}` only "
        f"(not the validated 3-sleeve blend). "
        f"{meta.get('n_folds_requested', '?')} folds, "
        f"{meta['eval_start']} -> {meta['eval_end']}.",
        "",
        "| Config | Sharpe | CAGR | MaxDD |",
        "|---|---:|---:|---:|",
    ]
    lines += [_fmt_row(r) for r in results]
    lines.append("")

    p = next((r for r in results if r["name"] == "regime_p"), None)
    if p and p.get("ok"):
        lines.append(
            f"New Regime P baseline (this reduced construction): Sharpe "
            f"{p['sharpe']:.2f} vs the validated full-study headline "
            f"{VALIDATED_HEADLINE_SHARPE:.2f} / MaxDD "
            f"{VALIDATED_HEADLINE_MAXDD:.2f}% "
            f"(NOT directly comparable -- 1 sleeve/6 folds here vs 3 "
            f"sleeves/17 folds there)."
        )
    else:
        lines.append("Regime P run FAILED -- no honest re-baseline produced this run.")

    best_f = max(
        (r for r in results if r["name"].startswith("regime_f") and r.get("ok")),
        key=lambda r: r["sharpe"],
        default=None,
    )
    if p and p.get("ok") and best_f:
        verdict = (
            "Regime F still behind Regime P"
            if best_f["sharpe"] < p["sharpe"]
            else "Regime F now AHEAD of Regime P"
        )
        lines.append(
            f"Recommendation: {verdict} at this sleeve/fold budget -- keep "
            f"live sizing as-is either way pending the full 3-sleeve confirmation."
        )
    else:
        lines.append("Recommendation: inconclusive -- at least one config failed, see logs.")

    msg = "\n".join(lines)
    token, chat_id = load_telegram_credentials()
    send_telegram(msg, token, chat_id, format_markdown=True)


def run_full(eval_start: str, eval_end: str, n_jobs: int, out_path: str) -> None:
    from ggTrader.lab.wfo import generate_folds

    n_folds_requested = len(
        generate_folds(pd.Timestamp(eval_start, tz="UTC"), pd.Timestamp(eval_end, tz="UTC"))
    )

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_config)(name, regime, slot_cap, eval_start, eval_end, SLEEVE_UNIVERSE)
        for name, regime, slot_cap in CONFIGS
    )
    meta = {
        "eval_start": eval_start,
        "eval_end": eval_end,
        "sleeve": f"{SLEEVE_STRATEGY}@{SLEEVE_UNIVERSE}",
        "n_folds_requested": n_folds_requested,
        "elapsed_sec": time.time() - t0,
        "n_jobs": n_jobs,
        # Effective overlay params actually passed to every run_blend() call
        # this run (see PRODUCTION_* constants above) -- recorded here so a
        # future reader never has to infer them from run_blend's own
        # (different) defaults. This is the fix for the 2026-08-18 bug where
        # the script silently inherited run_blend's max_leverage=2.0 default
        # instead of production's 1.0.
        "overlay_params": {
            "target_vol": PRODUCTION_TARGET_VOL,
            "window": PRODUCTION_BLEND_WINDOW,
            "max_leverage": PRODUCTION_MAX_LEVERAGE,
        },
    }

    out = {"meta": meta, "results": results}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] {time.time() - t0:.1f}s total, wrote {out_path}", flush=True)

    try:
        send_summary(results, meta)
        print("[telegram] summary sent", flush=True)
    except Exception:
        print("[telegram] FAILED to send summary:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "smoke"], default="full")
    p.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    p.add_argument("--eval-end", default=DEFAULT_EVAL_END)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs" / "research" / "_position_sizing_wfo_results.json"),
    )
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke(args.eval_start, args.eval_end, args.out)
    else:
        run_full(args.eval_start, args.eval_end, args.n_jobs, args.out)
