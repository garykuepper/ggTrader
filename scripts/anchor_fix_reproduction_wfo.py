"""Research script: does the validated SP500-core baseline (OOS Sharpe 1.12 /
CAGR 16.3% / MaxDD -11.0% vs SPY 0.58, 17-fold, gates 16/17 --
docs/research/RESEARCH_SNAPSHOT.md:21) still reproduce now that the
per-fold expanding-anchor leak fix (commit 2eafab1,
docs/research/2026-08-18-wfo-anchor-leakage-fix.md) is in?

Configuration match, and how it was determined
------------------------------------------------
`ggt.py lab --strategy ensemble --universe sp500 --wfo` with
`--eval-start 2021-01-31` (the CLI's own default, used everywhere the 1.12
baseline is cited) and `--eval-end 2026-04-30`.

The eval-end is NOT a guess: `generate_folds()` (ggTrader.lab.wfo, 12-month
train / 3-month test, rolling) produces EXACTLY 17 folds for
[2021-01-31, 2026-04-30) -- verified directly against the live code before
launching this run. This is also the exact window the July-13 leverage-
realistic 3-sleeve blend study used (docs/roadmap.md line 117-119), which is
the most recently pinned, audited, reproducible window in the project's
history for this comparison. It is NOT necessarily bit-identical to the
original, undocumented "now"-eval-end run that first produced 1.12 (roadmap
line 117-119 explicitly documents that the CLI's `--eval-end` defaults to
"now" and drifts run to run, and that re-running this exact 2026-04-30
window pre-anchor-fix gave the SP500 sleeve Sharpe 0.97, not 1.12 -- see
the module-level report for full discussion of this ambiguity). Given that
ambiguity, 2021-01-31 -> 2026-04-30 is the best-documented, audited,
17-fold window available and is what this script uses throughout.

Two runs, budget-prioritized per the task:
  (a) sp500 sleeve, standalone `run_wfo`, 17 folds, ensemble strategy --
      the direct comparator to the 1.12/16.3%/-11.0%/16-of-17 baseline.
  (b) 3-sleeve blend (sp500+midcap400+nasdaq100), `run_blend` at
      max_leverage=1.0 (the deployable/live config), same window -- the
      direct comparator to the 1.14 Sharpe / -5.39% MaxDD blend headline.

(a) is unconditionally run first (fits budget alone even in the worst
case); (b) runs only if (a) finishes with wall-clock room left in the
budget (--budget-sec, default 3h wall clock for the whole script).

Gate pass counts are reported per fold (not just the aggregate) since a
big change in gate behavior is itself the diagnostic signal for whether the
anchor-leak fix mattered -- the anchor is deployed exactly when gates fail.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_START = "2021-01-31"
EVAL_END = "2026-04-30"  # verified to produce exactly 17 folds, see docstring

CORE_STRATEGY = "ensemble"
CORE_UNIVERSE = "sp500"

BLEND_SLEEVES = [("ensemble", "sp500"), ("ensemble", "midcap400"), ("ensemble", "nasdaq100")]
PRODUCTION_TARGET_VOL = 0.068
PRODUCTION_BLEND_WINDOW = 60
PRODUCTION_MAX_LEVERAGE = 1.0

BASELINE_CORE = {"sharpe": 1.12, "cagr_pct": 16.3, "max_drawdown_pct": -11.0, "gates": "16/17"}
BASELINE_SPY = {"sharpe": 0.58}
BASELINE_BLEND = {"sharpe": 1.14, "max_drawdown_pct": -5.39}


def _fold_rows(fold_results: list[dict]) -> list[dict]:
    rows = []
    for r in fold_results:
        rows.append(
            {
                "fold_num": r["fold_num"],
                "test_start": str(pd.Timestamp(r["test_start"]).date()),
                "test_end": str(pd.Timestamp(r["test_end"]).date()),
                "ndh_passed": bool(r["ndh_passed"]),
                "dsr_passed": bool(r["dsr_passed"]),
                "gates_passed": bool(r["gates_passed"]),
                "used_anchor": bool(r["used_anchor"]),
                "wfe": r["wfe"],
                "oos_sharpe": r["oos_sharpe"],
            }
        )
    return rows


def run_core(eval_start: str, eval_end: str) -> dict:
    """Standalone SP500-core ensemble WFO -- the direct 1.12 comparator."""
    from ggTrader.lab.data import (
        STOCK_BASE_CONFIG,
        eligible_at,
        equity_universe_between,
        load_ohlcv,
    )
    from ggTrader.lab.metrics import curve_stats
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import run_wfo

    t0 = time.time()
    result: dict = {"name": "sp500_core_17fold", "eval_start": eval_start, "eval_end": eval_end}
    try:
        cfg = LabConfig()
        es = pd.Timestamp(eval_start, tz="UTC")
        ee = pd.Timestamp(eval_end, tz="UTC")
        warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
        data_start = str((es - pd.Timedelta(days=warmup_days)).date())

        universe = equity_universe_between(es, ee, universe=CORE_UNIVERSE)
        print(f"  [universe] {CORE_UNIVERSE}: {len(universe)} symbols", flush=True)
        ohlcv = load_ohlcv(universe + ["SPY"], data_start, eval_end, use_negative_cache=True)
        spy_close = ohlcv["SPY"]["close"].dropna()
        sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
        ohlcv = ohlcv[sym_cols]

        strategy_cls = STRATEGY_REGISTRY[CORE_STRATEGY]
        grid = build_grid(strategy_cls)
        print(f"WFO: {CORE_STRATEGY} | {len(grid)} param combos", flush=True)

        wfo_result = run_wfo(
            CORE_STRATEGY,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=eval_start,
            eval_end=eval_end,
            market="stock",
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
            universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe=CORE_UNIVERSE)[0],
        )
        oos_metrics = curve_stats(wfo_result.oos_equity)
        fold_rows = _fold_rows(wfo_result.fold_results)
        n_gate_pass = sum(1 for r in fold_rows if r["gates_passed"])
        n_folds = len(fold_rows)
        spy_common = spy_close.reindex(wfo_result.oos_equity.index).ffill().dropna()
        spy_metrics = (
            curve_stats(spy_common * (100000.0 / spy_common.iloc[0])) if len(spy_common) > 1 else {}
        )

        result.update(
            ok=True,
            sharpe=oos_metrics["sharpe"],
            cagr_pct=oos_metrics["cagr_pct"],
            max_drawdown_pct=oos_metrics["max_drawdown_pct"],
            spy_sharpe=spy_metrics.get("sharpe"),
            n_folds=n_folds,
            n_gate_pass=n_gate_pass,
            gate_pass_str=f"{n_gate_pass}/{n_folds}",
            fold_rows=fold_rows,
            table=wfo_result.table,
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced in JSON + Telegram, not swallowed
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_sec=time.time() - t0,
        )
    return result


def run_blend_3sleeve(eval_start: str, eval_end: str) -> dict:
    """3-sleeve leverage-realistic blend at max_leverage=1.0 -- the 1.14/-5.39% comparator."""
    from ggTrader.lab.blend import run_blend
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.metrics import curve_stats
    from ggTrader.lab.strategy import LabConfig

    t0 = time.time()
    result: dict = {"name": "3sleeve_blend_lev1.0", "eval_start": eval_start, "eval_end": eval_end}
    try:
        cfg = LabConfig()
        blend_result = run_blend(
            BLEND_SLEEVES,
            cfg,
            eval_start,
            eval_end,
            market="stock",
            base_config=dict(STOCK_BASE_CONFIG),
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
            table=blend_result.table,
            run_id=blend_result.run_id,
            elapsed_sec=time.time() - t0,
            overlay_params={
                "target_vol": PRODUCTION_TARGET_VOL,
                "window": PRODUCTION_BLEND_WINDOW,
                "max_leverage": PRODUCTION_MAX_LEVERAGE,
            },
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_sec=time.time() - t0,
        )
    return result


def run_smoke(out_path: str) -> dict:
    """Trivial-scale end-to-end check: 1-2 folds, tiny grid, both code paths,
    to prove the pipeline (incl. Telegram) works before the multi-hour launch.
    """
    from ggTrader.lab.data import (
        STOCK_BASE_CONFIG,
        eligible_at,
        equity_universe_between,
        load_ohlcv,
    )
    from ggTrader.lab.strategies.ensemble import EnsembleSignal
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo import run_wfo

    # 1 real fold's worth of window: 2021-01-31 -> 2022-04-30 (train 12mo +
    # test 3mo = the minimum generate_folds() needs to emit fold 1).
    smoke_start = "2021-01-31"
    smoke_end = "2022-04-30"
    cfg = LabConfig()
    es = pd.Timestamp(smoke_start, tz="UTC")
    ee = pd.Timestamp(smoke_end, tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    symbols = equity_universe_between(es, ee, universe=CORE_UNIVERSE)[:40]
    all_symbols = sorted(set(symbols) | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, smoke_end, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    spy_close = ohlcv["SPY"]["close"].dropna()
    syms = [s for s in symbols if s in available]

    tiny_grid = [
        {"min_agree": 2, "min_agree_exit": 1},
        {"min_agree": 3, "min_agree_exit": 1},
    ]

    t0 = time.time()
    table = run_wfo(
        CORE_STRATEGY,
        EnsembleSignal,
        cfg,
        ohlcv[syms],
        spy_close,
        eval_start=smoke_start,
        eval_end=smoke_end,
        market="stock",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=tiny_grid,
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe=CORE_UNIVERSE)[0],
    )
    out = {"smoke": {"table": table, "elapsed_sec": time.time() - t0}}
    print(f"[smoke] {time.time() - t0:.1f}s\n{table}\n", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _fmt_core_row(r: dict) -> str:
    if not r.get("ok"):
        return f"| SP500 core (17-fold, this run) | ERROR: {r.get('error')} |  |  |  |"
    return (
        f"| SP500 core (17-fold, this run) | {r['sharpe']:.2f} | {r['cagr_pct']:.2f}% "
        f"| {r['max_drawdown_pct']:.2f}% | {r['gate_pass_str']} |"
    )


def send_summary(core: dict, blend: dict | None, meta: dict) -> None:
    sys.path.insert(0, "/home/flynn/scripts")
    from notify import load_telegram_credentials, send_telegram

    lines = [
        "**Anchor-Fix Reproduction: SP500 Core Gated WFO**",
        "",
        f"Window: `{meta['eval_start']}` -> `{meta['eval_end']}` "
        f"({meta.get('n_folds', '?')} folds), post anchor-leak-fix (2eafab1).",
        "",
        "| Config | Sharpe | CAGR | MaxDD | Gates |",
        "|---|---:|---:|---:|---:|",
        _fmt_core_row(core),
        f"| **Baseline (pre-fix, validated)** | {BASELINE_CORE['sharpe']:.2f} "
        f"| {BASELINE_CORE['cagr_pct']:.1f}% | {BASELINE_CORE['max_drawdown_pct']:.1f}% "
        f"| {BASELINE_CORE['gates']} |",
        f"| SPY (this run) | {core.get('spy_sharpe', float('nan')):.2f} | -- | -- | -- |"
        if core.get("ok")
        else "| SPY (this run) | -- | -- | -- | -- |",
        "",
    ]

    if core.get("ok"):
        delta_sharpe = core["sharpe"] - BASELINE_CORE["sharpe"]
        verdict = "REPRODUCES" if abs(delta_sharpe) <= 0.05 else "DOES NOT reproduce"
        lines.append(
            f"**Verdict: baseline {verdict}.** Sharpe delta {delta_sharpe:+.2f} "
            f"({core['sharpe']:.2f} vs 1.12 baseline), gate pass {core['gate_pass_str']} vs 16/17."
        )
    else:
        lines.append(f"**Core run FAILED**: {core.get('error')}")
    lines.append("")

    if blend is not None:
        lines.append("| Config | Sharpe | MaxDD |")
        lines.append("|---|---:|---:|")
        if blend.get("ok"):
            lines.append(
                f"| 3-sleeve blend, lev=1.0 (this run) | {blend['sharpe']:.2f} "
                f"| {blend['max_drawdown_pct']:.2f}% |"
            )
        else:
            lines.append(f"| 3-sleeve blend, lev=1.0 (this run) | ERROR: {blend.get('error')} |  |")
        lines.append(
            f"| **Baseline (pre-fix, validated)** | {BASELINE_BLEND['sharpe']:.2f} "
            f"| {BASELINE_BLEND['max_drawdown_pct']:.2f}% |"
        )
    else:
        lines.append("3-sleeve blend: not run this session (budget or time-window cutoff).")

    lines.append("")
    lines.append(
        "Live paper trader keeps running on the deployed 3-sleeve blend config "
        "regardless of this result (Flynn's decision)."
    )

    msg = "\n".join(lines)
    token, chat_id = load_telegram_credentials()
    send_telegram(msg, token, chat_id, format_markdown=True)


def run_full(out_path: str, budget_sec: float, skip_blend: bool) -> None:
    t0 = time.time()
    print(f"[start] core sp500 17-fold WFO, budget {budget_sec}s", flush=True)
    core = run_core(EVAL_START, EVAL_END)
    elapsed = time.time() - t0
    print(f"[core done] {elapsed:.1f}s ok={core.get('ok')}", flush=True)

    blend = None
    remaining = budget_sec - elapsed
    if not skip_blend and core.get("ok") and remaining > 900:
        print(f"[start] 3-sleeve blend, {remaining:.0f}s remaining", flush=True)
        blend = run_blend_3sleeve(EVAL_START, EVAL_END)
        print(f"[blend done] ok={blend.get('ok')}", flush=True)
    elif skip_blend:
        print("[skip] blend explicitly skipped (--skip-blend)", flush=True)
    else:
        print(
            f"[skip] blend skipped -- {remaining:.0f}s remaining or core failed",
            flush=True,
        )

    meta = {
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "n_folds": core.get("n_folds"),
        "total_elapsed_sec": time.time() - t0,
        "anchor_fix_commit": "2eafab1",
    }
    out = {"meta": meta, "core": core, "blend": blend}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] {time.time() - t0:.1f}s total, wrote {out_path}", flush=True)

    try:
        send_summary(core, blend, meta)
        print("[telegram] summary sent", flush=True)
    except Exception:
        print("[telegram] FAILED to send summary:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "smoke"], default="full")
    p.add_argument("--budget-sec", type=float, default=3 * 3600)
    p.add_argument("--skip-blend", action="store_true", default=False)
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs" / "research" / "_anchor_fix_reproduction_results.json"),
    )
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke(args.out.replace(".json", "_smoke.json"))
    else:
        run_full(args.out, args.budget_sec, args.skip_blend)
