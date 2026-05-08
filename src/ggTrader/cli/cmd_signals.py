"""CLI Command: ``ggt signals`` — snapshot per-symbol live signal + regime state.

One-shot diagnostic. For each symbol the live trader is configured for, prints:

  - Entry / Exit booleans for the current bar.
  - Regime tier (BTC / Free) and the gate's allow decision.
  - Current price, BTC correlation, distance to the coin's own long EMA(200)
    (informational — shows how far each coin is from its own long-term trend).
  - WFO-selected strategy + exit + ATR (when applicable).
  - Position state (in active_positions or flat).
  - "Why blocked" reason for any symbol whose entry didn't fire.

Useful for answering "is X in the buy region right now?" without waiting for
the next event-loop tick.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


def register_signals_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "signals", help="Snapshot per-symbol entry/regime state for the live universe"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to run_results.json (default: auto-detect latest)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated subset of symbols to show (default: all in research)",
    )
    parser.add_argument(
        "--firing-only",
        action="store_true",
        help="Only print rows where entry=True",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print WFO params and signal-internal stop/fill prices",
    )


def run_signals(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from ggTrader.core.crypto_execution_engine import CryptoExecutionEngine
    from ggTrader.utils.state_manager import get_latest_research_run

    if args.results:
        results_source: object = args.results
    else:
        latest = get_latest_research_run()
        if not latest:
            print("Error: No research run found.", file=sys.stderr)
            sys.exit(1)
        results_source = latest

    from ggTrader.utils.result_db_manager import ResultDBManager
    from ggTrader.utils.run_config import full_pipeline_config, merge_run_config

    base_config = full_pipeline_config()
    config = merge_run_config(base_config, DRY_RUN=True)

    engine = CryptoExecutionEngine(
        config,
        results_path=results_source,
        db_manager=ResultDBManager(),
        run_id="LIVE",
    )

    requested_symbols: Optional[List[str]] = None
    if args.symbols:
        requested_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    df = engine._fetch_latest_data()
    if df.empty:
        print("Error: no OHLCV data fetched.", file=sys.stderr)
        sys.exit(1)

    signals = engine._compute_latest_signals(df)
    allowance = engine._compute_live_regime_allowance(df)

    from ggTrader.core.regime_filtering import (
        _compute_btc_correlations,
        _compute_btc_regime_mask,
    )
    threshold = float(config.get("LEADER_CORR_THRESHOLD", 0.7))
    btc_regime = (
        _compute_btc_regime_mask(df, config)
        if config.get("BTC_REGIME_FILTER", False) else None
    )
    btc_corrs = _compute_btc_correlations(df, config) if btc_regime is not None else {}
    btc_bull = bool(btc_regime.iloc[-1]) if btc_regime is not None else None

    ema_dist_span = 200

    try:
        from ggTrader.utils.fear_greed import fetch_fear_greed
        fg = fetch_fear_greed(limit=8)
    except Exception:
        fg = None
    fg_str = ""
    if fg is not None:
        suffix = ""
        hist = fg.get("history") or []
        if len(hist) >= 8:
            delta = fg["value"] - hist[7]["value"]
            sign = "+" if delta >= 0 else ""
            suffix = f" ({sign}{delta} vs 7d)"
        fg_str = f"  F&G={fg['value']} {fg['classification']} {fg['emoji']}{suffix}"

    print("=" * 120)
    print(f"  ggt signals — run_id={getattr(results_source, 'run_id', '<path>')}")
    btc_label = "BULL" if btc_bull else ("BEAR" if btc_bull is not None else "OFF")
    print(f"  BTC regime={btc_label}  threshold={threshold:.2f}{fg_str}")
    print("=" * 120)

    rows: List[Dict[str, Any]] = []
    universe = list(engine.symbols)
    if requested_symbols:
        universe = [s for s in universe if s in set(requested_symbols)]

    for sym in universe:
        info = engine.per_coin_params.get(sym, {})
        sig = signals.get(sym, {})
        in_pos = sym in engine.active_positions
        cb = float(btc_corrs.get(sym, float("nan")))

        cb_for_class = cb if pd.notna(cb) else 0.0
        if btc_bull is None:
            tier = "OFF"
            tier_ok = True
        elif cb_for_class >= threshold:
            tier = "BTC"
            tier_ok = bool(btc_bull)
        else:
            tier = "Free"
            tier_ok = True

        ema_long_value = float("nan")
        ema_dist_pct = float("nan")
        try:
            close = df[(sym, "close")].dropna()
            if len(close) > 0:
                ema_long = close.ewm(span=ema_dist_span, adjust=False).mean().iloc[-1]
                ema_long_value = float(ema_long)
                if ema_long_value > 0:
                    ema_dist_pct = (float(close.iloc[-1]) / ema_long_value - 1.0) * 100.0
        except Exception:
            pass

        allow = bool(allowance.get(sym, True))

        if sig.get("entry"):
            reason = "ENTRY"
        elif not info:
            reason = "no_params"
        elif in_pos:
            reason = "in_position"
        elif not allow:
            reason = f"blocked:{tier}_bear" if tier_ok is False else "blocked:regime"
        else:
            reason = "no_signal"

        rows.append({
            "symbol": sym,
            "strategy": info.get("strategy_name", "-"),
            "exit": info.get("exit_name", "-"),
            "entry": bool(sig.get("entry", False)),
            "exit_signal": bool(sig.get("exit", False)),
            "tier": tier,
            "corr_btc": cb,
            "allow": allow,
            "price": float(sig.get("current_price", float("nan"))),
            "ema_long": ema_long_value,
            "ema_dist_pct": ema_dist_pct,
            "atr": float(sig.get("atr_value", float("nan"))),
            "in_pos": in_pos,
            "reason": reason,
            "params": info.get("params", {}),
            "stop_price": float(sig.get("stop_price", float("nan"))),
        })

    if args.firing_only:
        rows = [r for r in rows if r["entry"]]

    if not rows:
        print("(no rows to display)")
        return

    header = (
        f"{'Symbol':<10} {'Strategy':<20} {'Exit':<14} "
        f"{'E/X':<5} {'Tier':<6} {'cBTC':>5} {'Allow':<5} "
        f"{'Price':>11} {'%vs_EMA':>8} {'ATR':>9} {'InPos':<5} {'Reason':<28}"
    )
    print(header)
    print("-" * len(header))

    def _fmt_corr(c: float) -> str:
        return f"{c:>5.2f}" if pd.notna(c) else f"{'n/a':>5}"

    def _fmt_dist(d: float) -> str:
        return f"{d:>+7.2f}%" if pd.notna(d) else f"{'n/a':>8}"

    for r in sorted(rows, key=lambda r: (not r["entry"], r["reason"], r["symbol"])):
        e_x = ("E" if r["entry"] else ".") + ("X" if r["exit_signal"] else ".")
        price_str = f"${r['price']:>10.4f}" if pd.notna(r['price']) else f"{'n/a':>11}"
        atr_str = f"{r['atr']:>9.4f}" if pd.notna(r["atr"]) else f"{'n/a':>9}"
        print(
            f"{r['symbol']:<10} {r['strategy']:<20} {r['exit']:<14} "
            f"{e_x:<5} {r['tier']:<6} {_fmt_corr(r['corr_btc'])} "
            f"{'yes' if r['allow'] else 'NO':<5} "
            f"{price_str:>11} {_fmt_dist(r['ema_dist_pct']):>8} {atr_str} "
            f"{'YES' if r['in_pos'] else '-':<5} {r['reason']:<28}"
        )

    if args.verbose:
        print()
        print("=== Verbose: WFO params + signal stops ===")
        for r in rows:
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            stop_str = f"stop=${r['stop_price']:.4f}" if pd.notna(r["stop_price"]) else "stop=n/a"
            print(f"  {r['symbol']:<10} {r['strategy']}+{r['exit']}: {params_str} | {stop_str}")

    n_total = len(rows)
    n_firing = sum(1 for r in rows if r["entry"])
    n_blocked = sum(1 for r in rows if not r["allow"])
    n_in_pos = sum(1 for r in rows if r["in_pos"])
    print()
    print(f"Summary: {n_total} symbol(s) | {n_firing} firing | {n_blocked} regime-blocked | {n_in_pos} in position")
