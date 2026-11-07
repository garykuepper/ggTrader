"""Tape pass: the ggTrader.engine presets against independent pandas formulas on the frozen tape.

Used by the vectorbt upgrade runbook (the one check that saw the 1.1.0 Bollinger error on real
data). Sections (each logged and recorded even if another fails):

* ``indicators``: configured presets against pandas formulas (RSI, EMA, ATR written inline here)
  on the frozen tape (max abs/rel difference, split before/after a 200-bar burn-in).
* ``voters``: entry/exit cells produced by the engine presets + vectorbt crossovers against the
  pre-rebuild hand-written voters kept as a test oracle (``tests/oracles/legacy_voters.py``);
  every cell that differs after the burn-in is tie-audited.

The P2 ``simulator`` section (engine against the deleted ``lab.simulate``) is gone with the
simulator: its outputs are frozen in ``tests/engine/golden/legacy_simulate.npz`` and checked by
``tests/engine/test_portfolio.py``. The original script is archived as
``research/programme/vectorbt-engine/legacy-scripts/engine_parity_evidence_p2_full.py``.

Reads the frozen Parquet snapshot only (no database, no network). Writes one JSON with a
provenance block via ``ggTrader.utils.results.write_result_json``.

Usage:
    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \\
        .venv/bin/python scripts/engine_parity_evidence.py --symbols 150
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ggTrader.engine import config
from ggTrader.engine import indicators as ind
from ggTrader.engine import signals as sig
from ggTrader.utils.results import write_result_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "data" / "snapshots" / "ohlcv_20261106.parquet"
#: Never the committed evidence: a default there made a bare run overwrite it (P8 R5).
DEFAULT_OUT = Path(tempfile.gettempdir()) / "engine_parity_evidence.json"
BURN_IN = 200
TIE_TOLERANCE = 1e-9


def load_panels(path: Path, n_symbols: int, min_bars: int) -> dict[str, pd.DataFrame]:
    """Wide close/high/low/volume frames for the first ``n_symbols`` symbols with full history."""
    long = pd.read_parquet(path)
    counts = long.groupby("symbol")["close"].count()
    symbols = sorted(counts[counts >= min_bars].index)[:n_symbols]
    long = long[long["symbol"].isin(symbols)]
    out = {}
    for field in ("close", "high", "low", "volume"):
        wide = long.pivot(index="timestamp", columns="symbol", values=field)
        wide.index = pd.DatetimeIndex(wide.index)
        out[field] = wide[symbols].copy()
    return out


def _burn_in_mask(close: pd.DataFrame, burn_in: int) -> np.ndarray:
    """True on bars at least ``burn_in`` bars after each symbol's own first valid close.

    A global bar count is wrong for late listings: an EMA seeded at bar 400 is still
    converging at global bar 500.
    """
    first = close.notna().to_numpy().argmax(axis=0)
    return np.arange(len(close))[:, None] >= (first + burn_in)[None, :]


def _diff(new: np.ndarray, old: np.ndarray, settled: np.ndarray) -> dict[str, Any]:
    """Max abs/rel difference over cells where both are finite: all bars and settled bars."""
    result = {}
    for label, keep in (("all_valid", np.ones_like(settled)), ("after_burn_in", settled)):
        a, b = np.where(keep, new, np.nan), np.where(keep, old, np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            result[label] = {"cells": 0}
            continue
        abs_diff = np.abs(a[ok] - b[ok])
        scale = np.maximum(np.abs(b[ok]), 1e-12)
        result[label] = {
            "cells": int(ok.sum()),
            "max_abs": float(abs_diff.max()),
            "max_rel": float((abs_diff / scale).max()),
            "only_new_valid": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
            "only_old_valid": int((~np.isfinite(a) & np.isfinite(b)).sum()),
        }
    return result


def indicator_section(p: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Configured presets against the legacy pandas formulas."""
    close, high, low = p["close"], p["high"], p["low"]
    settled = _burn_in_mask(close, BURN_IN)
    out: dict[str, Any] = {}

    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    legacy_rsi = 100.0 - 100.0 / (1.0 + ag / al.replace(0, np.nan))
    out["wilder_rsi_14"] = _diff(
        ind.wilder_rsi(close, 14).rsi.to_numpy(), legacy_rsi.to_numpy(), settled
    )

    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std()
    bb = ind.bbands(close, 20, 2.0)
    out["bb_lower_20_2"] = _diff(bb.lower.to_numpy(), (mid - 2.0 * sd).to_numpy(), settled)
    out["bb_middle_20"] = _diff(bb.middle.to_numpy(), mid.to_numpy(), settled)

    for w in (10, 20, 50):
        legacy = close.ewm(span=w, adjust=False).mean()
        out[f"ema_{w}"] = _diff(ind.ema(close, w).ma.to_numpy(), legacy.to_numpy(), settled)

    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    m = ind.macd(close, 12, 26, 9)
    out["macd_line"] = _diff(m.macd.to_numpy(), line.to_numpy(), settled)
    out["macd_signal"] = _diff(m.signal.to_numpy(), signal.to_numpy(), settled)
    out["macd_hist"] = _diff(m.hist.to_numpy(), (line - signal).to_numpy(), settled)

    prev = close.shift(1)
    tr = pd.DataFrame(
        np.maximum(np.maximum(high - low, (high - prev).abs()), (low - prev).abs()),
        index=close.index,
        columns=close.columns,
    )
    legacy_atr = tr.rolling(14, min_periods=14).mean()
    out["atr_14"] = _diff(
        ind.atr(high, low, close, 14).atr.to_numpy(), legacy_atr.to_numpy(), settled
    )
    return out


def _flip_report(
    new: pd.DataFrame, old: pd.DataFrame, margin: pd.DataFrame, settled: np.ndarray
) -> dict[str, Any]:
    """XOR of two boolean cell frames, split by burn-in, with a tie audit after the burn-in.

    A flip is explained by a tie when the compared quantity is within 1e-9 of its threshold on
    the flip bar or the bar before: vectorbt needs a strict dip before a cross, the legacy
    ``prev <= b`` form does not.
    """
    new = new.reindex_like(old).fillna(False).to_numpy(dtype=bool)
    old_arr = old.to_numpy(dtype=bool)
    xor = new != old_arr
    late = xor & settled
    rows, cols = np.nonzero(late)
    ties = []
    for r, c in zip(rows, cols, strict=True):
        near = abs(margin.iat[r, c]) < TIE_TOLERANCE or (
            r > 0 and abs(margin.iat[r - 1, c]) < TIE_TOLERANCE
        )
        ties.append(bool(near))
    first_burn = xor & ~settled
    return {
        "cells_legacy": int(old_arr.sum()),
        "cells_engine": int(new.sum()),
        "xor_total": int(xor.sum()),
        "xor_within_burn_in": int(first_burn.sum()),
        "xor_after_burn_in": int(late.sum()),
        "after_burn_in_explained_by_tie": int(sum(ties)),
        "after_burn_in_unexplained": int(len(ties) - sum(ties)),
    }


def voter_section(p: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Voter entry/exit cells: engine presets + vectorbt crossovers against the legacy functions."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.oracles.legacy_voters import bb_signals, ema_signals, macd_signals, rsi_signals

    close = p["close"]
    settled = _burn_in_mask(close, BURN_IN)
    out: dict[str, Any] = {}

    bb = ind.bbands(close, 20, 2.0)
    lower, middle = bb.lower.copy(), bb.middle.copy()
    lower.columns = middle.columns = close.columns
    old_e, old_x = bb_signals(close, 20, 2.0)
    out["bb_entry"] = _flip_report(sig.cross_below(close, lower), old_e, close - lower, settled)
    out["bb_exit"] = _flip_report(sig.cross_above(close, middle), old_x, close - middle, settled)

    rsi = ind.wilder_rsi(close, 14).rsi.copy()
    rsi.columns = close.columns
    old_e, old_x = rsi_signals(close, 14, 30, 50)
    out["rsi_entry"] = _flip_report(sig.cross_below(rsi, 30.0), old_e, rsi - 30.0, settled)
    out["rsi_exit"] = _flip_report(sig.cross_above(rsi, 50.0), old_x, rsi - 50.0, settled)

    fast = ind.ema(close, 20).ma.copy()
    slow = ind.ema(close, 50).ma.copy()
    fast.columns = slow.columns = close.columns
    old_e, old_x = ema_signals(close, 20, 50)
    out["ema_entry"] = _flip_report(sig.cross_above(fast, slow), old_e, fast - slow, settled)
    out["ema_exit"] = _flip_report(sig.cross_below(fast, slow), old_x, fast - slow, settled)

    hist = ind.macd(close, 12, 26, 9).hist.copy()
    hist.columns = close.columns
    old_e, old_x = macd_signals(close, 12, 26, 9, 10)
    out["macd_exit"] = _flip_report(sig.cross_below(hist, 0.0), old_x, hist, settled)
    price_low = close.rolling(10, min_periods=10).min()
    hist_low = hist.rolling(10, min_periods=10).min()
    new_e = ((close <= price_low) & (hist > hist_low)).fillna(False)
    new_e.iloc[: 26 + 9 + 10] = False  # the legacy warm-up mask (slow + signal + window)
    out["macd_entry_ggtrader_rule_on_engine_hist"] = _flip_report(
        new_e, old_e, hist - hist_low, settled
    )
    return out


def _guarded(name: str, func: Callable[[], Any], record: dict[str, Any]) -> None:
    """Run a section; on error log it and keep going (the other sections still matter)."""
    started = time.perf_counter()
    try:
        record[name] = func()
    except Exception as exc:  # noqa: BLE001 - one failed section must not lose the others
        traceback.print_exc()
        record[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"[{name}] {time.perf_counter() - started:.1f}s", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT, help="frozen Parquet tape")
    parser.add_argument("--symbols", type=int, default=150, help="number of tape symbols")
    parser.add_argument("--min-bars", type=int, default=1500, help="minimum bars per symbol")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="result JSON (default: a scratch file, never the committed evidence)",
    )
    args = parser.parse_args(argv)

    record: dict[str, Any] = {"snapshot": str(args.snapshot), "symbols_requested": args.symbols}
    if args.snapshot.exists():
        panels = load_panels(args.snapshot, args.symbols, args.min_bars)
        record["tape"] = {
            "symbols": panels["close"].shape[1],
            "bars": panels["close"].shape[0],
            "first": str(panels["close"].index[0].date()),
            "last": str(panels["close"].index[-1].date()),
        }
        _guarded("indicators", lambda: indicator_section(panels), record)
        _guarded("voters", lambda: voter_section(panels), record)
    else:
        record["tape"] = "snapshot not found; no sections ran"
    config.set_engine("rust")
    path = write_result_json(args.out, record)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
