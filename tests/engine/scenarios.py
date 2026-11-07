"""Seeded golden scenarios shared by the L2 tests and the golden generator.

Three synthetic panels (NaN holes, a price gap, a point-in-time mask) go through
``run_signals`` and ``run_weights``, plus a 2-fold mini walk-forward built only from engine
functions (single-pass simulation, calendar folds, window Sharpe, per-window trade counts).
Every output is a plain JSON-able dict so it can be stored as a golden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ggTrader.engine import indicators as ind
from ggTrader.engine import portfolio as pfm
from ggTrader.engine import signals as sig
from ggTrader.engine import stats, windows

from .panels import make_panel

BASE = {
    "START_CASH": 10000.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0005,
    "FREQ": "1D",
    "SIGNAL_POSITION_SIZE": 0.03,
}
SEEDS = {"A": 20261106, "B": 20261107, "C": 20261108}


def _random_signals(
    panel: dict[str, pd.DataFrame], seed: int, rate: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shape = panel["close"].shape
    index, columns = panel["close"].index, panel["close"].columns
    entries = pd.DataFrame(rng.random(shape) < rate, index=index, columns=columns)
    exits = pd.DataFrame(rng.random(shape) < rate, index=index, columns=columns)
    return entries, exits


def _grouped(frame: pd.DataFrame, n_groups: int) -> tuple[pd.DataFrame, list[str]]:
    """Split the columns of a panel into n contiguous groups named g0..gN."""
    per = frame.shape[1] // n_groups
    groups = [f"g{i}" for i in range(n_groups) for _ in range(per)]
    return frame.iloc[:, : per * n_groups], groups


def _summarise(result: pfm.SimResult) -> dict[str, Any]:
    return {
        "value": {str(g): result.value[g].tolist() for g in result.value.columns},
        "n_trades": {str(g): int(n) for g, n in result.n_trades.items()},
        "entry_idx": result.trades["entry_idx"].astype(int).tolist(),
        "exit_idx": result.trades["exit_idx"].astype(int).tolist(),
    }


def signals_trailing() -> dict[str, Any]:
    """Panel A: 60 symbols x 400 bars, 3 groups, 5% trailing stop."""
    panel = make_panel(seed=SEEDS["A"], n_bars=400, n_syms=60)
    entries, exits = _random_signals(panel, SEEDS["A"] + 1, 0.03)
    close, groups = _grouped(panel["close"], 3)
    out = pfm.run_signals(
        entries[close.columns],
        exits[close.columns],
        close,
        {**BASE, "ts_stop": 0.05},
        groups=groups,
    )
    return _summarise(out)


def signals_masked_time_stop() -> dict[str, Any]:
    """Panel B: NaN holes and a gap, a PIT mask, a 7-bar time stop and a take-profit."""
    panel = make_panel(seed=SEEDS["B"], n_bars=400, n_syms=60, holes=True)
    entries, exits = _random_signals(panel, SEEDS["B"] + 1, 0.04)
    rng = np.random.default_rng(SEEDS["B"] + 2)
    mask = pd.DataFrame(
        rng.random(entries.shape) < 0.7, index=entries.index, columns=entries.columns
    )
    close, groups = _grouped(panel["close"], 4)
    cols = close.columns
    out = pfm.run_signals(
        entries[cols],
        exits[cols],
        close,
        {**BASE, "tp_stop": 0.08},
        groups=groups,
        mask=mask[cols],
        td_stop=7,
    )
    return _summarise(out)


def weights_monthly() -> dict[str, Any]:
    """Panel C: monthly target weights over 40 symbols, 2 groups, with a late listing."""
    panel = make_panel(seed=SEEDS["C"], n_bars=500, n_syms=40, holes=True)
    close, groups = _grouped(panel["close"], 2)
    rng = np.random.default_rng(SEEDS["C"] + 1)
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for bar in range(30, len(close), 21):
        for g in sorted(set(groups)):
            cols = [i for i, label in enumerate(groups) if label == g]
            picks = rng.choice(cols, 5, replace=False)
            row = np.zeros(len(close.columns))
            row[picks] = 0.18
            weights.iloc[bar, cols] = row[cols]
    out = pfm.run_weights(weights, close, BASE, groups=groups)
    return _summarise(out)


def _folds(eval_start: str, eval_end: str) -> list[tuple[pd.Timestamp, ...]]:
    """12-month train / 3-month test / 3-month step, the lab.wfo recurrence."""
    folds, cursor, end = [], pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    while True:
        train_end = cursor + pd.DateOffset(months=12)
        test_end = train_end + pd.DateOffset(months=3)
        if test_end > end:
            return folds
        folds.append((cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(months=3)


class _Fold:
    def __init__(self, ts: pd.Timestamp, te: pd.Timestamp, ss: pd.Timestamp, se: pd.Timestamp):
        self.train_start, self.train_end, self.test_start, self.test_end = ts, te, ss, se


def mini_wfo() -> dict[str, Any]:
    """2-fold walk-forward on engine functions only: one simulation, sliced per window."""
    panel = make_panel(seed=SEEDS["A"] + 10, n_bars=700, n_syms=30, start="2020-01-01")
    close = panel["close"]
    rsi = ind.wilder_rsi(close, 14).rsi
    rsi.columns = close.columns
    grid = [(o, a) for o in (25.0, 30.0, 35.0) for a in (2.0, 2.5)]
    blocks_e, blocks_x, blocks_c, groups = [], [], [], []
    for oversold, alpha in grid:
        bb = ind.bbands(close, 20, alpha)
        lower, middle = bb.lower, bb.middle
        lower.columns = middle.columns = close.columns
        entries = sig.cross_below(rsi, oversold) | sig.cross_below(close, lower)
        exits = sig.cross_above(close, middle)
        blocks_e.append(entries)
        blocks_x.append(exits)
        blocks_c.append(close)
        groups += [f"rsi{int(oversold)}_bb{alpha}"] * close.shape[1]
    entries = pd.concat(blocks_e, axis=1)
    exits = pd.concat(blocks_x, axis=1)
    all_close = pd.concat(blocks_c, axis=1)
    entries.columns = exits.columns = all_close.columns = pd.RangeIndex(all_close.shape[1])
    sim = pfm.run_signals(entries, exits, all_close, {**BASE, "ts_stop": 0.08}, groups=groups)

    folds = [_Fold(*f) for f in _folds("2020-07-01", "2022-02-28")]
    table = windows.fold_positions(close.index, folds)
    out: dict[str, Any] = {"folds": []}
    for _, row in table.iterrows():
        train = stats.window_stats(sim.returns, row.train_start, row.train_end).loc["sharpe"]
        winner = str(train.idxmax())
        test = stats.window_stats(sim.returns[[winner]], row.test_start, row.test_end)
        counts = stats.trade_counts(
            sim.trades, row.test_start, row.test_end, groups=list(sim.value)
        )
        out["folds"].append(
            {
                "winner": winner,
                "train_sharpe": float(train[winner]),
                "test_sharpe": float(test.loc["sharpe", winner]),
                "test_max_drawdown": float(test.loc["max_drawdown", winner]),
                "test_trades": int(counts[winner]),
                "n_finite_train_sharpes": int(np.isfinite(train).sum()),
            }
        )
    return out


def run_all() -> dict[str, Any]:
    """All golden scenarios."""
    return {
        "signals_trailing": signals_trailing(),
        "signals_masked_time_stop": signals_masked_time_stop(),
        "weights_monthly": weights_monthly(),
        "mini_wfo": mini_wfo(),
    }
