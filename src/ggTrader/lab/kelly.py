"""Pooled, causal Kelly-criterion sizing for signal-based lab strategies.

Given a strategy's own (entries, exits, close) signal matrices, this module
reconstructs the round-trip trades those signals imply, estimates a pooled
expanding-window Kelly fraction from them, and turns that into a per-bar,
per-symbol position-size matrix suitable for SignalTargets.sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _symbol_trades(
    entry_positions: np.ndarray, exit_positions: np.ndarray, prices: np.ndarray
) -> list[tuple[int, int, float]]:
    """Pair one symbol's entry/exit bar positions into closed round trips.

    Mirrors vbt's from_signals semantics: while a position is open, further
    entries are ignored; a trade closes on the next exit at or after its
    entry. A trailing entry with no closing exit left is unrealized and
    dropped (it can't contribute a win/loss to the edge estimate yet).
    """
    trades: list[tuple[int, int, float]] = []
    ei = 0
    xi = 0
    n_entries = len(entry_positions)
    n_exits = len(exit_positions)
    in_position = False
    entry_idx = -1
    while True:
        if not in_position:
            if ei >= n_entries:
                break
            entry_idx = entry_positions[ei]
            ei += 1
            in_position = True
            while xi < n_exits and exit_positions[xi] < entry_idx:
                xi += 1
        else:
            if xi >= n_exits:
                break  # no closing exit remains -> unrealized, drop it
            exit_idx = exit_positions[xi]
            xi += 1
            ret = float(prices[exit_idx] / prices[entry_idx] - 1.0)
            trades.append((entry_idx, exit_idx, ret))
            in_position = False
            while ei < n_entries and entry_positions[ei] <= exit_idx:
                ei += 1
    return trades


def extract_trades(entries: pd.DataFrame, exits: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Pair each symbol's entry/exit signal bars into completed round-trip trades.

    Returns columns [symbol, entry_time, exit_time, ret], sorted by exit_time
    ascending (ties broken by symbol) — the ordering the rest of the
    Kelly-sizing pipeline assumes.
    """
    index = entries.index
    records = []
    for col in entries.columns:
        entry_positions = np.flatnonzero(entries[col].to_numpy())
        exit_positions = np.flatnonzero(exits[col].to_numpy())
        prices = close[col].to_numpy()
        for entry_idx, exit_idx, ret in _symbol_trades(entry_positions, exit_positions, prices):
            records.append(
                {
                    "symbol": col,
                    "entry_time": index[entry_idx],
                    "exit_time": index[exit_idx],
                    "ret": ret,
                }
            )
    trades = pd.DataFrame.from_records(
        records, columns=["symbol", "entry_time", "exit_time", "ret"]
    )
    if trades.empty:
        return trades
    return trades.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
