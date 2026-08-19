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


def expanding_kelly_fraction(trades: pd.DataFrame, min_trades: int = 10) -> pd.Series:
    """Pooled expanding Kelly fraction f* = W - (1-W)/R, one value per row of
    `trades` (must be pre-sorted by exit_time ascending, as extract_trades
    returns). f*.iloc[i] is computed only from trades[0:i+1] — it never
    looks at a later trade.

    NaN until `min_trades` trades have closed, or while there are zero wins
    or zero losses in the pool so far (the payoff ratio R is undefined
    without both).
    """
    if trades.empty:
        return pd.Series(dtype=float)
    ret = trades["ret"].to_numpy()
    is_win = ret > 0
    is_loss = ret < 0
    n = np.arange(1, len(ret) + 1)
    win_count = np.cumsum(is_win)
    loss_count = np.cumsum(is_loss)
    win_rate = win_count / n
    sum_win = np.cumsum(np.where(is_win, ret, 0.0))
    sum_loss = np.cumsum(np.where(is_loss, -ret, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_win = np.where(win_count > 0, sum_win / np.maximum(win_count, 1), np.nan)
        avg_loss = np.where(loss_count > 0, sum_loss / np.maximum(loss_count, 1), np.nan)
        payoff_ratio = avg_win / avg_loss
        f_star = win_rate - (1.0 - win_rate) / payoff_ratio
    valid = (n >= min_trades) & (win_count > 0) & (loss_count > 0)
    f_star = np.where(valid, f_star, np.nan)
    return pd.Series(f_star, index=trades["exit_time"], name="f_star")


def kelly_fraction_asof(f_star: pd.Series, asof: pd.Timestamp) -> float:
    """f* using only trades closed strictly before `asof`; NaN if none qualify.

    `f_star` must be indexed by exit_time, sorted ascending (the output of
    expanding_kelly_fraction).
    """
    if f_star.empty:
        return float("nan")
    pos = f_star.index.searchsorted(asof, side="left")
    if pos == 0:
        return float("nan")
    return float(f_star.iloc[pos - 1])


def kelly_sizes(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    close: pd.DataFrame,
    *,
    kelly_multiplier: float,
    base_size: float,
    max_size: float,
    min_trades: int = 10,
) -> pd.DataFrame:
    """Per-bar, per-symbol position size (time x symbol) for SignalTargets.sizes.

    NaN everywhere `entries` is False. Where `entries` is True:
    `kelly_multiplier * f*`, capped at `max_size`, using only the causal
    pooled expanding Kelly fraction as of that entry's time. Falls back to
    `base_size` whenever there isn't yet a positive measurable edge (fewer
    than `min_trades` closed trades, or f* <= 0).

    Vectorized: all of a column's entry times are looked up against the
    shared `f_star` index in one `searchsorted` call (batching what
    `kelly_fraction_asof` does per scalar timestamp -- "strictly before
    asof" is `side="left"`, position 0 -> NaN, same as the scalar version),
    then the per-entry size formula runs as one masked numpy expression
    instead of a Python-level branch per entry.
    """
    trades = extract_trades(entries, exits, close)
    f_star = expanding_kelly_fraction(trades, min_trades=min_trades)

    sizes = pd.DataFrame(np.nan, index=entries.index, columns=entries.columns)
    if f_star.empty:
        f_vals = np.array([])
        f_index = pd.DatetimeIndex([])
    else:
        f_vals = f_star.to_numpy()
        f_index = f_star.index

    for col in entries.columns:
        entry_times = entries.index[entries[col].to_numpy()]
        if len(entry_times) == 0:
            continue
        if len(f_index) == 0:
            f = np.full(len(entry_times), np.nan)
        else:
            pos = f_index.searchsorted(entry_times, side="left")
            f = np.where(pos > 0, f_vals[np.clip(pos - 1, 0, None)], np.nan)

        has_edge = ~np.isnan(f) & (f > 0)
        size = np.where(has_edge, np.minimum(kelly_multiplier * f, max_size), base_size)
        sizes.loc[entry_times, col] = size
    return sizes
