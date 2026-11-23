"""Build explicit end-of-series rules from release metadata."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeadStockFrames:
    """Masks and assumptions for series ending within a window."""

    exits: pd.DataFrame
    entry_ok: pd.DataFrame
    weight_ok: pd.DataFrame
    meta: dict[str, Any]


def _positions(
    close: pd.DataFrame, spans: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not isinstance(close.index, pd.DatetimeIndex) or not close.index.is_monotonic_increasing:
        raise ValueError("close index must be a monotonic increasing DatetimeIndex")
    if not {"symbol", "last_date"}.issubset(spans.columns):
        raise ValueError("spans requires symbol and last_date")
    if spans["symbol"].duplicated().any():
        raise ValueError("duplicate symbols in spans")
    selected = spans.loc[spans.symbol.isin(close.columns)].copy()
    dates = pd.to_datetime(selected.last_date)
    if close.empty:
        return np.array([], dtype=int), np.array([], dtype=int), selected.iloc[:0]
    eligible = dates.notna() & (dates >= close.index[0]) & (dates <= close.index[-1])
    selected = selected.loc[eligible]
    dates = dates.loc[eligible]
    rows = close.index.searchsorted(dates, side="right") - 1
    cols = close.columns.get_indexer(selected.symbol)
    return rows, cols, selected


def forced_exit_frames(close: pd.DataFrame, spans: pd.DataFrame) -> DeadStockFrames:
    """Build exit and order masks using metadata dates, not missing prices.

    Args:
        close: Prices indexed by trading day.
        spans: Symbols and their actual last dates.

    Returns:
        Masks aligned to close and metadata describing their bias.
    """
    rows, cols, _ = _positions(close, spans)
    exits = np.zeros(close.shape, dtype=bool)
    exits[rows, cols] = True
    times = np.arange(len(close))[:, None]
    end = np.full(len(close.columns), len(close), dtype=int)
    end[cols] = rows
    return DeadStockFrames(
        pd.DataFrame(exits, index=close.index, columns=close.columns),
        pd.DataFrame(times < end, index=close.index, columns=close.columns),
        pd.DataFrame(times <= end, index=close.index, columns=close.columns),
        {
            "n_ended": len(rows),
            "biases": [
                "Exit at the last observed price is optimistic.",
                "Exit uses knowledge that the series ends.",
            ],
        },
    )


def apply_signal_rule(
    entries: pd.DataFrame, exits: pd.DataFrame, frames: DeadStockFrames
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Suppress entries at and after the end and add forced exits."""
    kept = entries & frames.entry_ok
    return kept, exits | frames.exits, int((entries & ~frames.entry_ok).to_numpy().sum())


def apply_weight_rule(weights: pd.DataFrame, frames: DeadStockFrames) -> pd.DataFrame:
    """Set an exit weight to zero and leave later bars without an order."""
    return weights.where(frames.weight_ok).mask(frames.exits, 0.0)


def haircut_close(close: pd.DataFrame, spans: pd.DataFrame, loss: float) -> pd.DataFrame:
    """Reduce terminal prices; a zero close may need care in the simulator.

    Args:
        close: Prices indexed by trading day.
        spans: End dates and optional successor and total-loss flags.
        loss: Fraction of terminal price lost, between zero and one.

    Returns:
        A copy with eligible terminal prices reduced.
    """
    if not 0 <= loss <= 1:
        raise ValueError("loss must be between zero and one")
    rows, cols, selected = _positions(close, spans)
    successor = selected.get("has_successor", pd.Series(False, index=selected.index)).to_numpy()
    total_loss = selected.get("total_loss", pd.Series(False, index=selected.index)).to_numpy()
    factors = np.where(total_loss, 0.0, np.where(successor, 1.0, 1.0 - loss))
    result = close.copy()
    for row, col, factor in zip(rows, cols, factors, strict=True):
        result.iloc[row, col] = result.iloc[row, col] * factor
    return result
