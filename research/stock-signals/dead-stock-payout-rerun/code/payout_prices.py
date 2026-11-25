"""Per-stock dead-stock exit prices, evidence-set pin parsing and payout coverage counts."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

import ft_replay
import pandas as pd

READINGS: tuple[float, ...] = (1.0, 0.5, 0.0)
EVIDENCE_SETS: tuple[str, ...] = ("confirmed", "all")
ANCHOR_SET = "none"
VALID_EVIDENCE_SETS: frozenset[str] = frozenset((*EVIDENCE_SETS, ANCHOR_SET))
_SPELL_SUFFIX_RE = re.compile(r"~\d+$")


def base_symbol(column: str) -> str:
    """Return a price-column name with any trailing ``~k`` later-spell suffix removed.

    Args:
        column: Price-frame column name (for example ``PERMNO-5~2`` or ``AAPL``).

    Returns:
        The base symbol (``PERMNO-5`` for ``PERMNO-5~2``; unchanged if there is no suffix).
    """
    return _SPELL_SUFFIX_RE.sub("", column)


def _validate_loss(value: float, label: str) -> float:
    """Ensure a loss fraction is finite and lies in ``[0.0, 1.0]``."""
    loss = float(value)
    if not math.isfinite(loss) or not 0.0 <= loss <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1, got {value!r}")
    return loss


def pinned_losses(rows: Sequence[Mapping[str, str]], evidence_set: str) -> dict[str, float]:
    """Map base symbol to its pinned loss fraction for one evidence set.

    Args:
        rows: Payout-table rows (text values with ``symbol``, ``suggested_loss`` and
            ``match_confidence``).
        evidence_set: ``"confirmed"`` (numeric loss and ``match_confidence == "confirmed"``),
            ``"all"`` (every row with a numeric loss), or ``"none"`` (empty map).

    Returns:
        Dictionary of base symbol to loss fraction in ``[0.0, 1.0]``.

    Raises:
        ValueError: Unknown ``evidence_set``, non-numeric or out-of-range loss, blank symbol on a
            pinned row, or two rows for the same symbol with conflicting numeric losses.
    """
    if evidence_set not in VALID_EVIDENCE_SETS:
        raise ValueError(
            f"unknown evidence_set {evidence_set!r}; expected one of {sorted(VALID_EVIDENCE_SETS)}"
        )
    if evidence_set == ANCHOR_SET:
        return {}

    pinned: dict[str, float] = {}
    for row in rows:
        confidence = (row.get("match_confidence") or "").strip()
        if evidence_set == "confirmed" and confidence != "confirmed":
            continue
        raw_loss = str(row.get("suggested_loss") or "").strip()
        if not raw_loss:
            continue
        try:
            parsed = float(raw_loss)
        except ValueError as exc:
            raise ValueError(f"non-numeric suggested_loss {raw_loss!r} in payout table") from exc
        loss = _validate_loss(parsed, f"suggested_loss for {row.get('symbol')!r}")
        sym = base_symbol(str(row.get("symbol") or "").strip())
        if not sym:
            raise ValueError("pinned payout row has an empty symbol")
        if sym in pinned:
            if pinned[sym] != loss:
                raise ValueError(
                    f"conflicting suggested_loss values for {sym}: {pinned[sym]!r} vs {loss!r}"
                )
            continue
        pinned[sym] = loss
    return pinned


def exit_prices(
    close: pd.DataFrame,
    spans: pd.DataFrame,
    reading: float,
    pinned: Mapping[str, float],
) -> pd.DataFrame:
    """Compute per-column exit prices using ``reading`` by default and ``pinned`` overrides.

    Calls ``ft_replay.exit_close`` once per distinct loss value actually needed (``reading`` plus
    each distinct pinned loss present in ``close.columns``) and splices whole columns, so living
    columns and successor spells keep the exact behavior of ``ft_replay.exit_close``.

    Args:
        close: Daily closes, trading days by symbols (never mutated).
        spans: Release span rows for ending series (never mutated).
        reading: Fallback dead-stock loss fraction in ``[0.0, 1.0]`` for unpinned dead stocks.
        pinned: Base symbol to pinned loss fraction in ``[0.0, 1.0]``.

    Returns:
        Fresh DataFrame with dying stocks' last bars written down to their applicable sale prices.

    Raises:
        ValueError: Any loss outside ``[0.0, 1.0]``.
    """
    reading_loss = _validate_loss(reading, "reading")
    for sym, val in pinned.items():
        _validate_loss(val, f"pinned loss for {sym!r}")

    cols_by_loss: dict[float, list[Any]] = {}
    for col in close.columns:
        base = base_symbol(str(col))
        if base in pinned:
            col_loss = float(pinned[base])
            if col_loss != reading_loss:
                cols_by_loss.setdefault(col_loss, []).append(col)

    result = ft_replay.exit_close(close, spans, reading_loss)
    if not cols_by_loss:
        return result

    # Copy once before splicing so a custom or mocked exit_close cannot alias close.
    out = result.copy()
    for loss_val in sorted(cols_by_loss):
        cols = cols_by_loss[loss_val]
        frame = ft_replay.exit_close(close, spans, loss_val)
        out[cols] = frame[cols]
    return out


def held_dead_symbols(trades: Iterable[Any], successors: frozenset[str]) -> list[str]:
    """Return sorted distinct base symbols of held non-successor dead-stock exits.

    Args:
        trades: Executed ``ft_replay.Trade`` records for one ledger run.
        successors: Full column names whose series continue in a later CRSP spell.

    Returns:
        Sorted list of unique base symbols sold with ``DEAD_EXIT_REASON`` and not in
        ``successors``.
    """
    return sorted(
        {
            base_symbol(str(trade.symbol))
            for trade in trades
            if trade.reason == ft_replay.DEAD_EXIT_REASON and str(trade.symbol) not in successors
        }
    )


def coverage_of(
    held: Sequence[str],
    pinned: Mapping[str, float],
    table_symbols: Collection[str],
) -> dict[str, Any]:
    """Count how many held dead stocks are pinned, unpinned, or missing from the payout table.

    Args:
        held: Base symbols of held dead stocks (duplicates are collapsed).
        pinned: Base symbol to pinned loss for the active evidence set.
        table_symbols: All ``symbol`` values present in the payout table.

    Returns:
        Coverage dictionary with counts and sorted symbol lists (``missing_from_table`` is a
        subset of ``unpinned_names``).
    """
    held_names = sorted(set(held))
    table_set = {base_symbol(str(sym).strip()) for sym in table_symbols if str(sym).strip()}
    pinned_names = [sym for sym in held_names if sym in pinned]
    unpinned_names = [sym for sym in held_names if sym not in pinned]
    missing_from_table = [sym for sym in unpinned_names if sym not in table_set]
    return {
        "held": len(held_names),
        "pinned": len(pinned_names),
        "unpinned": len(unpinned_names),
        "unpinned_names": unpinned_names,
        "missing_from_table": missing_from_table,
        "pinned_names": pinned_names,
    }
