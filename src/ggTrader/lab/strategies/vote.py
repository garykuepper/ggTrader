"""N-of-M vote over precomputed voter arrays, with an independent exit and grid stacking.

vectorbt has no vote primitive (spec 5.1), so this is ggTrader glue. Each voter is computed
once per distinct parameter tuple (by ``voters.py``); a combination of the grid then only
*sums* the right precomputed boolean arrays and thresholds them. The per-combination results
are stacked side by side into one wide frame, ready for a single grouped simulation.

Semantics are the legacy ``EnsembleSignal._generate_signals`` ones:

* ``entries = sum(entry votes) >= min_agree``;
* ``exits = independent_exit | (sum(exit votes) >= min_agree_exit)`` where the independent
  voter's exit also counts as a vote, and ``min_agree_exit = None`` means ``min_agree``;
* a voter listed twice, or two voters sharing an array, vote twice (the volume-BB "double
  vote", spec OD3a, is kept on purpose).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BoolPair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class VoterArrays:
    """Precomputed arrays of one voter.

    Attributes:
        name: voter name (``bb``, ``rsi``, ...).
        keys: names of the combination parameters this voter depends on.
        arrays: ``param tuple (values of keys in order) -> (entry, exit)`` boolean
            ``dates x symbols`` arrays.
    """

    name: str
    keys: tuple[str, ...]
    arrays: Mapping[tuple[Hashable, ...], BoolPair]

    def param(self, combo: Mapping[str, Any]) -> tuple[Hashable, ...]:
        """The tuple that selects this voter's arrays for ``combo``."""
        return tuple(combo[k] for k in self.keys)

    def pick(self, combo: Mapping[str, Any]) -> BoolPair:
        """The ``(entry, exit)`` arrays for ``combo``."""
        return self.arrays[self.param(combo)]


def threshold(combo: Mapping[str, Any], entry_key: str, exit_key: str) -> tuple[int, int]:
    """``(min_agree, min_agree_exit)`` of a combination; a missing/None exit means entry."""
    entry = int(combo[entry_key])
    raw = combo.get(exit_key)
    return entry, entry if raw is None else int(raw)


def vote(
    voters: Sequence[VoterArrays],
    combos: Sequence[Mapping[str, Any]],
    index: pd.Index,
    columns: pd.MultiIndex,
    *,
    independent_exit: str | None = "rsi",
    entry_key: str = "min_agree",
    exit_key: str = "min_agree_exit",
    on_progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack the vote result of every combination into wide entry and exit frames.

    Args:
        voters: the active voters, in any order.
        combos: resolved combinations (every key the voters and thresholds need).
        index: the bar index of the arrays.
        columns: ``(levels..., symbol)`` index with ``len(combos) * n_symbols`` entries, one
            block of symbols per combination in ``combos`` order.
        independent_exit: voter whose exit fires on its own, or ``None``.
        entry_key, exit_key: combination names of the two thresholds.
        on_progress: optional callback with the number of combinations finished.

    Returns:
        ``(entries, exits)`` bool frames with ``columns``.
    """
    if not voters:
        raise ValueError("vote needs at least one voter")
    first = next(iter(voters[0].arrays.values()))[0]
    n_bars, n_sym = first.shape
    if len(columns) != len(combos) * n_sym:
        raise ValueError(f"columns has {len(columns)} entries, expected {len(combos) * n_sym}")
    by_name = {v.name: v for v in voters}
    if independent_exit is not None and independent_exit not in by_name:
        independent_exit = None  # legacy: only when the voter is active

    entries = np.empty((n_bars, len(combos) * n_sym), dtype=bool)
    exits = np.empty_like(entries)
    sums: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    for i, combo in enumerate(combos):
        key = tuple(v.param(combo) for v in voters)
        if key not in sums:
            ev = np.zeros((n_bars, n_sym), dtype=np.int8)
            xv = np.zeros((n_bars, n_sym), dtype=np.int8)
            for voter in voters:
                entry, exit_ = voter.pick(combo)
                ev += entry
                xv += exit_
            sums[key] = (ev, xv)
        ev, xv = sums[key]
        min_entry, min_exit = threshold(combo, entry_key, exit_key)
        sl = slice(i * n_sym, (i + 1) * n_sym)
        entries[:, sl] = ev >= min_entry
        exits[:, sl] = xv >= min_exit
        if independent_exit is not None:
            exits[:, sl] |= by_name[independent_exit].pick(combo)[1]
        if on_progress is not None:
            on_progress(i + 1)
    return (
        pd.DataFrame(entries, index=index, columns=columns),
        pd.DataFrame(exits, index=index, columns=columns),
    )
