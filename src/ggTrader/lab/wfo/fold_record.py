"""``FoldRecord``: the 28 values of one walk-forward fold, and the legacy dict they make."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

#: The fold dict's keys, in the order callers and the golden files see them.
FOLD_KEYS: tuple[str, ...] = (
    "fold_num",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "winner_combo",
    "winner_params",
    "train_score",
    "oos_score",
    "ndh_passed",
    "dsr_passed",
    "gates_passed",
    "ndh_density",
    "dsr_value",
    "wfe",
    "is_sharpe",
    "oos_sharpe",
    "halted",
    "halted_entering",
    "used_anchor",
    "anchor_combo",
    "anchor_params",
    "anchor_reason",
    "deployed_combo",
    "halt_cleared",
    "winner_trades_in_window",
    "winner_trades_whole_simulation",
    "od12_ndh_flip",
)


@dataclass(frozen=True)
class FoldRecord:
    """One fold's values; ``as_dict`` returns the legacy fold dict with the same objects.

    The annotations document intent and do not coerce: some values are numpy scalars, and the
    record holds exactly what the fold loop computed.

    ``anchor_combo`` and ``anchor_params`` are the fallback parameter set of this fold (fit on
    data up to ``train_end``); ``deployed_combo`` is what was deployed, which is the whole
    out-of-sample book when the anchor is used (review MAJOR-3).
    """

    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    winner_combo: str
    winner_params: dict[str, Any]
    train_score: float
    oos_score: float
    ndh_passed: bool
    dsr_passed: bool
    gates_passed: bool
    ndh_density: float
    dsr_value: float
    wfe: float | None
    is_sharpe: float
    oos_sharpe: float
    halted: bool
    halted_entering: bool
    used_anchor: bool
    anchor_combo: str
    anchor_params: dict[str, Any]
    anchor_reason: str | None
    deployed_combo: str
    halt_cleared: bool
    winner_trades_in_window: int
    winner_trades_whole_simulation: int
    od12_ndh_flip: bool

    def as_dict(self) -> dict[str, Any]:
        """The legacy fold dict: the same 28 keys in the same order, the same objects."""
        return {name: getattr(self, name) for name in FOLD_KEYS}
