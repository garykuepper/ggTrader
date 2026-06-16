"""Strategy protocol and config for the lab research bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Protocol, Union

import pandas as pd

Plan = List[Dict[str, Any]]  # JSON-able selection records, each with at least "symbol"


class SignalTargets(NamedTuple):
    """Return type for signal-based strategies' to_targets method."""

    entries: pd.DataFrame  # (time x symbol) boolean — True = entry bar
    exits: pd.DataFrame  # (time x symbol) boolean — True = exit bar


@dataclass
class LabConfig:
    """Tunables shared by lab strategies and the harness."""

    top_n: int = 50
    lookback: int = 252  # trailing bars for the momentum measurement window
    skip: int = 21  # most-recent bars excluded (12-1 momentum)
    min_history_bars: int = 400  # required non-NaN closes to be eligible
    max_stocks: int | None = None  # cap the per-rebalance universe (deterministic)


class Strategy(Protocol):
    """A lab strategy: point-in-time select, then a whole-window target matrix.

    ``target_kind`` is "weights" (simulated via Portfolio.from_orders) or
    "signals" (via from_signals — added in Plan 2).
    """

    name: str
    target_kind: str

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        """JSON-able selections; MUST be a pure function of data <= asof."""
        ...

    def to_targets(
        self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame
    ) -> Union[pd.DataFrame, SignalTargets]:
        """Whole-window target matrix from per-rebalance plans.

        Weight strategies return pd.DataFrame (time x symbol, weight values).
        Signal strategies return SignalTargets(entries, exits) boolean frames.
        """
        ...
