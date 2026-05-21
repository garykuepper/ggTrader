"""Strategy ABC (spec §5.7, refined in Phase 3.5).

A Strategy is a pure function ``(ts, feature_store) -> list[Signal]``. It must
not call brokers, place orders, or compute features inline.

Strategies declare all four attributes at the instance level (Phase 3.5 fix:
no more class/instance attr split). The ``FeatureStore`` Protocol lives in
``ggTrader.features.base`` and is widened to handle pair/universe features
via a ``DataFrame`` return type.

Usage patterns
--------------
The ``Universe`` API supports two patterns; both are valid:

  1. **iterate-all** (cross-sectional, pairs trading): the strategy iterates
     ``universe.members(ts)`` and emits one signal per member.
  2. **pick-one** (carry, regime gates): the strategy reaches into the Universe
     for a specific sub-element (e.g. ``universe.active_future(ts)``) and emits
     signals for a curated subset.

Use ``members(ts)`` when membership IS the signal set. Use a specialized
accessor on a Universe subclass when membership is a structured tuple
(e.g. ``CarryUniverse`` exposes ``spot`` and ``active_future``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ggTrader.core.signal import Signal
from ggTrader.core.universe import Universe
from ggTrader.features.base import FeatureStore


class Strategy(ABC):
    """Base class for all strategies. All four core attributes are set in
    ``__init__`` (instance-level — Phase 3.5 fix)."""

    def __init__(
        self,
        strategy_id: str,
        universe: Universe,
        required_features: list[str],
        timeframe: str,
    ) -> None:
        self.strategy_id = strategy_id
        self.universe = universe
        self.required_features = required_features
        self.timeframe = timeframe

    @abstractmethod
    def generate_signals(self, ts: datetime, feature_store: FeatureStore) -> list[Signal]:
        """Return signals for instruments in the universe at ``ts``."""


__all__ = ["FeatureStore", "Strategy"]
