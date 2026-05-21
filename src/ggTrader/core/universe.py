"""Universe: the set of Instruments a strategy operates on at a point in time.

Concrete universes (SingleSymbol, StaticList, TopNByDollarVolume, Pair,
SP500Members) live in this module as they are needed. Phase 0 only defines
the abstract base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ggTrader.core.instrument import Instrument


class Universe(ABC):
    @abstractmethod
    def members(self, ts: datetime) -> list[Instrument]:
        """Members at point-in-time. Must be survivorship-bias-free."""

    @abstractmethod
    def is_dynamic(self) -> bool:
        """True if membership changes over time (e.g., top-N rebalances)."""
