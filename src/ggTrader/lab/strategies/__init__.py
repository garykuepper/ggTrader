"""Lab strategies on the engine contract; ``vote`` is the N-of-M glue.

Importing this package scans it (``registry.discover``) and registers every module that
implements the strategy contract in ``REGISTRY``; there is nothing to edit here to add a
strategy (see ``docs/adding_a_strategy.md``). The pre-rebuild strategy classes were deleted in
P5c; those that were not ported are listed in ``registry.LEGACY_ONLY_STRATEGIES`` and run at
git tag ``legacy-engine-vbt0285`` (``scripts/legacy_env.sh``).
"""

from __future__ import annotations

from ggTrader.lab.strategies import registry
from ggTrader.lab.strategies.registry import REGISTRY, get, signal_names, weight_names

registry.discover()

__all__ = ["REGISTRY", "get", "signal_names", "weight_names"]
