"""Single-source derivation of strategy names/builders from STRATEGY_REGISTRY.

Helpers import STRATEGY_REGISTRY lazily (inside functions) to avoid an import
cycle: strategies/__init__.py imports the strategy modules to build the
registry, so those modules cannot read it at their own import time.
"""

from __future__ import annotations

from typing import Any

from ggTrader.lab.strategy import LabConfig


def _registry() -> dict[str, Any]:
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    return STRATEGY_REGISTRY


def signal_strategy_names() -> tuple[str, ...]:
    return tuple(n for n, c in _registry().items() if c.target_kind == "signals")


def weight_strategy_names() -> tuple[str, ...]:
    return tuple(n for n, c in _registry().items() if c.target_kind == "weights")


def all_strategy_names() -> tuple[str, ...]:
    return tuple(_registry())


def signal_registry() -> dict[str, Any]:
    return {n: c for n, c in _registry().items() if c.target_kind == "signals"}


def build_strategy(name: str, cfg: LabConfig) -> Any:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown strategy {name!r}. Available: {tuple(reg)}")
    return reg[name](cfg)
