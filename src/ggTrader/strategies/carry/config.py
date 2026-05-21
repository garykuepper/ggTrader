"""Legacy thin shim — Phase 3 had a strategy-specific YAML loader here.

Phase 3.5 replaces it with the generic ``ggTrader.strategies.loader``
(strategies declare ``from_config`` classmethods). Keep these names exported
so existing callers and tests keep working.
"""

from __future__ import annotations

from pathlib import Path

from ggTrader.strategies.carry.cash_and_carry import CashAndCarryBTC
from ggTrader.strategies.loader import build_strategy_from_yaml, load_strategy_yaml


def load_config(path: str | Path) -> dict:
    """Phase 3 compatibility: returns the raw dict instead of a Pydantic model."""
    return load_strategy_yaml(path)


def build_strategy(config: dict | object) -> CashAndCarryBTC:
    """Accept either a path-loaded dict or a Phase-3-style Pydantic model."""
    if isinstance(config, dict):
        return CashAndCarryBTC.from_config(config)
    # back-compat: Pydantic-model-style: dump to dict and recurse.
    if hasattr(config, "model_dump"):
        return CashAndCarryBTC.from_config(config.model_dump())
    raise TypeError(f"unsupported config type: {type(config)}")


__all__ = ["build_strategy", "build_strategy_from_yaml", "load_config"]
