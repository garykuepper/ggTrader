"""Generic strategy loader: read YAML, locate Strategy subclass by ``strategy_class``,
dispatch to its ``from_config`` classmethod.

YAML convention:

    strategy_id: cash_and_carry_btc
    strategy_class: ggTrader.strategies.carry.cash_and_carry:CashAndCarryBTC
    # ... arbitrary strategy-specific config follows

The loader pulls ``strategy_class`` out, imports it, and calls
``klass.from_config(raw_dict)``. Each new strategy ships its own
``from_config`` classmethod and any Pydantic schemas it wants for validation.
No more per-strategy boilerplate factory function (Phase 3.5 fix).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from ggTrader.strategies.base import Strategy


def load_strategy_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return raw


def build_strategy_from_yaml(path: str | Path) -> Strategy:
    raw = load_strategy_yaml(path)
    cls_path = raw.get("strategy_class")
    if not cls_path:
        raise KeyError(f"{path}: missing required 'strategy_class' field")
    module_name, _, attr_name = cls_path.partition(":")
    if not module_name or not attr_name:
        raise ValueError(
            f"{path}: strategy_class must be 'module.path:ClassName'; got {cls_path!r}"
        )
    module = importlib.import_module(module_name)
    klass = getattr(module, attr_name)
    if not (isinstance(klass, type) and issubclass(klass, Strategy)):
        raise TypeError(f"{cls_path} is not a Strategy subclass")
    from_config = getattr(klass, "from_config", None)
    if from_config is None:
        raise TypeError(f"{cls_path} must define a from_config(raw: dict) classmethod")
    result: Strategy = from_config(raw)
    return result


__all__ = ["build_strategy_from_yaml", "load_strategy_yaml"]
