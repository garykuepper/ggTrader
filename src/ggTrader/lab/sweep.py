"""Parameter sweep: grid generation, vectorized orchestration, results display."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Type


def _is_valid_combo(params: Dict[str, Any]) -> bool:
    """Filter combos where a 'fast' param is >= a corresponding 'slow' param."""
    fast_keys = sorted(k for k in params if "fast" in k)
    slow_keys = sorted(k for k in params if "slow" in k)
    for fk, sk in zip(fast_keys, slow_keys):
        if params[fk] >= params[sk]:
            return False
    return True


def build_grid(
    strategy_cls: Type,
    overrides: Optional[Dict[str, list]] = None,
) -> List[Dict[str, Any]]:
    """Cartesian product of sweep params, filtering invalid combos."""
    raw = strategy_cls.sweep_params()
    if overrides:
        raw = {**raw, **overrides}
    keys = sorted(raw.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*(raw[k] for k in keys))]
    return [c for c in combos if _is_valid_combo(c)]


def combo_name(strategy_name: str, params: Dict[str, Any]) -> str:
    """Deterministic label from strategy name + sorted param key-value pairs."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return strategy_name + "__" + "_".join(parts)
