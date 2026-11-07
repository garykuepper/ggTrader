"""Count raw and valid sweep combos per registry strategy (read-only)."""
import itertools
import sys
from ggTrader.lab.strategies import STRATEGY_REGISTRY
from ggTrader.lab.sweep import _is_valid_combo

for name, cls in STRATEGY_REGISTRY.items():
    raw = cls.sweep_params()
    keys = sorted(raw)
    prod = list(itertools.product(*(raw[k] for k in keys)))
    combos = [dict(zip(keys, v)) for v in prod]
    valid = [c for c in combos if _is_valid_combo(c)]
    print(f"{name}\t{cls.__name__}\t{cls.target_kind}\traw={len(combos)}\tvalid={len(valid)}\tkeys={','.join(keys)}")
print("total_registry", len(STRATEGY_REGISTRY))
