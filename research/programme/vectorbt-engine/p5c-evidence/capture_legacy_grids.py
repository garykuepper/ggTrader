"""Freeze the legacy parameter grids and constructor defaults of the ported strategies.

Run once BEFORE the legacy strategy classes were deleted (P5c). Output:
``tests/lab/strategies_v2/golden/legacy_grids.json`` (grid combinations in legacy ``build_grid``
order and the attributes a default-constructed legacy class carried), which the strategy tests
compare the new ``ParamGrid`` and ``defaults`` with. After P5c the script runs only at tag
legacy-engine-vbt0285.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.lab.strategies.commodity_trend import CommodityTrendStrategy  # noqa: E402,F401
from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.sweep import build_grid  # noqa: E402

PORTED = [
    "ensemble", "bb_reversion", "rsi_reversion", "ema_cross", "macd_divergence",
    "volume_bb_reversion", "conviction_bb", "xs_momentum", "cross_asset_sleeve", "retail_attention",
]
out = {}
for name in PORTED:
    cls = STRATEGY_REGISTRY[name]
    inst = cls(LabConfig())
    attrs = {k: v for k, v in vars(inst).items() if isinstance(v, (int, float, str, bool, type(None)))}
    attrs["voters"] = list(getattr(inst, "voters", [])) or None
    out[name] = {"grid": build_grid(cls), "constructor_attributes": attrs}
dest = ROOT / "tests/lab/strategies_v2/golden/legacy_grids.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest, {k: len(v["grid"]) for k, v in out.items()})
