"""Freeze the legacy weight-strategy targets and equity that ``test_weights.py`` compared with (P5c).

Run once BEFORE the legacy classes (``CrossSectionalMomentum``, ``CrossAssetSleeveStrategy``,
``RetailAttentionStrategy``) and ``lab/simulate.py`` were deleted. Output:
``tests/lab/strategies_v2/golden/legacy_weights.npz``. After P5c it runs only at the legacy tag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ggTrader.data.bars import to_legacy_frame  # noqa: E402
from ggTrader.lab.data import rebalance_dates  # noqa: E402
from ggTrader.lab.simulate import simulate_weights  # noqa: E402
from ggTrader.lab.strategies.cross_asset_sleeve import CrossAssetSleeveStrategy  # noqa: E402
from ggTrader.lab.strategies.momentum import CrossSectionalMomentum  # noqa: E402
from ggTrader.lab.strategies.retail_attention import RetailAttentionStrategy  # noqa: E402
from ggTrader.lab.strategies.v2.cross_asset_sleeve import UNIVERSE, CrossAssetSleeve  # noqa: E402
from ggTrader.lab.strategies.v2.xs_momentum import XsMomentum  # noqa: E402
from ggTrader.lab.strategies.v2.retail_attention import RetailAttention  # noqa: E402
from ggTrader.lab.strategies.v2.contract import ParamGrid  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from tests.lab.strategies_v2.helpers import make_bars  # noqa: E402
from tests.lab.strategies_v2.test_weights import (  # noqa: E402
    CFG,
    _legacy_targets,
    _rebal,
    _synthetic_interest,
)

out: dict[str, np.ndarray] = {}


def put(key: str, frame: pd.DataFrame) -> None:
    out[f"{key}|values"] = frame.to_numpy(dtype=float)
    out[f"{key}|columns"] = np.array(list(frame.columns), dtype=str)
    out[f"{key}|index"] = frame.index.asi8


mom_bars = make_bars(seed=41, n_bars=900, n_syms=14, holes=True)
rng = np.random.default_rng(0)
mask = rng.random(mom_bars.close.shape) < 0.8
eligible = pd.DataFrame(mask, index=mom_bars.close.index, columns=mom_bars.close.columns)
rebal = _rebal(mom_bars)
targets = {}
for i, combo in enumerate(XsMomentum.grid.combos()):
    t = _legacy_targets(
        CrossSectionalMomentum, combo, mom_bars, rebal,
        lambda a: [s for s in mom_bars.symbols if eligible.at[a, s]],
    )
    put(f"xs_momentum|{i}", t)
    targets[f"c{i}"] = t
_, old_value, _ = simulate_weights(targets, mom_bars.close, CFG)
out["xs_momentum|equity"] = old_value.to_numpy()

sleeve_bars = make_bars(seed=7, n_bars=700, n_syms=5, symbols=sorted(UNIVERSE))
rebal = _rebal(sleeve_bars, "2020-05-01")
for trend in (True, False):
    grid = CrossAssetSleeve.grid if trend else ParamGrid.single({"trend": False})
    for i, combo in enumerate(grid.combos()):
        t = _legacy_targets(
            CrossAssetSleeveStrategy, {**combo, "top_n": 50}, sleeve_bars, rebal,
            lambda a: [], trend=trend,
        )
        put(f"sleeve|{trend}|{i}", t)

rebal = _rebal(mom_bars, "2021-06-01")
loader = _synthetic_interest()
for i, combo in enumerate(RetailAttention.grid.combos()):
    strat = RetailAttentionStrategy(LabConfig(min_history_bars=1), _interest_loader=loader, **combo)
    data = to_legacy_frame(mom_bars)
    plans = {a: strat.select(a, data.loc[:a], mom_bars.symbols) for a in rebal}
    put(f"retail|{i}", strat.to_targets(plans, data))

dest = ROOT / "tests/lab/strategies_v2/golden/legacy_weights.npz"
np.savez_compressed(dest, **out)
print("wrote", dest, dest.stat().st_size, "bytes;", len(out), "arrays")
