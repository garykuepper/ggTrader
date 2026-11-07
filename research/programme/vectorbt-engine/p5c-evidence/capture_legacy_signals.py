"""Freeze the legacy per-combination signals the strategy-port tests compared with (P5c).

Run once BEFORE the legacy strategy classes were deleted. For the fixtures of
``tests/lab/strategies_v2/test_ensemble.py``, ``test_voter_strategies.py`` and
``test_conviction_bb.py`` (seeded synthetic bars) it records, per legacy ``build_grid`` combination,
the entry/exit cells (bit-packed) and the conviction sizes, into
``tests/lab/strategies_v2/golden/legacy_signals.npz``. The tests then apply their unchanged
"equal after burn-in" check to these frozen arrays. After P5c the script runs only at tag
legacy-engine-vbt0285.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ggTrader.data.bars import to_legacy_frame  # noqa: E402
from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: E402
from ggTrader.lab.strategies.ensemble import EnsembleSignal  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.sweep import build_grid, combo_name  # noqa: E402
from tests.lab.strategies_v2.helpers import make_bars  # noqa: E402

out: dict[str, np.ndarray] = {}


def pack(prefix: str, name: str, frame) -> None:
    arr = frame.to_numpy(dtype=bool)
    out[f"{prefix}|{name}|shape"] = np.array(arr.shape)
    out[f"{prefix}|{name}"] = np.packbits(arr, axis=None)


def record(strategy: str, cls, bars, sizes: bool = False) -> None:
    legacy = cls(LabConfig()).sweep_signals(build_grid(cls), bars.symbols, to_legacy_frame(bars))
    for cname, targets in legacy.items():
        pack(f"{strategy}|{cname}", "entries", targets.entries)
        pack(f"{strategy}|{cname}", "exits", targets.exits)
        if sizes:
            out[f"{strategy}|{cname}|sizes"] = targets.sizes.to_numpy(dtype=float)


# ensemble: 48 combinations on the test_ensemble fixture + the legacy-default control
bars = make_bars(seed=21, n_bars=900, n_syms=10, holes=True)
record("ensemble", EnsembleSignal, bars)
old = EnsembleSignal(LabConfig())._generate_signals(bars.close, bars.volume)
pack("ensemble_live_control", "entries", old.entries)
pack("ensemble_live_control", "exits", old.exits)

# the five voters on the test_voter_strategies fixture, plus default-constructor exits
bars = make_bars(seed=33, n_bars=800, n_syms=8, holes=True)
for name in ("ema_cross", "bb_reversion", "rsi_reversion", "macd_divergence", "volume_bb_reversion"):
    cls = STRATEGY_REGISTRY[name]
    record(name, cls, bars)
    legacy = cls(LabConfig())
    targets = legacy.to_targets(
        {bars.close.index[-1]: [{"symbol": s, "weight": 0.0} for s in bars.symbols]},
        to_legacy_frame(bars),
    )
    pack(f"{name}_defaults", "exits", targets.exits)

# conviction_bb on the test_conviction_bb fixture, with sizes
bars = make_bars(seed=51, n_bars=800, n_syms=8, holes=True)
record("conviction_bb", STRATEGY_REGISTRY["conviction_bb"], bars, sizes=True)

dest = ROOT / "tests/lab/strategies_v2/golden/legacy_signals.npz"
np.savez_compressed(dest, **out)
print("wrote", dest, dest.stat().st_size, "bytes;", len(out), "arrays")
