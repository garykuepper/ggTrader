"""Freeze the legacy ``lab.simulate`` outputs that ``tests/engine/test_portfolio.py`` compared with.

Run once, BEFORE ``lab/simulate.py`` was deleted (P5c). The inputs are the seeded fixtures of
that test module; the outputs go to ``tests/engine/golden/legacy_simulate.npz`` so the engine
keeps an independent equity oracle after the legacy simulator is gone. After P5c this script runs
only at git tag legacy-engine-vbt0285 (scripts/legacy_env.sh) with the test fixtures copied over.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import ggTrader.lab.strategy as _strategy  # noqa: E402


class SignalTargets(NamedTuple):
    """The legacy signal container (``lab.strategy.SignalTargets``, trimmed away later in P5c)."""

    entries: pd.DataFrame
    exits: pd.DataFrame
    sizes: pd.DataFrame | None = None


_strategy.SignalTargets = SignalTargets  # test_portfolio imports it from here
from ggTrader.lab.simulate import compute_atr_stop, simulate_signals, simulate_weights  # noqa: E402
from tests.engine.panels import make_panel  # noqa: E402
from tests.engine.test_portfolio import (  # noqa: E402
    BASE,
    STOCK,
    _signals,
    _two_strategy_inputs,
)

out: dict[str, np.ndarray] = {}

extras = {
    "plain": {},
    "trail": {"ts_stop": 0.05},
    "take_profit": {"tp_stop": 0.08},
    "trail_and_tp": {"ts_stop": 0.05, "tp_stop": 0.08},
    "fees": {"FEES": 0.001},
}
for key, extra in extras.items():
    panel, targets, _e, _x, _c, _g = _two_strategy_inputs()
    _, value, diag = simulate_signals(targets, panel["close"], {**STOCK, **extra})
    out[f"signals_{key}_value"] = value.to_numpy()
    out[f"signals_{key}_n_trades"] = np.array([diag["alpha"]["n_trades"], diag["beta"]["n_trades"]])

panel, targets, *_ = _two_strategy_inputs()
rets, _, _ = simulate_signals(targets, panel["close"], STOCK)
out["signals_returns"] = rets.to_numpy()

panel = make_panel(seed=4, n_bars=400, n_syms=4)
e, x = _signals(panel, seed=21)
sizes = pd.DataFrame(
    np.random.default_rng(1).uniform(0.01, 0.05, e.shape), index=e.index, columns=e.columns
)
_, value, _ = simulate_signals(
    {"s": SignalTargets(entries=e, exits=x, sizes=sizes)}, panel["close"], STOCK
)
out["conviction_value"] = value.to_numpy()

panel = make_panel(seed=6, n_bars=500, n_syms=4)
e, x = _signals(panel, seed=31)
e.iloc[:30] = False
ohlcv = pd.concat(
    {s: pd.DataFrame({f: panel[f][s] for f in ("high", "low", "close")}) for s in panel["close"]},
    axis=1,
)
cfg = {**STOCK, "atr_mult": 2.0, "atr_period": 14}
_, value, _ = simulate_signals({"s": SignalTargets(entries=e, exits=x)}, panel["close"], cfg, ohlcv=ohlcv)
out["atr_value"] = value.to_numpy()

panel = make_panel(seed=6, n_bars=300, n_syms=5, holes=True)
with np.errstate(invalid="ignore"):
    old = compute_atr_stop(panel["high"], panel["low"], panel["close"], 14, 2.0)
    out["atr_stop_ffilled"] = old.ffill().fillna(np.inf).to_numpy()

panel = make_panel(seed=8, n_bars=400, n_syms=6)
rng = np.random.default_rng(2)
targets = {}
for name, syms in {"w1": ["S00", "S01", "S02"], "w2": ["S03", "S04", "S05"]}.items():
    w = pd.DataFrame(np.nan, index=panel["close"].index, columns=syms)
    for d in range(20, 400, 21):
        w.iloc[d] = rng.dirichlet(np.ones(3)) * 0.9
    targets[name] = w
_, value, diag = simulate_weights(targets, panel["close"], BASE)
out["weights_value"] = value.to_numpy()
out["weights_n_trades"] = np.array([diag["w1"]["n_trades"], diag["w2"]["n_trades"]])

panel = make_panel(seed=8, n_bars=120, n_syms=3)
w = pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)
w.iloc[10] = 0.3
_, value, _ = simulate_weights({"live": w, "empty": w.iloc[:, :0]}, panel["close"], BASE)
out["weights_empty_value"] = value.to_numpy()

dest = ROOT / "tests/engine/golden/legacy_simulate.npz"
np.savez_compressed(dest, **out)
print("wrote", dest, dest.stat().st_size, "bytes;", len(out), "arrays")
