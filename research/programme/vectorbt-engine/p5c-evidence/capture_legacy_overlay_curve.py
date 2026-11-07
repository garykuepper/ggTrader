"""Freeze the legacy EnsembleSignal + simulate_signals sleeve curve of ``tests/paper/test_overlay.py``.

Run once BEFORE ``lab/simulate.py`` and the legacy ensemble were deleted (P5c). Output:
``tests/paper/golden/legacy_overlay_curve.npz`` (values and microsecond UTC index of the curve
that the rerouted ``paper.overlay.compute_sleeve_curve`` must still equal). After P5c it runs only
at the legacy tag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ggTrader.lab.data import STOCK_BASE_CONFIG  # noqa: E402
from ggTrader.lab.simulate import simulate_signals  # noqa: E402
from ggTrader.lab.strategies.ensemble import EnsembleSignal  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from tests.paper.test_overlay import _mock_ohlcv  # noqa: E402

symbols = [f"S{i:02d}" for i in range(14)]
ohlcv = _mock_ohlcv(symbols, n_days=700, end="2026-07-10")
asof = pd.Timestamp("2026-07-10", tz="UTC")
trim_start = asof - pd.Timedelta(days=90 + 20)
sym_cols = list(ohlcv.columns.get_level_values(0).unique())
close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)
ensemble = EnsembleSignal(LabConfig(min_history_bars=60))
plans = {}
for bar in close.loc[str(trim_start.date()) :].index:
    plan = ensemble.select(bar, ohlcv, sym_cols)
    if plan:
        plans[bar] = plan
targets = ensemble.to_targets(plans, ohlcv)
_r, equity_df, _d = simulate_signals({"ensemble": targets}, close, dict(STOCK_BASE_CONFIG), ohlcv=ohlcv)
legacy = equity_df["ensemble"].dropna()
legacy = legacy.loc[legacy.index >= trim_start]
dest = ROOT / "tests/paper/golden/legacy_overlay_curve.npz"
dest.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dest, values=legacy.to_numpy(), index=legacy.index.asi8, unit=str(legacy.index.unit))
print("wrote", dest, len(legacy), str(legacy.index.unit))
