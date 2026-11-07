"""D4: ggTrader's wfo.py re-simulates the 48-combo grid on ohlcv.loc[:window_end] for EVERY fold
(anchor sweep + train sweep + test sweep). Because signals and vbt's simulator are causal, the equity
on bar t does not depend on later bars, so one full-history simulation sliced per fold should be
identical. This script checks that claim numerically with vbt.Portfolio.from_signals.

Run: PYTHONPATH=/home/flynn/ggTrader/src python d4_prefix_causality.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, "/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/D-examples")
from d1_wfo_splitter import Ens, synth  # noqa: E402

from ggTrader.lab.metrics import curve_stats  # noqa: E402
from ggTrader.lab.wfo import generate_folds  # noqa: E402

close = synth(40)
GRID = dict(min_agree=[2, 3, 4], min_agree_exit=[1, 2], bb_alpha=[2.0, 2.5], rsi_oversold=[25, 30], ema_fast=[10, 20])


def simulate(px: pd.DataFrame) -> pd.Series:
    ens = Ens.run(px, **GRID, param_product=True)
    pf = vbt.Portfolio.from_signals(
        px, ens.entries, ens.exits, size=0.03, size_type="percent", fees=0.0005, slippage=0.0005,
        init_cash=100_000.0, cash_sharing=True, group_by=[0, 1, 2, 3, 4], freq="D",
    )
    return pf.value()  # (T, 48 groups)


full = simulate(close)
folds = generate_folds(pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC"))
worst_eq = 0.0
worst_sh = 0.0
for i in (0, 8, 16):
    f = folds[i]
    prefix = simulate(close.loc[: f.train_end])  # what wfo._sweep_fold does for the train/anchor sweeps
    common = prefix.index
    d = (prefix - full.loc[common]).abs().max().max()
    worst_eq = max(worst_eq, float(d))
    # fold metric exactly as wfo._sweep_fold computes it, from each equity source
    def sharpe_of(eq: pd.DataFrame) -> pd.Series:
        out = {}
        for col in eq.columns:
            w = eq[col].loc[f.train_start : f.train_end].dropna()
            out[col] = curve_stats(100_000 * (w / w.iloc[0]))["sharpe"]
        return pd.Series(out)

    s_pre, s_full = sharpe_of(prefix), sharpe_of(full)
    sd = float(np.nanmax(np.abs(s_pre.values - s_full.values)))
    worst_sh = max(worst_sh, sd)
    print(f"fold {i+1:2d}: prefix sim to {f.train_end.date()} vs full sim sliced -> max |equity diff| = {d:.3e}; "
          f"max |Sharpe diff| over 48 combos = {sd:.3e}")
print(f"WORST: equity {worst_eq:.3e}, sharpe {worst_sh:.3e}")
