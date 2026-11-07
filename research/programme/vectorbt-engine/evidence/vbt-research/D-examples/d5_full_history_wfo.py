"""D5: the vbt-native shape of ggTrader's WFO at full scale (500 symbols x 48 combos x 17 folds).

ONE Ens.run + ONE Portfolio.from_signals over the whole history (24,000 columns, 48 cash-sharing
groups), then per-fold train/test metrics by slicing the returns frame. Reports wall time and peak RSS.

Run: python d5_full_history_wfo.py [--symbols 500]
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, "/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/D-examples")
sys.path.insert(0, "/home/flynn/ggTrader/src")
from d1_wfo_splitter import Ens, synth  # noqa: E402

from ggTrader.lab.wfo import generate_folds  # noqa: E402


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


ap = argparse.ArgumentParser()
ap.add_argument("--symbols", type=int, default=500)
n = ap.parse_args().symbols
close = synth(n)
folds = generate_folds(pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC"))

t0 = time.time()
ens = Ens.run(close, min_agree=[2, 3, 4], min_agree_exit=[1, 2], bb_alpha=[2.0, 2.5], rsi_oversold=[25, 30],
              ema_fast=[10, 20], param_product=True)
t1 = time.time()
pf = vbt.Portfolio.from_signals(close, ens.entries, ens.exits, size=0.03, size_type="percent", fees=0.0005,
                                slippage=0.0005, init_cash=100_000.0, cash_sharing=True,
                                group_by=[0, 1, 2, 3, 4], freq="D")
t2 = time.time()
rets = pf.returns()  # (T, 48)
t3 = time.time()
rows = []
for k, f in enumerate(folds):
    tr = rets.loc[f.train_start : f.train_end].iloc[1:]
    te = rets.loc[f.test_start : f.test_end].iloc[1:]
    a = tr.vbt.returns(freq="D", year_freq="252 days")
    b = te.vbt.returns(freq="D", year_freq="252 days")
    s_tr = a.sharpe_ratio().replace([np.inf, -np.inf], np.nan)
    s_te = b.sharpe_ratio().replace([np.inf, -np.inf], np.nan)
    w = s_tr.idxmax()
    rows.append((k + 1, w, s_tr[w], s_te[w], a.max_drawdown()[w]))
t4 = time.time()
print(f"symbols={n} T={len(close)} cols={ens.entries.shape[1]} groups={rets.shape[1]}")
print(f"Ens.run {t1-t0:.1f}s | from_signals {t2-t1:.1f}s | returns {t3-t2:.1f}s | 17 folds slice+select {t4-t3:.2f}s")
print(f"peak RSS {rss():.0f} MB")
print(pd.DataFrame(rows, columns=["fold", "winner", "train_sharpe", "test_sharpe(winner)", "train_maxdd"]).head(5).to_string(index=False))
