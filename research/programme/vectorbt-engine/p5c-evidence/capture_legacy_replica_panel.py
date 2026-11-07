"""Capture the LEGACY replica signal panel (pre-deletion) on the frozen tape for a port check."""
import sys, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "/home/flynn/ggTrader-vbt12/src")
from ggTrader.data.bars import load_bars, to_legacy_frame
from ggTrader.lab.data import eligible_at, equity_universe_between
from ggTrader.lab.replica import build_signal_panel, run_replica, ReplicaConfig
from ggTrader.lab.strategy import LabConfig

OUT = sys.argv[1]
es, ee = pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2026-04-29", tz="UTC")
uni = equity_universe_between(es, ee, universe="sp500")
syms = sorted(uni)[:120]
bars = load_bars(sorted(set(syms) | {"SPY"}), start="2022-01-01", end="2026-04-30", source="snapshot")
legacy = to_legacy_frame(bars)
cfg = LabConfig(min_history_bars=60)
fn = lambda d, past: eligible_at(d, past, cfg, universe="sp500")[0]
panel = build_signal_panel(legacy, es, ee, fn)
res = run_replica(panel.close, panel.entries, panel.exits, panel.spy_close)
np.savez_compressed(OUT, entries=panel.entries.to_numpy(), exits=panel.exits.to_numpy(),
    close=panel.close.to_numpy(), cols=np.array(panel.entries.columns), idx=panel.entries.index.astype("int64").to_numpy(),
    equity=res.equity.to_numpy(), eq_idx=res.equity.index.astype("int64").to_numpy())
print(panel.entries.shape, int(panel.entries.sum().sum()), int(panel.exits.sum().sum()), float(res.equity.iloc[-1]))
