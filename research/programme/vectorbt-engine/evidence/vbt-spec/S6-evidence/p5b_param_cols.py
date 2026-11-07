"""Gotcha: indicator outputs carry a parameter column level; the mask must be broadcast across it."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, vectorbt as vbt
from pit import pit_mask
close = pd.read_parquet("tape_close.parquet").iloc[:, :50]
mask = pit_mask(close, 60)
r = vbt.RSI.run(close, window=[14, 21])
ent = r.rsi.vbt.crossed_below(30)
print("entries columns:", ent.columns.names, ent.shape, "| mask columns:", mask.columns.names, mask.shape)
try:
    ent & mask
except Exception as e: print("ent & mask ->", type(e).__name__, str(e)[:70])
tiled = mask.vbt.tile(2, keys=ent.columns.levels[0])
print("tiled mask columns:", tiled.columns.names, tiled.shape, "aligned:", tiled.columns.equals(ent.columns))
m = ent & tiled
print("masked entries:", int(m.values.sum()), "of", int(ent.values.sum()))
# vbt-native broadcast alternative
a, b = ent.vbt.broadcast(mask.droplevel(0, axis=1) if False else tiled)
print("vbt.broadcast ok", a.shape == b.shape)
pf = vbt.Portfolio.from_signals(close, m, r.rsi.vbt.crossed_above(50), size=0.03, size_type="percent", freq="1D") if False else None
