"""Sanity check that the 0.0 differences in bench_indicators are real, not a harness artefact."""
import sys, os, numpy as np, pandas as pd, vectorbt as vbt
sys.path.insert(0, os.path.dirname(__file__))
from common import make_prices
px = make_prices(1900, 500)
n = px.vbt.rolling_mean(20, engine="numba").values; r = px.vbt.rolling_mean(20, engine="rust").values; p = px.rolling(20).mean().values
print("rolling_mean #elements != (numba,rust):", int(np.sum(~np.isclose(n, r, rtol=0, atol=0, equal_nan=True))),
      "| #!= (numba,pandas):", int(np.sum(~np.isclose(n, p, rtol=0, atol=0, equal_nan=True))), "| max |x|", np.nanmax(np.abs(n)))
ns = px.vbt.rolling_std(20, engine="numba").values; rs = px.vbt.rolling_std(20, engine="rust").values; ps = px.rolling(20).std().values
print("rolling_std #!=(numba,rust):", int(np.sum((ns != rs) & ~(np.isnan(ns) & np.isnan(rs)))), "(numba,pandas):", int(np.sum(~np.isclose(ns, ps, rtol=0, atol=0, equal_nan=True))),
      "max abs (numba,pandas):", float(np.nanmax(np.abs(ns - ps))))
m = vbt.MSTD.run(px, window=20, engine="numba").mstd.values
print("MSTD(vbt, default ddof) vs pandas ddof=0 max abs:", float(np.nanmax(np.abs(m - px.rolling(20).std(ddof=0).values))),
      "| vs ddof=1:", float(np.nanmax(np.abs(m - px.rolling(20).std(ddof=1).values))))
