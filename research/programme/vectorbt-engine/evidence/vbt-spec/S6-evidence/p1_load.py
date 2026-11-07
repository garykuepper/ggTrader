import sys, time, resource
sys.path.insert(0, '.')
import pandas as pd
from ggTrader.lab.data import equity_universe_between
from tsdata import ReadOnlyLoader
t0=time.time()
uni = equity_universe_between(pd.Timestamp("2019-03-01", tz="UTC"), pd.Timestamp("2026-11-05", tz="UTC"))
print("universe", len(uni), round(time.time()-t0,2), flush=True)
L = ReadOnlyLoader()
for n in (40, 100, len(uni)):
    syms = uni[:n] + ["SPY"]
    t=time.time(); w = L._db_fetch(syms, "1d", pd.Timestamp("2019-03-01", tz="UTC"), None)
    print(n, "symbols -> shape", w.shape, "idx", w.index.tz, w.index[0], w.index[-1], "secs", round(time.time()-t,1), flush=True)
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
