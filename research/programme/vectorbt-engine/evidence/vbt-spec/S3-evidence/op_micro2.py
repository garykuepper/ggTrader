"""Same micro-benchmark but on a frame built the way extract_close builds it (concat of Series, axis=1)."""
import time

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
idx = pd.bdate_range("2021-01-01", periods=860, tz="UTC")
cols = [f"S{i}" for i in range(100)]
base = pd.DataFrame(100 + rng.normal(0, 1, (860, 100)).cumsum(0), index=idx, columns=cols)
wide = pd.concat({c: pd.DataFrame({"close": base[c], "open": base[c]}) for c in cols}, axis=1)  # (sym, field) like ohlcv
close = pd.concat({s: wide[s]["close"] for s in cols}, axis=1)  # == extract_close


def best(fn, n=15):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


def rsi(c):
    d = c.diff()
    g = d.clip(lower=0.0)
    l = -d.clip(upper=0.0)
    ag = g.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = l.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    r = 100.0 - (100.0 / (1.0 + rs))
    return ((r.shift(1) >= 30) & (r < 30)).fillna(False).astype(bool)


print("pandas", pd.__version__, "| blocks in extract_close frame:", close._mgr.nblocks, "| consolidated copy blocks:", close.copy()._mgr.nblocks)
print(f"rsi_signals-like on concat-built frame : {best(lambda: rsi(close)):8.2f} ms")
print(f"rsi_signals-like on same frame, .copy() : {best(lambda: rsi(close.copy())):8.2f} ms")
print(f"rsi_signals-like on np-built 2D frame   : {best(lambda: rsi(base)):8.2f} ms")
