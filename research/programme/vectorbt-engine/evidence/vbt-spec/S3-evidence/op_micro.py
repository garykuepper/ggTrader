"""Per-op micro-benchmark on a (860 x 100) frame -- locates the pandas-3 slowdown in rsi_signals."""
import time

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
idx = pd.bdate_range("2021-01-01", periods=860, tz="UTC")
cols = [f"S{i}" for i in range(100)]
close = pd.DataFrame(100 + rng.normal(0, 1, (860, 100)).cumsum(0), index=idx, columns=cols)


def best(fn, n=40):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


delta = close.diff()
gain = delta.clip(lower=0.0)
avg = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
rs = avg / avg.replace(0, np.nan)
rsi = 100.0 - (100.0 / (1.0 + rs))
b = (rsi.shift(1) >= 30) & (rsi < 30)
ops = {
    "diff": lambda: close.diff(),
    "clip": lambda: delta.clip(lower=0.0),
    "neg": lambda: -delta,
    "ewm.mean": lambda: gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(),
    "replace(0,nan)": lambda: avg.replace(0, np.nan),
    "div": lambda: avg / avg,
    "scalar arith": lambda: 100.0 - (100.0 / (1.0 + rs)),
    "shift": lambda: rsi.shift(1),
    "cmp scalar": lambda: rsi < 30,
    "and": lambda: (rsi.shift(1) >= 30) & (rsi < 30),
    "fillna(False).astype(bool)": lambda: b.fillna(False).astype(bool),
    "rolling.mean": lambda: close.rolling(20, min_periods=20).mean(),
    "rolling.std": lambda: close.rolling(20, min_periods=20).std(),
}
print("pandas", pd.__version__)
for k, f in ops.items():
    print(f"{k:28s} {best(f):8.3f} ms")
