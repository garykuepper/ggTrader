"""Shared synthetic data + safe-run helper for S8 compat probes."""
import traceback, warnings
import numpy as np, pandas as pd
warnings.simplefilter("default")
def make_data(n=1500, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = rng.normal(0.0004, 0.012, (n, k))
    close = pd.DataFrame(100*np.exp(np.cumsum(r, 0)), index=idx, columns=[f"S{i}" for i in range(k)])
    high = close*(1+rng.uniform(0, .01, close.shape)); low = close*(1-rng.uniform(0, .01, close.shape))
    open_ = close.shift(1).fillna(close.iloc[0]); vol = pd.DataFrame(rng.integers(1e5, 1e6, close.shape).astype(float), index=idx, columns=close.columns)
    return open_, high, low, close, vol
def run(name, fn):
    try:
        out = fn(); print(f"OK   {name}: {out}")
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]} | {tb[-3].strip()[:150]}")
