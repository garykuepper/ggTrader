"""Shared helpers for the vectorbt benchmark scripts (runs on 0.28.5 and 1.2.0)."""
import time
import numpy as np
import pandas as pd


LAST_FIRST_CALL_S = None


def make_prices(n_days: int, n_sym: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.018, size=(n_days, n_sym))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    return pd.DataFrame(px, index=idx, columns=[f"S{i:04d}" for i in range(n_sym)])


def best_of(fn, repeat: int = 3, warmup: int = 1):
    """Return (best_seconds, all_seconds, last_result)."""
    out = None
    global LAST_FIRST_CALL_S
    for _ in range(warmup):
        t0 = time.perf_counter()
        out = fn()
        LAST_FIRST_CALL_S = time.perf_counter() - t0
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), ts, out


def install_rust_counters():
    """Wrap every *_rs callable in vectorbt_rust.* so we can prove Rust dispatched.

    vectorbt imports the kernels lazily at call time (`from vectorbt_rust.x import y_rs`),
    so replacing the module attribute is enough.
    """
    counts = {}
    try:
        import importlib
        import vectorbt_rust  # noqa: F401
    except ImportError:
        return counts
    for m in ["generic", "indicators", "labels", "portfolio", "records", "returns", "signals"]:
        mod = importlib.import_module("vectorbt_rust." + m)
        for n in dir(mod):
            if n.endswith("_rs") and callable(getattr(mod, n)):
                orig = getattr(mod, n)

                def wrap(*a, __orig=orig, __k=f"{m}.{n}", **k):
                    counts[__k] = counts.get(__k, 0) + 1
                    return __orig(*a, **k)

                setattr(mod, n, wrap)
    return counts
