"""Cold-start probe, fresh process: import cost + first simulate_signals / simulate_weights call.

Usage: coldstart_probe.py [rust|numba|auto]   (engine selection only matters on vectorbt>=1.0)
Prints one JSON line. Wall AND process CPU seconds (box is shared, CPU time is steadier).
"""
import json
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
t0, c0 = time.perf_counter(), time.process_time()
res = {"mode": mode, "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"), "cpu_name": os.environ.get("NUMBA_CPU_NAME")}


def lap(key, t, c):
    res[key + "_wall"] = round(time.perf_counter() - t, 2)
    res[key + "_cpu"] = round(time.process_time() - c, 2)


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import vectorbt as vbt  # noqa: E402

res["vectorbt"] = vbt.__version__
res["pandas"] = pd.__version__
if hasattr(vbt, "settings") and mode in ("numba", "rust"):
    try:
        vbt.settings["engine"] = mode
    except Exception as e:  # 0.28.5 has no engine setting
        res["engine_set_error"] = repr(e)
try:
    from vectorbt import _engine

    res["rust_available"] = bool(_engine.is_rust_available())
except Exception:  # 0.28.5 has no _engine module
    res["rust_available"] = None
lap("import_vbt", t0, c0)

t1, c1 = time.perf_counter(), time.process_time()
import ggTrader.paper.trader  # noqa: E402,F401

lap("import_trader_chain", t1, c1)

from ggTrader.lab.data import STOCK_BASE_CONFIG  # noqa: E402
from ggTrader.lab.simulate import simulate_signals, simulate_weights  # noqa: E402
from ggTrader.lab.strategy import SignalTargets  # noqa: E402

rng = np.random.default_rng(0)
n, m = 400, 40
idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
syms = [f"S{i}" for i in range(m)]
close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, m)), axis=0)), index=idx, columns=syms)
ohlcv = pd.concat(
    {s: pd.DataFrame({"open": close[s], "high": close[s] * 1.01, "low": close[s] * 0.99, "close": close[s], "volume": 1e6}) for s in syms},
    axis=1,
)
entries = pd.DataFrame(rng.random((n, m)) < 0.03, index=idx, columns=syms)
exits = pd.DataFrame(rng.random((n, m)) < 0.03, index=idx, columns=syms)
cfg = dict(STOCK_BASE_CONFIG)
cfg["atr_mult"] = 3.0  # live config uses the ATR trailing stop path (sl_stop frame + sl_trail)
t2, c2 = time.perf_counter(), time.process_time()
r, v, d = simulate_signals({"a": SignalTargets(entries, exits)}, close, cfg, ohlcv=ohlcv)
lap("first_simulate_signals", t2, c2)
t3, c3 = time.perf_counter(), time.process_time()
r, v, d = simulate_signals({"a": SignalTargets(entries, exits)}, close, cfg, ohlcv=ohlcv)
lap("second_simulate_signals", t3, c3)

w = pd.DataFrame(0.0, index=idx, columns=syms)
w.iloc[::21, :5] = 0.2
t4, c4 = time.perf_counter(), time.process_time()
simulate_weights({"w": w}, close, cfg)
lap("first_simulate_weights", t4, c4)
lap("total", t0, c0)
print(json.dumps(res))
