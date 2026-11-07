"""vbt rolling/EWM/indicator micro-benchmarks: numba vs rust vs pandas (1.2.0 venv only)."""
import json, sys, os, platform
import numpy as np, pandas as pd, vectorbt as vbt
sys.path.insert(0, os.path.dirname(__file__))
import common
from common import make_prices, best_of, install_rust_counters

DAYS, SYMS, SWEEP_SYMS, N_COMBO, REPEAT = 1900, 500, 100, 48, 5
counts = install_rust_counters()
px = make_prices(DAYS, SYMS)
px_s = px.iloc[:, :SWEEP_SYMS]
res = {"vbt": vbt.__version__, "numpy": np.__version__, "pandas": pd.__version__, "cases": {},
       "shape": [DAYS, SYMS], "sweep_shape": [DAYS, SWEEP_SYMS], "n_combo": N_COMBO}


def arr(x):
    if isinstance(x, (pd.DataFrame, pd.Series)): return x.values
    if hasattr(x, "to_numpy"): return x.to_numpy()
    return np.asarray(x)


def rsi_pandas(c, w):  # same definition as vbt RSI(ewm=False): simple-MA of gains/losses
    d = c.diff(); up = d.clip(lower=0).rolling(w).mean(); dn = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + up / dn)


def case(name, numba_fn, rust_fn, pandas_fn=None, outsel=lambda o: o):
    r = {}
    outs = {}
    for lab, fn in [("numba", numba_fn), ("rust", rust_fn), ("pandas", pandas_fn)]:
        if fn is None: continue
        counts.clear()
        t, ts, o = best_of(fn, repeat=REPEAT)
        outs[lab] = arr(o if lab == "pandas" else outsel(o))
        r[lab + "_s"] = t
        r[lab + "_rust_calls"] = sum(counts.values())
    with np.errstate(all="ignore"):
        r["maxabs_rust_vs_numba"] = float(np.nanmax(np.abs(outs["rust"] - outs["numba"])))
        if "pandas" in outs:
            r["maxabs_numba_vs_pandas"] = float(np.nanmax(np.abs(outs["numba"] - outs["pandas"])))
            r["maxabs_rust_vs_pandas"] = float(np.nanmax(np.abs(outs["rust"] - outs["pandas"])))
            r["nan_mask_equal_numba_vs_pandas"] = bool(np.array_equal(np.isnan(outs["numba"]), np.isnan(outs["pandas"])))
    res["cases"][name] = r
    print(name, {k: (float("%.3g" % v) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


a = px.vbt
case("rolling_mean w=20 (500 sym)", lambda: a.rolling_mean(20, engine="numba"), lambda: a.rolling_mean(20, engine="rust"),
     lambda: px.rolling(20).mean())
case("rolling_std w=20 ddof=1 (500 sym)", lambda: a.rolling_std(20, engine="numba"), lambda: a.rolling_std(20, engine="rust"),
     lambda: px.rolling(20).std())
from vectorbt.generic import dispatch as gd
case("ewm_mean span=20 adjust=False via accessor (pandas-3 shortcut, accessors.py:454)", lambda: a.ewm_mean(20, adjust=False, engine="numba"),
     lambda: a.ewm_mean(20, adjust=False, engine="rust"), lambda: px.ewm(span=20, adjust=False).mean())
case("ewm_mean span=20 adjust=False via generic.dispatch (500 sym)", lambda: gd.ewm_mean(px.values, 20, minp=0, adjust=False, engine="numba"),
     lambda: gd.ewm_mean(px.values, 20, minp=0, adjust=False, engine="rust"), lambda: px.ewm(span=20, adjust=False).mean())
case("ewm_mean span=20 adjust=True (500 sym)", lambda: a.ewm_mean(20, adjust=True, engine="numba"),
     lambda: a.ewm_mean(20, adjust=True, engine="rust"), lambda: px.ewm(span=20, adjust=True).mean())
case("MA.run w=20 (500 sym)", lambda: vbt.MA.run(px, window=20, engine="numba"), lambda: vbt.MA.run(px, window=20, engine="rust"),
     lambda: px.rolling(20).mean(), outsel=lambda o: getattr(o, "ma", o))
case("MA.run ewm=True w=20 (500 sym)", lambda: vbt.MA.run(px, window=20, ewm=True, engine="numba"),
     lambda: vbt.MA.run(px, window=20, ewm=True, engine="rust"), lambda: px.ewm(span=20, adjust=False).mean(), outsel=lambda o: o.ma)
case("MSTD.run w=20 (500 sym)", lambda: vbt.MSTD.run(px, window=20, engine="numba"), lambda: vbt.MSTD.run(px, window=20, engine="rust"),
     lambda: px.rolling(20).std(ddof=0), outsel=lambda o: o.mstd)
case("RSI.run w=14 simple-MA (500 sym)", lambda: vbt.RSI.run(px, window=14, engine="numba"), lambda: vbt.RSI.run(px, window=14, engine="rust"),
     lambda: rsi_pandas(px, 14), outsel=lambda o: o.rsi)
case("BBANDS.run w=20 a=2 upper (500 sym)", lambda: vbt.BBANDS.run(px, window=20, alpha=2, engine="numba"),
     lambda: vbt.BBANDS.run(px, window=20, alpha=2, engine="rust"),
     lambda: px.rolling(20).mean() + 2 * px.rolling(20).std(ddof=0), outsel=lambda o: o.upper)
case("MACD.run 12/26/9 default macd_ewm=False (500 sym)", lambda: vbt.MACD.run(px, engine="numba"), lambda: vbt.MACD.run(px, engine="rust"),
     None, outsel=lambda o: o.macd)
case("MACD.run 12/26/9 macd_ewm=True signal_ewm=True (500 sym)",
     lambda: vbt.MACD.run(px, macd_ewm=True, signal_ewm=True, engine="numba"),
     lambda: vbt.MACD.run(px, macd_ewm=True, signal_ewm=True, engine="rust"),
     lambda: (px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()), outsel=lambda o: o.macd)
# 48-combo sweeps (100 sym x 48 windows = 4800 columns)
wins = np.arange(5, 5 + N_COMBO)
case("MA.run 48 windows x100 sym", lambda: vbt.MA.run(px_s, window=wins, engine="numba"), lambda: vbt.MA.run(px_s, window=wins, engine="rust"),
     lambda: pd.concat([px_s.rolling(int(w)).mean() for w in wins], axis=1), outsel=lambda o: o.ma)
case("RSI.run 48 windows x100 sym", lambda: vbt.RSI.run(px_s, window=wins, engine="numba"), lambda: vbt.RSI.run(px_s, window=wins, engine="rust"),
     lambda: pd.concat([rsi_pandas(px_s, int(w)) for w in wins], axis=1), outsel=lambda o: o.rsi)
case("BBANDS.run 48 windows x100 sym", lambda: vbt.BBANDS.run(px_s, window=wins, engine="numba"), lambda: vbt.BBANDS.run(px_s, window=wins, engine="rust"),
     None, outsel=lambda o: o.upper)
json.dump(res, open(os.path.join(os.path.dirname(__file__), os.environ.get("OUTJSON", "indicators_venv120.json")), "w"), indent=1)
