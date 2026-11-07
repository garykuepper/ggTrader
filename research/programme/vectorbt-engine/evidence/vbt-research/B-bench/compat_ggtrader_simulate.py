"""Run ggTrader's own lab/simulate.py (loaded read-only from the repo by path) under a given venv.

Synthetic inputs only. Prints a digest so outputs can be compared across venvs
(0.28.5/pandas2 vs 1.0.0+rust/pandas2 vs 1.2.0/pandas3).  Usage: python compat_ggtrader_simulate.py OUT.npz [engine]
"""
import importlib.util, sys, os, warnings, hashlib
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
import vectorbt as vbt

SIM = "/home/flynn/ggTrader/src/ggTrader/lab/simulate.py"
spec = importlib.util.spec_from_file_location("ggt_simulate", SIM)
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)

out = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] != "default":
    vbt.settings["engine"] = sys.argv[2]
print("vbt", vbt.__version__, "pandas", pd.__version__, "numpy", np.__version__, "engine setting",
      vbt.settings["engine"] if "engine" in vbt.settings else "n/a (0.28.5)")

rng = np.random.default_rng(7)
idx = pd.bdate_range("2021-01-04", periods=400)
syms = [f"S{i:03d}" for i in range(60)]
px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, (400, 60)), axis=0)), index=idx, columns=syms)
cfg = {"START_CASH": 100_000.0, "FEES": 0.0005, "SLIPPAGE": 0.0005, "FREQ": "1D", "SIGNAL_POSITION_SIZE": 0.03}

# --- weight path (from_orders targetpercent, call_seq='auto', cash_sharing, group_by)
tg = {}
for k, name in enumerate(["stratA", "stratB", "stratC"]):
    cols = syms[k * 15 : k * 15 + 20]
    w = pd.DataFrame(np.nan, index=idx, columns=cols)
    for r in range(0, 400, 21):
        wt = rng.dirichlet(np.ones(len(cols)))
        w.iloc[r] = wt * 0.95
    tg[name] = w
tg["empty"] = pd.DataFrame(index=idx, columns=pd.Index([], dtype=object), dtype=float)
ret_w, eq_w, diag_w = sim.simulate_weights(tg, px, cfg)

# --- signal path (from_signals, trailing SL, size percent, group_by)
class ST:
    def __init__(s, e, x): s.entries, s.exits = e, x
sg = {}
for k, name in enumerate(["sigA", "sigB"]):
    cols = syms[k * 25 : k * 25 + 30]
    sg[name] = ST(pd.DataFrame(rng.random((400, 30)) < 0.02, index=idx, columns=cols),
                  pd.DataFrame(rng.random((400, 30)) < 0.03, index=idx, columns=cols))
res = {}
for label, extra in [("sl_trail", {"ts_stop": 0.08}), ("plain", {}), ("tp+sl", {"ts_stop": 0.1, "tp_stop": 0.15})]:
    r, e, d = sim.simulate_signals(sg, px, {**cfg, **extra})
    res[label] = (r, e, d)
arrays = {"w_ret": ret_w.values, "w_eq": eq_w.values}
for k, (r, e, d) in res.items():
    arrays[f"s_{k}_ret"] = r.values; arrays[f"s_{k}_eq"] = e.values
np.savez(out, **arrays)
print("weights diag", diag_w)
for k, (r, e, d) in res.items(): print("signals", k, d)
print("final equity weights", eq_w.iloc[-1].round(4).to_dict())
print("final equity sig_trail", res["sl_trail"][1].iloc[-1].round(4).to_dict())
