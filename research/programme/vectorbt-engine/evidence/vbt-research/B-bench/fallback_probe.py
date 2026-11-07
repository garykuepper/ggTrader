"""Probe when vectorbt 1.2.0 engine='auto' silently uses numba vs rust, and what engine='rust' does when unsupported."""
import sys, warnings, numpy as np, pandas as pd, vectorbt as vbt
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import make_prices, install_rust_counters
from numba import njit
warnings.simplefilter("always")
c = install_rust_counters()
px = make_prices(200, 6); rng = np.random.default_rng(0)
ent = pd.DataFrame(rng.random(px.shape) < .05, index=px.index, columns=px.columns); ex = pd.DataFrame(rng.random(px.shape) < .06, index=px.index, columns=px.columns)

def sim_calls(): return c.get("portfolio.simulate_from_signals_rs", 0) + c.get("portfolio.simulate_from_orders_rs", 0)
def probe(name, fn):
    for eng in ("auto", "rust"):
        c.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fn(eng); r = "ok"
            except Exception as e:
                r = f"{type(e).__name__}: {str(e)[:110]}"
        print(f"{name:55s} engine={eng:5s} -> sim kernel rust calls={sim_calls()}  {r}  warnings={[str(x.message)[:60] for x in w][:2]}")
probe("baseline float64 signals", lambda e: vbt.Portfolio.from_signals(px, ent, ex, engine=e))
probe("close float32", lambda e: vbt.Portfolio.from_signals(px.astype("float32"), ent, ex, engine=e))
probe("close int64 (rounded)", lambda e: vbt.Portfolio.from_signals(px.round().astype("int64"), ent, ex, engine=e))
probe("call_seq='auto' + cash_sharing group", lambda e: vbt.Portfolio.from_orders(px, 0.1, size_type="targetpercent", cash_sharing=True, group_by=True, call_seq="auto", engine=e))
@njit
def adj_sl(c): return c.curr_stop, c.curr_trail
probe("adjust_sl_func_nb callback", lambda e: vbt.Portfolio.from_signals(px, ent, ex, sl_stop=0.1, adjust_sl_func_nb=adj_sl, engine=e))
probe("reject_prob=0.2 (random)", lambda e: vbt.Portfolio.from_orders(px, 100., reject_prob=0.2, seed=1, engine=e))
probe("pandas string-dtype column labels (pandas>=3)", lambda e: vbt.Portfolio.from_signals(px.set_axis(pd.Index(px.columns, dtype="str"), axis=1), ent.set_axis(pd.Index(px.columns, dtype="str"), axis=1), ex.set_axis(pd.Index(px.columns, dtype="str"), axis=1), engine=e))
# read-only inputs (pandas 3 CoW .values is read-only)
ro = px.values.copy(); ro.flags.writeable = False
probe("read-only numpy close", lambda e: vbt.Portfolio.from_signals(ro, ent.values, ex.values, engine=e))
# version mismatch -> warning + numba
import vectorbt._engine as eng_mod, vectorbt_rust
orig = vectorbt_rust.__version__
vectorbt_rust.__version__ = "1.1.9"; eng_mod.clear_engine_cache()
probe("SIMULATED vectorbt-rust 1.1.9 vs vectorbt 1.2.0", lambda e: vbt.Portfolio.from_signals(px, ent, ex, engine=e))
vectorbt_rust.__version__ = orig; eng_mod.clear_engine_cache()
print("global setting default:", vbt.settings["engine"])
