"""Portfolio micro-benchmark mirroring ggTrader's usage (from_signals / from_orders).

Usage: python bench_portfolio.py --engine {none,numba,rust,auto} --out OUT.json --npz OUT.npz
`none` = do not pass engine= at all (the only option on vectorbt 0.28.5).
"""
import argparse, json, sys, platform
import numpy as np
import pandas as pd
import vectorbt as vbt
sys.path.insert(0, __import__("os").path.dirname(__file__))
import common
from common import make_prices, best_of, install_rust_counters

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="none")
ap.add_argument("--days", type=int, default=1900)
ap.add_argument("--syms", type=int, default=500)
ap.add_argument("--groups", type=int, default=48)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--npz", default=None)
a = ap.parse_args()

counts = install_rust_counters() if a.engine in ("rust", "auto", "none") else {}
kw = {} if a.engine == "none" else {"engine": a.engine}
res = {"vbt": vbt.__version__, "numpy": np.__version__, "pandas": pd.__version__, "engine": a.engine,
       "python": platform.python_version(), "cases": {}}
import numba; res["numba"] = numba.__version__
arrays = {}


def run_case(name, builder):
    import time
    counts.clear()
    t, ts, _ = best_of(builder, repeat=a.repeat); first = common.LAST_FIRST_CALL_S
    pf = builder()  # fresh instance so value()/trades are not already cached
    counts.clear()
    t0 = time.perf_counter(); val = pf.value(); t_val = time.perf_counter() - t0
    t0 = time.perf_counter(); tc = pf.trades.count(); t_tr = time.perf_counter() - t0
    v = val.values if hasattr(val, "values") else np.asarray(val)
    oc = pf.orders.count()
    res["cases"][name] = {"sim_best_s": t, "sim_all_s": ts, "sim_first_call_s": first, "value_s": t_val, "trades_count_s": t_tr,
                          "n_orders": int(np.sum(oc)), "n_trades": int(np.sum(tc)),
                          "final_value_sum": float(np.nansum(v[-1])), "rust_kernel_calls_post": dict(counts)}
    arrays[name] = v
    print(name, f"sim={t:.3f}s value={t_val:.3f}s trades.count={t_tr:.3f}s", flush=True)


px = make_prices(a.days, a.syms)
rng = np.random.default_rng(1)
# ---- Case 1: ggTrader-like -- 1 shared-cash group, ~500 symbols, signals, trailing SL, percent size
ent = pd.DataFrame(rng.random(px.shape) < 0.01, index=px.index, columns=px.columns)
ex = pd.DataFrame(rng.random(px.shape) < 0.02, index=px.index, columns=px.columns)
grp1 = pd.Index(["g"] * a.syms, name="strategy")
run_case("signals_1group_500sym_trailSL_pct", lambda: vbt.Portfolio.from_signals(
    px, ent, ex, size=0.03, size_type="percent", init_cash=100_000., fees=0.0005, slippage=0.0005,
    sl_stop=0.08, sl_trail=True, freq="1D", cash_sharing=True, group_by=grp1, **kw))
# ---- Case 2: same, no stops
run_case("signals_1group_500sym_nostop_pct", lambda: vbt.Portfolio.from_signals(
    px, ent, ex, size=0.03, size_type="percent", init_cash=100_000., fees=0.0005, slippage=0.0005,
    freq="1D", cash_sharing=True, group_by=grp1, **kw))
# ---- Case 3: grid -- 48 groups (param combos) x 100 syms = 4800 cols, shared cash per group, trailing SL
n_per = min(100, a.syms)
px3 = pd.concat([px.iloc[:, :n_per]] * a.groups, axis=1)
px3.columns = pd.MultiIndex.from_product([range(a.groups), px.columns[:n_per]], names=["combo", "sym"])
ent3 = pd.concat([ent.iloc[:, :n_per].shift(i % 5, fill_value=False) for i in range(a.groups)], axis=1); ent3.columns = px3.columns
ex3 = pd.concat([ex.iloc[:, :n_per]] * a.groups, axis=1); ex3.columns = px3.columns
sl3 = pd.DataFrame(np.repeat(np.linspace(0.03, 0.15, a.groups)[None, :], a.days, 0).repeat(n_per, axis=1), index=px.index, columns=px3.columns)
grp3 = px3.columns.get_level_values("combo")
run_case("signals_48grp_x100sym_trailSL_pct", lambda: vbt.Portfolio.from_signals(
    px3, ent3, ex3, size=0.03, size_type="percent", init_cash=100_000., fees=0.0005, slippage=0.0005,
    sl_stop=sl3, sl_trail=True, freq="1D", cash_sharing=True, group_by=grp3, **kw))
# ---- Case 4: from_orders targetpercent, call_seq='auto' (ggTrader's weight path), 48 groups x 100
w = pd.DataFrame(np.nan, index=px.index, columns=px3.columns)
rb = np.arange(0, a.days, 21)
tgt = rng.dirichlet(np.ones(n_per) * 0.3, size=(len(rb), a.groups)).transpose(0, 1, 2)  # (rb, groups, n_per)
vals = np.concatenate([tgt[:, g, :] for g in range(a.groups)], axis=1)
w.iloc[rb] = vals
run_case("orders_48grp_x100sym_targetpct_auto", lambda: vbt.Portfolio.from_orders(
    px3, w, size_type="targetpercent", init_cash=100_000., fees=0.0005, slippage=0.0005, freq="1D",
    cash_sharing=True, group_by=grp3, call_seq="auto", **kw))
# ---- Case 5: from_orders, 1 group 500 syms, call_seq=auto
w5 = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
w5.iloc[rb] = rng.dirichlet(np.ones(a.syms) * 0.3, size=len(rb))
run_case("orders_1group_500sym_targetpct_auto", lambda: vbt.Portfolio.from_orders(
    px, w5, size_type="targetpercent", init_cash=100_000., fees=0.0005, slippage=0.0005, freq="1D",
    cash_sharing=True, group_by=grp1, call_seq="auto", **kw))

json.dump(res, open(a.out, "w"), indent=1)
if a.npz:
    np.savez_compressed(a.npz, **arrays)
