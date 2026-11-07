"""Randomised parity fuzz of Portfolio.from_signals / from_orders. Same seeds in every venv; arrays saved for cross-venv compare.
Usage: python fuzz_parity.py OUT.npz [engine|none]"""
import sys, warnings, numpy as np, pandas as pd, vectorbt as vbt
warnings.filterwarnings("ignore")
out = sys.argv[1]; eng = sys.argv[2] if len(sys.argv) > 2 else "none"
kw = {} if eng == "none" else {"engine": eng}
N = 120
store = {}; errs = {}
for i in range(N):
    rng = np.random.default_rng(1000 + i)
    T, C = int(rng.integers(40, 260)), int(rng.integers(2, 14))
    idx = pd.bdate_range("2021-01-04", periods=T); cols = [f"c{j}" for j in range(C)]
    px = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, C)), axis=0))
    if rng.random() < 0.6:  # NaN holes + leading NaNs
        px[rng.random((T, C)) < 0.04] = np.nan; px[: int(rng.integers(0, 10)), : max(1, C // 3)] = np.nan
    close = pd.DataFrame(px, index=idx, columns=cols)
    ngroups = int(rng.integers(1, min(C, 4) + 1)); grp = pd.Index(np.sort(rng.integers(0, ngroups, C)), name="g") if rng.random() < .7 else None
    cs = bool(rng.random() < 0.7) if grp is not None else bool(rng.random() < 0.5)
    common = dict(init_cash=float(rng.choice([1e4, 1e5])), fees=float(rng.choice([0, 0.0005, 0.002])), slippage=float(rng.choice([0, 0.0005])),
                  fixed_fees=float(rng.choice([0, 1.0])), freq="1D", cash_sharing=cs, group_by=grp)
    try:
        if i % 2 == 0:
            ent = pd.DataFrame(rng.random((T, C)) < 0.05, index=idx, columns=cols); ex = pd.DataFrame(rng.random((T, C)) < 0.06, index=idx, columns=cols)
            skw = dict(common)
            skw["size"] = float(rng.choice([0.02, 0.05, 0.2, 1.0])); skw["size_type"] = str(rng.choice(["percent", "amount", "value", "targetpercent"]))
            if skw["size_type"] == "amount": skw["size"] = float(rng.choice([1, 5, 20]))
            if skw["size_type"] == "value": skw["size"] = float(rng.choice([500, 2000]))
            if rng.random() < .6: skw["sl_stop"] = float(rng.choice([0.03, 0.08, 0.2])); skw["sl_trail"] = bool(rng.random() < .6)
            if rng.random() < .4: skw["tp_stop"] = float(rng.choice([0.05, 0.15]))
            skw["direction"] = str(rng.choice(["longonly", "longonly", "shortonly", "both"]))
            skw["accumulate"] = bool(rng.random() < .3)
            if rng.random() < .3: skw["upon_opposite_entry"] = str(rng.choice(["close", "reverse", "reverseReduce", "ignore", "closeReduce"]))
            if cs and grp is not None and rng.random() < .3: skw["call_seq"] = "auto"
            if rng.random() < .3: skw["price"] = close.shift(1).bfill()  # alt price path
            pf = vbt.Portfolio.from_signals(close, ent, ex, **skw, **kw)
        else:
            tp = pd.DataFrame(np.nan, index=idx, columns=cols); rb = np.arange(0, T, int(rng.integers(5, 25)))
            tp.iloc[rb] = rng.dirichlet(np.ones(C), size=len(rb)) * float(rng.choice([0.5, 0.95, 1.0]))
            if rng.random() < .2: tp.iloc[rb[1:], 0] = 0.0
            okw = dict(common); okw["size_type"] = "targetpercent"
            if cs and rng.random() < .7: okw["call_seq"] = "auto"
            pf = vbt.Portfolio.from_orders(close, tp, **okw, **kw)
        v = pf.value().values if hasattr(pf.value(), "values") else np.asarray(pf.value())
        store[f"v{i}"] = v; store[f"n{i}"] = np.atleast_1d(pf.trades.count().values if hasattr(pf.trades.count(), 'values') else pf.trades.count())
        store[f"o{i}"] = np.atleast_1d(pf.orders.count().values if hasattr(pf.orders.count(), 'values') else pf.orders.count())
    except Exception as e:
        errs[i] = f"{type(e).__name__}: {str(e)[:100]}"
np.savez(out, **store)
print(eng, vbt.__version__, "cases ok:", N - len(errs), "errors:", len(errs), dict(list(errs.items())[:5]))
import json; json.dump(errs, open(out + ".errs.json", "w"))
