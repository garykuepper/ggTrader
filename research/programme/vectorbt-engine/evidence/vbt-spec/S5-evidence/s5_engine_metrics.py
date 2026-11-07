"""S5 prototype step 4 (venv120): engine dispatch/parity, invariants and metric definitions.

Inputs: the read-only tape snapshot and arm D ensemble signals. Writes engine_metrics.json.
"""

import argparse
import importlib
import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def install_rust_counters():
    counts = {}
    import vectorbt_rust  # noqa: F401

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    import vectorbt as vbt

    counts = install_rust_counters()
    ev = a.evidence
    close = pd.read_parquet(os.path.join(ev, "tape_close.parquet"))
    ent = pd.read_parquet(os.path.join(ev, "arms/D_vbt_matched_vbtcross/ENS_entries.parquet"))
    ext = pd.read_parquet(os.path.join(ev, "arms/D_vbt_matched_vbtcross/ENS_exits.parquet"))
    out = {"vectorbt": vbt.__version__, "pandas": pd.__version__, "engine": {}, "invariants": {},
           "metrics": {}}

    # ---- (e) engine dispatch + numba vs Rust parity on real-shaped inputs ----
    cfgs = {
        "plain_cash_shared": dict(cash_sharing=True, group_by=True),
        "trail8_cash_shared": dict(cash_sharing=True, group_by=True, sl_stop=0.08, sl_trail=True),
        "trail8_tp15_per_symbol": dict(sl_stop=0.08, sl_trail=True, tp_stop=0.15),
    }
    pfs = {}
    for name, kw in cfgs.items():
        res = {}
        for eng in ("numba", "rust", "auto"):
            before = sum(counts.values())
            pf = vbt.Portfolio.from_signals(
                close, ent, ext, size=0.03, size_type="percent", init_cash=10000.0,
                fees=0.0, slippage=0.0005, freq="1D", engine=eng, **kw)
            val = np.asarray(pf.value())
            res[eng] = {"rust_kernel_calls": sum(counts.values()) - before,
                        "orders": int(pf.orders.count().sum()),
                        "trades": int(pf.trades.count().sum()), "val": val}
            pfs[(name, eng)] = pf
        res_out = {e: {k: v for k, v in r.items() if k != "val"} for e, r in res.items()}
        res_out["max_abs_value_diff_numba_rust"] = float(np.nanmax(np.abs(res["numba"]["val"] - res["rust"]["val"])))
        res_out["max_abs_value_diff_numba_auto"] = float(np.nanmax(np.abs(res["numba"]["val"] - res["auto"]["val"])))
        out["engine"][name] = res_out
    # indicator kernels numba vs Rust
    ind = {}
    for nm, fn in {
        "BBANDS_ddof1_lower": lambda e: vbt.BBANDS.run(close, 20, alpha=2.0, ddof=1, engine=e).lower,
        "RSI_27_ewm": lambda e: vbt.RSI.run(close, 27, ewm=True, engine=e).rsi,
        "MACD_ewm_hist": lambda e: vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True, engine=e).hist,
        "MA_50_ewm": lambda e: vbt.MA.run(close, 50, ewm=True, engine=e).ma,
    }.items():
        try:
            before = sum(counts.values())
            n = np.asarray(fn("numba"), dtype=float)
            mid = sum(counts.values())
            r = np.asarray(fn("rust"), dtype=float)
            ind[nm] = {"rust_calls_numba_run": mid - before, "rust_calls_rust_run": sum(counts.values()) - mid,
                       "max_abs_diff": float(np.nanmax(np.abs(n - r))),
                       "nan_mask_equal": bool((np.isnan(n) == np.isnan(r)).all())}
        except Exception as exc:  # record, do not crash
            ind[nm] = {"error": repr(exc)[:200]}
    out["engine"]["indicators_numba_vs_rust"] = ind

    # ---- (f) invariants ----
    pf = pfs[("trail8_cash_shared", "rust")]
    cash = np.asarray(pf.cash())
    out["invariants"]["min_cash_cash_shared"] = float(np.nanmin(cash))
    pf_fee = vbt.Portfolio.from_signals(close, ent, ext, size=0.03, size_type="percent",
                                        init_cash=10000.0, fees=0.001, slippage=0.0005,
                                        freq="1D", cash_sharing=True, group_by=True)
    rec = pf_fee.orders.records_readable
    exp_fees = float((rec["Size"] * rec["Price"] * 0.001).sum())
    out["invariants"]["fees_recorded_vs_expected"] = [float(rec["Fees"].sum()), exp_fees]
    # PIT mask: random 70% membership mask ANDed into entries; no entry fill where mask False.
    rng = np.random.default_rng(0)
    mask = pd.DataFrame(rng.random(close.shape) < 0.7, index=close.index, columns=close.columns)
    pf_m = vbt.Portfolio.from_signals(close, ent & mask, ext, size=0.03, size_type="percent",
                                      init_cash=10000.0, freq="1D", cash_sharing=True, group_by=True)
    orr = pf_m.orders.records
    buys = orr[orr["side"] == 0]
    out["invariants"]["buy_fills_outside_mask"] = int(
        (~mask.to_numpy()[buys["idx"], buys["col"]]).sum())
    out["invariants"]["buy_fills_on_non_entry_bar"] = int(
        (~(ent & mask).to_numpy()[buys["idx"], buys["col"]]).sum())

    # ---- metric definitions: ggTrader curve_stats vs vbt ----
    spec = importlib.util.spec_from_file_location(
        "ggt_metrics", "/home/flynn/ggTrader/src/ggTrader/lab/metrics.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    eq = pd.Series(np.asarray(pfs[("trail8_cash_shared", "rust")].value()).ravel(), index=close.index)
    eq = eq.loc["2021-01-31":"2026-04-30"]
    g = gm.curve_stats(eq)
    r = eq.pct_change(fill_method=None).dropna()
    m = {"ggtrader_curve_stats": g}
    for yf in ("365 days", "252 days"):
        acc = r.vbt.returns(freq="1D", year_freq=yf)
        m[f"vbt_year_freq_{yf.split()[0]}"] = {
            "sharpe": float(acc.sharpe_ratio()), "sortino": float(acc.sortino_ratio()),
            "annualized_return_pct": float(acc.annualized()) * 100,
            "ann_vol_pct": float(acc.annualized_volatility()) * 100,
            "max_drawdown_pct": float(acc.max_drawdown()) * 100,
            "deflated_sharpe_ntrials48": float(acc.deflated_sharpe_ratio(nb_trials=48)),
        }
    flat = pd.Series(0.0, index=r.index)
    m["flat_curve_sharpe_vbt"] = float(flat.vbt.returns(freq="1D", year_freq="252 days").sharpe_ratio())
    m["flat_curve_sharpe_ggt"] = gm.curve_stats(pd.Series(100.0, index=r.index))["sharpe"]
    tiny = pd.Series(0.0, index=r.index)
    tiny.iloc[5] = 1e-4  # one tiny gain, zero downside
    m["one_gain_sortino_vbt"] = float(tiny.vbt.returns(freq="1D", year_freq="252 days").sortino_ratio())
    m["one_gain_sortino_ggt"] = gm.curve_stats((1 + tiny).cumprod() * 100)["sortino"]
    out["metrics"] = m
    out["rust_kernel_names_seen"] = sorted(counts)[:40]
    with open(os.path.join(ev, "engine_metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("engine", "invariants", "metrics")}, indent=1, default=str))


if __name__ == "__main__":
    main()
