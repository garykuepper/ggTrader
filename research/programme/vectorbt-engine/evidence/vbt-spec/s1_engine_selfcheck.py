"""S1 prototype: explicit engine pin + startup self-check that proves Rust dispatch."""
import functools
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt import _engine


def assert_engine(required: str) -> dict:
    """Pin vbt.settings['engine'] and prove the kernel actually ran on that engine."""
    vbt.settings["engine"] = required  # never leave 'auto'
    _engine.clear_engine_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # version-mismatch warning becomes fatal
        avail = _engine.is_rust_available()
    if required == "rust" and not avail:
        raise RuntimeError("engine=rust requested but vectorbt-rust missing or minor-mismatched")
    import vectorbt_rust.portfolio as rp

    calls = {"n": 0}
    orig = rp.simulate_from_signals_rs

    @functools.wraps(orig)
    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    rp.simulate_from_signals_rs = counted
    try:
        close = pd.DataFrame(np.linspace(10, 12, 50)[:, None] * [1.0, 1.1])
        ent = pd.DataFrame(False, index=close.index, columns=close.columns)
        ent.iloc[3] = True
        pf = vbt.Portfolio.from_signals(
            close, ent, ~ent, size=0.5, size_type="percent", sl_stop=0.05, sl_trail=True,
            cash_sharing=True, group_by=True, freq="1D",
        )
        pf.value()
    finally:
        rp.simulate_from_signals_rs = orig
    dispatched = "rust" if calls["n"] > 0 else "numba"
    if dispatched != required:
        raise RuntimeError(f"engine pinned to {required} but kernel ran on {dispatched}")
    return {"vectorbt": vbt.__version__, "vectorbt_rust": __import__("vectorbt_rust").__version__,
            "engine": dispatched, "rust_calls": calls["n"]}


if __name__ == "__main__":
    print(assert_engine("rust"))
    try:
        print(assert_engine("numba"))
    except RuntimeError as e:
        print("numba pin:", e)
    # extra API facts used by the spec
    c = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (300, 3)).cumsum(0) + 100)
    b0 = vbt.BBANDS.run(c, window=20, alpha=2.0)
    try:
        b1 = vbt.BBANDS.run(c, window=20, alpha=2.0, ddof=1)
        print("BBANDS ddof kwarg ok; lower diff", float((b0.lower - b1.lower).abs().max().max()))
    except Exception as e:  # noqa: BLE001
        print("BBANDS ddof kwarg FAILS:", type(e).__name__, e)
    e = c.vbt.crossed_below(c.rolling(5).mean())
    print("fshift ok", hasattr(e.vbt, "fshift"), "signals.fshift", hasattr(e.vbt.signals, "fshift"))
    r = vbt.Portfolio.from_signals(c, e, ~e, freq="1D").returns()
    print("sharpe 365 vs 252:", float(r.vbt.returns(freq="1D").sharpe_ratio().iloc[0]),
          float(r.vbt.returns(freq="1D", year_freq="252 days").sharpe_ratio().iloc[0]))
