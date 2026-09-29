"""Quick, non-WFO sanity checks on simple rules from the 2026-09-28 research batch.

Not walk-forward results: frozen rules, no parameter search, 1 bp per side on
switches, cash earns 0, raw daily Sharpe (no risk-free rate). Prices are
yfinance auto-adjusted closes pulled directly (DBC/UUP/UDN are not in the
ohlcv table), so compare rows with each other, not with RESEARCH_SNAPSHOT.
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY", "IEF", "TLT", "GLD", "DBC", "UUP", "UDN"]
COST = 0.0001
WINDOWS = {
    "PINNED 2021-02-01..2026-04-30": ("2021-02-01", "2026-04-30"),
    "LONG 2011-03-01..2026-04-30": ("2011-03-01", "2026-04-30"),
    "2022 only": ("2022-01-01", "2022-12-31"),
}


def stats(ret: pd.Series, pos: pd.Series | None = None) -> str:
    """CAGR / Sharpe / MaxDD line for a daily return series, net of switch costs."""
    ret = ret.dropna()
    if pos is not None:
        turn = pos.diff().abs().fillna(0).reindex(ret.index).fillna(0)
        if isinstance(turn, pd.DataFrame):
            turn = turn.sum(axis=1)
        ret = ret - turn * COST
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return f"CAGR {ann * 100:6.1f}%  Sharpe {sh:5.2f}  MaxDD {dd * 100:6.1f}%"


def rebalance_threshold_signal(r: pd.DataFrame, band: float = 0.02) -> pd.Series:
    """True on days whose *next* return should be in IEF (equities overweight).

    Synthetic 60/40 SPY/IEF book drifts with returns and resets to 60/40 when
    the equity weight leaves 60% +/- band (Harvey-Mazzoleni-Melone threshold
    idea). Loop is over dates only -- a path-dependent state, not a signal
    that vectorizes.
    """
    w_eq, out = 0.6, []
    for rs, rb in zip(r["SPY"].values, r["IEF"].values):
        w_eq = w_eq * (1 + rs) / (w_eq * (1 + rs) + (1 - w_eq) * (1 + rb))
        over = w_eq - 0.6 > band
        out.append(over)
        if abs(w_eq - 0.6) > band:
            w_eq = 0.6
    return pd.Series(out, index=r.index)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    px = yf.download(
        SYMBOLS, start="2008-01-01", end="2026-05-01", auto_adjust=True, progress=False
    )["Close"]
    r = px.pct_change()

    # Slow cross-asset TSMOM sleeve: each of TLT/GLD/DBC held when its trailing
    # 252-day return > 0, else cash; equal weight; signal lagged one day.
    legs = ["TLT", "GLD", "DBC"]
    on = (px[legs].pct_change(252) > 0).shift(1).astype(float)
    sleeve = (r[legs] * on).mean(axis=1)
    sleeve_bh = r[legs].mean(axis=1)
    blend = 0.8 * r["SPY"] + 0.2 * sleeve
    blend_bh = 0.8 * r["SPY"] + 0.2 * sleeve_bh

    # Threshold rebalancing tilt: hold IEF instead of SPY the day after the
    # synthetic 60/40 book's equity weight breaches 62%.
    sig = rebalance_threshold_signal(r[["SPY", "IEF"]].dropna()).shift(1, fill_value=False)
    tilt = r["SPY"].where(~sig.reindex(r.index, fill_value=False), r["IEF"])
    tilt_pos = sig.astype(float)

    # Dollar: no rate-differential data here, so only buy-and-hold context.
    for name, (a, b) in WINDOWS.items():
        w = slice(a, b)
        print(f"\n=== {name} ===")
        print("  SPY buy&hold                  ", stats(r["SPY"][w]))
        print("  TLT/GLD/DBC equal-wt B&H      ", stats(sleeve_bh[w]))
        print(
            "  TSMOM sleeve (12m, long/flat) ",
            stats(sleeve[w], on[w] / 3),
            f" avg exposure {on[w].mean().mean() * 100:.0f}%",
        )
        print("  80% SPY + 20% TSMOM sleeve    ", stats(blend[w]))
        print("  80% SPY + 20% sleeve B&H      ", stats(blend_bh[w]))
        print("  corr(sleeve, SPY) daily       ", f"{sleeve[w].corr(r['SPY'][w]):.2f}")
        print(
            "  rebal threshold tilt SPY->IEF ",
            stats(tilt[w], tilt_pos[w]),
            f" days in IEF {tilt_pos[w].mean() * 100:.1f}%",
        )
        print("  UUP buy&hold                  ", stats(r["UUP"][w]))
        print("  UDN buy&hold                  ", stats(r["UDN"][w]))


if __name__ == "__main__":
    main()
