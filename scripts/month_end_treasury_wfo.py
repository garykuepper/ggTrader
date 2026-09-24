"""A10 month-end Treasury sleeve: frozen-rule test, 12-combo WFO, SPY-overlay portfolio.

Pre-registered in docs/research/briefs/2026-09-24-month-end-treasury-sleeve.md
before any run:
  1. primary: IEF, last 3 trading days (enter at the close of the 4th-to-last
     trading day, exit at the close of the last), SIGNAL_POSITION_SIZE 1.0,
     on the pinned window plus 2019-01 -> 2021-01 and 2011 -> 2018 (reference).
  2. WFO: days_before_end {2,3,4,5} x {IEF, TLT, EDV}, pinned 17 folds, lab
     default cost (5 bp slippage/side).
  3. portfolio: 80% SPY fixed + 20% that sits in SPY except the last 3 trading
     days of each month, when it sits in IEF; vs 100% SPY.

Each section is checkpointed into the JSON output; re-running skips sections
already recorded (use --force to redo them).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategies.month_end_treasury import (
    MONTH_END_TREASURY_UNIVERSE,
    MonthEndTreasuryStrategy,
    month_end_signals,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs/research/_month_end_treasury_results.json"

EVAL_START = "2021-01-31"
EVAL_END = "2026-04-30"
#: load_ohlcv's end is exclusive; one extra day keeps 2026-04-30 (April's exit bar).
DATA_END = "2026-05-01"
DATA_START = "2010-02-01"

WINDOWS = {
    "pinned": (EVAL_START, EVAL_END),
    "post_pub": ("2019-01-01", "2021-01-31"),
    "paper_is": ("2011-01-01", "2018-12-31"),
}
COSTS_BP = [1, 3, 5]
PRIMARY_SYMBOL = "IEF"
PRIMARY_DAYS = 3
OVERLAY_WEIGHT = 0.20
SIZE_CFG = {"SIGNAL_POSITION_SIZE": 1.0}


def held_mask(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """True on bars whose close-to-close return the position earns: (entry, exit]."""
    open_ = entries.cumsum() - exits.cumsum()
    return open_.shift(1, fill_value=0) > 0


def sleeve_run(close: pd.DataFrame, symbol: str, days: int, cost_bp: float) -> dict:
    """Simulate the rule through the lab's own simulate_signals on one window."""
    syms = list(close.columns)
    st = month_end_signals(close.index, syms, symbol, days)
    cfg = {**STOCK_BASE_CONFIG, **SIZE_CFG, "SLIPPAGE": cost_bp / 1e4, "FEES": 0.0}
    _r, eq, diag = simulate_signals({"s": st}, close, cfg)
    curve = eq["s"]
    held = held_mask(st.entries[symbol], st.exits[symbol])
    ret = close[symbol].pct_change()
    inv, other = ret[held].dropna(), ret[~held].dropna()
    yearly = curve.resample("YE").last().pct_change()
    yearly.iloc[0] = curve.resample("YE").last().iloc[0] / curve.iloc[0] - 1
    return {
        **curve_stats(curve),
        "n_trades": int(diag["s"].get("n_trades", 0)),
        "exposure_pct": float(held.mean() * 100),
        "mean_ret_invested_day_bp": float(inv.mean() * 1e4),
        "mean_ret_other_days_bp": float(other.mean() * 1e4),
        "welch_t_invested_vs_other": float(
            (inv.mean() - other.mean()) / np.sqrt(inv.var() / len(inv) + other.var() / len(other))
        ),
        "hit_rate_pct": float(
            (curve[st.exits[symbol]].to_numpy() > curve[st.entries[symbol]].to_numpy()).mean() * 100
        )
        if st.entries[symbol].sum() == st.exits[symbol].sum()
        else None,
        "yearly_return_pct": {str(d.year): float(v * 100) for d, v in yearly.items()},
        "curve": curve,
    }


def buy_hold(close: pd.Series) -> dict:
    return curve_stats(close / close.iloc[0])


def overlay_run(close: pd.DataFrame, cost_bp: float) -> dict:
    """80% SPY + 20% rotating SPY->IEF for the month's last 3 trading days.

    Weights reset daily (ponytail: ignores 3-day drift of the 20% slice, a few
    bp of weight at most). Each switch trades 20% out of one ETF and 20% into
    the other: 2 legs x 20% x cost, charged on the switch bar.
    """
    st = month_end_signals(close.index, list(close.columns), PRIMARY_SYMBOL, PRIMARY_DAYS)
    held = held_mask(st.entries[PRIMARY_SYMBOL], st.exits[PRIMARY_SYMBOL])
    r_spy = close["SPY"].pct_change().fillna(0.0)
    r_ief = close["IEF"].pct_change().fillna(0.0)
    r = r_spy.where(~held, (1 - OVERLAY_WEIGHT) * r_spy + OVERLAY_WEIGHT * r_ief)
    switches = st.entries[PRIMARY_SYMBOL] | st.exits[PRIMARY_SYMBOL]
    r = r - switches * 2 * OVERLAY_WEIGHT * cost_bp / 1e4
    port = (1 + r).cumprod()
    spy = (1 + r_spy).cumprod()
    sleeve_ret = r_ief.where(held, 0.0)
    return {
        "portfolio": curve_stats(port),
        "spy": curve_stats(spy),
        "corr_sleeve_vs_spy_daily": float(sleeve_ret[held].corr(r_spy[held])),
        # What the 20% slice gives up: SPY's own return on the swapped days.
        "spy_mean_ret_held_days_bp": float(r_spy[held].mean() * 1e4),
        "spy_mean_ret_other_days_bp": float(r_spy[~held].mean() * 1e4),
        "ief_mean_ret_held_days_bp": float(r_ief[held].mean() * 1e4),
    }


def window(close: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = close.index.tz_localize(None)
    return close[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]


def run_primary(close: pd.DataFrame) -> dict:
    out: dict = {}
    for wname, (s, e) in WINDOWS.items():
        c = window(close, s, e)
        w: dict = {"start": str(c.index[0].date()), "end": str(c.index[-1].date())}
        for bp in COSTS_BP:
            res = sleeve_run(c, PRIMARY_SYMBOL, PRIMARY_DAYS, bp)
            res.pop("curve")
            w[f"{bp}bp"] = res
        w["bh"] = {s_: buy_hold(c[s_]) for s_ in ["SPY", "IEF", "TLT", "EDV"]}
        out[wname] = w
    return out


def run_sweep_standalone(close: pd.DataFrame) -> dict:
    """All 12 combos on the pinned window, un-walk-forwarded (context for the WFO)."""
    c = window(close, EVAL_START, EVAL_END)
    out = {}
    for combo in build_grid(MonthEndTreasuryStrategy):
        sym = MONTH_END_TREASURY_UNIVERSE[combo["duration_rank"]]
        res = sleeve_run(c, sym, combo["days_before_end"], 5)
        out[f"{sym}_d{combo['days_before_end']}"] = {
            k: res[k] for k in ["sharpe", "cagr_pct", "max_drawdown_pct", "exposure_pct"]
        }
    return out


def run_portfolio(close: pd.DataFrame) -> dict:
    out: dict = {}
    for wname in ["pinned", "post_pub"]:
        c = window(close, *WINDOWS[wname])
        out[wname] = {f"{bp}bp": overlay_run(c, bp) for bp in COSTS_BP}
    return out


def run_wfo_section(ohlcv: pd.DataFrame, spy_close: pd.Series) -> dict:
    from collections import Counter

    cfg = LabConfig()
    grid = build_grid(MonthEndTreasuryStrategy)
    res = run_wfo(
        "month_end_treasury",
        MonthEndTreasuryStrategy,
        cfg,
        ohlcv,
        spy_close,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market="stock",
        base_config={**STOCK_BASE_CONFIG, **SIZE_CFG},
        grid=grid,
        # Fixed ETF list -- no index membership, so no PIT mask to build.
        universe_fn=lambda asof, past: MONTH_END_TREASURY_UNIVERSE,
    )
    rows = [
        {
            k: (str(pd.Timestamp(v).date()) if k.endswith(("_start", "_end")) else v)
            for k, v in r.items()
            if k not in ("winner_params",)
        }
        for r in res.fold_results
    ]
    oos = curve_stats(res.oos_equity)
    spy = spy_close.reindex(res.oos_equity.index).ffill().dropna()
    winners = Counter(r["winner_combo"] for r in rows)
    top, top_n = winners.most_common(1)[0]
    return {
        "oos": oos,
        "spy": curve_stats(spy),
        "n_folds": len(rows),
        "n_gate_pass": sum(r["gates_passed"] for r in rows),
        "n_ndh_pass": sum(r["ndh_passed"] for r in rows),
        "n_dsr_pass": sum(r["dsr_passed"] for r in rows),
        "n_used_anchor": sum(r["used_anchor"] for r in rows),
        "halted_any": any(r["halted"] for r in rows),
        "top_winner": top,
        "top_winner_folds": top_n,
        "winner_counts": dict(winners),
        "live_params": res.live_params,
        "fold_rows": rows,
        "table": res.table,
    }


def verdict(out: dict) -> dict:
    p = out["primary"]["pinned"]["1bp"]
    p3 = out["primary"]["pinned"]["3bp"]
    years = [p["yearly_return_pct"].get(str(y), float("nan")) for y in range(2021, 2026)]
    w = out["wfo"]
    port = out["portfolio"]["pinned"]

    def c1(x: dict) -> bool:
        yrs = [x["yearly_return_pct"].get(str(y), float("nan")) for y in range(2021, 2026)]
        return x["sharpe"] >= 0.5 and x["total_return_pct"] > 0 and sum(v > 0 for v in yrs) >= 3

    def c4(x: dict) -> bool:
        return (
            x["portfolio"]["sharpe"] > x["spy"]["sharpe"]
            and x["portfolio"]["max_drawdown_pct"] >= x["spy"]["max_drawdown_pct"]
        )

    checks = {
        "1_primary_pinned": c1(p),
        "2_post_pub_positive": out["primary"]["post_pub"]["1bp"]["total_return_pct"] > 0,
        "3_wfo_gates_12_no_halt_stable_8": w["n_gate_pass"] >= 12
        and not w["halted_any"]
        and w["top_winner_folds"] >= 8,
        "4_portfolio_beats_spy": c4(port["1bp"]),
        "5_criteria_1_4_at_3bp": c1(p3) and c4(port["3bp"]),
    }
    return {"checks": checks, "go": all(checks.values()), "years_2021_2025_pct": years}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-run recorded sections")
    args = ap.parse_args()

    out = json.loads(args.out.read_text()) if args.out.exists() and not args.force else {}
    out["meta"] = {
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "signal_position_size": 1.0,
        "wfo_cost": "STOCK_BASE_CONFIG SLIPPAGE 5bp/side, FEES 0",
        "cash_rate": 0.0,
    }

    ohlcv = load_ohlcv(MONTH_END_TREASURY_UNIVERSE + ["SPY"], DATA_START, DATA_END)
    close = extract_close(ohlcv, MONTH_END_TREASURY_UNIVERSE + ["SPY"]).dropna()

    sections = {
        "primary": lambda: run_primary(close),
        "sweep_standalone": lambda: run_sweep_standalone(close),
        "portfolio": lambda: run_portfolio(close),
        "wfo": lambda: run_wfo_section(
            ohlcv[MONTH_END_TREASURY_UNIVERSE], ohlcv["SPY"]["close"].dropna()
        ),
    }
    for name, fn in sections.items():
        if name in out and "error" not in out[name]:
            print(f"[skip] {name} already recorded", flush=True)
            continue
        t0 = time.time()
        print(f"[run] {name}", flush=True)
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- recorded, later sections still run
            out[name] = {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
            print(f"[fail] {name}: {exc}", flush=True)
        out[name + "_elapsed_s"] = round(time.time() - t0, 1)
        args.out.write_text(json.dumps(out, indent=2, default=str))

    if all("error" not in out[k] for k in sections):
        out["verdict"] = verdict(out)
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
