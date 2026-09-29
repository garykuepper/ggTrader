"""Cross-asset sleeve: frozen static/trend arms, reported diagnostics, lookback WFO.

Pre-registered in docs/research/briefs/2026-09-28-cross-asset-sleeve.md before
any run:
  1. static arm: 80% SPY + 20% equal-weight TLT/GLD/PDBC, month-end rebalance,
     1 bp/side, on the pinned window, 2016-01 -> 2026-04, and calendar 2022.
  2. trend arm: identical, each leg -> BIL when its trailing 252-day total
     return is <= 0 at the month-end close (one lookback, frozen).
  3. WFO robustness (trend arm): lookback {126, 189, 252}, pinned 17 folds.
  4. reported only: sleeve 10%/30%, leave-one-leg-out, C4 pre-screen, and
     correlation to SPY in SPY's worst 10% of months.

Every book goes through the lab's own select() -> to_targets() ->
simulate_weights() path. Each section is checkpointed into the JSON output;
re-running skips sections already recorded (use --force to redo them).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv, rebalance_dates
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_weights
from ggTrader.lab.strategies.cross_asset_sleeve import (
    CROSS_ASSET_SLEEVE_UNIVERSE,
    LEGS,
    CrossAssetSleeveStrategy,
)
from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import generate_folds, run_wfo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs/research/_cross_asset_sleeve_results.json"

EVAL_START = "2021-01-31"
EVAL_END = "2026-04-30"
#: load_ohlcv's end is exclusive; one extra day keeps 2026-04-30.
DATA_END = "2026-05-01"
#: PDBC's first bar is 2014-11-07; everything the books trade exists from here.
DATA_START = "2014-11-10"
REFERENCE_SYMBOLS = ["IEF", "IAU"]

_FOLDS = generate_folds(pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC"))
WINDOWS = {
    "pinned": (EVAL_START, EVAL_END),
    "long": ("2016-01-01", EVAL_END),
    "y2022": ("2022-01-01", "2022-12-31"),
    # The span the WFO's OOS curve (and the snapshot's SPY 0.78 / core+SPY 0.80) covers.
    "wfo_oos": (str(_FOLDS[0].test_start.date()), EVAL_END),
}
CORE_SPY_SWEEP_SHARPE = 0.80  # docs/research/2026-09-24-core-spy-blend-keep.md, not re-run
COSTS_BP = [1, 3]


def book_curve(ohlcv: pd.DataFrame, start: str, end: str, cost_bp: float, **kw) -> dict:
    """Run one frozen book over [start, end]; select() sees history before start."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    win = ohlcv.loc[start_ts:end_ts]
    strat = CrossAssetSleeveStrategy(LabConfig(lookback=kw.pop("lookback", 252)), **kw)
    dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
    plans = {d: strat.select(d, ohlcv.loc[:d], []) for d in dates}
    return simulate_plans(plans, win, cost_bp) | {"plans": plans}


def fixed_curve(ohlcv: pd.DataFrame, start: str, end: str, weights: dict, cost_bp: float) -> dict:
    """A fixed-weight book rebalanced at every month-end (60/40, sleeve-only, ...)."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    win = ohlcv.loc[start_ts:end_ts]
    plan = [{"symbol": s, "weight": w} for s, w in weights.items()]
    plans = {d: plan for d in rebalance_dates(ohlcv.index, start_ts, end_ts)}
    return simulate_plans(plans, win, cost_bp)


def simulate_plans(plans: dict, win: pd.DataFrame, cost_bp: float) -> dict:
    targets = CrossAssetSleeveStrategy(LabConfig()).to_targets(plans, win)
    prices = extract_close(win, list(targets.columns))
    cfg = {**STOCK_BASE_CONFIG, "SLIPPAGE": cost_bp / 1e4, "FEES": 0.0}
    _r, eq, diag = simulate_weights({"b": targets}, prices, cfg)
    curve = eq["b"]
    return {"stats": stats(curve), "curve": curve, "n_trades": int(diag["b"].get("n_trades", 0))}


def stats(curve: pd.Series) -> dict:
    yearly = curve.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / curve.iloc[0] - 1
    return {
        **curve_stats(curve),
        "yearly_return_pct": {str(d.year): v * 100 for d, v in yr.items()},
    }


def buy_hold(close: pd.Series) -> dict:
    return stats(close.dropna() / close.dropna().iloc[0])


def leg_on_fraction(plans: dict) -> dict:
    """Share of month-end decisions each leg was held (trend arm diagnostic)."""
    n = len(plans)
    return {
        leg: sum(any(s["symbol"] == leg and s["weight"] > 0 for s in p) for p in plans.values()) / n
        for leg in LEGS
    }


def run_primary(ohlcv: pd.DataFrame, close: pd.DataFrame, wname: str) -> dict:
    s, e = WINDOWS[wname]
    c = close.loc[pd.Timestamp(s, tz="UTC") : pd.Timestamp(e, tz="UTC")]
    out: dict = {"start": str(c.index[0].date()), "end": str(c.index[-1].date())}
    for bp in COSTS_BP:
        static = book_curve(ohlcv, s, e, bp, trend=False)
        trend = book_curve(ohlcv, s, e, bp, trend=True)
        out[f"{bp}bp"] = {
            "static": static["stats"] | {"n_trades": static["n_trades"]},
            "trend": trend["stats"]
            | {"n_trades": trend["n_trades"], "leg_on_fraction": leg_on_fraction(trend["plans"])},
            "spy_ief_60_40": fixed_curve(ohlcv, s, e, {"SPY": 0.6, "IEF": 0.4}, bp)["stats"],
        }
    out["bh"] = {sym: buy_hold(c[sym]) for sym in c.columns}
    return out


def run_reported(ohlcv: pd.DataFrame, close: pd.DataFrame) -> dict:
    out: dict = {}
    for wname in ["pinned", "long"]:
        s, e = WINDOWS[wname]
        w: dict = {}
        for sw in [0.10, 0.30]:
            for arm, trend in [("static", False), ("trend", True)]:
                w[f"{arm}_sleeve{int(sw * 100)}"] = book_curve(
                    ohlcv, s, e, 1, trend=trend, sleeve_weight=sw
                )["stats"]
        for drop in LEGS:
            legs = [x for x in LEGS if x != drop]
            for arm, trend in [("static", False), ("trend", True)]:
                w[f"{arm}_drop_{drop}"] = book_curve(ohlcv, s, e, 1, trend=trend, legs=legs)[
                    "stats"
                ]
        # C4 pre-screen: a sleeve funded from SPY raises book Sharpe iff
        # SR_sleeve > rho(sleeve, SPY) * SR_SPY (small-weight approximation).
        sleeve = fixed_curve(ohlcv, s, e, {leg: 1 / 3 for leg in LEGS}, 1)["curve"]
        spy = close["SPY"].loc[sleeve.index]
        r_sl, r_spy = sleeve.pct_change().dropna(), spy.pct_change().dropna()
        rho = float(r_sl.corr(r_spy))
        sr_sl, sr_spy = curve_stats(sleeve)["sharpe"], curve_stats(spy)["sharpe"]
        m_sl = sleeve.resample("ME").last().pct_change().dropna()
        m_spy = spy.resample("ME").last().pct_change().dropna()
        worst = m_spy <= m_spy.quantile(0.10)
        w["c4_prescreen"] = {
            "sleeve_only": curve_stats(sleeve),
            "rho_daily": rho,
            "sleeve_sharpe": sr_sl,
            "rho_x_spy_sharpe": rho * sr_spy,
            "passes": bool(sr_sl > rho * sr_spy),
        }
        w["tail_months"] = {
            "n_months": int(worst.sum()),
            "rho_monthly_all": float(m_sl.corr(m_spy)),
            "rho_monthly_spy_worst10pct": float(m_sl[worst].corr(m_spy[worst])),
            "sleeve_mean_ret_pct_spy_worst10pct": float(m_sl[worst].mean() * 100),
            "spy_mean_ret_pct_spy_worst10pct": float(m_spy[worst].mean() * 100),
            "legs_mean_ret_pct_spy_worst10pct": {
                leg: float(
                    close[leg]
                    .loc[sleeve.index]
                    .resample("ME")
                    .last()
                    .pct_change()[m_spy.index][worst]
                    .mean()
                    * 100
                )
                for leg in LEGS
            },
        }
        out[wname] = w
    return out


def run_post_hoc(ohlcv: pd.DataFrame, close: pd.DataFrame, n_boot: int = 2000) -> dict:
    """POST-HOC, not pre-registered: is book Sharpe - SPY Sharpe distinguishable from 0?

    Paired circular block bootstrap (21-day blocks) of daily returns; reports the
    share of resamples where the static book's Sharpe exceeds SPY's.
    """
    rng = np.random.default_rng(20260928)
    out: dict = {}
    for wname in ["pinned", "long"]:
        s, e = WINDOWS[wname]
        book = book_curve(ohlcv, s, e, 1, trend=False)["curve"]
        r = pd.concat([book, close["SPY"].loc[book.index]], axis=1).pct_change().dropna().values
        n, block = len(r), 21
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, n, size=n // block + 1)
            idx = (starts[:, None] + np.arange(block)).ravel()[:n] % n
            x = r[idx]
            sr = x.mean(axis=0) / x.std(axis=0) * np.sqrt(252)
            diffs[i] = sr[0] - sr[1]
        out[wname] = {
            "p_book_sharpe_gt_spy": float((diffs > 0).mean()),
            "diff_median": float(np.median(diffs)),
            "diff_ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
        }
    return out


def run_wfo_section(ohlcv: pd.DataFrame) -> dict:
    grid = build_grid(CrossAssetSleeveStrategy)
    res = run_wfo(
        "cross_asset_sleeve",
        CrossAssetSleeveStrategy,
        LabConfig(),
        ohlcv,
        ohlcv["SPY"]["close"].dropna(),
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market="stock",
        base_config={**STOCK_BASE_CONFIG, "SLIPPAGE": 1e-4, "FEES": 0.0},
        grid=grid,
        # Fixed ETF list -- no index membership, so no PIT mask to build.
        universe_fn=lambda asof, past: CROSS_ASSET_SLEEVE_UNIVERSE,
    )
    rows = [
        {
            k: (str(pd.Timestamp(v).date()) if k.endswith(("_start", "_end")) else v)
            for k, v in r.items()
            if k != "winner_params"
        }
        for r in res.fold_results
    ]
    spy = ohlcv["SPY"]["close"].reindex(res.oos_equity.index).ffill().dropna()
    winners = Counter(r["winner_combo"] for r in rows)
    top, top_n = winners.most_common(1)[0]
    return {
        "grid": grid,
        "oos": curve_stats(res.oos_equity),
        "oos_start": str(res.oos_equity.index[0].date()),
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
    def p(w: str, bp: int = 1) -> dict:
        return out[f"primary:{w}"][f"{bp}bp"]

    def spy(w: str) -> dict:
        return out[f"primary:{w}"]["bh"]["SPY"]

    def beats(w: str, arm: str, bp: int) -> tuple[bool, bool]:
        b = p(w, bp)[arm]
        return b["sharpe"] > spy(w)["sharpe"], b["max_drawdown_pct"] >= spy(w)["max_drawdown_pct"]

    wins = ["pinned", "long"]
    c1 = all(beats(w, "static", 1)[0] for w in wins)
    c2 = all(beats(w, "static", 1)[1] for w in wins)
    c3 = p("y2022")["static"]["total_return_pct"] >= spy("y2022")["total_return_pct"]
    c4 = all(all(beats(w, "static", 3)) for w in wins)
    c5 = all(
        out["reported"][w][f"static_drop_{leg}"]["sharpe"] > spy(w)["sharpe"]
        for w in wins
        for leg in LEGS
    )
    static_checks = {
        "1_sharpe_gt_spy_both_windows": c1,
        "2_maxdd_no_worse_both_windows": c2,
        "3_2022_return_no_worse": c3,
        "4_c1_c2_at_3bp": c4,
        "5_c1_with_any_leg_removed": c5,
    }
    st, tr = p("pinned")["static"], p("pinned")["trend"]
    d_sh = tr["sharpe"] - st["sharpe"]
    d_dd = tr["max_drawdown_pct"] - st["max_drawdown_pct"]
    wfo = out["wfo"]
    trend_checks = {
        "a_sharpe_plus_0.05_or_dd_plus_3_no_sharpe_loss": d_sh >= 0.05 or (d_dd >= 3 and d_sh >= 0),
        "b_wfo_gates_ge_12": wfo["n_gate_pass"] >= 12,
        "c_no_regime_halt": not wfo["halted_any"],
        "d_same_lookback_ge_8_folds": wfo["top_winner_folds"] >= 8,
    }
    static_go = all(static_checks.values())
    trend_pref = all(trend_checks.values())
    winner = "trend" if trend_pref else "static"
    oos = out["primary:wfo_oos"]["1bp"][winner]["sharpe"]
    return {
        "static_checks": static_checks,
        "static_go": static_go,
        "trend_checks": trend_checks,
        "trend_minus_static_sharpe_pinned": d_sh,
        "trend_minus_static_maxdd_pts_pinned": d_dd,
        "trend_preferred": trend_pref,
        "winning_arm": winner,
        "winning_arm_sharpe_wfo_oos_span": oos,
        "spy_sharpe_wfo_oos_span": spy("wfo_oos")["sharpe"],
        "beats_core_spy_sweep_0.80": bool(oos > CORE_SPY_SWEEP_SHARPE),
    }


HOLDOUT_OUT = REPO_ROOT / "docs/research/_cross_asset_sleeve_holdout_results.json"
HOLDOUT_WINDOW = ("2007-06-01", "2011-02-28")
HOLDOUT_LEGS = ("TLT", "GLD", "DBC")


def run_holdout(out_path: Path) -> dict:
    """Pre-registered unseen-data holdout (briefs/2026-09-28-cross-asset-sleeve-holdout.md).

    TLT/GLD/BIL/DBC have no 2007-2011 rows in ``ohlcv``, so prices come straight
    from yfinance (auto_adjust, same basis as the DB tape) rather than
    backfilling the live-shared table. The 2011-03 -> 2026-04 overlap is
    checked against ``ohlcv`` first.
    """
    import yfinance as yf

    syms = ["SPY", *HOLDOUT_LEGS, "BIL"]
    raw = yf.download(syms, start="2007-05-01", end=DATA_END, auto_adjust=True, progress=False)
    yclose = raw["Close"][syms]
    yclose.index = pd.DatetimeIndex(yclose.index).tz_localize("UTC")

    db = extract_close(
        load_ohlcv(["SPY", "TLT", "GLD"], "2011-03-01", DATA_END), ["SPY", "TLT", "GLD"]
    )
    db.index = db.index.tz_convert("UTC").normalize()
    ov = db.pct_change().join(yclose.pct_change(), rsuffix="_yf", how="inner").dropna()
    overlap_check = {
        s: {"n_days": len(ov), "max_abs_daily_diff": float((ov[s] - ov[s + "_yf"]).abs().max())}
        for s in ["SPY", "TLT", "GLD"]
    }

    s, e = HOLDOUT_WINDOW
    c = yclose.loc[pd.Timestamp(s, tz="UTC") : pd.Timestamp(e, tz="UTC")]
    if c.isna().any().any():
        raise ValueError(f"holdout tape has NaNs: {c.isna().sum().to_dict()}")
    ohlcv = pd.concat({sym: yclose[[sym]].set_axis(["close"], axis=1) for sym in syms}, axis=1)
    ohlcv = ohlcv.loc[pd.Timestamp("2007-05-30", tz="UTC") :]  # BIL inception

    res: dict = {
        "window": {"start": str(c.index[0].date()), "end": str(c.index[-1].date())},
        "prices": "yfinance auto_adjust, fetched at run time (not ohlcv)",
        "overlap_check_vs_ohlcv": overlap_check,
    }
    for bp in COSTS_BP:
        book = book_curve(ohlcv, s, e, bp, trend=False, legs=HOLDOUT_LEGS)
        res[f"{bp}bp"] = {"static": book["stats"] | {"n_trades": book["n_trades"]}}
    res["bh"] = {sym: buy_hold(c[sym]) for sym in syms}
    res["reported_loo_1bp"] = {
        f"drop_{d}": book_curve(
            ohlcv, s, e, 1, trend=False, legs=[x for x in HOLDOUT_LEGS if x != d]
        )["stats"]
        for d in HOLDOUT_LEGS
    }
    spy = res["bh"]["SPY"]
    beats = {
        bp: (
            res[f"{bp}bp"]["static"]["sharpe"] > spy["sharpe"],
            res[f"{bp}bp"]["static"]["max_drawdown_pct"] >= spy["max_drawdown_pct"],
        )
        for bp in COSTS_BP
    }
    checks = {
        "1_sharpe_gt_spy": beats[1][0],
        "2_maxdd_no_worse": beats[1][1],
        "3_c1_c2_at_3bp": all(beats[3]),
    }
    res["verdict"] = {
        "checks": checks,
        "pass": all(checks.values()),
        "reading": "pass = weak evidence (2008 flatters the sleeve); fail withdraws the GO",
    }
    out_path.write_text(json.dumps(res, indent=2, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-run recorded sections")
    ap.add_argument("--skip-wfo", action="store_true", help="frozen-rule sections only")
    ap.add_argument(
        "--holdout",
        action="store_true",
        help=f"run only the pre-registered 2007-06 -> 2011-02 holdout (writes {HOLDOUT_OUT.name})",
    )
    args = ap.parse_args()
    if args.holdout:
        res = run_holdout(HOLDOUT_OUT)
        print(json.dumps({k: res[k] for k in ["window", "verdict"]}, indent=2))
        return

    out = json.loads(args.out.read_text()) if args.out.exists() and not args.force else {}
    out["meta"] = {
        "windows": WINDOWS,
        "costs_bp_per_side": COSTS_BP,
        "wfo_cost": "SLIPPAGE 1bp/side, FEES 0",
        "rebalance": "month-end close decision, applied next bar (lab rebalance_dates)",
        "prices": "yfinance auto_adjust closes (total return) from ohlcv, venue yfinance",
        "sharpe_basis": "lab curve_stats: daily, rf=0, sqrt(252)",
        "core_spy_sweep_sharpe_cited": CORE_SPY_SWEEP_SHARPE,
    }

    syms = CROSS_ASSET_SLEEVE_UNIVERSE + REFERENCE_SYMBOLS
    ohlcv = load_ohlcv(syms, DATA_START, DATA_END)
    close = extract_close(ohlcv, syms)
    if (
        close.index.duplicated().any()
        or close.loc[:, CROSS_ASSET_SLEEVE_UNIVERSE].isna().any().any()
    ):
        bad = close[CROSS_ASSET_SLEEVE_UNIVERSE].isna().sum().to_dict()
        raise ValueError(f"tape not aligned: dup index or NaNs per symbol {bad}")
    out["meta"]["data"] = {
        "first_bar": str(close.index[0].date()),
        "last_bar": str(close.index[-1].date()),
        "n_bars": len(close),
    }

    sections = {f"primary:{w}": (lambda w=w: run_primary(ohlcv, close, w)) for w in WINDOWS}
    sections["reported"] = lambda: run_reported(ohlcv, close)
    sections["post_hoc"] = lambda: run_post_hoc(ohlcv, close)
    if not args.skip_wfo:
        sections["wfo"] = lambda: run_wfo_section(ohlcv[CROSS_ASSET_SLEEVE_UNIVERSE])
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

    if "wfo" in out and all("error" not in out[k] for k in sections):
        out["verdict"] = verdict(out)
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
