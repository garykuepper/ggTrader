"""Run the approved historical daily-bar model with pinned windows and audit stops."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.fred_data import load_fred_series
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.persist import _sanitize
from ggTrader.lab.replica import ReplicaConfig, SignalPanel, build_signal_panel, run_replica
from ggTrader.lab.significance import ExcessStats, excess_stats, passes_pretax_bar, years_for_t
from ggTrader.lab.strategy import LabConfig

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"primary": ("2021-01-31", "2026-04-30"), "holdout": ("2010-01-31", "2020-12-31")}
WARMUP_DAYS = 420
SEAM_THRESHOLD = 0.40
HOLDOUT_COVERAGE_FLOOR = 0.70
ARMS = {
    "primary": {},
    "lag0": {"signal_lag_bars": 0},
    "cost3": {"stock_cost_bps": 3.0},
    "cost10": {"stock_cost_bps": 10.0},
    "cash0": {},
}
DEFAULT_OUT = ROOT / "docs/research/_production_replica_results.json"
DEFAULT_CURVES = ROOT / "docs/research/_production_replica_curves.csv"
AMENDMENT = "docs/research/briefs/2026-10-19-production-replica-parity-amendment.md"
LOG = logging.getLogger(__name__)


def tbill_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill cached TB3MS annual percent and convert to simple daily rates."""
    start = str((index.min() - pd.Timedelta(days=120)).date())
    frame = load_fred_series("TB3MS", start, str(index.max().date()))
    if frame.empty:
        raise ValueError("TB3MS cache is empty; the audit cannot substitute zero cash yield")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True))
    rates = pd.Series(frame.value.to_numpy(dtype=float), index=dates).sort_index()
    target = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    aligned = rates.reindex(target, method="ffill")
    if aligned.isna().any():
        raise ValueError("TB3MS cache does not cover the first audit session")
    return pd.Series((1 + aligned.to_numpy() / 100) ** (1 / 252) - 1, index=index)


def exposure_matched(
    spy_close: pd.Series, exposure: pd.DataFrame, cash_rate: pd.Series, start: float
) -> pd.Series:
    """Compound SPY and cash using the replica's previous-close exposures."""
    spy_return = spy_close.reindex(exposure.index).ffill().pct_change(fill_method=None).fillna(0)
    weights = exposure.shift(1)
    returns = ((weights.stocks + weights.spy) * spy_return + weights.cash * cash_rate).fillna(0)
    return (start * (1 + returns).cumprod()).rename("exposure_matched")


def seam_scan(close: pd.DataFrame, threshold: float) -> list[dict]:
    """Flag large adjacent observed returns without filling missing stock prices."""
    returns = close.pct_change(fill_method=None)
    rows, cols = np.where(returns.abs().to_numpy() >= threshold)
    return [
        {
            "symbol": str(close.columns[c]),
            "date": str(close.index[r]),
            "return": float(returns.iat[r, c]),
        }
        for r, c in zip(rows, cols)
    ]


def _coverage_rows(close: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Retain member-day counts so the holdout floor uses the registered denominator."""
    rows = []
    for date in dates:
        members = {normalize_yf_ticker(m) for m in universe_members_asof("sp500", date)}
        prices = close.loc[date].reindex(sorted(members)).to_numpy(dtype=float)
        count = int(np.isfinite(prices).sum())
        rows.append(
            dict(
                year=date.year,
                observed=count,
                expected=len(members),
                fraction=count / len(members) if members else 0.0,
            )
        )
    return pd.DataFrame(rows)


def coverage(close: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[int, float]:
    """Return each year's mean daily point-in-time membership coverage."""
    return _coverage_rows(close, dates).groupby("year").fraction.mean().to_dict()


def verdict(results: dict) -> dict:
    """Apply the registered primary-first pre-tax decision ordering."""
    windows = results.get("windows", {})
    primary = windows.get("primary", {})
    stopped = dict(verdict="INCONCLUSIVE", stage="stopped", reasons=[])
    if primary.get("stop_flags"):
        return {**stopped, "reasons": primary["stop_flags"]}
    p = primary.get("arms", {}).get("primary")
    if p is None:
        return {**stopped, "reasons": ["primary run incomplete"]}
    if not passes_pretax_bar(ExcessStats(**p["excess_vs_spy"])):
        return dict(verdict="NO-GO", stage="pretax_fail", reasons=["primary fails pre-tax bar"])
    holdout = windows.get("holdout", {})
    if holdout.get("stop_flags"):
        return {**stopped, "reasons": holdout["stop_flags"]}
    cov = holdout.get("mean_coverage")
    if cov is not None and cov < HOLDOUT_COVERAGE_FLOOR:
        return {**stopped, "reasons": ["holdout coverage below 70%"]}
    h = holdout.get("arms", {}).get("primary")
    if h is None or cov is None:
        return {**stopped, "reasons": ["holdout run incomplete"]}
    if not passes_pretax_bar(ExcessStats(**h["excess_vs_spy"])):
        return dict(verdict="NO-GO", stage="pretax_fail", reasons=["holdout fails pre-tax bar"])
    return dict(verdict="INCONCLUSIVE", stage="pretax_pass", reasons=["after-tax replay required"])


def tainted_trades(trades: pd.DataFrame, seams: list[dict], index: pd.DatetimeIndex) -> list[dict]:
    """Find same-symbol trades within five index bars of a confirmed artifact."""
    matches = []
    for seam in seams:
        if not seam.get("artifact"):
            continue
        position = index.get_indexer([pd.Timestamp(seam["date"])])[0]
        if position < 0:
            raise ValueError("Artifact date is absent from the source tape")
        dates = index[max(0, position - 5) : position + 6]
        hits = trades[(trades.symbol == seam["symbol"]) & trades.date.isin(dates)]
        matches.extend(
            dict(date=str(row.date), symbol=row.symbol, artifact_date=seam["date"])
            for row in hits.itertuples()
        )
    return matches


def _checkpoint(out: Path, results: dict) -> None:
    """Atomically save results, retaining a global stop over any provisional verdict."""
    flags = [
        f"{name}:{flag}"
        for name, window in results["windows"].items()
        for flag in window.get("stop_flags", [])
    ]
    results["verdict"] = (
        dict(verdict="INCONCLUSIVE", stage="stopped", reasons=flags) if flags else verdict(results)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(_sanitize(results), indent=2, allow_nan=False) + "\n")
    temporary.replace(out)


def _verify_seams(close: pd.DataFrame, state: dict, results: dict, out: Path) -> None:
    """Compare flagged returns to fresh adjusted data and checkpoint each symbol."""
    seams = state.setdefault("seam_scan", seam_scan(close, SEAM_THRESHOLD))
    symbols = sorted({row["symbol"] for row in seams if "artifact" not in row})
    for symbol in symbols:
        pending = [row for row in seams if row["symbol"] == symbol and "artifact" not in row]
        dates = [pd.Timestamp(row["date"]) for row in pending]
        try:
            fresh = yf.download(
                symbol,
                start=str((min(dates) - pd.Timedelta(days=10)).date()),
                end=str((max(dates) + pd.Timedelta(days=5)).date()),
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30,
            )
            prices = fresh["Close"]
            if isinstance(prices, pd.DataFrame):
                prices = prices[symbol]
            prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
            fresh_returns = prices.pct_change(fill_method=None)
            for row in pending:
                date = pd.Timestamp(row["date"]).tz_localize(None).normalize()
                value = fresh_returns.get(date, np.nan)
                row["fresh_return"] = float(value) if np.isfinite(value) else None
                row["artifact"] = bool(not np.isfinite(value) or abs(row["return"] - value) > 0.05)
        except Exception as exc:
            LOG.warning("Fresh seam comparison failed for %s: %s", symbol, type(exc).__name__)
            for row in pending:
                row.update(artifact=True, fresh_return=None, comparison_error=type(exc).__name__)
        LOG.info("Seam verification %s: %d flagged bars", symbol, len(pending))
        _checkpoint(out, results)


def _save_curves(path: Path, frame: pd.DataFrame, window: str, arm: str) -> None:
    """Replace a single arm's CSV rows so a resumed run cannot duplicate them."""
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[~((previous.window == window) & (previous.arm == arm))]
        frame = pd.concat([previous, frame], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_window(
    name: str, results: dict, out: Path, curves: Path, max_symbols: int | None, force: bool
) -> bool:
    """Run or resume one pinned window; return False immediately on an audit stop."""
    start, end = WINDOWS[name]
    state = results["windows"].setdefault(
        name, dict(eval_start=start, eval_end=end, arms={}, stop_flags=[])
    )
    if state["stop_flags"]:
        LOG.error("Existing registered stop in %s: %s", name, state["stop_flags"])
        return False
    if not force and all(arm in state["arms"] for arm in ARMS):
        LOG.info("%s already complete", name)
        return True
    _checkpoint(out, results)
    es, ee = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    universe = equity_universe_between(es, ee, "sp500")
    if max_symbols is not None:
        universe = universe[:max_symbols]
    LOG.info("Loading %s: %d symbols, %s through %s", name, len(universe), start, end)
    data = load_ohlcv(
        universe + ["SPY"],
        str((es - pd.Timedelta(days=WARMUP_DAYS)).date()),
        end,
        use_negative_cache=True,
    )
    before = data.index[data.index < es]
    if not len(before):
        raise ValueError("Audit needs a pre-window lag bar")
    lag_bar = before[-1]
    cfg = LabConfig(min_history_bars=60)
    full = build_signal_panel(
        data, lag_bar, ee, lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0]
    )
    all_close = full.close.assign(SPY=full.spy_close)
    _verify_seams(all_close, state, results, out)
    dates = list(full.close.index[(full.close.index >= es) & (full.close.index <= ee)])
    counts = _coverage_rows(full.close, dates)
    state["coverage"] = counts.groupby("year").fraction.mean().to_dict()
    state["coverage_member_days"] = dict(
        observed=int(counts.observed.sum()), expected=int(counts.expected.sum())
    )
    state["mean_coverage"] = float(counts.observed.sum() / counts.expected.sum())
    if (
        name == "holdout"
        and max_symbols is None
        and state["mean_coverage"] < HOLDOUT_COVERAGE_FLOOR
    ):
        state["stop_flags"].append("holdout_coverage_below_70pct")
        _checkpoint(out, results)
        return False
    mask = (full.close.index >= lag_bar) & (full.close.index <= ee)
    panel = SignalPanel(*(frame.loc[mask] for frame in full))
    rates = tbill_daily(panel.close.index)
    spy = panel.spy_close.ffill()
    if spy.isna().any():
        raise ValueError("SPY has no mark at the window's first bar")
    _checkpoint(out, results)
    for arm, settings in ARMS.items():
        if arm in state["arms"] and not force:
            continue
        LOG.info("Running %s/%s", name, arm)
        config = replace(ReplicaConfig(), **settings)
        cash_rate = rates * 0 if arm == "cash0" else rates
        replay = run_replica(*panel, config, cash_rate_daily=cash_rate)
        spy_curve = config.start_cash * spy / spy.iloc[0]
        matched = exposure_matched(spy, replay.exposure, cash_rate, config.start_cash)
        over_spy = excess_stats(replay.equity, spy_curve)
        tainted = tainted_trades(replay.trades, state["seam_scan"], full.close.index)
        arm_result = dict(
            config=asdict(config),
            stats=curve_stats(replay.equity),
            spy_stats=curve_stats(spy_curve),
            excess_vs_spy=asdict(over_spy),
            exposure_matched_stats=curve_stats(matched),
            excess_vs_exposure_matched=asdict(excess_stats(replay.equity, matched)),
            mean_exposure=replay.exposure.mean().to_dict(),
            trade_count=len(replay.trades),
            trade_counts=replay.trades.groupby(["symbol", "side"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records"),
            halted_days=replay.halted_days,
            truncated_days=replay.truncated_days,
            years_for_t=years_for_t(over_spy.information_ratio),
            tainted_trades=tainted,
            equity_has_nan=bool(replay.equity.isna().any()),
            min_cash_exposure=float(replay.exposure.cash.min()),
            pretax_pass=passes_pretax_bar(over_spy),
        )
        frame = pd.DataFrame(
            dict(
                date=replay.equity.index,
                window=name,
                arm=arm,
                equity=replay.equity.to_numpy(),
                spy=spy_curve.to_numpy(),
                exposure_matched=matched.to_numpy(),
                stocks_weight=replay.exposure.stocks.to_numpy(),
                spy_weight=replay.exposure.spy.to_numpy(),
                cash_weight=replay.exposure.cash.to_numpy(),
            )
        )
        _save_curves(curves, frame, name, arm)
        state["arms"][arm] = arm_result
        if tainted:
            state["stop_flags"].append(f"tainted_trade:{arm}")
        _checkpoint(out, results)
        LOG.info(
            "%s/%s: t=%.3f, excess CAGR=%+.3f pp, halted=%d, tainted=%d",
            name,
            arm,
            over_spy.t_stat,
            over_spy.cagr_diff_pp,
            replay.halted_days,
            len(tainted),
        )
        if tainted:
            return False
    return True


def main() -> None:
    """Run the production-replica audit according to the pre-registered protocol."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["primary", "holdout", "all"], default="all")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    args = parser.parse_args()

    results: dict = {
        "status": "in_progress",
        "citable": args.max_symbols is None,
        "windows": {},
    }
    if args.out.exists():
        results = json.loads(args.out.read_text())
        results["citable"] = results.get("citable", True) and args.max_symbols is None

    windows_to_run = ["primary", "holdout"] if args.window == "all" else [args.window]

    for w in windows_to_run:
        ok = run_window(w, results, args.out, args.curves, args.max_symbols, args.force)
        if not ok:
            LOG.warning("Window %s stopped; halting audit", w)
            break

    results["status"] = "complete"
    _checkpoint(args.out, results)
    LOG.info("Audit run complete. Final verdict: %s", results.get("verdict"))


if __name__ == "__main__":
    main()
