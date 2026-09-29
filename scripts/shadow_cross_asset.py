"""Paper shadow of the static 80/20 cross-asset book vs the live account and SPY.

The static book (80% SPY + TLT/GLD/PDBC, monthly rebalance) passed its
pre-registered bar as a paper-shadow GO only
(docs/research/2026-09-28-cross-asset-sleeve.md). This recomputes that book
from prices since SHADOW_START on every run -- no state, nothing persisted
except the price top-up -- and prints it beside the live paper account and
SPY buy-and-hold, all rebased to the live NAV on SHADOW_START.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv, rebalance_dates
from ggTrader.lab.persist import get_engine
from ggTrader.lab.simulate import simulate_weights
from ggTrader.lab.strategies.cross_asset_sleeve import (
    CROSS_ASSET_SLEEVE_UNIVERSE,
    CrossAssetSleeveStrategy,
)
from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results/shadow_cross_asset.csv"
SHADOW_START = "2026-09-29"
COST_BP = 1.0  # same per-side cost as the research's primary test


def shadow_curve(
    ohlcv: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost_bp: float = COST_BP
) -> pd.Series:
    """Static 80/20 equity curve over [start, end], normalized to 1.0 at start.

    Same path as the research driver's primary test: the committed strategy's
    plans at `rebalance_dates`, each applied on the next bar, simulated by
    `simulate_weights`.
    """
    win = ohlcv.loc[start:end]
    strat = CrossAssetSleeveStrategy(LabConfig(), trend=False)
    plans = {d: strat.select(d, ohlcv.loc[:d], []) for d in rebalance_dates(win.index, start, end)}
    targets = strat.to_targets(plans, win)
    prices = extract_close(win, list(targets.columns))
    cfg = {**STOCK_BASE_CONFIG, "SLIPPAGE": cost_bp / 1e4, "FEES": 0.0}
    _returns, equity, _diag = simulate_weights({"shadow": targets}, prices, cfg)
    curve = equity["shadow"]
    return curve / curve.iloc[0]


def compare(books: dict[str, pd.Series], start_value: float) -> pd.DataFrame:
    """Value, return and max drawdown per book, each rebased to `start_value`."""
    rows = {}
    for name, series in books.items():
        s = series.dropna()
        rebased = s / s.iloc[0] * start_value
        rows[name] = {
            "value": rebased.iloc[-1],
            "return_pct": (rebased.iloc[-1] / start_value - 1) * 100,
            "max_dd_pct": (rebased / rebased.cummax() - 1).min() * 100,
        }
    return pd.DataFrame(rows).T


def live_nav(start: str) -> pd.Series:
    """Live paper-account NAV from `paper_snapshots` (12:45 PT, not the close)."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT run_date, portfolio_value FROM paper_snapshots "
                "WHERE run_date >= :start ORDER BY run_date"
            ),
            {"start": start},
        ).all()
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows]).tz_localize("UTC")
    return pd.Series([float(v) for _, v in rows], index=idx, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=SHADOW_START, help="shadow start date (YYYY-MM-DD)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="daily series output")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp.now(tz="UTC").normalize()
    ohlcv = load_ohlcv(
        CROSS_ASSET_SLEEVE_UNIVERSE, args.start, str((end + pd.Timedelta(days=1)).date())
    )
    live = live_nav(args.start)
    if ohlcv.loc[start:].shape[0] < 2 or live.empty:
        print(f"Shadow starts {args.start}; not enough data yet (need 2 bars and a live snapshot).")
        return

    spy = extract_close(ohlcv.loc[start:], ["SPY"])["SPY"]
    books = {"shadow 80/20": shadow_curve(ohlcv, start, end), "live": live, "SPY": spy}
    table = compare(books, start_value=float(live.iloc[0]))
    print(f"Shadow since {args.start} (rebased to live NAV ${live.iloc[0]:,.0f}):")
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))

    daily = pd.DataFrame({k: v / v.dropna().iloc[0] for k, v in books.items()})
    daily.index = daily.index.date
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.csv, index_label="date", float_format="%.6f")


if __name__ == "__main__":
    main()
