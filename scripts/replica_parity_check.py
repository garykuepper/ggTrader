"""Explain persisted live strategy orders using the replica's production signal panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import load_ohlcv
from ggTrader.lab.persist import get_engine
from ggTrader.lab.replica import SignalPanel, build_signal_panel
from ggTrader.paper.cash_sweep import SWEEP_TRADE_REASON
from ggTrader.paper.catastrophe_stop import CATASTROPHE_STOP_REASON

ROOT = Path(__file__).resolve().parents[1]
NON_STRATEGY_REASONS = {SWEEP_TRADE_REASON, CATASTROPHE_STOP_REASON}


def explain_orders(
    orders: pd.DataFrame, panel: SignalPanel, prior_buys: dict[str, pd.Timestamp]
) -> dict:
    """Explain buys, last-bar exits and catch-up exits without broker access."""
    last_buys = {symbol: pd.Timestamp(date).normalize() for symbol, date in prior_buys.items()}
    rows: list[dict] = []
    daily: dict[str, dict] = {}
    for order in orders.sort_values(["run_date", "id"]).to_dict("records"):
        date = pd.Timestamp(order["run_date"], tz="UTC").normalize()
        prior = panel.entries.index[panel.entries.index.normalize() < date]
        bar = prior[-1] if len(prior) else None
        symbol, side = order["symbol"], order["side"].upper()
        strategy_order = order["reason"] not in NON_STRATEGY_REASONS
        explanation = "excluded sweep or catastrophe stop"
        explained = not strategy_order
        if strategy_order:
            explained = False
            explanation = "no prior signal bar or symbol absent from the panel"
            if bar is not None and symbol in panel.entries:
                if side == "BUY":
                    explained = bool(panel.entries.at[bar, symbol])
                    explanation = "entry signal" if explained else "no entry signal"
                elif side == "SELL":
                    regular = bool(panel.exits.at[bar, symbol])
                    buy_date = last_buys.get(symbol)
                    catch_up = False
                    if buy_date is not None:
                        dates = panel.exits.index
                        mask = (dates.normalize() > buy_date) & (dates <= bar)
                        catch_up = bool(panel.exits.loc[mask, symbol].any())
                    explained = regular or catch_up
                    explanation = (
                        "last-bar exit"
                        if regular
                        else "catch-up exit"
                        if catch_up
                        else "no exit since last live buy"
                    )
            key = str(date.date())
            if key not in daily:
                daily[key] = {
                    "signal_bar": str(bar) if bar is not None else None,
                    "entry_candidates": int(panel.entries.loc[bar].sum()) if bar is not None else 0,
                    "live_buys": 0,
                }
            daily[key]["live_buys"] += side == "BUY"
            rows.append(
                {
                    "id": int(order["id"]),
                    "date": key,
                    "symbol": symbol,
                    "side": side,
                    "reason": order["reason"],
                    "signal_bar": str(bar) if bar is not None else None,
                    "last_live_buy": str(last_buys.get(symbol)),
                    "explained": explained,
                    "explanation": explanation,
                }
            )
        if side == "BUY":
            last_buys[symbol] = date
    count = len(rows)
    n_explained = sum(row["explained"] for row in rows)
    return {
        "orders": rows,
        "strategy_orders": count,
        "explained_orders": n_explained,
        "explained_pct": 100.0 * n_explained / count if count else 0.0,
        "daily": daily,
        "notes": [],
        "stop_flags": ["unexplained_orders"]
        if n_explained < count
        else ["no_strategy_orders"]
        if not count
        else [],
    }


def main() -> None:
    """Read orders and cached prices, then write the parity evidence as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-09-29")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    since = pd.Timestamp(args.since, tz="UTC")
    today = pd.Timestamp.now(tz="UTC").normalize()
    out = args.out or ROOT / "docs/research" / f"_replica_parity_{today:%Y%m%d}.json"
    # A read-only transaction protects the live tables even if this script changes later.
    with get_engine().connect().execution_options(postgresql_readonly=True) as conn:
        reasons = conn.execute(text("SELECT DISTINCT reason FROM paper_trades ORDER BY reason"))
        reason_values = [row[0] for row in reasons]
        print("SELECT DISTINCT reason:", reason_values, flush=True)
        orders = pd.read_sql(
            text(
                "SELECT id, run_date, side, symbol, reason FROM paper_trades "
                "WHERE run_date >= :since ORDER BY run_date, id"
            ),
            conn,
            params={"since": since.date()},
        )
        prior = conn.execute(
            text(
                "SELECT symbol, max(run_date) FROM paper_trades "
                "WHERE side = 'BUY' AND run_date < :since GROUP BY symbol"
            ),
            {"since": since.date()},
        )
        prior_buys = {row[0]: pd.Timestamp(row[1], tz="UTC") for row in prior}
    members = sorted({normalize_yf_ticker(m) for m in universe_members_asof("sp500", today)})
    data = load_ohlcv(
        members + ["SPY"], str((since - pd.Timedelta(days=500)).date()), str(today.date())
    )
    if len(data.index) < 62:
        raise ValueError("Insufficient warm-up data for the parity panel")
    panel = build_signal_panel(data, data.index[60], data.index[-1], lambda asof, past: members)
    result = explain_orders(orders, panel, prior_buys)
    result.update(
        {
            "since": args.since,
            "through": str(today.date()),
            "distinct_reasons": reason_values,
            "universe_size": len(members),
            "panel_first_bar": str(panel.close.index[0]),
            "panel_last_bar": str(panel.close.index[-1]),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["strategy_orders", "explained_orders", "explained_pct", "stop_flags"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
