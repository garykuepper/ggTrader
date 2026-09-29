"""Read-only live-vs-backtest parity probe for the paper trader.

Flags the three ways the live book has drifted from the strategy the backtest
measured: exits that fired but were never acted on, holdings outside the
strategy's universe (no exit can ever fire), and positions held far longer
than a days-long reversion trade should last. Also flags dead data: a
holding whose tape has no recent bar, or whose broker price is far from our
latest close (AVB merged into EQR in 2026 and the broker price froze).
Places no orders.
"""

import argparse
from datetime import date

import pandas as pd
from sqlalchemy import text

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.persist import get_engine
from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.cash_sweep import sweep_symbol
from ggTrader.paper.persist import get_last_buy_dates
from ggTrader.paper.signal_runner import generate_signals

STALE_HOLD_DAYS = 30  # a reversion exit normally fires within days
TAPE_GAP_PCT = 5.0  # broker price vs our latest close; more than this is a data problem
TAPE_STALE_DAYS = 5  # calendar days without a bar before a holding counts as stale


def latest_closes(symbols: list[str]) -> dict[str, tuple]:
    """{symbol: (last bar date, close)} from the yfinance daily tape."""
    q = text(
        "SELECT DISTINCT ON (symbol) symbol, timestamp::date, close FROM ohlcv "
        "WHERE venue = 'yfinance' AND interval = '1d' AND symbol = ANY(:syms) "
        "ORDER BY symbol, timestamp DESC"
    )
    with get_engine().connect() as conn:
        return {sym: (d, c) for sym, d, c in conn.execute(q, {"syms": symbols}).all()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="sp500")
    args = parser.parse_args()

    positions = {s: p for s, p in AlpacaBroker().get_positions().items() if s != sweep_symbol()}
    members = {
        normalize_yf_ticker(t)
        for t in universe_members_asof(args.universe, pd.Timestamp.now(tz="UTC").normalize())
    }
    signals = generate_signals(args.universe)
    last_exit = signals.get("last_exit", {})
    entries = get_last_buy_dates()
    today = date.today()
    tape = latest_closes(sorted(positions))

    rows = []
    for sym, pos in sorted(positions.items()):
        entry = entries.get(sym)
        entry = entry.date() if hasattr(entry, "date") and not isinstance(entry, date) else entry
        exit_d = last_exit.get(sym)
        rows.append(
            {
                "symbol": sym,
                "entry": entry,
                "days_held": (today - entry).days if entry else None,
                "last_exit_bar": exit_d,
                "missed_exit": bool(entry and exit_d and date.fromisoformat(exit_d) > entry),
                "in_universe": sym in members,
                "unrealized_pct": round(100 * pos.get("unrealized_plpc", 0.0), 1),
                "tape_last_bar": tape.get(sym, (None, None))[0],
                "tape_vs_broker_pct": (
                    round(100 * (tape[sym][1] / pos["current_price"] - 1), 1)
                    if sym in tape and pos.get("current_price")
                    else None
                ),
            }
        )
    df = pd.DataFrame(rows)
    print(f"signals as_of {signals['as_of']}; {len(df)} strategy positions\n")
    print(df.to_string(index=False))
    if df.empty:
        return
    print("\nmissed exits:", df.loc[df.missed_exit, "symbol"].tolist())
    print("outside universe (no exit path):", df.loc[~df.in_universe, "symbol"].tolist())
    stale = df.loc[df.days_held.fillna(0) > STALE_HOLD_DAYS, "symbol"].tolist()
    print(f"held > {STALE_HOLD_DAYS} days:", stale)
    print("no BUY record:", df.loc[df.entry.isna(), "symbol"].tolist())
    gap = df.tape_vs_broker_pct.abs() > TAPE_GAP_PCT
    print(f"tape vs broker price gap > {TAPE_GAP_PCT}%:", df.loc[gap, "symbol"].tolist())
    age = df.tape_last_bar.map(lambda d: (today - d).days if d else None)
    print(
        f"no tape bar in {TAPE_STALE_DAYS}+ days (halt, merger, delisting?):",
        df.loc[age.fillna(999) > TAPE_STALE_DAYS, "symbol"].tolist(),
    )


if __name__ == "__main__":
    main()
