"""Print an advisory month-end A23 SPY/SSO rebalance check; never place orders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.lab.a23_rebalance import Lot, check_rebalance  # noqa: E402


def read_lots(path: Path, asof: date) -> dict[str, list[Lot]]:
    """Read independently acquired lots and reject malformed account data."""
    lots: dict[str, list[Lot]] = {"SPY": [], "SSO": []}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "shares", "cost_basis_total", "acquired_date"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"lots CSV needs columns: {', '.join(sorted(required))}")
        for row_number, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("extra or missing columns")
                symbol = row["symbol"].strip().upper()
                shares = float(row["shares"])
                basis_total = float(row["cost_basis_total"])
                acquired = date.fromisoformat(row["acquired_date"])
                if symbol not in lots or not math.isfinite(shares) or shares <= 0:
                    raise ValueError("symbol or shares invalid")
                if not math.isfinite(basis_total) or basis_total <= 0 or acquired > asof:
                    raise ValueError("cost basis or acquisition date invalid")
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid lots CSV row {row_number}: {exc}") from exc
            lots[symbol].append(Lot(pd.Timestamp(acquired), shares, basis_total / shares))
    return lots


def read_prices(path: Path, asof: date) -> tuple[dict[str, float], dict[str, str]]:
    """Read dated unadjusted closes from an override CSV."""
    prices: dict[str, float] = {}
    dates: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"symbol", "close", "date"}.issubset(reader.fieldnames):
            raise ValueError("prices CSV needs columns: symbol, close, date")
        for row_number, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("extra or missing columns")
                symbol = row["symbol"].strip().upper()
                close = float(row["close"])
                price_date = date.fromisoformat(row["date"])
                if symbol not in ("SPY", "SSO") or symbol in prices:
                    raise ValueError("unexpected or duplicate symbol")
                if not math.isfinite(close) or close <= 0 or price_date > asof:
                    raise ValueError("close or price date invalid")
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid prices CSV row {row_number}: {exc}") from exc
            prices[symbol], dates[symbol] = close, str(price_date)
    if set(prices) != {"SPY", "SSO"}:
        raise ValueError("prices CSV must contain one SPY and one SSO close")
    return prices, dates


def fetch_prices(asof: date) -> tuple[dict[str, float], dict[str, str]]:
    """Fetch the last available unadjusted closes through the requested date."""
    prices: dict[str, float] = {}
    dates: dict[str, str] = {}
    for symbol in ("SPY", "SSO"):
        frame = yf.Ticker(symbol).history(
            start=str(asof - timedelta(days=10)),
            end=str(asof + timedelta(days=1)),
            auto_adjust=False,
        )
        if frame.empty or "Close" not in frame:
            raise ValueError(f"no yfinance close for {symbol} on or before {asof}")
        latest = frame.loc[frame.index.date <= asof].tail(1)
        if latest.empty:
            raise ValueError(f"no yfinance close for {symbol} on or before {asof}")
        prices[symbol] = float(latest["Close"].iloc[0])
        dates[symbol] = str(latest.index[-1].date())
    return prices, dates


def session_dates(asof: date) -> tuple[bool, date, date]:
    """Find the final available NYSE close and the next session."""
    if asof > date.today():
        raise ValueError(f"future --asof is not allowed: {asof}")
    nyse = mcal.get_calendar("NYSE")
    first = date(asof.year, asof.month, 1)
    last = (pd.Timestamp(first) + pd.offsets.MonthEnd(0)).date()
    month_schedule = nyse.schedule(start_date=str(first), end_date=str(last))
    past = nyse.schedule(start_date=str(asof - timedelta(days=10)), end_date=str(asof))
    if past.empty:
        raise ValueError(f"no NYSE session found on or before {asof}")
    if (
        asof == date.today()
        and past.index[-1].date() == asof
        and past.iloc[-1]["market_close"] > datetime.now(timezone.utc)
    ):
        if past.index[-1] == month_schedule.index[-1]:
            raise ValueError("no action: month-end close not yet final")
        past = past.iloc[:-1]
    effective = past.index[-1].date()
    month_end = effective == month_schedule.index[-1].date()
    future = nyse.schedule(
        start_date=str(effective + timedelta(days=1)),
        end_date=str(effective + timedelta(days=10)),
    )
    if future.empty:
        raise ValueError(f"no NYSE session found after {effective}")
    return month_end, effective, future.index[0].date()


def annual_review_note(asof: date) -> str | None:
    """Remind the owner of the fixed review choices near September 30."""
    if asof.year < 2027 or abs((asof - date(asof.year, 9, 30)).days) > 30:
        return None
    return (
        "Annual review: permitted owner decisions are hold, rebalance to band, "
        "or step down to 100% SPY. This tool does not recommend a step-down."
    )


def main() -> None:
    """Validate inputs, run the pure check, and print text or JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lots", type=Path, required=True, help="CSV: symbol,shares,cost_basis_total,acquired_date"
    )
    parser.add_argument("--cash", type=float, default=0.0, help="available existing cash, dollars")
    parser.add_argument("--new-cash", type=float, default=0.0, help="available new cash, dollars")
    parser.add_argument(
        "--asof", type=date.fromisoformat, default=date.today(), help="check date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--prices", type=Path, help="override CSV: symbol,close,date (unadjusted closes)"
    )
    parser.add_argument(
        "--st-rate", type=float, default=0.24, help="short-term tax fraction (default 0.24)"
    )
    parser.add_argument(
        "--lt-rate", type=float, default=0.15, help="long-term tax fraction (default 0.15)"
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = parser.parse_args()
    try:
        month_end, effective, next_session = session_dates(args.asof)
        lots = read_lots(args.lots, args.asof)
        prices, dates = (
            read_prices(args.prices, args.asof) if args.prices else fetch_prices(effective)
        )
        if dates != {"SPY": str(effective), "SSO": str(effective)}:
            raise ValueError(f"price dates must both equal NYSE session {effective}; got {dates}")
        plan = check_rebalance(
            lots,
            prices,
            pd.Timestamp(effective),
            pd.Timestamp(next_session),
            month_end,
            cash=args.cash,
            new_cash=args.new_cash,
            st_rate=args.st_rate,
            lt_rate=args.lt_rate,
        )
        plan["price_dates"] = dates
        plan["review_reminder"] = annual_review_note(args.asof)
        plan["fund_risk_note"] = (
            "Check SSO expense ratio, fee waiver, and closure notices; no automatic action."
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print(f"A23 advisory | {plan['asof']} | SPY/SSO 70/30")
    print(f"SSO weight: {plan['sso_weight']:.2%} (band 22–38%)")
    print(f"Prices: SPY ${prices['SPY']:.2f} ({dates['SPY']})")
    print(f"        SSO ${prices['SSO']:.2f} ({dates['SSO']})")
    print(f"Status: {plan['status']}")
    print(f"Leftover cash: ${plan['leftover_cash']:,.2f}")
    if plan["status"] == "rebalance proposed":
        print(f"Indicative next-session trades for {plan['next_session']} (fills may differ):")
        for sale in plan["sales"]:
            print(
                f"  SELL {sale['shares']:.6f} {sale['symbol']} (lot {sale['acquired_date']}, "
                f"{sale['term']}); proceeds ${sale['estimated_proceeds']:,.2f}; "
                f"gain/loss ${sale['estimated_gain']:,.2f}; "
                f"bracket tax effect ${sale['estimated_tax']:,.2f} before wash/netting"
            )
        for buy in plan["buys"]:
            print(f"  BUY  {buy['shares']:.6f} {buy['symbol']}; cost ${buy['estimated_cost']:,.2f}")
    for warning in plan["warnings"]:
        print(warning)
    if plan["review_reminder"]:
        print(plan["review_reminder"])
    print(plan["fund_risk_note"])


if __name__ == "__main__":
    main()
