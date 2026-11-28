"""Summarize the last seven days of forward collection without sending messages."""

import argparse
import datetime as dt
from contextlib import closing
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars
import psycopg2
from dotenv import load_dotenv

from ggTrader.data.forward import common_candidate
from ggTrader.utils.config import get_db_connection_string


def weekly_report(conn: Any, end_day: dt.date) -> dict[str, Any]:
    """Report daily bar coverage, gaps, absent trading days and newly ended symbols."""
    start = end_day - dt.timedelta(days=6)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT l.load_id, l.day, "
            "(SELECT count(*) FROM forward.daily_bars b WHERE b.load_id = l.load_id), "
            "(SELECT count(*) FROM forward.gaps g WHERE g.load_id = l.load_id "
            "AND g.reason <> 'listing_missing'), "
            "(SELECT count(*) FROM forward.gaps g WHERE g.load_id = l.load_id "
            "AND g.reason = 'listing_missing') "
            "FROM forward.loads l WHERE l.day BETWEEN %s AND %s "
            "AND l.source <> 'yahoo-shares' ORDER BY l.day, l.load_id",
            (start, end_day),
        )
        rows = cur.fetchall()
        load_ids = [row[0] for row in rows]
        cur.execute(
            "SELECT load_id, symbol, security_name, is_etf, is_test "
            "FROM forward.listing_snapshots WHERE load_id=ANY(%s)",
            (load_ids,),
        )
        listings = cur.fetchall()
        cur.execute(
            "SELECT load_id, symbol FROM forward.daily_bars WHERE load_id=ANY(%s)", (load_ids,)
        )
        bars_by_load = cur.fetchall()
        cur.execute(
            "SELECT DISTINCT symbol FROM forward.symbol_status "
            "WHERE ended_on BETWEEN %s AND %s ORDER BY symbol",
            (start, end_day),
        )
        ended = [row[0] for row in cur.fetchall()]
        cur.execute(
            "SELECT count(*), COALESCE(sum((SELECT count(*) FROM forward.shares_outstanding s "
            "WHERE s.load_id = l.load_id)), 0) FROM forward.loads l "
            "WHERE l.source = 'yahoo-shares' AND l.day BETWEEN %s AND %s",
            (start, end_day),
        )
        shares_loads, shares_count = cur.fetchone()
        cur.execute(
            "SELECT count(DISTINCT (l.day, g.symbol)) FROM forward.loads l "
            "JOIN forward.gaps g ON g.load_id=l.load_id "
            "WHERE l.source='yahoo-shares' AND l.day BETWEEN %s AND %s "
            "AND g.reason='shares_failed' AND NOT EXISTS ("
            "SELECT 1 FROM forward.shares_outstanding s JOIN forward.loads done "
            "ON done.load_id=s.load_id WHERE done.source='yahoo-shares' "
            "AND done.day=l.day AND done.load_id > l.load_id AND s.symbol=g.symbol) "
            "AND NOT EXISTS (SELECT 1 FROM forward.gaps terminal "
            "JOIN forward.loads resolved ON resolved.load_id=terminal.load_id "
            "WHERE resolved.source='yahoo-shares' AND resolved.day=l.day "
            "AND resolved.load_id > l.load_id "
            "AND terminal.symbol=g.symbol AND terminal.reason='shares_unavailable')",
            (start, end_day),
        )
        shares_pending = cur.fetchone()[0]
    by_day: dict[dt.date, list[int]] = {}
    missing_listings = sorted({day for _, day, _, _, missing in rows if missing})
    load_days = {load: day for load, day, _, _, _ in rows}
    candidate_by_day: dict[dt.date, set[str]] = {}
    bars_by_day: dict[dt.date, set[str]] = {}
    for _load, day, _, gaps, _ in rows:
        totals = by_day.setdefault(day, [0, 0])
        totals[1] += gaps
    for load, symbol, name, is_etf, is_test in listings:
        day = load_days[load]
        if common_candidate(
            {"symbol": symbol, "security_name": name, "is_etf": is_etf, "is_test": is_test}
        ):
            candidate_by_day.setdefault(day, set()).add(symbol)
    for load, symbol in bars_by_load:
        bars_by_day.setdefault(load_days[load], set()).add(symbol)
    coverage = {
        day: len(symbols & bars_by_day.get(day, set())) / len(symbols)
        for day, symbols in candidate_by_day.items()
        if symbols
    }
    expected = [d.date() for d in calendars.get_calendar("NYSE").valid_days(start, end_day)]
    return {
        "start_day": start,
        "end_day": end_day,
        "days": len(by_day),
        "mean_coverage": sum(coverage.values()) / len(coverage) if coverage else None,
        "worst_day": min(coverage, key=coverage.get) if coverage else None,
        "worst_coverage": min(coverage.values()) if coverage else None,
        "gaps": sum(gaps for _, gaps in by_day.values()),
        "ended": ended,
        "missing_days": [d for d in expected if d not in by_day],
        "missing_listings": missing_listings,
        "shares_loaded": bool(shares_count),
        "shares_count": int(shares_count),
        "shares_pending": int(shares_pending),
    }


def format_message(report: dict[str, Any]) -> str:
    """Build a plain-text health message without sending it."""
    lines = [f"Forward collection: {report['start_day']} to {report['end_day']}"]
    if report["mean_coverage"] is None:
        lines.append("No collections with measurable coverage.")
    else:
        lines.append(f"Mean coverage: {report['mean_coverage']:.1%} across {report['days']} days.")
        lines.append(f"Lowest: {report['worst_day']} ({report['worst_coverage']:.1%}).")
    lines.append(f"Missing price rows: {report['gaps']}.")
    lines.append("Missing trading days: " + (", ".join(map(str, report["missing_days"])) or "none"))
    lines.append("Missing listings: " + (", ".join(map(str, report["missing_listings"])) or "none"))
    lines.append(
        f"Shares: loaded ({report['shares_count']} observations)."
        if report["shares_loaded"]
        else "Shares: missing this week."
    )
    if report["shares_pending"]:
        lines.append(f"Shares: {report['shares_pending']} unresolved symbols.")
    lines.append("Newly ended: " + (", ".join(report["ended"]) or "none"))
    if report["worst_coverage"] is not None and report["worst_coverage"] < 0.9:
        lines.append("Alert: at least one day had less than 90% coverage.")
    return "\n".join(lines)


def main() -> None:
    """Read the report from the configured database and print it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-day", type=dt.date.fromisoformat, help="Last report date")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    end = args.end_day or dt.datetime.now(ZoneInfo("America/Los_Angeles")).date()
    url = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
    with closing(psycopg2.connect(url)) as conn:
        conn.set_session(readonly=True)
        print(format_message(weekly_report(conn, end)))


if __name__ == "__main__":
    main()
