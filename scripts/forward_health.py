"""Summarize the last seven days of forward collection without sending messages."""

import argparse
import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars
import psycopg2
from dotenv import load_dotenv

from ggTrader.utils.config import get_db_connection_string


def weekly_report(conn: Any, end_day: dt.date) -> dict[str, Any]:
    """Report daily bar coverage, gaps, absent trading days and newly ended symbols."""
    start = end_day - dt.timedelta(days=6)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT l.day, "
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
    by_day: dict[dt.date, list[int]] = {}
    missing_listings = sorted({day for day, _, _, missing in rows if missing})
    for day, bars, gaps, _ in rows:
        totals = by_day.setdefault(day, [0, 0])
        totals[0] += bars
        totals[1] += gaps
    coverage = {day: bars / (bars + gaps) for day, (bars, gaps) in by_day.items() if bars + gaps}
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
        "shares_loaded": bool(shares_loads),
        "shares_count": int(shares_count),
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
    with psycopg2.connect(url) as conn:
        conn.set_session(readonly=True)
        print(format_message(weekly_report(conn, end)))


if __name__ == "__main__":
    main()
