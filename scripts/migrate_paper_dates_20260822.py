"""One-time repair for the paper-tables day-shift bug (2026-08-22).

Root cause (see the fix + comment in ``src/ggTrader/paper/trader.py``):
``paper_trades.run_date`` / ``paper_snapshots.run_date`` were stamped from
``signals["as_of"]`` -- the date of the last *completed* daily OHLCV bar the
signal was computed from. Two effects compounded to shift that date a full
session-plus-a-day early relative to the actual ET calendar date of the run:

1. By design, ``as_of`` trails the run by one trading session (a run at
   12:45 PT / 15:45 ET sees only the prior session's completed bar).
2. Until 2026-08-21 (commit ``ef4e15f``), ``CachedYFinanceLoader`` wrote
   equity OHLCV bars with a tz-AWARE timestamp into the naive
   ``timestamp WITHOUT time zone`` column, so Postgres rebased the offset
   into the session timezone and stamped every bar one calendar day EARLY.
   That OHLCV bug is already fixed and the ``ohlcv`` table already migrated
   in place -- this script does not touch ``ohlcv``.

Combined, every paper run's persisted ``run_date`` landed one full calendar
day earlier than the prior trading session's real date -- e.g. a Tuesday
run (whose real prior session is Monday) stamped Sunday. The observed
signature: a weekday census of ``paper_snapshots.run_date`` of Sun-Thu with
zero Fridays.

This script shifts the affected date columns +1 day so history reads
correctly. It is NOT a fix for the write path (see trader.py, already
fixed to stamp ``_today_et()`` directly) -- it only repairs rows written
before that fix landed.

Usage::

    python scripts/migrate_paper_dates_20260822.py --dry-run
    python scripts/migrate_paper_dates_20260822.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ggTrader.lab.persist import get_engine

_log = logging.getLogger(__name__)

BACKUP_SNAPSHOTS = "paper_snapshots_preshift_backup_20260822"
BACKUP_TRADES = "paper_trades_preshift_backup_20260822"


def _weekday_census(conn: Connection, table: str, date_col: str) -> dict[str, int]:
    rows = conn.execute(
        text(f"SELECT to_char({date_col}, 'Dy') AS dow, count(*) FROM {table} GROUP BY 1")
    ).all()
    return {dow.strip(): count for dow, count in rows}


def _row_count(conn: Connection, table: str) -> int:
    return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def _already_migrated(census: dict[str, int]) -> bool:
    """The idempotency guard: only run if the corrupted signature is present.

    Corrupted signature: Fridays == 0 and Sundays > 0 (a session-only table
    can never have real Sunday rows -- their presence, with zero Fridays, is
    the exact fingerprint of the day-shift bug). Once migrated, Fridays > 0
    and Sundays == 0, and a second run refuses to proceed."""
    return not (census.get("Fri", 0) == 0 and census.get("Sun", 0) > 0)


def _check_migratable(conn: Connection) -> None:
    """Raise if either table's weekday census no longer matches the bug
    signature -- i.e. the migration already ran (or the corruption was never
    present)."""
    for table in ("paper_snapshots", "paper_trades"):
        census = _weekday_census(conn, table, "run_date")
        if _already_migrated(census):
            raise RuntimeError(
                f"{table}.run_date census {census} does not show the day-shift "
                "signature (Fridays == 0 and Sundays > 0). Refusing to run -- "
                "either already migrated, or there is nothing to fix."
            )


def run_migration(dry_run: bool) -> None:
    engine = get_engine()
    with engine.connect() as conn:
        _check_migratable(conn)

        before_snap_census = _weekday_census(conn, "paper_snapshots", "run_date")
        before_trade_census = _weekday_census(conn, "paper_trades", "run_date")
        before_snap_count = _row_count(conn, "paper_snapshots")
        before_trade_count = _row_count(conn, "paper_trades")

        _log.info("paper_snapshots.run_date census (before): %s", before_snap_census)
        _log.info("paper_trades.run_date census (before): %s", before_trade_census)
        _log.info(
            "paper_snapshots rows: %d, paper_trades rows: %d", before_snap_count, before_trade_count
        )

        if dry_run:
            print("[dry-run] Would create backup tables:")
            print(f"[dry-run]   {BACKUP_SNAPSHOTS} <- paper_snapshots ({before_snap_count} rows)")
            print(f"[dry-run]   {BACKUP_TRADES} <- paper_trades ({before_trade_count} rows)")
            print("[dry-run] Would shift paper_snapshots.run_date += 1 day")
            print("[dry-run] Would shift paper_trades.run_date += 1 day")
            print(f"[dry-run] paper_snapshots.run_date census (before): {before_snap_census}")
            print(f"[dry-run] paper_trades.run_date census (before): {before_trade_census}")
            return

    # Backup + shift in one transaction: either both tables end up backed up
    # and shifted, or neither does.
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {BACKUP_SNAPSHOTS} AS TABLE paper_snapshots"))
        conn.execute(text(f"CREATE TABLE {BACKUP_TRADES} AS TABLE paper_trades"))

        # paper_snapshots.run_date is the PRIMARY KEY, and Postgres checks a
        # non-deferrable PK per row as the UPDATE executes rather than at
        # statement end. Shifting the whole table with one `run_date =
        # run_date + 1` statement can therefore collide mid-flight: a row
        # shifted to X+1 can momentarily match another row still sitting at
        # its un-shifted X+1. The final mapping is injective (no two rows
        # ever really collide), so shifting in descending run_date order
        # side-steps it -- each row's new value (old+1) is always greater
        # than every not-yet-shifted row's (smaller, still-old) value, and
        # always greater than every already-shifted row's new value (whose
        # original was itself smaller). paper_trades has no such constraint
        # on run_date (its PK is the `id` serial column), so a single
        # statement is safe there.
        dates = [
            row[0]
            for row in conn.execute(
                text("SELECT run_date FROM paper_snapshots ORDER BY run_date DESC")
            ).all()
        ]
        for d in dates:
            conn.execute(
                text(
                    "UPDATE paper_snapshots SET run_date = run_date + INTERVAL '1 day' "
                    "WHERE run_date = :d"
                ),
                {"d": d},
            )
        conn.execute(text("UPDATE paper_trades SET run_date = run_date + INTERVAL '1 day'"))

        after_snap_census = _weekday_census(conn, "paper_snapshots", "run_date")
        after_trade_census = _weekday_census(conn, "paper_trades", "run_date")
        after_snap_count = _row_count(conn, "paper_snapshots")
        after_trade_count = _row_count(conn, "paper_trades")

    print(f"Backed up paper_snapshots -> {BACKUP_SNAPSHOTS} ({before_snap_count} rows)")
    print(f"Backed up paper_trades -> {BACKUP_TRADES} ({before_trade_count} rows)")
    print("Shifted paper_snapshots.run_date and paper_trades.run_date by +1 day.")
    print()
    print(f"paper_snapshots.run_date census (before): {before_snap_census}")
    print(f"paper_snapshots.run_date census (after):  {after_snap_census}")
    print(f"paper_snapshots rows (before/after): {before_snap_count} / {after_snap_count}")
    print()
    print(f"paper_trades.run_date census (before): {before_trade_census}")
    print(f"paper_trades.run_date census (after):  {after_trade_census}")
    print(f"paper_trades rows (before/after): {before_trade_count} / {after_trade_count}")

    if after_snap_count != before_snap_count or after_trade_count != before_trade_count:
        raise RuntimeError("Row count changed across the shift -- this should be impossible.")
    if after_snap_census.get("Sun", 0) or after_trade_census.get("Sun", 0):
        _log.warning("Sunday rows remain after the shift -- investigate before trusting this run.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the paper-tables day-shift: back up paper_snapshots and "
            "paper_trades, then shift run_date +1 day. Idempotent -- refuses "
            "to run a second time (see _already_migrated)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to the database.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        run_migration(dry_run=args.dry_run)
    except RuntimeError as exc:
        _log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
