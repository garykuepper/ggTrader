"""One-off, manual-run tool -- NOT part of the app, NOT imported by anything.

Restates paper_snapshots.positions['MNST'] (and portfolio_value) for
2026-08-11 through today to reflect the corrected (2x) market_value /
unrealized_pl, now that persistent split-state correction (see trader.py's
`_compute_split_corrections`) exists going forward and would otherwise
leave this historical window carrying the broker's phantom (pre-split) P&L.

Per docs/next_steps.md R1's "one-off restatement" ask, following the
2026-08-20 withdrawn-bug-report reconciliation (see MEMORY.md
`project_mnst_split_report_withdrawn_2026-08-20`).

Run once, by hand, against the live DB:

    docker compose exec ggtrader_live python scripts/one_off/restate_mnst_snapshots_20260911.py

(or natively with DB_HOST=localhost -- see AGENTS.md's Docker-research note
-- whichever session has DB credentials for it).

Verification (expect MNST unrealized_pl ~+$86 across the range, not the
~-$900 broker figure):

    docker exec ggtrader_db psql -U <user> -d <db> -c \\
      "SELECT run_date, positions->'MNST' FROM paper_snapshots \\
       WHERE run_date BETWEEN '2026-08-26' AND '2026-09-08' ORDER BY run_date;"

This script is idempotent (see the mv > 1500 guard below) and safe to
re-run, but it is still a direct UPDATE against the live paper_snapshots
table -- do not wire it into any cron, CLI, or import path.
"""

import json
from datetime import date

from sqlalchemy import text

from ggTrader.lab.persist import get_engine
from ggTrader.paper.split_check import corrected_market_value, corrected_unrealized_pl

FACTOR = 2.0  # MNST 2-for-1, ex 2026-08-11
# Widened 2026-09-22: every snapshot outside 08-20..08-25 (the only days the
# lookback-bound correction ran) carries the phantom half value, both before
# the correction shipped and after its lookback expired.
START, END = date(2026, 8, 11), date.today()


def main() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT run_date, positions, portfolio_value FROM paper_snapshots "
                "WHERE run_date BETWEEN :start AND :end ORDER BY run_date"
            ),
            {"start": START, "end": END},
        ).all()

        for run_date, positions, portfolio_value in rows:
            pos = positions if isinstance(positions, dict) else json.loads(positions)
            mnst = pos.get("MNST")
            if not mnst:
                continue
            cost_basis = mnst.get("cost_basis", 0.0)
            mv = mnst.get("market_value", 0.0)
            # Idempotency guard: skip a row already corrected by a prior run
            # of this script (true post-split MV for 20.8041 shares is
            # >$1800; anything already above ~$1500 is not the halved
            # broker figure).
            if mv > 1500:
                print(f"{run_date}: already looks corrected (mv={mv:.2f}), skipping")
                continue
            new_mv = corrected_market_value(mv, FACTOR)
            new_pl = corrected_unrealized_pl(mv, FACTOR, cost_basis)
            pos["MNST"] = {
                **mnst,
                "market_value": new_mv,
                "unrealized_pl": new_pl,
                "unrealized_plpc": (new_pl / cost_basis)
                if cost_basis
                else mnst.get("unrealized_plpc"),
            }
            conn.execute(
                text(
                    "UPDATE paper_snapshots SET positions = :positions, "
                    "portfolio_value = :pv WHERE run_date = :run_date"
                ),
                # NAV is cash + corrected position values (as on 08-20..08-25),
                # so it moves by the same delta as MNST's market value.
                {
                    "positions": json.dumps(pos),
                    "pv": portfolio_value + (new_mv - mv),
                    "run_date": run_date,
                },
            )
            print(f"{run_date}: mv {mv:.2f} -> {new_mv:.2f}, unrealized_pl -> {new_pl:.2f}")
        conn.commit()


if __name__ == "__main__":
    main()
