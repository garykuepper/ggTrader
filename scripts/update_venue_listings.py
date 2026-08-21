"""BROKEN -- ImportError on load; this script does not run.

It imports ``ggTrader.data.core.venue_listings``, deleted in 82931a4
("remove legacy trading/backtesting code"). ``data/core/`` now holds only
base_loader, constants, index_constituents, stock_constants. It belongs to
the crypto venue-registry pipeline, parked by choice (see docs/roadmap.md).
Reviving it means restoring venue_listings.py; otherwise delete this file.

CLI: write per-venue availability snapshots (Layer 1 of the universe pipeline).

    python scripts/update_venue_listings.py --venue all
    python scripts/update_venue_listings.py --venue binanceus

Writes data/universe/{venue}_listings.json. See
docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md.
"""

from __future__ import annotations

import argparse
import os
import sys

from ggTrader.data.core.venue_listings import (
    DEFAULT_LISTINGS_DIR,
    SUPPORTED_VENUES,
    write_venue_listings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-venue availability snapshots.")
    parser.add_argument(
        "--venue",
        type=str,
        default=(os.getenv("EXCHANGE") or "kraken").lower(),
        choices=[*sorted(SUPPORTED_VENUES), "all"],
        help="Venue to snapshot, or 'all' (default: $EXCHANGE env, else 'kraken').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_LISTINGS_DIR,
        help=f"Output directory (default: {DEFAULT_LISTINGS_DIR}).",
    )
    args = parser.parse_args()

    venues = sorted(SUPPORTED_VENUES) if args.venue == "all" else [args.venue]
    failures = []
    for venue in venues:
        print(f"Fetching live listings for {venue}...")
        try:
            out_path = write_venue_listings(venue, listings_dir=args.out_dir)
            print(f"  Wrote {out_path}")
        except Exception as e:
            # One venue failing (e.g. network) shouldn't block the others; the
            # existing snapshot for that venue is left untouched. Report at the end.
            failures.append(venue)
            print(f"  FAILED {venue}: {type(e).__name__}: {e}")

    if failures:
        print(f"\nCompleted with failures: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
