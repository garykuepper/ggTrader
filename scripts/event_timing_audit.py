#!/usr/bin/env python3
"""CLI script for the Event-Timing Audit Tool.

Implements the owner-approved event-timing audit for PEAD and Insider Cluster strategies
per docs/research/briefs/2026-10-21-event-timing-audit.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ggTrader.lab.event_timing_audit import EventTimingAuditRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Event-Timing Audit Tool for PEAD and Insider Cluster Strategies"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (<= 2 items per stratum, <= 50 SEC requests, output to /tmp/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume prior run using persisted checkpoints and response caches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Override default output JSON path (default: /tmp/event_timing_audit_smoke.json "
            "for smoke, docs/research/_event_timing_audit.json for full)"
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    runner = EventTimingAuditRunner(
        smoke=args.smoke,
        resume=args.resume,
        output_file=args.output,
    )
    results = runner.run()
    print(f"\nAudit complete. Artifact written to: {runner.output_file}")
    print(f"PEAD Final Status: {results['pead']['status']}")
    print(f"Insider Cluster Final Status: {results['insider_cluster']['status']}\n")


if __name__ == "__main__":
    main()
