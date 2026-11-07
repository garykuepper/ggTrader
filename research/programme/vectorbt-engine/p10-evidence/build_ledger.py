"""P10: build the deletion ledger (path, lines, importers found, decision, evidence) BEFORE deleting.

Runs ``import_graph.py`` over the candidate files and writes ``P10-ledger.csv``. Decisions are
written here, not inferred: ``delete`` only when no kept file imports the module and no research
report cites it; ``archive`` when a research report cites it (the file moves to
``legacy-src/`` or ``legacy-scripts/``); ``keep`` when a kept file still needs it.

Usage: python build_ledger.py <repo-root> <out.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: path -> (decision, reason). Importers and citations are measured, not listed here.
CANDIDATES: dict[str, tuple[str, str]] = {
    "src/ggTrader/data/historical/postgres_ingestor.py": (
        "delete",
        "dead: no importer in src, scripts or tests; only `ggt ingest` (itself a stub) named it",
    ),
    "src/ggTrader/cli/cmd_ingest.py": (
        "delete",
        "documented non-functional stub; its only importer is the `ingest` registration in cli/main.py, removed with it",
    ),
    "src/ggTrader/lab/regime.py": (
        "archive",
        "dead in production (only tests/lab/test_regime.py); cited by research reports, so archived",
    ),
    "src/ggTrader/lab/train_gate.py": (
        "archive",
        "trainer of the falsified ML gate (only its own test imports it); cited by reports",
    ),
    "src/ggTrader/paper/feature_gate.py": (
        "archive",
        "falsified ML gate, off by default; the live signal path no longer imports it; cited by reports",
    ),
    "src/ggTrader/lab/headline_sentiment_data.py": (
        "archive",
        "loader for headline_sentiment, a strategy that now exists only at the legacy tag; table untouched",
    ),
    "src/ggTrader/lab/short_interest_data.py": (
        "archive",
        "loader for short_interest (legacy tag only); table untouched",
    ),
    "src/ggTrader/lab/short_volume_data.py": (
        "archive",
        "loader for short_volume_ratio (legacy tag only); table untouched",
    ),
    "src/ggTrader/lab/house_ptr_data.py": (
        "archive",
        "loader for congress_trades (legacy tag only); table untouched",
    ),
    "src/ggTrader/lab/fomc_calendar.py": (
        "archive",
        "loader for fomc_drift (legacy tag only); no table",
    ),
    "scripts/headline_sentiment_backfill.py": ("archive", "backfill of the archived headline loader"),
    "scripts/short_interest_backfill.py": ("archive", "backfill of the archived short-interest loader"),
    "scripts/short_volume_backfill.py": ("archive", "backfill of the archived short-volume loader"),
    "scripts/house_ptr_backfill.py": ("archive", "backfill of the archived House PTR loader"),
    "scripts/fomc_calendar_backfill.py": ("archive", "backfill of the archived FOMC calendar loader"),
    "tests/lab/test_regime.py": ("archive", "tests only the archived lab/regime.py"),
    "tests/lab/test_train_gate.py": ("archive", "tests only the archived lab/train_gate.py"),
    "tests/paper/test_feature_gate.py": ("archive", "tests only the archived paper/feature_gate.py"),
    "tests/lab/test_ml_screen.py": ("archive", "tests only feature_gate.extract_features"),
    "tests/lab/test_headline_sentiment_data.py": ("archive", "tests only the archived loader"),
    "tests/lab/test_short_interest_data.py": ("archive", "tests only the archived loader"),
    "tests/lab/test_short_volume_data.py": ("archive", "tests only the archived loader"),
    "tests/lab/test_house_ptr_data.py": ("archive", "tests only the archived loader"),
    "tests/lab/test_fomc_calendar.py": ("archive", "tests only the archived loader"),
    # Kept on purpose (importers or readers exist).
    "src/ggTrader/lab/form4_data.py": (
        "keep",
        "its table form4_transactions is read by lab/event_timing_audit.py (owner-undecided event-timing tooling)",
    ),
    "src/ggTrader/lab/earnings_surprise_data.py": (
        "keep",
        "its table earnings_surprise is read by lab/event_timing_audit.py (owner-undecided event-timing tooling)",
    ),
    "src/ggTrader/lab/fred_data.py": ("keep", "used by static-leverage, century-leverage and replica scripts"),
    "src/ggTrader/lab/google_trends_data.py": ("keep", "imported by the ported strategy retail_attention"),
    "src/ggTrader/lab/data.py": ("keep", "eligible_at is used by the replica audit tools and as the PIT oracle"),
}


def main(argv: list[str] | None = None) -> int:
    """Write the ledger CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    proc = subprocess.run(
        [sys.executable, str(HERE / "import_graph.py"), str(args.root), *CANDIDATES],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        info = json.loads(line)
        decision, reason = CANDIDATES[info["path"]]
        imp = info["importers"]
        found = [f for kind in ("src", "scripts", "tests") for f in imp[kind]]
        reports = [
            c for c in info["cited_in"] if c.startswith("research/") and c.endswith(".md")
            and not Path(c).name.startswith("P1") and "/P5c" not in c and "/P9" not in c
        ]
        rows.append(
            {
                "path": info["path"],
                "lines": info["lines"],
                "importers_found": "; ".join(found) or "none",
                "decision": decision,
                "evidence": f"{reason}. kept importers: src {len(imp['src'])}, scripts "
                f"{len(imp['scripts'])}, tests {len(imp['tests'])}. research reports citing the "
                f"file name: {len(reports)}"
                + (f" (e.g. {reports[0]})" if reports else ""),
            }
        )
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
