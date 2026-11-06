# Intraday practice journal: build and handoff

**Built and verified offline; switched off. Final capped suite: 1,857 passed.**

The computer can now keep a separate journal of when the fixed SPY rule would
buy and sell, using hypothetical money. It cannot place an order. It has no
installed schedule and has not collected real prospective data.

## Try the simulated demonstration

From this checkout with its existing virtual environment:

```bash
.venv/bin/python scripts/intraday_observe.py --help
.venv/bin/python scripts/intraday_observe.py plan
.venv/bin/python scripts/intraday_observe.py rehearse --work-dir /tmp/my-intraday-demo
```

Choose a new directory for each rehearsal. The last command invents three days
of prices, records the decisions and writes `rehearsal_report.json` under that
directory. Its output is a software demonstration, not a trading result.
A [saved demonstration](_intraday_observer_rehearsal.json) and
[verification record](_intraday_observer_build_verification.json) are retained.

## What was verified

Sonnet built the core recorder, thin CLI and tests in `937e487`. Independent Opus
review identified eight Important defects. Parent reproduced them, fixed them
in `4ab01d1`, and added regression coverage. All 118 focused tests pass, including
clean twenty-session observation, failed-checkpoint halt, missing data, delayed
writes, crashes, altered evidence, calendar changes, costs/tax arithmetic and
real issuer-workbook selection. Repository Ruff check/format passed. A fresh
three-day simulated rehearsal has a valid evidence chain.

The corrected implementation's final capped full suite passed **1,857 tests**,
with 30 integration tests deselected and one existing websockets deprecation
warning, in 393.99 seconds (exit 0). Agy Flash independently checked source/test
hashes, synthetic-report identity/hash, Ruff and whitespace; all passed. It read
the full-suite log before completion, so the parent verified the final count. [Review/fix record](2026-10-25-intraday-observer-code-review.md).

## What it records

The observer keeps its own folder containing the fixed rules/calendar, original
source snapshots, decisions, timing evidence, subsequent corrections and a
hypothetical account journal. Missing decisions stay missing. Uncertain account
values stop new hypothetical positions. A failed month-one operational check
also stops new positions; it cannot disappear from the record.

Late writes require an append-only completion receipt before they can count as
timely. Deleted payloads and changed manifest bindings are detected. Snapshots
are not overwritten with a more favorable revision. The hypothetical account and
fixed trading rule are reported separately when outages cause missed positions.

Its report describes outcomes and evidence quality; it never declares a
profitability pass. Timestamps come from the host computer, not an independent
witness. Hypothetical bar prices and fractional positions are not broker fills.

## Before switching on

Real Alpaca access, delay and calendar provenance must be checked and recorded.
The verified readiness record must match the exact prepared manifest; live-mode
activation refuses an approval label or checksum alone. A readiness record is
operator evidence, not an authentication system or proof against a deliberately
false operator assertion. No readiness file for a real observer was created here.

Review the exact standalone schedule and storage/request limits before activation.
No production cron, paper broker, database or Docker changes are included. Actual
issuer currency and blank-capital-gain semantics still need verification; until
then the SPY comparison remains provisional or incomplete. No source convention
is silently guessed to create a completed comparison.

Full-year runtime and real source outage/retry behavior are not measured by the
small rehearsal. Optional revision commands consume the same daily request budget
as required captures and are not scheduled by this build. See the [CLI guide](../../../docs/cli_reference.md#prospective-intraday-practice-journal)
and [protocol](2026-10-25-intraday-prospective-observation.md).
