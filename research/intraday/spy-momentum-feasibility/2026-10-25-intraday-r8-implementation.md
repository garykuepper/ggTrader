# Intraday r8 mocked implementation and execution preflight

**Status: mocked implementation complete; reviews and checks passed; approval 2 pending.**
No candidate price/return analysis or actual r8 collection was performed. This is infrastructure
verification, not a strategy GO/NO-GO. The owner approved protocol finalization
and mocked implementation by saying "Continue" after the r8 memo's approval
question. The [approved protocol](briefs/2026-10-25-intraday-feasibility-r8.md)
remains unchanged; the original Stage 0A verdict remains unresolved.

## What was built

The isolated modules `lab/intraday_momentum.py` and
`lab/intraday_momentum_data.py`, invoked by
`scripts/intraday_momentum_research.py`, implement a bounded resumable collector,
a signal-blind validator and a single-shot scorer. They do not integrate with
the strategy registry, vectorbt, DB or production paper trader. Statsmodels
0.15.0 was recorded as a dependency before installation; existing packages were
not upgraded.

The validator requires four valid endpoint prices for every scored full session
and the immediately preceding session's correct closing-minute anchor, including
half-days. Missing required prices stop the study instead of deleting or zeroing
trades. Calendar dates/hours are checked against an independent offline NYSE
schedule. Scoring rehashes the actual stored chunks, protocol, code, runtime and
manual vendor-review evidence before claiming the collection directory.

The scorer implements the fixed long-only rule, cost cases, corrected NW-5
statistic, annual positive-P&L tax debits and terminal tax. It retains flat days,
reports the registered diagnostics and reinvested-distribution SPY comparison,
and cannot turn a stress case or diagnostic into a performance pass. The bar
prices and fractional sizing remain an idealized proxy, not executable fills.

## Verification and review record

Opus implemented the code; Flash checked the protocol fields, hash and offline
CLI inventory. The parent independently verified the protocol hash, CLI behavior,
constant-series statistical refusal and the new dependency/runtime identities.
The [independent task review](2026-10-25-intraday-r8-task-review.md) passed the
core specification and quality while identifying readiness conditions.

The implementation was corrected for zero-variance statistics, alternate-output
rescoring, jointly missing calendar/data dates, corporate-action schemas and
stop reasons, same-ticker name changes, cash capital-gains payouts, runtime
binding and corrupt chunks. An active documented vendor-adjustment review is
now hash-bound and required before collection or scoring. The
[bounded documentation record](_intraday_r8_vendor_adjustment_review_2026-10-25.json)
is tracked with the code, without editing the frozen protocol.

Before the final numerical fix, the two new test files passed 276 focused
cases (314 including Stage 0A). The final fix's owned-file run passed 235 cases.
The complete suite on corrected code `6e89281` passed **1,739 tests**, with
30 integration tests deselected and one existing `websockets.legacy` warning,
in 334.53s. Repository Ruff check and format-check passed for 277 files.

Opus's task and scoped fix reviews passed. Its final branch attempt failed an
API rate limit and produced no verdict. The fallback Codex final review found a
mandatory-benchmark overflow that could retain GO with null metrics. The focused
fix and [final scoped review](2026-10-25-intraday-r8-final-fix-review.md) closed that
finding without changing thresholds. Failed mandatory numerical calculations now
produce named errors and an unresolved diagnostic verdict.

Initial implementation was not strictly test-first; subsequent fixes had failing
regression tests and mutation checks. This process deviation is retained in the
record. All verification used synthetic data, not candidate returns. The prepared
manifest records the exact tested code, runtime and verification summary.

## Concrete proposed approval 2 scope

The [prepared manifest](_intraday_r8_execution_preflight_2026-10-25.json) records
all 132 logical requests and the exact protocol, code, runtime and vendor-review
identities. The manifest is prepared offline, matches the corrected code, and is **not yet
approved for execution**.
It requests SPY only:

- One paper-calendar request for 2016-01-04 through 2026-09-30.
- One raw SIP daily-bar request for the registered window.
- One all-types corporate-action request, filtered by process dates from
  2015-12-04 through 2026-10-31; records are placed by their documented effective
  or ex-date, not by substituting process dates.
- 129 raw SIP minute-bar chunks for January 2016 through September 2026,
  including extended hours; only the registered session endpoints determine
  the score. No IEX or auction request is included.

The proposed execution uses one directory,
`data/external/intraday_momentum_r8`. Complete chunks are immutable and resumable;
incomplete pages never count as successful chunks. Requests have a 30-second
total deadline, at least 0.65 seconds between starts, at most three attempts per
page and a bounded run. Credentials are inherited and never logged; redirects,
ambient proxy configuration and dotenv loading are disabled.

After collection, run one signal-blind validation. Only a matching PASS permits
one automatic score into `docs/research/_intraday_momentum_results.json` and its
Markdown report. Otherwise record UNRESOLVED (data), without changing the window,
price convention, missingness policy or gates. No shadow, order, production
change, live allocation or discretionary re-score is part of approval 2.

## Limits that the proposed approval must acknowledge

- Alpaca does not document complete historical corporate-action coverage or a
  processing-delay bound. The 31-day margin and quarterly-dividend sanity check
  do not certify absence of splits or other events.
- The manual vendor review is a bounded documentation check. Undocumented changes,
  omitted payout legs and later vendor revisions cannot be ruled out.
- Missing or empty cash-payout currency is an explicit USD assumption for this
  USD-SPY benchmark and is counted; non-USD data is unresolved, with no conversion.
- Page-body hashes are fingerprints; raw response bodies are not retained. The
  binding for analysis is the hash of the stored whitelisted chunk bytes.
- A claim is per named collection directory. A crash after claiming prevents a
  retry, and an intentionally copied directory is outside the single approved run.
- Runtime changes, even patch releases, invalidate resume/scoring identities.
  Pin this environment for the run. Actual source access, full-window endpoint
  coverage and candidate performance remain unverified.

## Implementation decisions recorded during review

- Runtime versions live in a separate r8 execution appendix, keeping the approved
  protocol hash stable. A stale runtime requires refreshed evidence before a run.
- The manifest includes distributions, split metadata and raw daily closes needed
  by mandatory diagnostics. Missing ancillary coverage stays unresolved rather
  than producing a partial comparison.
- Official calendar dates/hours are checked against an independent offline NYSE
  schedule. A disagreement remains unresolved for review; dates are not changed.
- A confirmed same-ticker name-only record is metadata rather than a valuation
  stop. Identity changes, splits and unvalued material actions retain explicit
  failure reasons; misclassification could invalidate the benchmark.
- A hash-bound manual review enforces documented adjustment stops. Its bounded
  search cannot exclude undocumented changes; a newly identified applicable
  unresolved change stops the run.

These decisions implement the frozen specification and do not change the
performance gates. The preceding limitations remain part of the proposed run.
