# Prospective Intraday Observation Implementation Plan

> **For agentic workers:** After protocol approval, use superpowers:subagent-driven-development or superpowers:executing-plans task by task. This document is a proposed plan, not authorization to start collection, schedules or a shadow ledger.

**Goal:** Build a reliable forward SPY archive and a separately accounted hypothetical record without orders or production changes.

**Architecture:** An isolated recorder saves originally observed vendor data and availability metadata. A deterministic hypothetical ledger consumes only frozen inputs and records decisions before their modeled entry. Corrections append evidence; they never overwrite the original observation.

**Tech stack:** Existing Python environment, requests, pandas, existing calendar and safe GET/checkpoint helpers. No new service, database or dependency is proposed.

**Spec:** Task 1 produces the candidate-specific prospective brief. Historical context: [r8](../../research/briefs/2026-10-25-intraday-feasibility-r8.md), [completed gap investigation](../../research/2026-10-25-intraday-gap-investigation.md), [existing forward standard](../../research/briefs/2026-10-23-forward-paper-standard.md).

## Status and scope

Owner requested this plan after accepting that the historical gaps cannot be repaired.
Frozen r8 remains UNRESOLVED (data); there is no historical strategy GO or NO-GO.
Prospective observation is a new research design. It does not replace the missing
long-history test, satisfy the existing stock-candidate standard, or authorize capital.

Recommended approach: delayed SIP REST snapshots plus a hypothetical ledger.
A real-time stream adds operational complexity without improving availability of
the morning signal by the afternoon decision. Broker paper orders would introduce
execution and account risks outside the current research scope. The delayed-source
capability and timestamp semantics must be verified before adopting this design.

## Global constraints

- SPY only, long-only, no borrowing; free data; taxable-account context.
- No changes to `src/ggTrader/paper/`, its cron, Docker, strategy registry or DB.
- No broker orders, support messages, new accounts, pushes or deployment in this plan.
- Preserve all r8 artifacts, hashes and unresolved findings.
- Retain every scheduled session, including flat days and evidence faults.
- No imputation, neighboring-minute replacement, retrospective decisions or favorable restarts.
- Hypothetical bar prices and fractional sizing are research proxies, not fill evidence.
- All tests use `scripts/run_tests.sh`; retain its memory cap and use mocked network/clock.
- Parent handles credentials and deployment. Children get bounded inputs and named output ownership.

## Proposed milestones and time commitment

1. Freeze a prospective brief and review it before the first observation.
2. Implement and verify the recorder and hypothetical ledger offline.
3. Activate a separate observer only after readiness review and owner authorization.
4. Review operations after the first **20 registered observation slots**. No profitability verdict.
5. Publish a descriptive report after **252 registered observation slots**, including every
   fault and elapsed day. Roughly one trading year, not a significance guarantee.

The 20-session check is a checkpoint within the 252-slot period, not a restart.
The ledger is frozen before session 1. Performance is not used to tune or extend
the study. A longer confirmatory test needs its own sample-size and monitoring
design before it begins; 252 observations cannot be advertised as replacing r8.

## Task 1: Freeze the prospective protocol

**Owner:** Parent drafts; Claude Opus reviews statistical and protocol decisions.
**File:** Create `docs/research/briefs/2026-10-25-intraday-prospective-observation.md`.

- [ ] Register the same single signal: 09:59 close divided by the immediately
  preceding official-session closing minute, minus one; long only if positive.
  Half-days are not trade sessions, but their closing minutes anchor the next session.
- [ ] Propose a 15:25 ET decision, modeled entry at 15:30 minute open and exit at
  15:59 minute close. Fetch inputs before decision, archive the decision before entry,
  and value later. Source latency must never be confused with execution latency.
- [ ] Freeze no-trade/fault behavior: missing or late decision inputs mean no new
  hypothetical position, record an operational fault, retain the session. Missing
  entry/exit valuations mean unresolved P&L, never zero. No valid aggregate return
  or performance verdict while any required valuation remains unresolved.
- [ ] Preserve r8 cost cases: 1 bp/side base, 2 bp/side stress, zero idle-cash yield,
  idealized fractional full-equity sizing. Preserve its 24% annual-positive-P&L
  tax-policy proxy, with terminal accrual; label it simplified, not personal tax accounting.
- [ ] Register the complete issuer distribution schedule for the SPY diagnostic,
  including how future issuer publications are captured and corrections disclosed.
  Define missing currency/capital-gain fields and decimal rounding before use.
- [ ] Freeze the exact first and last calendar sessions and bootstrap anchor before
  activation. Count 252 registered observation slots even when operational records are faulty.
  A late start requires a new future registration before observations, not backfilling decisions.
- [ ] Define monthly reports as descriptive only; no interim GO, parameter change or
  extension based on outcomes. Preserve safety halts and show unresolved observations.
- [ ] State that any displayed NW-5 statistic or >=3% after-tax-policy CAGR is
  descriptive in this observation phase, not a passing research verdict. Include
  sample-size/power arithmetic before day 1; assumptions must not be sold as evidence.
- [ ] Freeze decision timestamp evidence: local hashes detect alteration but do not
  independently prove time. Choose and authorize an independent receipt mechanism
  before claiming witnessed prospective decisions. Without one, label decisions
  locally timestamped and prohibit any stronger timing claim. No automatic publication.
- [ ] Review the 15:45 ET production SPY sweep collision explicitly: isolation is a
  file-based ledger, not positions in the existing paper account. Do not alter the sweep.
- [ ] Parent verifies Opus's objections; obtain approval of the concrete brief before building.

## Task 2: Implement an isolated recorder and ledger offline

**Owner:** Claude Sonnet implements; parent reviews and verifies.
**Proposed files:** `src/ggTrader/lab/intraday_observer.py`,
`scripts/intraday_observe.py`, `tests/lab/test_intraday_observer.py`.
Keep core logic in `src/`; do not refactor or change the hash-bound r8 modules.

- [ ] Reuse safe GET deadlines, timestamp parsing, calendar validation and atomic
  checkpoints from `intraday_probe.py` where compatible; do not call r8's scorer.
- [ ] Expose a thin CLI with `--help`, offline `plan`, input capture, decision,
  post-close capture and report commands. Credentials come only from inherited env.
- [ ] Preserve source parameters, UTC requested/retrieved times, raw payload or a
  lossless snapshot, byte hashes, protocol/code/runtime hashes and session identifiers.
  Maintain separate original and subsequently corrected source records.
- [ ] Require an exclusive session-stage claim before writes; reruns reconcile or
  refuse duplicates rather than replacing decisions or generating a second trade.
- [ ] Write red-first tests for early-close anchors, DST, late/missing inputs,
  duplicates, stale credentials, delayed availability, revised bars, restart/crash,
  calendar mismatch, flat decisions and missing entry/exit valuation. Network tests
  use fixtures; no test reads real `.env` or production persistence.
- [ ] Test that decision generation cannot access entry/exit data and that retrospective
  input availability cannot become a timely decision. Test no network on help/plan.
- [ ] Run focused capped tests, then the full capped suite and repository Ruff gates
  after final code changes. Add observer documentation to CLI/architecture/README
  references and changelog. Do not install or schedule it as part of this task.

## Task 3: Verify readiness and activate deliberately

**Owner:** Parent performs source/capability preflight; Claude Opus reviews readiness.
**Output:** Dated readiness JSON/report and a proposed standalone timer definition.

- [ ] Verify free delayed SIP entitlement, timestamp/availability semantics and the
  prior-session anchor with bounded GETs. Fail unresolved rather than switch to IEX.
- [ ] Rehearse at least three synthetic sessions including a half-day transition
  and an outage/restart. Check exactly one immutable decision per scheduled session.
- [ ] Propose captures at 15:15 ET conditional prior-archive recovery, inputs at 15:20 ET, decision at 15:25 ET, and full-session archive at
  16:20 ET; validate these times against actual source delay. Scheduling must follow
  America/New_York and the exchange calendar, independent of host timezone.
- [ ] Require session-stage prerequisites; a service restart cannot replay a missed
  decision as timely. Record missed work as a fault.
- [ ] Present exact paths, source calls/day, retry/runtime/storage bounds, receipt
  behavior, process permissions, timer contents and start/end dates for activation.
  Do not change existing cron or deploy to the production container.
- [ ] Activate only after owner authorization. Preserve that record and the frozen
  identities. No broker paper order capability is included.

## Task 4: Observe and report without tuning

**Owner:** Parent; agy Flash only for extraction/checking of specified records.

- [ ] At 20 registered observation slots, report timely decisions, faults, missing
  valuations, source revisions, duplicate prevention and local timestamp coverage.
  Proposed operational acceptance: all decisions timely and all endpoint valuations
  present, no unresolved faults. Failure halts new hypothetical trades pending review;
  data recording can continue, preserving dates. No restart erases failed sessions.
- [ ] Monthly, publish session counts, costs and hypothetical outcomes with evidence
  limitations. No interim performance pass or rule adjustment.
- [ ] At slot 252, publish immutable JSON plus a report: gross/net/tax-policy
  returns, SPY total-return diagnostic, drawdown, signal counts, uncertainty and
  all faults. If required inputs remain missing, report unresolved instead.
- [ ] End the observation phase as registered. Recommend a separately registered
  confirmatory study or other research based on the full record, with no automatic
  extension, capital allocation or equivalence to production broker paper.

## Review focus

- Delayed/revised data must not become information claimed available before entry.
- Missing observations must not disappear from the calendar or become zero P&L.
- Session overlap and process restarts must not create duplicate decisions.
- Short samples and repeated monthly reporting must not be labeled statistical proof.
- Hypothetical accounting and timestamps must not be described as broker fills or witnessed evidence without receipts.

## Execution boundary

This task creates the plan only. No agents are dispatched, code written, data
collected, schedules installed or hypothetical trades recorded by creating it.
Previous model preferences remain: Opus for hard review, Sonnet for bounded
implementation, Flash for mechanically verifiable extraction, parent for integration.


## Task 1 review rulings (2026-10-25)

Opus found six ambiguities; parent revised the concrete brief. Its explicit rules
supersede earlier proposed shorthand in this plan: activation-relative first date;
252 frozen slots with no extension for closures; canonical prior-session anchor;
valuation grace strictly before the next official session's 15:20; separate
one-session versus halting faults; owner-approved prospective resume; no displayed
performance threshold comparisons. Local publication means repository files only.
No independent timestamp service, 10:20 input stage, automatic revision re-fetch
or whole-window revision collection is added. These changes remain proposed for
owner protocol approval before Task 2 implementation.


## Build execution status

Owner said "ok build it" after the plain-language explanation. Sonnet implemented
in937e487; parent reproduced/corrected the eight Important review findings in
4ab01d1. Focused tests118passed and offline rehearsal completed. Final full suite
passed1,857tests (30deselected); see the [build record](../../research/2026-10-25-intraday-observer-build.md)
for final evidence. Task3 real-source preflight/activation and Task4 prospective
observation remain unperformed. No timer or broker integration was installed.
