# Intraday Practice Journal Readiness and Activation Plan

> **For agentic workers:** Execute task by task after authorization. The owner requested this plan only. Do not make source requests, install services or activate observations merely because this document exists.

**Goal:** Check that the built practice journal can receive real data reliably, prepare its daily schedule, and present a concrete switch-on package.

**Architecture:** Keep the observer independent of production. Verify the real feed with bounded read-only requests, preserve evidence, and prepare a fixed local release plus a separate file-based journal and standalone schedule. Activate only after the owner sees the tested package.

**Tech stack:** Existing Python environment and observer CLI; proposed user-level systemd timers. No new database, container, subscription or broker order integration.

**Spec:** [Prospective observation protocol](../../research/briefs/2026-10-25-intraday-prospective-observation.md).
**Baseline:** [Completed build](../../research/2026-10-25-intraday-observer-build.md), code `4ab01d1`, documented in `8ea98fc`; 1,857 tests passed at that build, 118 observer tests, clean Ruff and a synthetic rehearsal. Recheck repository status before execution; these are recorded results, not a fresh test claim.

## The plan in everyday language

1. Check that Alpaca actually gives us the prices and trading dates we need.
2. Check that the recorder still behaves correctly with a year's worth of records.
3. Prepare the separate daily jobs, folders and a simple local status report.
4. Show the owner exactly what will run and when. Leave it off until approved.
5. After approval, start recording hypothetical trades and check the first day.
6. After about a month, assess recording reliability; after about a year, report observations without declaring the strategy proven.

## Global constraints

- Free data, SPY only, long-only hypothetical positions; no orders or account changes.
- Preserve historical r8 as UNRESOLVED (data). No repair, re-score or date selection based on outcomes.
- No changes to production paper code, account positions, existing cron, Docker or database.
- Parent alone handles existing credentials; never print, copy or edit `.env`.
- Planning does not authorize authenticated requests or scheduling. Execution of Tasks 1–3 prepares readiness; Task 4 explicitly reserves activation approval.
- Bound the proposed readiness investigation to **20 direct HTTP GET attempts total**, including pages, retries, redirects and document downloads. Use an explicitly counted client with automatic redirects/retries off. Stop unresolved when the budget or required evidence is exhausted.
- Readiness calls are separate from the observer's frozen **12 GET attempts per New York date** once running. Do not silently increase either bound.
- Any code change must precede the final release/protocol/runtime/manifest hashes. No edits to an initialized study to make its bindings match.
- Tests use `scripts/run_tests.sh` with its unchanged memory cap; network fixtures never read real credentials.
- Local timestamps and hypothetical bar fills remain limited evidence. No profitability pass, capital route or independent timestamp claim.

## Task 1: Verify the real source without starting a study

**Owner:** Parent. **Flash:** extract exact fields from specified saved responses or issuer documents; parent verifies them.
**Outputs:** `docs/research/2026-10-26-intraday-observer-readiness.md` and `_intraday_observer_readiness_preflight.json`; separate original source snapshots under a dated readiness artifact folder, outside r8 and outside a live study ledger.

- [ ] Record source URLs, request count, exact parameters, UTC request/receipt times, status codes, payload hashes, code/runtime identities and elapsed times. Retain source payloads without headers or credentials.
- [ ] Check the official paper-calendar GET against the offline NYSE schedule for the proposed observation span, including early closes and the bootstrap anchor. A mismatch blocks readiness; do not choose the more convenient calendar.
- [ ] Check raw SIP one-minute bars for two completed sessions and the exact 09:59, 15:30 and closing-minute records, plus the immediately preceding anchor. Use an early-close anchor where available. This checks access and shape, not archive-wide completeness.
- [ ] Check the currently documented entitlement/delay and make bounded observations of availability. Distinguish a documented delay allowance from observed arrival time; neither guarantees future availability or absence of revisions. Do not mark `source_delay` verified from a CLI number alone.
- [ ] Measure normal response time, payload size and requests per stage, including bootstrap/recovery. Calculate storage needs for 252 slots and verify the frozen 12-attempt daily cap is workable. If not, propose a protocol amendment before any study starts.
- [ ] Verify issuer workbook identity, currency, date format and blank-capital-gain semantics from authoritative material. Preserve a complete SPY schedule and overlap checks. If semantics remain unknown, explicitly mark the benchmark unresolved/provisional and state whether that prevents readiness; do not claim a complete comparison.
- [ ] Classify each readiness check PASS or UNRESOLVED with supporting artifacts. Failed authentication, lack of free SIP access or unavailable required valuations stops preparation; no paid upgrade, IEX substitution or guessed values.

**Done when:** the source checks have evidence, the proposed delay is justified, and all unresolved limitations are clearly separated from verified facts. A successful sample is not proof that future sessions will always work.

## Task 2: Test the remaining operational risks

**Owner:** Parent; Sonnet implements only demonstrated missing behavior. **Opus:** judgment review in Task 3.
**Files, only if needed:** `src/ggTrader/lab/intraday_observer.py`, `scripts/intraday_observe.py`, `tests/lab/test_intraday_observer.py`; a small separate scheduler wrapper and its focused tests if the existing CLI cannot cleanly skip non-session days. Core logic stays in `src/`.

- [ ] Rehearse outage, delayed valuation, process restart, holiday, early-close transition, DST transition, unexpected closure and failed month-one checkpoint. Preserve every fault and verify no duplicated or retrospective decision.
- [ ] Build a synthetic state of roughly the final year's expected record/payload size. Measure actual decision runtime, including clock check, evidence verification and durable writes; it must fit the registered 15:25–15:26 window with margin. Do not use real strategy outcomes to select fixtures.
- [ ] Check host clock synchronization read-only and record failure behavior. Missing/unsynchronized status must never count as timely valid evidence.
- [ ] Verify scheduling can skip holidays and stages outside the registered window without producing misleading failures. Previously scheduled early closes get archive/recovery only, not new hypothetical trades.
- [ ] Verify recovery of the last observation before its next-session grace deadline without creating a 253rd observation or extending the endpoint.
- [ ] For a demonstrated code defect, first reproduce it in a failing test, fix the shared cause and rerun focused tests. After final code changes, run the full capped suite and Ruff once. If no code changes, do not repeat the entire suite just to refresh a date.

**Done when:** normal and fault paths behave as registered, and decision timing has been measured on representative-sized evidence.

## Task 3: Prepare an exact, separate switch-on package

**Owner:** Parent; Sonnet may prepare bounded wrapper/unit files. **Opus:** independent readiness review. **Flash:** mechanical comparison of hashes, fields and schedule values only.
**Outputs:** reviewed readiness report/JSON; exact proposed service/timer files under `deploy/intraday_observer/` if that location remains suitable after checking repository conventions; operator commands and a local status/report procedure.

- [ ] Use a fixed reviewed release outside the disposable T3 worktree. Proposed release root: `/home/flynn/.local/share/ggtrader/intraday-observer/releases/<commit>/`. Proposed state root: `/home/flynn/.local/share/ggtrader/intraday-observer/studies/<study-id>/`. Resolve exact paths in the activation package; do not create them during planning.
- [ ] Pin the reviewed source files and record the exact existing interpreter/runtime. A mutable `current` checkout or a later dependency upgrade must not silently change an active study. A runtime mismatch should refuse, not auto-upgrade.
- [ ] Propose separate user-level jobs at **15:15 recovery, 15:20 inputs, 15:25 decision and 16:20 archive, America/New_York**. Use calendar-aware dispatch, no randomized delay, no catch-up/backdated decisions and no edits to production cron or existing services. Specify service timeouts and their fit to each stage's cutoff.
- [ ] Include the final post-window recovery/report handling. Optional revisions must not starve required captures; do not schedule them automatically.
- [ ] Specify owner-only permissions, credential inheritance, log location, bounded storage, request budgets, local health reporting and a disable procedure that preserves the journal. No external notification channel or message sending is authorized.
- [ ] After code is final, prepare the exact calendar/manifest for the proposed activation date: first full observation is the third full session strictly after that date. Freeze 252 slots, checkpoint 20, endpoint and anchor before observations. If the approval date moves, regenerate the unactivated package; never roll an active inception date.
- [ ] Produce the live-mode readiness file required by `activate`, bound to the exact manifest, delay and calendar snapshot with verified check assertions. Preserve the underlying evidence. Operator JSON is a recorded attestation, not a substitute for performing the checks or an authentication system.
- [ ] Opus reviews this concrete package against the protocol and outstanding build limitations. Parent fixes small substantiated findings and verifies them; no repeated broad strategy reviews. Flash may mechanically check final identities after changes.

**Done when:** the owner can inspect exactly what will run, where data will live, what the checks established, what remains uncertain, and how to stop it. No services/timers are loaded or enabled yet.

## Task 4: Present the concrete activation decision

- [ ] Show a plain-language summary: hypothetical journal only, no orders; exact first date and review dates; separate daily jobs; remaining evidence limits.
- [ ] Request approval to install/enable only the reviewed standalone observer configuration. This is the separate activation decision reserved by the existing plan/protocol, not another approval of the strategy idea.
- [ ] If approved, install the fixed release/state and named units, record that authorization, activate using the matching readiness file, and verify the next scheduled runs. Preserve production settings.
- [ ] Verify the first actual input, decision and archive records, their completion times, source references and integrity. If any check fails, report and follow the registered fault/halt policy; do not manufacture a clean first day.
- [ ] Keep a simple local daily status record so a silent missed job is visible. No external alerts unless separately instructed.

## Follow-through after activation

The first 20 registered slots test operational reliability. Monthly reports are descriptive. At slot 252, retain all faults and produce the registered observation report. No early performance pass, tuning, favorable restart or automatic move to broker orders. This plan authorizes none of those outcomes by itself.

## Handoff rule

Do not recreate the recorder, rerun the closed historical study or commission another general strategy review. Start at Task 1, record evidence, and progress to the concrete activation package. If a required source/readiness check is unresolved, report that precise blocker and preserve the work already completed.
