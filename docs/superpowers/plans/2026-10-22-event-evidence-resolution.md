# Event Evidence Resolution Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the
> owner's explicitly delegated execution. Track the checkboxes below. Gemini
> Flash on agy performs the evidence work; Codex validates claims and acquisitions.

**Goal:** Resolve each of the five outstanding evidence gaps to a supported
finding or an exact remaining dependency, then finish the registered per-signal
feasibility audit. Resolution does not promise a PASS or an investable strategy.

**Architecture:** Reuse the existing audit, raw-source parsers, immutable SEC
cache, reconstruction and requirement matrix. Start with a single consolidated
offline evidence pass over all five gaps; collect only specifically justified,
reviewed inputs afterwards. No new ingestion framework or return detector.

**Tech stack:** Existing Python environment, SEC JSON/XML/HTML, existing audit
metadata/export contracts and Markdown/JSON evidence artifacts.

**Spec:** [Approved audit](../../research/briefs/2026-10-21-event-timing-audit.md),
[source manifest](../../research/briefs/2026-10-21-event-timing-audit-sources.md),
[approved filing-date amendment](../../research/briefs/2026-10-22-event-timing-source-conflict-proposal.md).
This plan extends the earlier insider evidence plan; completed collection and
collector reviews are not repeated.

## Global constraints

- Owner requested this plan and delegation to agy Flash. Begin the delegated
  offline work now; no additional plan confirmation is needed.
- No returns, price changes, CAR, P&L, post-event price paths, backtests, new
  detectors, paid purchases, account creation, sales outreach or production work.
- No edits to `paper/`, `data/live/`, `.env`, containers or database configuration.
  Never delete raw data. No database writes. Initial delegation makes no DB calls.
- Latest collection lineage: `data/external/event_timing_audit/identity-inventory-we_55vr8/`,
  79 SEC attempts, zero Yahoo attempts, 104.3013715306297 seconds. Reconcile the
  actual ledger before any later acquisition; never sum inherited prefixes twice.
- SEC ≤5 requests/sec; ≤20,000 SEC and ≤2,000 Yahoo attempts total including
  retries; initial attempt plus three retries; six-hour cumulative runtime cap.
  First new 403/429 stops external calls. Preserve prior documentation-stop history.
- Completed 26-URL batch authorization is consumed. Initial Flash work is offline;
  it prepares exact follow-up requests, not unbounded collection. Codex checks the
  concrete acquisition against existing owner authorization before execution.
  Only a genuine scope/protocol amendment or explicit restart requires a new decision.
- Contact authorization persists, process-only. Do not read or print private
  environment values, include contact in reports, or send emails.
- Observation cutoff 2026-04-30; insider candidate complete years 2015–2025;
  2026 is diagnostic. Keep original timestamp conflicts and the uniform approved
  date-only assessment separate; observed dissemination remains UNVERIFIED.
- Fixed A9 SHA-256 ordering, no sample substitution; all 136 NYSE month-end prefix
  checks. S&P PIT denominators, quarantines, failures and right censoring stay intact.
- A6 requires identity, retrieval and mapped-price coverage each ≥95%; both
  unexplained-missingness ratios ≤2%. A7 additionally requires every mandatory
  annual check and at least five contiguous qualifying complete years.
- Preserve R1–R9 routes and the two-hour cumulative vendor-documentation search
  ceiling. Unknown remaining search budget is not a fresh two hours. No new Vane
  or other external search in the initial pass. Unsupported route properties stay unknown.
- Earlier Flash investigations contained unsupported causation and coverage claims.
  Rebuild claims from exact raw references; never reuse their conclusions as evidence.
- A missing source is UNVERIFIED, not PASS or proof of infeasibility. Operational
  NOC/COST examples cannot qualify a population. No strategy edge is validated here.

## Files and ownership

- Plan/controller: this file, `docs/next_steps.md` and the durable
  `docs/research/event-timing-audit-implementation.md` remain Codex-owned.
- Flash creates only `docs/research/2026-10-22-event-evidence-resolution-workbook.md`
  and `docs/research/_event_evidence_resolution_workbook.json` during the initial pass.
  Scratch analysis may live in `/tmp`; no operational-cache writes.
- Read existing `event_timing_audit.py`, `event_timing_sources.py`,
  `event_timing_reconstruction.py` under `src/ggTrader/lab/`, the reviewed
  artifacts `_insider_identity_inventory_acquisition.json` and
  `_insider_identity_inventory_resolution_results.json`, and relevant raw caches.
- Final controller deliverables remain `docs/research/_event_timing_audit.json`
  and `docs/research/2026-10-21-event-timing-audit-report.md`, following the registered
  audit schema. Never overwrite them with a partial operational workbook.
- Runtime code changes are conditional on a reproduced defect, not a prerequisite.
  Flash proposes them; Codex owns implementation, targeted tests and review.

## Evidence interface and review focus

Each workbook requirement has: stable ID, applicable signal, registered predicate,
current PASS/FAIL/UNVERIFIED, scope/dates, exact source path/URL/hash and raw
locator, observation, inference (separate), missing evidence, smallest next action,
acquisition/query dependencies, and whether an approved rule would change.
Use the existing audit schema for audit inputs; do not invent a second runner.

Review especially: repeated annual observations mistaken for continuous identity;
page descriptors silently repaired; absent detector flags mistaken for clean tape;
amendments/co-owners collapsing distinct events; revised consensus mistaken for
pre-announcement information. No newly created review receipt may claim its own
unsupported interpretation is verified.

## Task 1: Historical identity and legal security continuity

**Consumes:** 24 cached annual reports and source-linked observations; PIT member
file; dated ownership records. **Produces:** identity/class timeline evidence and
precise gaps in the workbook, not inferred validity intervals.

- [ ] Verify relevant raw hashes and annual cover/Item 5 statements; record filing
  date separately from fiscal-period date and any explicit effective date.
- [ ] Extract the exact 2015 and 2022 COST charter references already identified
  in the results artifact. Compare actual hyperlink targets with exhibit captions.
  Preserve the 2019 caption/link disagreement. Both charter contents are uncached.
- [ ] Prepare a two-document request list with provenance and expected questions:
  operative common-stock/par-value provisions, amendments and effective dates.
  The 2020 charter and 2020/2022 Exhibit 4.8 descriptions are conditional follow-ups,
  not automatic additional requests. Do not infer a typo or a class conversion.
- [ ] After reviewed acquisition, inspect the operative language. Explain whether
  it resolves the 2014–16 $.005 / 2017–21 $.01 / 2022–25 $.005 observations and
  the associated accounting entries. Record exactly what remains unsupported.
- [ ] Distinguish issuer continuity, security-class continuity, ticker validity
  and vendor-price identity. Identify equivalent missing evidence for NOC and
  the broader PIT universe; no annual interpolation or survivor-only substitute.

**Done when:** every proposed validity assertion has dated legal/source support,
or a named unresolved dependency; no unsupported identity is passed downstream.

## Task 2: Northrop inventory conflict disposition

**Consumes:** cached root descriptor, pages 001/002 and anomalous filing.
**Produces:** reproducible inventory incident and a supported disposition.

- [ ] Confirm 2,000 page-001 rows, anomalous accession `0001225208-17-005319`,
  descriptor end 2017-03-01, API/header filing date 2017-03-02; check duplicates
  and preserve all rows. Page 002 has 214 rows inside its declared range.
- [ ] Separate source inconsistency from missing-page retrieval (now closed).
  Inventory any already-cached authoritative documentation of descriptor semantics;
  if none exists, say so. No universal pagination claim from this one example.
- [ ] Propose the smallest independent corroboration or source refresh needed,
  with exact URLs and the effect on registered completeness. Do not retry the
  earlier blocked documentation URL or silently refresh immutable cached inputs.
- [ ] If a different completeness interpretation is necessary, draft the precise
  predicate amendment and its consequences for all pages; leave the gate uncleared
  until approved. A source observation alone cannot authorize a protocol change.

**Done when:** supported disposition or exact documentary/protocol dependency is
recorded; original and any alternate assessments remain distinguishable.

## Task 3: Historical price mapping and flags-only tape metadata

**Consumes:** supported identities, existing detector/export contract and local
artifact inventory. **Produces:** metadata availability map and minimal read plan.

- [ ] Inspect `load_existing_artifact_flags()` and `ReadOnlySnapshot` in
  `event_timing_audit.py`; record actual fields and provenance requirements.
- [ ] Locate existing flags/counts-only artifacts without reading price values
  or return paths. List available date/security coverage and hashes; missing export
  is UNVERIFIED, not zero flags. Do not rerun a detector.
- [ ] Prepare exact read-only metadata SELECT/export proposals for missing inputs:
  vendor/security identifiers, session dates, per-date bar counts and existing
  detector flags. No price columns, outcomes or DB credentials in artifacts.
  The controller checks the proposals before a separate read-only execution step.
- [ ] After inputs exist, verify exactly one mapped bar on the tradeable NYSE
  session and no existing detector flag for each event; keep failures in denominators.
  Price existence does not independently prove historical security linkage.

**Done when:** mapped identity and tape predicates are supported per security-date,
or the precise mapping/export dependency is recorded for the final matrix.

## Task 4: Insider raw reconstruction and qualification

**Consumes:** Tasks 1–3, cached ownership documents and existing reconstruction.
**Produces:** requirement-level reconstruction findings and annual gate inputs.

- [ ] Run only pure/offline supported reconstruction on copied inputs or scratch
  outputs. Inventory every owner, role, lot, direct/indirect holding and security
  class; preserve 4/A, private-P ambiguity, missing fields and unknown linkages.
- [ ] Assess amendments at their own public times, collision/duplicate accounting,
  late filings, carry-in/lookback/load boundaries and the fixed 136 prefix checks.
  Identify missing raw accessions/URLs and cost, not invented test successes.
- [ ] Keep cluster events distinct from filings/lots; preserve all quarantined and
  censored units. Validate date-only and original timestamp namespaces separately.
- [ ] Build annual A6/A7 input coverage only from the registered full denominator;
  incomplete NOC/COST examples are diagnostics. Cost any population expansion
  before collection and stop if required evidence or budgets are unavailable.

**Done when:** each mandatory reconstruction check has a source-backed predicate
or exact missing input; no qualifying year is manufactured from operational examples.

## Task 5: PEAD announcement, actual EPS and consensus provenance

**Consumes:** cached source-search evidence, stored-schema contracts and approved
R1–R9 manifest. **Produces:** per-route evidence matrix and narrow remaining questions.

- [ ] Inventory existing evidence once; do not repeat the completed vendor search.
  For each relevant route record fields, history/delisted coverage, PIT availability
  and individual access/price evidence with exact primary citations already retained.
- [ ] Separate original release/issuer/fiscal-period matching and actual-EPS basis
  from filing acceptance; consensus snapshot must precede announcement, not merely
  acceptance. Missing announcement time or estimate lineage stays UNVERIFIED.
- [ ] Distinguish existing cached source contents from unvisited URLs. Determine
  documented spent search time and remaining allowance; uncertain accounting stays
  an explicit dependency, not a reset. Vane only after a specific reviewed question
  and budget/stop clearance, using public technical questions without private data.
- [ ] Record candidate mixed routes only over a common ≥5-complete-year window.
  Silent documentation does not exclude a route. No purchase, trial or outreach.

**Done when:** every relevant route/requirement is supported or explicitly unresolved,
without overstating NO VERIFIED PATH or dropping PEAD from the joint audit.

## Task 6: One consolidated acquisition review and final audit disposition

- [ ] Flash submits the complete offline workbook for all five tasks, including
  exact requests/metadata queries, dependency order, nominal and worst-case costs,
  protocol decisions and hard blockers. Stop repeating speculative research.
- [ ] Codex checks every material claim against raw evidence and reviews one
  consolidated acquisition package. Group independent approved requests; preserve
  separate operational and A9-sampled work, cumulative state, immutable caches,
  exclusive claims, source/spec hashes, stop behavior and checkpoint recovery.
- [ ] Execute only cleared bounded work within authorization, then review its raw
  results. No user reconfirmation for routine authorized work. Ask only for a
  concrete changed predicate, unavailable required input or explicit restart.
- [ ] Any runtime fix starts with a failing fixture, then the smallest shared-code
  fix. Use `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q`, targeted
  reconstruction tests if affected, Ruff check/format and fresh code review.
  Documentation-only work needs JSON/link checks and `git diff --check`, not tests.
- [ ] Populate the registered final JSON/report with actual annual denominators,
  missingness, per-requirement predicates, longest qualifying window and separate
  per-signal verdicts. If acquisition cannot establish a mandatory property, finish
  honestly UNRESOLVED with exact dependency and evidence, not another vague plan.
- [ ] Update the handoff/worklist and commit coherent outputs. No push or deployment.
  Only a verified data route can advance to a separately approved Phase 2 brief;
  no result here establishes performance against SPY.

**Completion:** All five gaps have an evidence-backed disposition, final registered
artifacts explain what is and is not feasible, and every remaining external or
owner decision is named precisely. Missing evidence is never disguised as resolution.
