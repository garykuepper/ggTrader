# Insider Evidence Validation Plan

> **For agentic workers:** Use superpowers:executing-plans for sequential execution, or superpowers:subagent-driven-development for explicitly delegated tasks. Track the checkboxes below. Read the approved audit brief before execution.

**Goal:** Determine what evidence and acquisition work are still needed for a fair insider-cluster test, using the reviewed audit and its cached smoke responses.

**Architecture:** Reuse the existing audit, source parsers, reconstruction and immutable response cache. Investigate Northrop and Costco as operational examples; keep population qualification and deterministic sampling under the registered protocol. No new strategy, backtest or general-purpose ingestion framework.

**Tech stack:** Existing Python environment, SEC JSON/XML, read-only metadata queries, existing audit fixtures; Vane only for specific unresolved documentary questions.

**Spec:** [Approved event-timing audit](../../research/briefs/2026-10-21-event-timing-audit.md), especially §8, and [source-search manifest](../../research/briefs/2026-10-21-event-timing-audit-sources.md). This plan implements those rules; it does not amend them.

## Global constraints

- No returns, price changes, P&L, drift, CAR, post-event price paths or new return-based detectors.
- Production, `.env`, container configuration and database writes remain out of scope. Export the already-authorized contact only into collection processes; never include it in artifacts or research queries.
- Preserve all raw caches and their accounting. SEC ≤5 requests/second; total ≤20,000 SEC and ≤2,000 Yahoo attempts, including retries; stop external calls at the first 403/429; cumulative runtime cap six hours. Existing smoke consumed 50 SEC attempts and zero Yahoo attempts; inspect the durable ledger before calculating remaining capacity.
- A9 sampling, A6 denominators, A7 predicates and the 2026-04-30 observation cutoff stay fixed. Insider complete candidate years are 2015–2025; 2026 YTD is diagnostic only.
- Northrop/Costco examples cannot replace failed sampled records or qualify an annual population. Current ticker matches are discovery candidates, not historical identity proof.
- Missing evidence remains UNVERIFIED. A completed audit may honestly finish UNRESOLVED. No paid purchase or Phase 2 return test is authorized by this plan.
- Only Codex/agy delegation is permitted; use Codex while agy remains quota-limited. No Claude. A separate fresh reviewer checks any code change before further collection.

## Starting evidence

- Reviewed code: `45ddee9`; 124 focused tests passed in the builder run. Independent residual review cleared all four findings and reran 14 residual cases.
- Smoke cache: `data/external/event_timing_audit/network-smoke-5b1q16b6/`. All 50 responses returned HTTP 200 and passed independent SHA-256 checks. Resume made no new requests.
- Northrop (CIK 1133421): recent inventory retained, historical inventory incomplete. Its first historical page declares an end date of 2017-03-01 but contains accession `0001225208-17-005319`, dated 2017-03-02. Declared/parsed page counts both equal 2,000. Its second historical page was not fetched.
- Costco (CIK 909832): parser reported a complete 2,733-filing inventory. That is not proof of dated identity, complete raw transaction reconstruction or annual coverage.
- Smoke supplied no reviewed historical identities, earnings provenance or detector export. Both signals remain UNRESOLVED; zero matrix PASS.
- Review/verification notes currently live in `/tmp/event-timing-*.md`; preserve the relevant observations and cache references in the durable implementation handoff before relying on temporary files in a future session.

## Files and responsibilities

- Update `docs/research/event-timing-audit-implementation.md`: durable smoke findings, source citations, operational-case evidence references, unresolved requirements and acquisition decision.
- Update `docs/next_steps.md`: only the next one or two actions and link to this plan.
- Preserve `data/external/event_timing_audit/`: raw responses, manifests, accounting and checkpoints. Store new raw evidence there only under the reviewed acquisition plan.
- Conditional code changes only: `src/ggTrader/lab/event_timing_sources.py`, `event_timing_audit.py`, `event_timing_reconstruction.py`, and `tests/lab/test_event_timing_audit.py`. Reuse current interfaces. No new module or dependency is planned.
- Final audit deliverables remain `docs/research/_event_timing_audit.json` and `docs/research/2026-10-21-event-timing-audit-report.md`; intermediate examples stay explicitly partial, with reports under `/tmp` and raw evidence retained.

## Review focus

1. A stale SEC page descriptor must not become either invented completeness or silently discarded filings: Task 1 preserves both and identifies the exact discrepancy.
2. Today's ticker/CIK match must not establish an earlier security identity: Task 2 requires dated supporting records.
3. Amendments and co-reporting owners must not rewrite earlier public information: Task 2 checks the existing raw/public-time reconstruction and preserves unknown linkage.
4. A new run must not renew the spent request/runtime budget: Task 3 requires cumulative ledger reconciliation before any request.
5. Two operational examples must not become a selected performance sample or an annual PASS: Tasks 2–4 label their scope and retain registered population requirements.

## Task 1: Resolve the inventory discrepancy using cached evidence first

**Inputs:** Cached submissions JSON and historical page; their manifest hashes; existing `enumerate_sec()` parser and interval/count checks.

**Deliverable:** A source-linked incident entry in the implementation handoff, separating observed disagreement from inferred cause.

- [ ] Recompute the relevant cache hashes and record descriptor dates/count, actual row dates, accession and exact raw locations. Confirm whether the anomalous accession is duplicated elsewhere in already-cached responses. No new download is needed for this step.
- [ ] Establish what SEC documentation actually guarantees about historical page descriptors. If multiple sources are needed, use Vane with: “What do SEC submissions API filingFrom, filingTo and filingCount mean, and what official guarantees or documented limitations apply to historical pagination?” Follow the answer to exact SEC documentation; Vane synthesis is discovery, not evidence by itself.
- [ ] Record one of: documented source inconsistency, documented parser misinterpretation, or unresolved semantics. Do not infer a universal rule from one page.
- [ ] Keep inventory completeness UNVERIFIED until independently reconciled. If a code fix is justified within the approved rules, add the cached-shape fixture, show the current failure, make the smallest fix, run the focused tests and obtain independent review. If the fix changes an approved predicate or source interpretation that controls status, prepare a dated amendment for owner approval before implementing it.

**Done when:** The discrepancy has reproducible raw evidence and a supported disposition or an explicit unresolved question. No rule was weakened to get a pass.

## Task 2: Build an evidence checklist for the two operational examples

**Inputs:** Existing Northrop and Costco cached filings; registered event-timing evidence schema and field-review contract. Read `EventTimingAuditRunner`, `collect_filing()` and `validate_evidence_record()` rather than creating a parallel parser.

**Deliverable:** Per-company evidence references and a requirement-by-requirement gap list, not a feasibility verdict.

- [ ] Inventory what the cache actually contains: submissions pages, raw Form 4/4-A XML, header/API receipt pairs, owners, lots, plan-field states and amendment candidates. Count missing documents explicitly; do not treat a successful HTTP response as successful reconciliation.
- [ ] Identify primary evidence needed for historical security↔issuer↔price-symbol validity intervals. Verify dates, share classes, reorganizations and symbol reuse. Do not infer “OTHER” status from index survival or substitute today's ticker map for history.
- [ ] If historical identity cannot be resolved from known primary documents, use one targeted Vane question about SEC-documented ways to reconstruct dated ticker/CIK/security history, including reorganizations and delisted issuers. Open and verify the cited primary pages. Record unsupported periods as unknown; do not broaden the approved commercial route manifest.
- [ ] Compare available raw owner/lot inventories with the existing stored-schema semantics without repairing the database. Check acceptance/header agreement, permitted date fallback, amendments from their own public times, late disclosures and missing co-owners. Matching receipt times do not prove dissemination.
- [ ] Record exactly which missing documents require requests and which facts require human documentary review. If a software defect is demonstrated, use one focused failing regression and minimal shared-code fix; otherwise write no code.

**Done when:** Each mandatory insider requirement has either referenced evidence or a named gap and acquisition/review action. The two examples remain operational diagnostics; no year or source route is declared verified.

## Task 3: Prepare and review the next bounded acquisition

**Deliverable:** An executable input/cache/request plan derived from Tasks 1–2, with a supported go/stop decision before requests.

- [ ] Enumerate the exact missing URLs and dependency order, including Northrop's unfetched page. Preserve A9 hash ordering for registered sample work; do not substitute easier issuers. Separate operational incident work from the registered sample and count both against the overall limits.
- [ ] Sum existing request reservations and elapsed accounting across every related run. State remaining ceilings. A new directory or changed code revision cannot reset the total allowance. If current entry points cannot enforce the remaining allowance, fix that narrowly and test it before sending requests.
- [ ] Choose validated immutable inputs, contact handling, no-write DB metadata access if needed, partial output locations and cache/resume bindings. Existing smoke state cannot be relabelled as a full run; preserve it. Any new registration must explicitly carry cumulative accounting forward without bypassing fail-closed recovery.
- [ ] Have a fresh Codex reviewer check only this acquisition plan and any code changes. Execute the reviewed bounded work under the owner's existing audit authorization; request a decision only for a genuine scope/protocol change or missing required input.
- [ ] Verify actual request outcomes, checksums, accounting, stop behavior and partial reconstruction. Report unresolved evidence honestly. Do not expand to a whole-universe run merely because endpoints respond successfully.

**Done when:** The bounded work has traceable evidence and the next acquisition's cost/runtime can be justified within the fixed limits, or a specific blocker is documented.

## Task 4: Decide whether to complete the registered feasibility audit

- [ ] Assemble the remaining input matrix: dated identities, complete inventories, raw reconstruction, public timing, existing flags-only detector export and mapped bar-existence metadata. Neither document count nor product-history overlap substitutes for A6/A7 predicates.
- [ ] Recommend expansion only if the inputs and registered budget support it. Otherwise identify the concrete unresolved requirement and stop that branch; do not invent coverage or initiate a purchase. Preserve both signals in the approved audit; insider-first is execution order, not permission to silently omit PEAD.
- [ ] PEAD continues to require original-announcement matching, actual-EPS lineage and strictly pre-announcement consensus evidence. Do not repeat the completed vendor search without a new specific question. Any further R1–R9 documentation search must account for the already-spent time within the approved two-hour total; if no budget remains, record unresolved evidence or seek an explicit amendment.
- [ ] Finish the registered JSON/Markdown report with the actual per-signal status, including UNRESOLVED/no qualifying window when appropriate. Any strategy return test requires a separately written, reviewed and owner-approved Phase 2 brief.

**Done when:** The owner receives a defensible data-feasibility decision and its raw evidence, not another promising-looking backtest.

## Verification and handoff

Documentation-only work needs link/path and `git diff --check` validation, not new unit tests. For a code change, first reproduce the concrete defect in a focused fixture, then run `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q` and Ruff check/format checks on changed Python files. Broaden testing only for a demonstrated cross-module concern. Commit coherent code/documentation changes; no push, merge or deployment is implied.

Use Codex for execution and fresh Codex review while other approved providers are quota-limited. This plan does not authorize Vane to receive private account data, contact details, secrets or raw proprietary records. Use public technical questions only and verify primary citations directly.
