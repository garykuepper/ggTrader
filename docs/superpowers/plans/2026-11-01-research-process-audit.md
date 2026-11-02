# Research Process Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or, if the owner explicitly selects delegation, superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit whether ggTrader's research understood, implemented, tested and judged its hypotheses fairly, while preserving existing decisions and finding evidence gaps.

**Architecture:** Build a sourced inventory and claim ledger, trace shared infrastructure across historical versions, calibrate an evidence rubric on representative studies, then assess the remaining claims and synthesize findings. The deliverable is a research audit, not a trading system or strategy selection.

**Tech Stack:** Markdown, JSON, read-only Git history and source inspection, Python standard library for artifact validation; project test wrapper only if tests are independently authorized and needed.

**Spec:** [Research process audit design](../specs/2026-11-01-research-process-audit-design.md).

## Global constraints

- Read [AGENTS.md](../../../AGENTS.md), the specification, [next steps](../../next_steps.md) and the [research snapshot](../../research/RESEARCH_SNAPSHOT.md) first.
- Taxable account only; stocks prioritized, crypto in scope; no paid datasets or subscriptions.
- Do not recommend retiring the stock sleeve, ending paper trading, shutting the project down or changing account type.
- Preserve production freeze, existing NO-GOs, stopped runs, owner decisions and event-study parking.
- This plan is research and documentation only: no market-return reruns, new data acquisition, production changes or orders. Propose separately scoped work where needed.
- Existing reviewed windows stay reused evidence; never call them fresh OOS. Report negative results and limitations accurately.
- Do not hand-edit the generated research snapshot. Existing untracked external data and unrelated workspace files are untouched.
- Cite specific reports, code paths, revisions and raw artifact keys; label observations, calculations, hypotheses and unknowns.

## Review focus

1. **Incomplete inventory:** a registry item or dated report absent from the ledger is surfaced as a coverage gap (Task 1).
2. **Stale or missing provenance:** a version or raw artifact cannot be located; mark unknown and limit inference (Tasks 1–2).
3. **Misleading confidence:** a threshold failure without precision sufficient to exclude a useful effect is not called proof of no edge (Tasks 3–4).
4. **Repeated or contaminated evidence:** related variants and previously inspected windows are linked, not counted as independent confirmations (Tasks 1, 3–4).
5. **Unmatched execution assumptions:** stock and crypto strategies have different data calendars, benchmark, funding, cost and tax needs; investigate before extrapolating (Tasks 2–4).

## Files and interfaces

- Create `docs/research/_research_process_audit_ledger.json`: machine-readable evidence ledger.
- Create `docs/research/2026-11-01-research-process-audit.md`: human-readable coverage, methods, case studies and synthesis.
- Update `docs/research/RESEARCH_INDEX.md` only to link the completed audit; do not replace historical classifications.
- JSON top-level object: `schema_version` (integer 1), `as_of` (date string), `coverage` (object with `sources`, `counts`, `exclusions`), `claims` (array), and `shared_findings` (array). Each claim has stable `id`, `family`, `kind`, `question`, `original_decision`, `decision_source`, `implementation_fidelity`, `evidence_validity`, `inference`, `follow_up`, `sources`, `dependencies`, `variants`, `limitations`, and `status_notes`. Use arrays for `sources`, `dependencies`, `variants`, `limitations`; use `unknown` for absent scalar facts and explain why in `status_notes`. Preserve specific verdicts and authority in `original_decision`/`decision_source`; audit classifications are separate.
- Allowed `kind`: `tested`, `candidate_group`, `method`, `allocation`. Fidelity: `faithful`, `materially_changed`, `unknown`, `not_applicable`. Validity: `usable`, `qualified`, `invalid`, `unresolved`, `not_assessed`. Inference: `not_demonstrated`, `evidence_against_meaningful_effect`, `adverse_point_estimate`, `inconclusive`, `not_assessed`. Follow-up: `none_identified`, `documentary_clarification`, `bounded_technical_diagnosis`, `separate_study_proposal`.
- Each shared finding records `id`, `observation`, `evidence`, `affected_claims`, `impact_status` (`verified`, `possible`, `unknown`), and `limitations`.

## Task 1: Inventory claims and provenance

**Inputs:** Research index, snapshot, roadmap, worklist, registry, dated reports, briefs and candidate register.

**Outputs:** Initial ledger and a coverage section in the report.

- [x] Enumerate registry names directly from `src/ggTrader/lab/strategies/__init__.py`; reconcile the list with the snapshot and index. Record counts and differences rather than assume the index is complete.
- [x] Inventory dated research reports, non-registry levers, allocation studies and data/methods reports. Make one ledger entry per material claim or explicitly grouped identical mechanism; record registry-to-claim mappings and source links.
- [x] Group literature-only candidates by register batch/family with links back to their individual entries. Record exclusions, missing reports, artifacts and actual Git-tracking status; do not count candidates as performed tests.
- [x] Record for each tested claim its original decision, pinned window if recoverable, benchmark, data and limitations, provenance, dependent mechanism and related variants. Use `unknown` for unrecoverable values; do not invent fields.
- [x] Verify all registry entries and dated reports are represented by at least one ledger entry or listed in `coverage.exclusions` with a reason. Spot-check at least five source links, including one missing or unresolved case, against the files.

## Task 2: Trace shared research machinery

**Inputs:** Task 1 ledger, lab harness and history, correction reports, raw results.

**Outputs:** `shared_findings` entries and a shared-machinery section with a dated fix-to-study dependency map.

- [x] Trace fold generation, overlapping OOS periods, anchor training, gate outcomes, halt/release logic and the parameters actually deployed. Record what is observed at each run's historical revision and flag unproven historical applicability.
- [x] Trace decision/fill timing, calendar and annualization, cost, cash, leverage, sizing, benchmark and risk metrics for stock and crypto paths. In particular check whether `curve_stats` with 252 periods per year was used on seven-day crypto data; if the call path cannot be reconstructed, record impact as unknown.
- [x] Map anchor-leak, PIT, DSR, timing and price-tape fixes to affected studies where supported; distinguish possible exposure from measured effect and retain pre-fix gate limitations.
- [x] Specify prospective diagnostic controls and power/minimum-detectable-effect calculations, their assumptions and what failures would mean. Do not run market backtests or assert a fixed effect-size threshold for all strategies.
- [x] For every shared finding cite a concrete line, report or artifact and at least one affected claim or mark the mapping unknown. Verify failed gates/halt select anchors, not automatically cash, and record FOMC clustered statistics separately from unclustered statistics.

## Task 3: Calibrate the rubric with representative cases

**Inputs:** Tasks 1–2, source papers where accessible, reports, briefs, code and raw artifacts.

**Outputs:** Four case-study sections and calibrated ledger entries; a fifth crypto feasibility case when records permit.

- [x] Trace pairs: intended mechanism, 1:1 spread, cadence, holding/ranking behavior, cache key, version, borrow/cost assumptions, corrected rerun anomaly and actual measured result. Distinguish what is verified from a hypothetical daily or hedged alternative; retain the original negative outcome.
- [x] Trace momentum: French proxy vs project top-N implementation; registered hypotheses, Holm family, post-1994 cost-based hurdle, sensitivity and uncertainty; distinguish source mechanism from the implemented strategy and from its screening decision.
- [x] Trace production replica: rules, portfolio and exposure-matched benchmarks, stock exposure, reported excess, order-parity denominator, missing decision inputs and stopped holdout. Keep model NO-GO, inconclusive holdout and unverified live parity separate.
- [x] Check FOMC's day-clustered versus unclustered statistics and the economic/funding benchmark. Assess how a different statistical unit changes the inference without treating t as return or claiming a zero effect.
- [x] Inspect a crypto feasibility or metric example; document source/venue/calendar restrictions and any unsupported annualization attribution. If no eligible evidence exists, explicitly mark it unresolved.
- [x] For each case write source-to-rule trace, original verdict, separate evidence classification, strongest supporting evidence and counterargument, and a bounded follow-up or reason none is identified. Confirm every reported number against its cited artifact/report; do not infer causation from a subperiod split.

## Task 4: Apply criteria and classify the full inventory

**Inputs:** Calibrated rubric, all Task 1 claims, shared dependency map.

**Outputs:** Completed ledger with an auditable classification for every entry, or a disclosed unresolved/not-assessed reason.

- [x] Assess each material tested claim for source fidelity, data and temporal integrity, implementation/market conventions, benchmark, funding leg, risks, costs, tax and decision standard. Use qualified or unresolved where evidence is missing.
- [x] For each NO-GO distinguish failing an economic or statistical decision rule from evidence excluding a predefined meaningful effect. Record uncertainty and measurement units; do not derive a confidence interval without a valid, documented method and inputs.
- [x] Review Part A, B0 and B1 for appropriate scope, repeated testing, power and false-positive/false-negative tradeoffs, and whether source proxies support the intended claim. Preserve the approved stock-specific rules and inclusive 15% stop; do not apply them automatically to crypto.
- [x] Choose any further deep dives using coverage of unexamined mechanisms/harnesses, materiality and available evidence, never attractive realized returns. Record the rule and exclusions. Keep event studies parked.
- [x] Document a fair second look only where there is a specific reproducible defect, missing question or substantive mechanism mismatch; state original verdict, possible direction, data constraints, what observation could disconfirm it and why a rerun would not be independent. It remains a proposal.

## Task 5: Synthesize, validate and hand off

**Inputs:** Completed ledger, deep dives and shared findings.

**Outputs:** Final audit report and one link in the research index.

- [x] Explain supported reasons for NO-GOs, unresolved causes, evidence that remains negative, competing explanations and their counterarguments. Do not infer market efficiency or publication-driven decay from the count of failures.
- [x] Summarize process gaps, candidate research questions and data dependencies without choosing a strategy. If proposing different evaluation standards, describe alternatives and error tradeoffs without changing the approved standard.
- [x] Record raw source locations, code revisions, available artifact keys and any gaps that prevent reproduction. Do not create new numerical return claims without a reproducible artifact and separately authorized computation.
- [x] Validate JSON with `python3 -m json.tool docs/research/_research_process_audit_ledger.json >/dev/null`; check all local Markdown links resolve, `git diff --check`, and registry/report coverage against the ledger. Spot-check at least one claim per deep-dive category against primary artifacts and report exceptions.
- [x] Verify no unapproved data acquisition, rerun, production change, verdict override, paid data suggestion or account-type recommendation appears in the deliverables. Link the final report from `RESEARCH_INDEX.md`; report outstanding unknowns and the scope of verification.

## Execution handoff

Execution begins only after the owner reviews this plan and chooses the execution method. An audit report is not authorization for the separately scoped follow-up work it identifies.
