# Plan: resolve the event-timing audit to an owner decision

**Date:** 2026-10-23. **Branch:** `research/event-timing-audit`.
**Supersedes, for scheduling only:** [complete-event-feasibility-audit](2026-10-22-complete-event-feasibility-audit.md),
[next-evidence-actions](2026-10-22-next-evidence-actions.md) and
[event-evidence-resolution](2026-10-22-event-evidence-resolution.md). Their recorded evidence stands.
**Binding rules:** [audit brief](../../research/briefs/2026-10-21-event-timing-audit.md) §8 and its
[source manifest](../../research/briefs/2026-10-21-event-timing-audit-sources.md), plus the approved
[source-conflict amendment](../../research/briefs/2026-10-22-event-timing-source-conflict-proposal.md).
No pass bar changes.

## Why this plan

Two days of evidence collection (38 commits) produced no verdict. The remaining blockers are
**source availability**, not parser bugs:
- pre-announcement consensus history (PEAD);
- a historical security-identity map and clean pre-2021 prices (both signals);
- detector export provenance.

More SEC or charter collection cannot supply any of these. The approved vendor-documentation
search (source manifest R1–R9) can, and it has not been run.

## Stop rule (effective now)

No further charter, predecessor, identity-inventory or broad SEC collection, and no further
parser or protocol work beyond Task 1. Any new evidence need is written to the decision memo
as a blocker, not pursued.

## Tasks

Tasks 1 and 2 are independent and run in parallel. Each is assigned to a Codex subagent, which
may hand Gemini Flash only bounded extraction work with checkable outputs (repo `AGENTS.md`,
delegation preference).

### Task 1 — Timestamp-source amendment (draft only; owner gate)

**Finding (senior quant, 2026-10-23, from cached raw bytes, no returns):** in all 25 cached
API/header pairs, the submissions-API `acceptanceDateTime` minus the full-submission header
`ACCEPTANCE-DATETIME` equals twice the New York UTC offset (8 h in DST, 10 h in standard time).
Read as New York time, the header falls within EDGAR's 06:00–22:00 ET hours and matches the
official filing date in 25/25 cases. Read as UTC, the API value violates one or both in 9/25.

Deliverable: `docs/research/briefs/2026-10-23-insider-acceptance-time-amendment.md`, proposing
that insider signal availability be the header `ACCEPTANCE-DATETIME` in America/New_York, with
the API field recorded as a known-defective diagnostic. It must include:
- a reproducible script under `scripts/` that regenerates the 25/25 vs 9/25 counts from the
  cache, offline, with `--help`, plus a focused test;
- a citation of SEC documentation for EDGAR filing hours and the filing-date rule, retrieved
  live. If the documentation can't be retrieved, say so and leave the citation UNVERIFIED.
- the explicit effect on the existing date-only amendment.

**Gate:** the status is DRAFT until the owner approves it. Do not implement it in the runner.

### Task 2 — Vendor-documentation search (already approved)

Execute the source manifest exactly: routes R1–R9, in its order, from public documentation only,
within its 2-hour cap. The four evidence items and the UNRESOLVED and exclusion rules apply as
written.

Deliverable: `docs/research/_event_source_search_ledger.json`, with one record per route and
evidence item. Each record holds the URL, the retrieval time, a verbatim excerpt (or "silent"),
and PASS, EXCLUDED or UNRESOLVED. Also record the listed or quoted individual-subscriber price,
and the history start date.

- Flash may fetch a given URL and return verbatim text spans.
- Codex judges every item, and verifies each Flash excerpt against the fetched page.
- No accounts, trials, sales contact or purchases.

### Task 3 — Per-signal status and owner decision memo (after 1–2)

Apply brief §5 and §8 A6–A7/A10 to the accepted evidence and the Task 2 ledger, for PEAD and for
insider separately.

Deliverable: `docs/research/2026-10-23-event-timing-audit-decision.md`, following the report
template, with a requirement matrix for each signal (PASS / FAIL / UNVERIFIED and the evidence
behind each).

- Per-signal status: FREE—VERIFIED, PAID—DOCUMENTED PATH, UNRESOLVED or NO VERIFIED PATH.
- For any PAID path, list the cheapest requirement-complete route combination, its annual cost,
  and its common qualifying window.
- End with the owner decision: buy (named data and cost) or park, for each signal.
  No recommendation may stop paper trading, retire a sleeve, or involve retirement accounts.

The registered `_event_timing_audit.json` is written only if the runner's completion guard
legitimately passes; otherwise the memo states why not.

### Task 4 — Independent review and closure

A separate Codex reviewer checks the Task 1–3 deliverables against the brief, the manifest and
the cited evidence: every number is traceable, and no status is overstated. Must-fix findings are
fixed and re-reviewed.

Then, in an add-only changelog entry plus `docs/next_steps.md` and `RESEARCH_SNAPSHOT.md`, record
the decision. The branch stays unmerged until the owner decides.

## Done when

The owner has (a) approved or rejected the Task 1 amendment, and (b) a reviewed memo giving
each signal's status and a priced buy-or-park choice.
