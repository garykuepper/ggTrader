# Retrospective verdict inventory and review protocol

**Date / cutoff:** 2026-11-10  
**Status:** Initial replica, stock and defensive batches reviewed; **INCOMPLETE**.  
**Inventory file:** [inventory.json](inventory.json)  
**Source revision:** `e51d46ce806a29b2c055aab97cda6664243f2b9d` (HEAD when inventory work began).  
**Dirty paths observed at start:** the untracked plan and design spec for this framework, and
`research/programme/benchmark-objectives/`. Those pre-existing paths were preserved.

## Scope and counts

The primary historical source is the 114-row `classifications` array in
[`_lens2_sensitivity.json`](../../audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json).
Each row receives a stable ID `lens2-<classification id>` and retains its original audit ID,
verdict text, `verdict_type`, and class as source-reference metadata. These copied values are
provenance only; they are not new annotations or judgments.

| Source population | Count | Inventory disposition |
| --- | ---: | --- |
| Lens-2 NO rows | 53 | 52 reviewed; 1 excluded as an aggregate |
| Lens-2 GO rows | 16 | 16 reviewed |
| Lens-2 TEST-INCONCLUSIVE rows | 3 | 3 reviewed |
| Lens-2 STATUS rows | 42 | 41 excluded as status-only; live parity separately reviewed as an operational status |
| Later membership evidence notes (2026-11-08 and 2026-11-09) | 2 | Excluded, each says it makes no candidate verdict change |
| Prompt-kit review with an explicit verdict section (2026-11-05) | 1 | Excluded as a review-only status; source remains linked |
| Engine synthesis and benchmark-framework documents screened after audit | 3 | Excluded with documented scope reasons |
| Additional text-search document candidates | 68 | Excluded with source-specific reasons after review: summaries, methods, status or duplicate material |
| **Total inventory records** | **188** | **0 pending; 116 excluded; 72 reviewed/included** |

One Lens-2 STATUS row, live parity, was separately reviewed as an operational status instead of being excluded; it is not an edge test.

The 114 audit rows are not 114 independent edge tests. The audit itself defines 72 test rows
(53 NO, 16 GO, 3 TEST-INCONCLUSIVE) and 42 STATUS items, explicitly saying status rows are not
edge tests. The 41 STATUS exclusions preserve their audit classification and do not close or erase
the underlying study. Seventy-one of the 72 test records have been reviewed; the July programme summary is excluded as an aggregate of other results rather than another independent verdict.

## Discovery and reconciliation performed

1. Counted and extracted every row in the raw `classifications` array; asserted 114 rows and
   verified the split by `verdict_type` against the audit's own verdict-audit report, which
   reports 72 test rows and 42 STATUS rows.
2. Read the research snapshot roster (last full regeneration 2026-11-05) and research index,
   including the post-audit dated study files available through the 2026-11-10 cutoff. The
   snapshot and index do not yet list the two later membership evidence notes.
3. Read both later membership notes. The 2026-11-08 note says “No verdict changes”; the
   2026-11-09 note says it changes no code, reference run or stored list. They report data
   checks/evidence about existing references, not new candidate verdicts; both are explicit
   exclusions rather than silent omissions.
4. Reviewed the 2026-11-05 prompt-kit review and excluded it as a review-only status, not a strategy verdict. Its recommendation remains linked from the inventory.
5. Screened the 2026-11-05 vectorbt-engine synthesis and the two 2026-11-10
   benchmark-objectives reports. The index describes the engine work as methods/tooling and
   says no strategy verdict changed; the benchmark reports are framework/claim-verification
   inputs, say they generate no new study result or approved rule, and do not change existing
   verdicts. Each has an explicit inventory exclusion with its own source and reason. Other
   dated files found in the scan are already represented by the snapshot/index or are methods,
   evidence framework, design or feasibility/status material rather than a post-audit
   candidate verdict.

A text search across Markdown for verdict headings, GO and NO-GO found 68 documents not already referenced in inventory or annotation data; each was added as a discovery candidate and reviewed, with exclusion reasons recorded for summaries, methods, status and documentation. The [intraday](2026-11-10-discovery-intraday.md), [engine](2026-11-10-discovery-engine.md) and [other](2026-11-10-discovery-other.md) batches received fresh same-family Sol review.

A filename-level cross-check found 21 `research/**/*nogo.md` reports and verified that each path appears in the inventory or annotation data. This check does not discover verdicts in differently named or unindexed files and is not a substantive review of each report.

This is a bounded inventory reconciliation, not proof that every historical research artifact
is discoverable from these three indexes. The source snapshot predates some cutoff-date work,
and there may be unindexed artifacts. Newly identified items must be appended with stable IDs;
none may be silently dropped. The two later membership notes are separately counted even
though neither introduces a verdict.

## Source precedence and record rules

- The original study brief, report and raw artifacts govern that study's registered scope and
  result. The research index and snapshot are discovery aids; if they conflict with an original
  source, retain the discrepancy for review and do not infer a correction from the index.
- The Lens-2 JSON is authoritative only for what that audit classified. Its verdict strings,
  `GO`/`NO`/`STATUS` types and sensitivity class are recorded as audit provenance, not as the
  verdict register's evidence or action labels.
- Every discovered test remains `pending` until a source review checks its original brief,
  report and available evidence. Missing power is recorded as unknown power; it does not alone
  invalidate an observed result. A source gap that prevents checking a particular claim remains
  unresolved for that claim, with the missing artifact named.
- Only items explicitly shown to be status-only or outside the verdict-bearing population may
  be excluded, and each exclusion has its own source and reason. Exclusion is not a judgment
  that the underlying research is unimportant.

## Review sequence and verification

When annotation review is authorized and undertaken, prioritize: (1) the production-rule
replica's distinct daily-bar model result, stopped/inconclusive holdout and unverified live
parity; (2) current stock-signal closures and defensive studies; (3) remaining included
records. The replica scopes must be checked separately against its original brief, closure
amendment, report and audit. Then process remaining records in bounded batches, preserving
original verdict language and using source reports/artifacts for any annotation. Post-hoc
analyses must be labeled descriptive where appropriate; no new threshold can be retroactively
preregistered.

For each batch, a reviewer who did not prepare that batch should sample-check source identity,
classification counts and record-to-source paths. The lead then independently checks every
figure that will be cited, samples the batch, reads the changed records and reruns the
inventory/register checks. An independent Sol reviewer checked all 114 copied audit IDs and classifications and sampled the additional exclusions; this was a same-family review. The reviewer identified the live-parity status as requiring separate review and cautioned that later and unindexed sources remain incompletely reconciled. No statistical or return calculation was run.

Completion requires zero unaccounted discovered verdict-bearing items, all included records
reviewed, every exclusion supported by an explicit reason and source, and successful register
validation. All 188 identified items have been reviewed or explicitly excluded. The text search and index reconciliation remain bounded and cannot establish that every unindexed or differently worded historical verdict has been found, so discovery remains INCOMPLETE. Current state: **188 identified inventory records, 0 pending, 116 explicit exclusions, 72 included/reviewed**. See the [replica batch](2026-11-10-review-replica.md), [stock batch](2026-11-10-review-stock.md), [remaining stock batch](2026-11-10-review-remaining.md), [defense batch](2026-11-10-review-defense.md) and [crypto batch](2026-11-10-review-crypto.md), [stock allocation batch](2026-11-10-review-more-stock.md) [other allocation batch](2026-11-10-review-more-allocation.md), [ensemble](2026-11-10-review-more-ensemble.md), [strategy](2026-11-10-review-more-strategy.md) [additional studies](2026-11-10-review-more-studies.md), and [final inventory batch](2026-11-10-review-final-inventory.md); current JSON is authoritative for counts.

## Limits

No market data, database, broker, or return-test access was used. No source study or historical
artifact was changed. This inventory does not claim the lens audit covered post-audit work, does
not treat the audit's 114 records as independent tests. Reviewed records carry scoped evidence and action annotations without authorizing new progression. The 2026-11-08 and 2026-11-09 evidence notes add no new
candidate verdict; the prompt-kit review is excluded as review-only; later or unindexed work
remains subject to discovery and append-only inventory additions.
