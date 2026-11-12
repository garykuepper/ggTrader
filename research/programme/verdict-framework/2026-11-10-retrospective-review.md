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
| Archive text-search candidates | 6 | Excluded: 4 engineering/operations documents and 2 historical summaries of existing decisions |
| Coverage-review candidate scopes | 19 | 18 reviewed/included; pipeline discovery parent excluded in favor of run-level records |
| Lens 8 crypto scopes | 4 | May recalibration included; snapshot grouping, tournament component and March pipeline parent excluded in favor of scoped records |
| Additional stock specification scopes | 7 | Reviewed/included as standalone and blend decisions |
| Cross-asset forward shadow scope | 1 | Included as a separate UNRESOLVED stage; rule authority and qualifying observations remain unresolved |
| March pipeline run identifiers | 29 | 27 reviewed/included as reported attempts; 2 excluded as example/wiring |
| Further MidCap and possible archived sizing scopes | 4 | 2 MidCap stages included; blend mapping and possible 10% trial remain pending |
| Dated Kraken snapshot scopes | 3 | Included; one original report unavailable, two external reports hash-recorded |
| **Total inventory records** | **261** | **2 pending; 128 excluded; 131 reviewed/included** |

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
validation. The original 188 identified items have been reviewed or explicitly excluded; a subsequent archive scan identified six documents; four engineering or operations documents and two summaries of existing decisions are excluded with reasons. The text search and index reconciliation remain bounded and cannot establish that every unindexed or differently worded historical verdict has been found, so discovery remains INCOMPLETE. Current state: **261 identified inventory records, 2 pending, 128 explicit exclusions, 131 included/reviewed**. See the [replica batch](2026-11-10-review-replica.md), [stock batch](2026-11-10-review-stock.md), [remaining stock batch](2026-11-10-review-remaining.md), [defense batch](2026-11-10-review-defense.md) and [crypto batch](2026-11-10-review-crypto.md), [stock allocation batch](2026-11-10-review-more-stock.md) [other allocation batch](2026-11-10-review-more-allocation.md), [ensemble](2026-11-10-review-more-ensemble.md), [strategy](2026-11-10-review-more-strategy.md) [additional studies](2026-11-10-review-more-studies.md), and [final inventory batch](2026-11-10-review-final-inventory.md); current JSON is authoritative for counts.

## Limits

No market data, database, broker, or return-test access was used. No source study or historical
artifact was changed. This inventory does not claim the lens audit covered post-audit work, does
not treat the audit's 114 records as independent tests. Reviewed records carry scoped evidence and action annotations without authorizing new progression. The 2026-11-08 and 2026-11-09 evidence notes add no new
candidate verdict; the prompt-kit review is excluded as review-only; later or unindexed work
remains subject to discovery and append-only inventory additions.

The archive scan reviewed two historical summary documents against the register and original-source links. Their repeated verdicts remain governed by the existing study records; engineering and operational status items are not new candidate verdicts. Five linked research documents were not directly cited by the register: event-timing source/implementation/amendment, paper-freeze readiness, and accruals feasibility. These are source, operational or feasibility material, not newly completed strategy verdicts; this scan was not an exhaustive source audit.

A subsequent independent coverage review identified potentially collapsed candidate specifications, a historical pipeline experiment, and status exclusions containing comparative decisions. The initially identified candidate scopes were recorded as pending, not treated as validated independent verdicts; the pipeline entry may require further subdivision. Original family rows and existing decisions remain unchanged pending source verification.

The allocation/crypto and stock source audits identified four additional Lens 8 crypto records and seven separate PEAD, congressional and insider scopes; the latter have since been reviewed and annotated. Source-backed corrections were appended to the core/SPY, IBIT, funding-carry and A23 revisions. The dual and cross-sectional momentum replay annotations now distinguish arithmetic-return MDE from the CAGR gate; the combined dual record is unresolved until its historical decisions are separately represented. Existing revisions remain visible.

Opus cross-family review identified inconsistent action rationale across batches and uncertainty about French-screen registration, among other questions. Those require record-by-record adjudication; no blanket label rewrite has been made. Replica closure, stop, approval and shadow-stage source corrections are appended; a separate shadow scope is pending. MidCap STATUS exclusion retains its audit history with a correction pointing to the pending test scope.

A subsequent action-consistency review identified documented closure decisions that had been recorded as HOLD. Ten bounded historical specifications now have appended CLOSE annotations with their original sources; evidence and protocol validity remain separate. The momentum screen retains its unresolved registration-timing qualification and the shadow HOLD explicitly concerns unresolved scope and observation records. These changes authorize no continuation, trading or production change. The other pending scopes and remaining action records still require source-specific review.

Source reviews added individual stock-screen hypotheses, standalone and blend decisions, historical equity configurations, and twelve March pipeline comparisons. Family/study summaries remain traceable and must not be counted as additional independent tests. The fifteen remaining March attempts are now included as source-reported outcomes; matching reported results do not prove independent replications. Original Holm families remain unchanged.

A further Opus review identified missing family relationships, summary double-counting risk, and historical pipeline and MidCap scope limitations. Newly added records now link to their family where available; family summaries are labeled in the register and included totals are not independent-test counts. Four further MidCap/sizing scopes are pending. The March dry runs with reported outcomes are recorded as attempts; a dry-run label alone is not grounds for exclusion.

The March pipeline source review recorded all fifteen additional source-reported attempts separately, with no formal verdict invented; windows and execution identity remain unresolved. The excluded Phase-3B wiring sample also has a reported outcome, now recorded in its exclusion reason.

The crypto and MidCap source review added three dated Kraken snapshots, the May recalibration, and separate MidCap standalone and breaker-change records. The grouped snapshot and tournament selection-score records are excluded as grouping/component material. Two original Kraken reports and the May report were inspected outside this worktree and their hashes recorded; they are not packaged in this checkout. Weight-specific MidCap outcomes and the existence of a 10% equity trial remain pending.

A cross-family source review identified incompatible three-month and 12-month shadow rules, plus missing contemporaneous qualification evidence. The shadow is now separately recorded as UNRESOLVED/HOLD; an earlier parent description of observations is corrected by an appended revision. The later B1 stock standard has not been applied retroactively.
