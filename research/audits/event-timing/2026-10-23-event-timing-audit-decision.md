# Event-timing audit: owner decision on data and timestamp convention

**Classification:** Internal Quantitative Research & Engineering Strategy

**Date:** 2026-10-23

**Audience:** Owner and quantitative research reviewers

**Disposition:** PEAD **UNRESOLVED**; insider cluster buying **UNRESOLVED**,
including its separately approved filing-date-only representation.

## 1. Executive Summary & Core Engine Audit

**Park both event-signal studies with the present evidence; no data purchase is
justified by a requirement-complete documented path.** This is a data-feasibility
decision under the [registered brief §5 and §8](briefs/2026-10-21-event-timing-audit.md),
not a new strategy-performance rejection. No common qualifying five-complete-year
window is established for either signal. Unresolved vendor evidence prevents
**NO VERIFIED PATH** under A10; priced execution-price components alone cannot
support **PAID—DOCUMENTED PATH**. The existing paper observation and production
freeze continue under the standing owner instructions.

This memo implements Task 3 of the
[resolution plan](../superpowers/plans/2026-10-23-resolve-event-audit.md).
Its inputs are the accepted
[workbook](2026-10-22-event-evidence-resolution-workbook.md),
[subsequent connected metadata and predecessor review](event-timing-audit-implementation.md#metadata-slice-cleared-exact-predecessor-acquired-and-compared),
the new [timestamp diagnostic](_insider_acceptance_diagnostic.json) and
[R1–R9 ledger](_event_source_search_ledger.json). Earlier workbook statements
about unavailable DB access, an unfetched predecessor, and only 24 timestamp
pairs are superseded by those later observations. Annual qualification is not.

The [Task 1 source amendment](briefs/2026-10-23-insider-acceptance-time-amendment.md)
is **DRAFT**. The offline diagnostic reproduces **25/25** header-consistent
pairs and **9/25** API-inconsistent pairs under its stated New York filing-hour
and same-filing-date tests. All **25/25** have the twice-absolute-offset
wall-clock pattern. These are all cached matching pairs, not the registered
25-filings-per-year sample. The SEC documentation citation remains
**UNVERIFIED** because `SEC_CONTACT_EMAIL` was unset; no SEC request was made.
Neither these counts nor owner approval of a source convention would establish
dissemination, population reconstruction or an annual A6/A7 pass.

## 2. Quantitative Performance Context

No returns, price changes, drift, P&L, new outcome summaries or return detectors
were calculated or inspected for this work. No OOS comparison against SPY is
produced, and no improvement over SPY is established. Historical PEAD and insider
NO-GO reports retain their historical status. Any future return test requires a
separately written, reviewed and owner-approved Phase 2 brief.

### Fixed qualification and evidence scope

For **each complete year**, A6 requires dated identity coverage ≥95%, successful
complete-inventory retrieval ≥95%, correctly mapped tradeable-date event prices
≥95%, and **both** unexplained-missingness ratios ≤2% (denominator units and
events separately). PEAD units are member-quarters and events are
security/fiscal-period earnings records; insider units are member-months and
events are reconstructed clusters. Failed requests, quarantines, unknown
tradeable dates and administratively censored events stay in their denominators.
Forward-window completeness is diagnostic only. Missing numerators or
denominators are UNVERIFIED, never zero or PASS.

A7 additionally requires every applicable mandatory annual predicate and the
longest contiguous qualifying run of at least five complete calendar years,
ties favouring the most recent run. The assessment ranges are **2002–2025** for
PEAD and **2015–2025** for insider; **2026 through April 30 is diagnostic only**.
The accepted workbook records the annual denominators, not passing numerators.
The frozen membership file's SHA-256 is
`b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472`.
The static Russell snapshot cannot qualify either signal.

The matrices below are documentary dispositions, not substituted runner output.
Unless a row explicitly says otherwise, UNVERIFIED applies to every complete
year in that signal's range. A local FAIL is scoped to its named observation;
it is not a fabricated failure rate or a claim that every year failed.

### PEAD requirement matrix

| Requirement / representation | Predicate and evidence | Status |
|---|---|---|
| A5 denominator provenance | Frozen PIT membership and annual member-quarter counts in [workbook JSON](_event_evidence_resolution_workbook.json), `population_denominators`; source hash rechecked for this memo. This passes denominator provenance only. | PASS |
| A6(a) historical identity | ≥95% dated security/issuer/ticker/CIK and correct vendor-price mapping. Dated issuer observations do not supply continuous population validity intervals; [metadata inputs](_event_audit_metadata_inputs.json) do not fill that gap. | UNVERIFIED |
| A6(b) retrieval / P3 | ≥95% resolved member-quarters have successful, nonempty matching earnings inventories. Attempted/resolved/fetched/stored annual counts and the required source-linked population are not established in the [workbook](2026-10-22-event-evidence-resolution-workbook.md#4-ownership-reconstruction-timing-and-qualification). | UNVERIFIED |
| A1/A3 announcement and availability / P1 | Original release matched to issuer/fiscal period, correct New York/NYSE clock, later of announcement and acceptance where both exist, distinct precision and inferred close. Date-only fallback does not certify an unknown original release. [R4 ledger](_event_source_search_ledger.json) documents fields, not matched historical release lineage; filing-regime handling remains required. | UNVERIFIED |
| A7 actual EPS / P2 | Original source receipt, issuer/period, GAAP or adjusted basis, currency and split basis agree. Stored `eps_reported` exists but its schema provides no such lineage: [workbook §5](2026-10-22-event-evidence-resolution-workbook.md#5-pead-lineage-and-retained-vendor-evidence), [metadata inputs](_event_audit_metadata_inputs.json). | UNVERIFIED |
| A3/A7 consensus / P2 | A snapshot strictly before the original announcement, matching period/basis/currency; vendor `surprise_pct` needs scaling, zero/negative-estimate, duplicate and missing-value reconciliation. [R1–R3](_event_source_search_ledger.json) do not establish a matched, accessible, delisted-inclusive historical package with all required lineage. Current/refetched agreement cannot certify PIT. | UNVERIFIED |
| A6(c) mapped event prices / P4 | ≥95% of all in-scope events have the correctly mapped price on a known tradeable date. Neither component price documentation nor stored row counts supplies the required annual numerator. | UNVERIFIED |
| A6(d) unexplained missingness | Both unexplained-unit and unexplained-event ratios ≤2%. Cause accounting and population numerators are absent; listing, merger/delisting, halts and administrative censoring must remain distinct. | UNVERIFIED |
| A7 tape integrity / P4 | Exactly one correctly mapped NYSE-session bar at each event date plus applicable existing detector screening, consumed without recomputation. [Recovered detector metadata](_event_audit_detector_provenance.json) lacks an explicit screened manifest; it cannot establish exhaustive clean intervals. R5–R7 prices do not themselves supply that evidence. | UNVERIFIED |
| A8/A9 quarantine and deterministic evidence | Retain all anomalous/failed records, fixed hash order, 12 events/year, and frozen classification before the 30 delisted/acquired + 70 other re-fetch selections. Full source-linked samples/classification are unavailable; operational examples are not substitutes. No new classification collection is pursued. | UNVERIFIED |
| A7 eligible window | Every mandatory row must pass annually over ≥5 contiguous complete years. No annual qualifying window is established. | UNVERIFIED |

**PEAD status: UNRESOLVED.** FREE—VERIFIED fails its evidence predicate;
PAID—DOCUMENTED PATH has no requirement-complete documented combination;
unresolved relevant routes bar NO VERIFIED PATH.

### Insider requirement matrix

The original timestamp and approved `insider_filing_date_only` representations
remain independently assessed. The proposed header representation is not an
approved audit input.

| Requirement / representation | Predicate and evidence | Status |
|---|---|---|
| A5 denominator provenance | Frozen PIT member-month denominators and source hash in [workbook JSON](_event_evidence_resolution_workbook.json), `population_denominators`; hash rechecked. Does not establish numerator coverage. | PASS |
| A6(a) historical identity | ≥95% dated issuer/class/ticker/CIK/vendor-security validity. Charter observations and a linked amendment lot cannot establish the population map. R7 expressly supplies current names/symbols rather than historical names; stable asset IDs alone do not fill the CIK link. | UNVERIFIED |
| A6(b) inventory / I3 | ≥95% of resolved units have complete Form 4/4-A lists; verified empty months count. NOC page001 has one filing dated 2017-03-02 beyond its 2017-03-01 descriptor. [Strict boundary predicate](2026-10-22-event-evidence-resolution-workbook.md#2-noc-inventory-incident) still FAILs; interval completeness and annual retrieval coverage remain unknown. No tolerance or waiver applied. | FAIL (local strict inventory check); annual coverage UNVERIFIED |
| A2/A8 original timestamps / I1 | Same-event API/header receipt agreement within one minute, otherwise retained conflict/quarantine. All 25 [cached diagnostic pairs](_insider_acceptance_diagnostic.json) remain conflicting under the existing interpretation. This is a failed reconciliation, not a demonstrated dissemination clock or population-wide timing rate. | FAIL (25 cached comparisons); annual timing UNVERIFIED |
| Approved filing-date-only timing / I1 | Exact corroborating API/header filing dates, uniform population representation, first NYSE close strictly after that date, no extra buffer; all other quarantines retained. [Approved amendment](briefs/2026-10-22-event-timing-source-conflict-proposal.md) remains binding. Cached corroboration supports convention diagnostics; full source-bound population execution/annual timing is not established. | UNVERIFIED annually |
| Proposed header timing / I1 | [DRAFT Task 1](briefs/2026-10-23-insider-acceptance-time-amendment.md) would add a uniform separate header-clock representation with A1 buffer. Its 25/25 consistency diagnostic is not authority to change A2/A8. It neither rescinds the approved date-only candidate nor clears original quarantines. | UNVERIFIED; owner gate pending |
| Dissemination / I1 | Acceptance is distinct from publication. No supported historical dissemination instant/bound is established by [R9 retained evidence](_event_source_search_ledger.json). Date-only convention compliance is not publication certification. | UNVERIFIED for all representations |
| Raw owners/lots/duplicates / I2 | Retain every reporting owner, roles/related entities, security classes, transaction codes and distinct lots; reconcile stored collisions/overwrites against raw sources. [Workbook](2026-10-22-event-evidence-resolution-workbook.md#4-ownership-reconstruction-timing-and-qualification) is a partial cache diagnostic, with no purchase-cluster demonstration. Stored parser/PK defects are Phase 2 build requirements, not proof that raw reconstruction is impossible. | UNVERIFIED annually |
| Amendments / I2 | Source-supported corrections apply only at their own availability, without replacing earlier history. [Reviewed specific link](_insider_predecessor_comparison.json) corroborates one F-lot share correction; it is not population amendment ingestion or a purchase-cluster replay. Repeated element identity does not establish unchanged referenced footnotes. | UNVERIFIED annually |
| Plan fields / I2 | Preserve PRESENT-TRUE, PRESENT-FALSE and ABSENT; missing does not mean discretionary/non-plan. [Workbook inventories](2026-10-22-event-evidence-resolution-workbook.md#4-ownership-reconstruction-timing-and-qualification) do not supply annual population regime counts. This diagnostic cannot justify selecting new filters. | UNVERIFIED at population scope |
| Load boundary / I3 | Complete carry-in and late disclosures, prior-year/amendment links; transaction-date loader boundaries cannot omit already-public information. Partial-cache checks and the one predecessor do not prove population completeness. | UNVERIFIED |
| Prefix invariance / I3 | Exact identical historical outputs at every A9 month-end close, separately by approved representation. [Workbook JSON](_event_evidence_resolution_workbook.json), `prefix_checkpoints`, retains all 136 decisions as UNVERIFIED for unavailable population inputs. No empty/partial replay promoted to PASS. | UNVERIFIED |
| A8 anomaly handling / I3 | Quarantine pre-1990, transaction-after-filing, future, missing/unparseable dates and unresolved clock conflicts without corrections or denominator removal. [Stored metadata counts](_event_audit_metadata_inputs.json) identify anomalies; full raw/population reconciliation remains unavailable. | UNVERIFIED annually |
| A6(c) mapped event prices / I4 | ≥95% of reconstructed clusters have correctly mapped tradeable-date prices. [23 operational filing-date bar counts](_event_audit_operational_bar_counts.json) are one-bar diagnostics, not clusters, historical identity or annual coverage. | UNVERIFIED |
| A6(d) unexplained missingness / I4 | Both unit/event ratios ≤2%, with failures/quarantines retained. Complete unit and cluster cause accounting is not established. | UNVERIFIED |
| A7 tape integrity / I4 | Unique correctly mapped NYSE-session bar plus applicable existing screening per event. [Detector provenance](_event_audit_detector_provenance.json) lacks the source-bound screened scope. The reviewed venue fix and metadata access do not clear this predicate. | UNVERIFIED |
| A9 sample and A7 window | 25 filings per year from the registered stored/enumerated frame, fixed hash order and recorded shortfalls; ≥5 contiguous complete years passing all annual predicates. Twenty-five cached pairs in total do not meet 25/year, and no window is established. | UNVERIFIED |

**Insider status: UNRESOLVED**, independently for original timestamp and approved
filing-date-only candidates. The DRAFT header candidate supplies no third
approved path. Unknown reconstruction inputs are not demonstrated prefix
failures, and no raw route is universally excluded.

## 3. Actionable Research Directions

### Public-documentation search and prices

The [machine ledger](_event_source_search_ledger.json) has nine routes in manifest
order and four evidence-item records each: fields, history/delisted-acquired,
PIT and individual access/price. Every item retains URL, retrieval interval,
excerpt or `silent`, predicate and rationale. R9 explicitly uses retained SEC
review evidence, not a fresh retrieval; the prior documentation 403 stop remains.
Task 2's documented search occupied **202 seconds**, with **378 seconds** elapsed
at ledger preparation, below its 7,200-second cap. Earlier historical cumulative
search time remains UNVERIFIED. Brief lead verification re-opened the three
existing Sharadar subscription/SEP and Norgate package URLs; no new route was
added. No account, trial, purchase, contact or SEC network request occurred.

| Route | Documentary route status | History start and individual price (USD) |
|---|---|---|
| R1 LSEG I/B/E/S | UNRESOLVED | Ordinary US 1976 / international 1987; matched PIT package start and individual price unknown. |
| R2 Zacks consensus | UNRESOLVED | Quarterly 1982; annual FAQ 1979 versus institutional-page annual/LTG 1978 retained separately. Individual historical entitlement/price unknown. |
| R3 S&P Capital IQ | UNRESOLVED | Ordinary North America 1999 / international 1997; PIT Snapshot August 2016. Individual Snapshot price unknown. |
| R4 Wall Street Horizon | UNRESOLVED | General history 2006; original-release-specific history and individual historical price unknown. |
| R5 CRSP daily stocks | UNRESOLVED | NYSE 1925-12-31; NYSE American 1962-07-02; Nasdaq 1972-12-14. Individual entitlement/price unknown; no academic-only exclusion established. |
| R6 Sharadar SEP | PASS, execution-price component only | Full history December 1997. Direct Prices 5/10/full: $9/$19/$39 monthly or $99/$199/$299 annually. Nasdaq entitlement/quote unknown. |
| R7 Norgate | PASS, Platinum/Diamond execution-price component only | Platinum 1990, $630/year ($346.50/6 months); Diamond 1950, $787.50/year ($433.13/6 months). Silver $270/year and Gold $360/year exclude delisted coverage. |
| R8 Sharadar SF2 | UNRESOLVED | January 2008. Direct Investors 5/10/full: $19/$29/$39 monthly or $99/$199/$299 annually. Nasdaq quote unknown. |
| R9 SEC EDGAR | UNRESOLVED | Free public access; electronic ownership filing mandatory 2003-06-30, not proof of complete inventory from that date. |

Sharadar's [direct subscription schedule](https://sharadar.com/subscribe) matches
history tiers to delivered tables; its schema identifies
[stocks as SEP](https://sharadar.com/docs/stocks). Full History Bundle is
$69/month or $499/year; the $29/month Bundle is five-year history. Those are
direct personal-use prices, not Nasdaq Data Link quotes. The
[Norgate package schedule](https://norgatedata.com/stockmarketpackages.php)
requires Platinum or Diamond for delisted securities; its FAQ cautions that
included security-type coverage is essentially complete only from late 1992,
with earlier exchange differences. A Windows updater/export requirement is an
operational constraint, not evidence that personal exported research is barred.

Execution-price PIT versioning is exempt under A10. The price-component PIT
records say PASS **because that property is not required**, not because the
vendor documents unrevised historical price vintages. Signal inputs still need
PIT evidence. R8's calendar-date field is excluded only as a substitute for an
intraday acceptance clock; its permitted date-only or SEC-supplemented routes
remain unresolved. No route is excluded just because documentation is silent.

### Cheapest requirement-complete combination

**None is documented for either signal. Annual cost and common qualifying
window are therefore unknown, not $0 and not an inferred cheap basket.**

For PEAD, every candidate combination needs R1/R2/R3 consensus, verified
original-release/actual-EPS provenance (R4 or supported SEC evidence), and
R5/R6/R7 prices plus dated security mapping and the remaining A6/A7 evidence.
Consensus entitlement/price and matched PIT/delisted lineage remain unresolved,
as do original release/actual-EPS matching, identity and tape scope.

For insider, R9 plus R6 or R7 is only a priced **partial** basket. R8 plus prices,
or R8 supplemented with SEC/date-only evidence, still lacks demonstrated
complete owners/amendment/PIT history, inventory/replay, identity and tape scope.
No such combination is documented as entirely ruled out either.

The cheapest listed ten-year price component is R6 at **$199/year**, so
R9 + R6 would list at **$199/year** before unresolved requirements. R6 full
history + free R9 lists at **$299/year**; Norgate Platinum + free R9 lists at
**$630/year**. R6 Prices + R8 Investors lists at **$398/year** for ten-year
tiers, while the ten-year Bundle lists at $399/year; full Bundle lists at
**$499/year**. These arithmetic component totals are not requirement-complete
offers. A rolling five-year tier purchased in October 2026 does not establish
five complete years before the fixed 2026-04-30 cutoff; exact tier delivery
starts and any common qualifying window remain unverified. No cheapest
complete purchase case can be ranked from unknown entitlements and lineage.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

This work does not reopen historical NO-GOs on performance. The reviewed
specific predecessor correction and metadata venue fix are already recorded;
do not repeat those acquisitions or treat their limited successes as annual
coverage. The original timestamp quarantine and approved date-only convention
remain distinct. Task 1 adds a DRAFT proposal only.

The resolution plan's stop rule applies: no new charter, predecessor,
identity-inventory or broad SEC collection, and no parser/runner/protocol work
beyond the Task 1 diagnostic and draft. New evidence needs listed here are
blockers, not a follow-on collection queue.

## 5. Operational Roadmap: Recommended First Action

**Owner choice for PEAD: PARK; do not buy data on this evidence.** Status is
UNRESOLVED, with no requirement-complete paid product combination, annual cost
or common qualifying window established.

**Owner choice for insider: PARK; do not buy data on this evidence.** Status is
UNRESOLVED for both currently approved representations. Separately, approve or
reject the [DRAFT header-source amendment](briefs/2026-10-23-insider-acceptance-time-amendment.md).
Its unresolved SEC documentation citation is explicit. Approval would authorize
a convention decision, not annual feasibility or Phase 2; no implementation is
performed in this branch's resolution work.

These are recommendations awaiting the owner, not recorded owner approvals.
The branch remains unmerged and unpushed. Paper observation, production freeze
and taxable-account standing decisions continue unchanged.

## 6. Contrarian Evaluation & Parked Research

**Could cheap delisted prices plus free SEC filings be enough?** They can cover
components at a modest listed price. They cannot establish the historical
security map, complete issuer-month inventories, amendment/carry-in/public-time
reconstruction or a source-bound detector-screened event tape. PEAD additionally
lacks strictly pre-announcement consensus snapshots and matched actual/release
lineage. Buying only prices would leave the decisive predicates unresolved.

The exact blockers are: dated historical identity/price mapping for both signals;
annual A6 numerator and missingness accounting; existing detector export with
source bytes and screened scope; full registered samples/populations; insider
complete inventory and prefix/amendment/load-boundary reconstruction; PEAD
release/actual/consensus lineage and matched individual entitlements/prices; and
the SEC-documentation citation and owner decision for Task 1. No further
collection or protocol change is pursued to resolve them here.

The registered `_event_timing_audit.json` and original final report are **not
written**. The [runner completion guard](../../src/ggTrader/lab/event_timing_audit.py)
at `AuditRunner.run` rejects canonical publication for smoke/offline/interrupted
work; completion requires a nonsmoke, nonoffline bounded registered execution
within its deadline. This task performs documentary synthesis, not that run.
Missing inputs and the collection stop prevent an authorized full execution;
Task 1/2 artifacts cannot be relabeled as its output. Source-search exhaustive
completion is also false while relevant documentation remains unresolved.

Verification for this memo: the Task 1 artifact was regenerated exactly, eight
focused tests passed through the capped wrapper, changed Python files passed
Ruff, the ledger's nine ordered routes and 36 evidence items were checked, and
the frozen membership hash was reproduced. Independent Task 4 review of
`58a1dcb..ddce4c4` reproduced the artifact, hashes, eight focused tests and Ruff,
and found no substantive correctness blockers. Its one minor R7 quotation
transcription issue was corrected to match the cited FAQ. Final closure
re-review is **CLEAN**, including that correction, the add-only changelog,
closure links, scoped snapshot/worklist updates and pending owner gates.
No database or new SEC request was needed for this synthesis.

**Decision submitted to the owner: PEAD — PARK; insider — PARK. No
requirement-complete purchase cost is established. Task 1 source amendment —
approve or reject separately; currently DRAFT and unimplemented.**
