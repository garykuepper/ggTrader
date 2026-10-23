# Event-timing audit implementation handoff — targeted review fixes

This is an implementation/re-review handoff, **not an audit report, collection
result, vendor approval or feasibility conclusion**. The original implementation
round started at `c964659`; this four-residual round starts at `4f6e3ab`, on the same
`research/event-timing-audit` branch. The
[approved brief](briefs/2026-10-21-event-timing-audit.md), especially A1–A12,
and [source manifest](briefs/2026-10-21-event-timing-audit-sources.md) are unchanged.

The earlier implementation follows all 14 findings in `/tmp/event-timing-rework-codex-review.md`
and the recovered original 17 in `/tmp/event-timing-original-code-review.md`.
The independent `/tmp/event-timing-vendor-evidence-review.md` supersedes the
unsupported vendor narrative. Neither vendor document is ingested as runtime
facts. No paid path, qualifying real window, price entitlement or guaranteed
SEC dissemination bound is established here. The independent follow-up
`/tmp/event-timing-fourteen-fix-rereview.md` judged the prior head **10 addressed,
4 partial (prior items 1, 2, 6, 10), 0 critical / 4 important**. This round changes
only those four confirmed residual blockers, pending fresh scoped re-review.

## Scope and operation

Changes are confined to the three audit modules under `src/ggTrader/lab/`,
focused tests and necessary documentation. The existing script delegates to
one parser/main. No dependencies, thresholds, strategy filters or detector
were added. No production, live-data, environment or container file was edited.
Existing external caches and untracked workspaces were left untouched.

The audit is offline by default. `--smoke` caps each selected stratum at two;
its PASS rows are downgraded and it cannot qualify a window. Smoke and offline
results remain partial `/tmp` JSON/Markdown. `--read-db` permits only explicit
read-only metadata/count/session-date SELECTs with a statement timeout;
connection strings and `.env` are never read into the audit artifact.
`--network` separately enables the counted SEC/Yahoo transport.

After separate authorization and input/cache readiness, the explicit final
output invocation is:

```bash
.venv/bin/python scripts/event_timing_audit.py --network --read-db \
  --evidence /path/to/reviewed-evidence.json \
  --detector-flags /path/to/existing-flags-export.json \
  --output docs/research/_event_timing_audit.json
```

This command was **not executed**. The adjacent registered Markdown report is
`2026-10-21-event-timing-audit-report.md`. A completed bounded audit may retain
FAIL/UNVERIFIED rows and an UNRESOLVED signal. Work completion, requirement
truth and manifest source-search completion are separate fields. Smoke,
offline and runtime-interrupted work cannot publish final artifacts. HTTP
stops, exhausted request budgets and missing documentation remain disclosed
unresolved evidence; they are not proof of infeasibility.

## Evidence validation contract

`event-timing-evidence-v1` remains the input envelope. A successful URL/hash
reference establishes byte integrity only. It cannot authorize historical
identity, original-release meaning, EPS lineage, entitlement or completeness.

Machine facts are reparsed from declared, checksummed raw responses using the
same collection parsers with a transport-free reference reader:

- SEC submissions must identify the requested issuer, include the pagination
  inventory, and supply every declared historical page with consistent date
  bounds. Accession syntax, duplicate/conflicting rows and historical-page
  intervals are checked. The supplied filing list and covered interval must
  match the reparsed inventory. The upper bound comes from actual manifest
  retrieval timestamps, not an annotation or the resume date.
- Ownership XML and headers must use canonical issuer/accession URLs. Issuer
  CIK is checked against the inventory; header readings require a matching
  header accession. Owners, lots, fingerprints, dates, plan states and raw
  completeness come from XML. Imported `raw_complete` flags cannot certify it.
- Yahoo inventories and release candidates are reparsed. Publisher metadata
  never automatically becomes a confirmed original release. PEAD 8-K receipt
  fields come from the matching inventoried accession/header, separately from
  reviewed original-release evidence.
- A detector wrapper must match its referenced raw flags-only export, including
  screened symbol/date intervals and all flags inside each interval. Unrelated
  checksum-correct bytes cannot certify screening. No detector is run.

Interpreted facts require **field-specific** `field_reviews`, for example:

```json
{
  "field": "valid_from",
  "value": "2000-01-01",
  "linkage": {
    "symbol": "A", "cik": "1", "security_id": "A-common",
    "valid_from": "2000-01-01", "valid_to": "2026-04-30"
  },
  "source_ref": {"url": "...", "sha256": "..."},
  "byte_start": 0, "byte_end": 123,
  "extract": "exact UTF-8 text at those decoded-response byte offsets",
  "reviewer": "actual reviewer identifier",
  "reviewed_at": "2026-10-21T00:00:00Z",
  "interpretation": "Why this specific extract establishes this scoped field"
}
```

Every used field needs exactly one receipt with the same typed value, exact
linkage, a nonempty reviewer/interpretation and parseable review time. Its source
must be among the record's successful checksum-validated references. Byte
bounds/extract must match exactly. The extract must contain the linked identity
and period/product context and literal scalar claims; a different period,
changed value or unrelated quotation is rejected. EPS receipts independently
bind issuer, fiscal period, value, basis, currency, split basis and consensus
snapshot time. Original-release observations need their own field reviews.
Routes bind product and representation; used documentary assessments/statements,
history, delisted/acquired scope and exclusion scope require their own receipts.
A decisive supported contradiction is evaluated before missing positive items.

These receipts make human interpretation concrete and reviewable; a reviewer
identifier is **not an expert certificate or proof of truth**. There is no global
trust/verified boolean. Unsupported facts stay UNVERIFIED. This implementation
accepts exact UTF-8 extracts; a PDF or other source without a supported extract
cannot silently pass. Actual review of real source meaning is still required.
Rejected candidates and their reasons remain in `input_errors` and affected
population uncertainty; dropping an annotation cannot manufacture an annual
population or a qualifying window.

## Four residual blockers after `4f6e3ab`

All four were reproduced against the reviewed code before editing. Their
implementation fixes and regressions are confined to the same audit path:

1. **Historical descriptor count.** `enumerate_sec()` now checks a supplied
   `filingCount` is an integer (not a boolean), nonnegative and exactly the
   number of rows returned by `filing_rows()`. That parser preserves every form;
   comparison occurs before form filtering or accession deduplication. A mismatch
   retains the page reference/bytes, observed rows and an explicit reason, with
   `complete=False`; imported revalidation therefore cannot certify it. No new
   coverage threshold or source assertion was added.
   `test_residual_historical_page_count_matches_all_parsed_rows` covers a missing
   third row with both interval endpoints retained, invalid count types/range,
   and a matching count including a non-Form-4 row.
2. **Supported-bound receipt comparison.** Timestamp PASS now requires both API
   acceptance and the correct-accession header acceptance, with the existing
   within-one-minute reconciliation. A supported bound alone cannot replace the
   independent reading. Missing or wrong-accession headers remain UNVERIFIED.
   The independently checked filing-date fallback remains permitted.
   `test_residual_bound_timestamp_needs_independent_matching_receipts` exercises
   missing/wrong headers, a matching-header positive control and the checked-date
   fallback. Its valid dissemination bound is synthetic, not real evidence.
3. **Executing code identity on resume.** Progress now binds the on-disk imported
   `index_constituents.py` helper as well as the three owned audit modules. These
   hashes include uncommitted file changes. Resume also rejects a changed Git
   HEAD and retains the actual current revision rather than overwriting it with
   a saved label. This deliberately rejects unrelated commits too: a single
   conservative registration rule avoids an exemption/dependency-analysis
   mechanism. Existing progress is preserved on rejection; no accounting is
   reset or cache migrated.
   `test_residual_resume_rejects_changed_revision_or_membership_code` simulates
   changed revision and changed helper bytes without editing the shared helper,
   creating commits or altering the branch; both must reject without changing
   progress bytes.
4. **Durable event observations.** `attach_population()` now completes each
   event's tape/diagnostic observations and query receipts, then forces its
   existing population checkpoint before starting the next event. This removes
   the two-pass tape staging and bypasses the normal 30-second checkpoint
   throttle for completed observations. Resume reuses those saved event results.
   `test_residual_completed_event_observation_survives_interruption_on_next_event`
   interrupts B's count query after A finishes. A's result and receipts must
   already be on disk; resume is forbidden from re-querying A and completes B.

No other review scope, strategy rules, thresholds, dependencies or production
paths changed. No subagent/reviewer or real collection was invoked. These are
fixture-verified implementation dispositions, not independent approval or a
feasibility result.

## Disposition of the 14 rework findings

The original builder dispositions below are retained as the 14-item map.
Independent review at `4f6e3ab` confirmed 10 and left 1, 2, 6 and 10 partial;
the preceding four-residual section records the targeted follow-up fixes.
These are implementation dispositions backed by offline fixtures, not external
review approval or real-data verification. Tests live in
[`test_event_timing_audit.py`](../../tests/lab/test_event_timing_audit.py);
`round2_...` below denotes the exact `test_round2_...` function name.

| # | Change and evidence |
|---|---|
| 1 | `validate_evidence_record`, `reviewed_fields` and field-specific consumers separate byte integrity from facts. Reparse issuer/accession/pagination/intervals; reject unrelated extracts and retain rejected populations. Regressions: `unrelated_bytes_cannot_authorize_identity`, `review_receipt_cannot_promote_unrelated_facts` (four cases), `inventory_is_rederived_not_asserted` (four cases). |
| 2 | Receipt reconciliation stays separate from dissemination. Without a reviewed bound, availability is null and execution uses the labelled next-session-close fallback. Checked filing-date fallback can pass; matching receipt times alone cannot. Regressions: `receipt_without_dissemination_uses_supported_fallback`, `receipt_agreement_alone_cannot_pass_mandatory_timestamp`. |
| 3 | Prefix/population readiness includes cumulative required disclosures, carry-in and earlier amendments, not just the current filing year. Unknown populations have null official denominators. Regression: `missing_prior_year_xml_blocks_later_population`, retaining an independent observed cluster. Existing later-amendment tests protect earlier years. |
| 4 | Complete `event-timing-accounting-v2` state is mandatory; required types, counters/reservations, runtime and stop state reconcile. Manifest responses require reservations. Regressions: `reduced_checkpoint_with_valid_manifest_is_rejected`, `manifest_requires_matching_reservation`. |
| 5 | `SafeRedirect` refuses every automatic redirect, including same-host, downgrade and non-443 targets. Responses remain unresolved within counted attempts. Regression: `redirects_are_not_automatic` (three cases); no endpoint call. |
| 6 | Progress binds detector, owned-module and imported membership-helper hashes, plus Git HEAD and the original protocol/input hashes. It persists stored snapshots/query metadata, frames, selected keys/outcomes, failures and stage progress. Resume reuses stored observations. Regressions: `resume_binds_detector`, `resume_keeps_stored_snapshot`, plus the original full resume fixture. |
| 7 | Identical complete empty reconstructions pass; no minimum-activity gate. Empty tape/collision aggregation requires explicit proven emptiness plus complete inputs. Regression: `proven_no_cluster_prefix_passes`; the complete-census fixture also checks honest zero-event years. |
| 8 | Stored losses compare each `(lot_id, owner)` pair; lot-level summaries derive from those pairs. Regression: `coowner_missing_lot_is_counted`. Raw recoverability stays separate from Phase 2 storage repair. |
| 9 | Malformed owners/fingerprints are retained as unassignable, excluded from reconstruction and reported UNVERIFIED. Regressions: `malformed_owner_survives_preparation`, `malformed_raw_owner_is_reported_by_full_orchestration`. |
| 10 | CPU unit boundaries check deadlines and checkpoint preparation, units, replay disclosures, prefix months, diagnostic periods and event observations. Indexes replace per-filing/per-period stored scans and repeated identity/inventory lookups. Lag counts use vectorized session searches. Regressions: `expired_cpu_preparation_checkpoints_remaining`, `expired_runner_emits_partial_without_cpu_or_db_work`, `cpu_replay_resumes_completed_disclosures`, `prepared_pead_resumes_without_recomputing_clocks`. Per-event tape/diagnostic observations and receipts now force a checkpoint before the next event (residual 4). No full-population benchmark was run. |
| 11 | Each selected PEAD key records a fresh-response comparison, unmatched/changed/missing/failed outcomes, fields, references and request failures without replacement. Matching is stability only; fiscal/issuer PIT lineage stays separate. Regression: `selected_pead_exact_key_reconciles_missing_and_changed`. |
| 12 | Scope uncertainty propagates to every affected year and official event population. Regression: `boundary_uncertainty_blocks_both_annual_populations`, with otherwise complete quarterly inputs. Temporarily restoring the old annual-only condition makes this test fail. |
| 13 | Completed bounded protocol work can publish truthful UNRESOLVED final artifacts while source search remains incomplete. Runtime interruption/smoke/offline stay partial. Regression: `completed_bounded_audit_can_publish_unresolved`, writing only below a synthetic `/tmp` root. |
| 14 | A supported contradiction scoped to all permitted representations can exclude a route without positive documentation of the other dimensions. Positive PASS still requires all applicable evidence; exact-time-only contradiction does not exclude fallback/mixed uses. Regression: `decisive_exclusion_does_not_require_positive_dimensions`; retained R8 fallback and corrected-price tests. |

No finding was rejected as invalid. A10 precedence is already fixed by the
brief and needs no owner decision. The two 63-session endpoint conventions
remain diagnostic only; they do not block audit implementation or qualification.

## Mapping to the recovered original 17

This replaces the earlier blanket disposition. “Implemented/tested” below is
limited to the code contract and synthetic tests; it is not a claim that real
facts or any year have passed.

| Original # | Implementation evidence after this round |
|---|---|
| 1 Coverage constants | Unit-derived arithmetic plus validated identity/inventory inputs; rework 1/3/12 and annual accounting tests. |
| 2 Full populations/event units | PIT units independent of samples; reconstructed clusters, null unknown denominators and cross-year uncertainty; rework 1/3/12. |
| 3 Raw versus stored defects | Raw collision recovery remains separate from storage defects; missing co-owner pairs now reported; rework 7/8. No DB repair. |
| 4 Acquisition/provenance | Existing counted parsers retained; raw facts rederived, semantic ingestion constrained, selected PEAD comparison implemented; rework 1/2/11. No actual sources collected. |
| 5 Prefix invariance | Existing 14-day/plan convention at all 136 NYSE closes retained; cumulative inputs, safe public-time representation and proven-empty comparisons; rework 2/3/7/10. |
| 6 Membership | Incumbents/intervening changes and unchanged periods retained with existing tests. No scope/threshold change. |
| 7 A10 routes | Fixed routes/common windows retained; field-specific evidence and decisive-exclusion logic; rework 1/14. Unknown relevant routes keep UNRESOLVED. |
| 8 Cutoff/censoring | Existing execution cutoff/diagnostics retained; deadline checks and affected-year propagation; rework 10/12. |
| 9 Quarantines | Existing date/conflict quarantine retained; malformed owners/fingerprints and rejected candidates remain visible/unused; rework 1/9. |
| 10 Sampling/classification | Existing union frames, quotas, hash order and shortfalls retained; frozen observations/selection and exact PEAD outcomes; rework 6/11. Real classification evidence missing. |
| 11 Detector availability | Missing/out-of-scope screening remains unknown; raw export content and resume hash now checked; rework 1/6. No detector computed. |
| 12 Runtime/resume | Counted retries retained; strict cumulative state, frozen observations and CPU stage checkpoints; rework 4/5/6/10. Full-scale cost unmeasured. |
| 13 Cache recovery | Existing immutable raw/hash defenses retained; reservations/manifest consistency and redirect refusal; rework 4/5. No automatic legacy recovery. |
| 14 Credentials | Existing omitted DSNs, sanitized errors and no-dotenv/read-only tests retained. No real secret or database used. |
| 15 Reproducible artifacts | Detector/code hashes, stored snapshots/frames, selected comparisons and honest final publication added; rework 6/11/13. |
| 16 Runner tests | Existing orchestration tests updated to authoritative synthetic source shapes; 31 additional regression instances cover the concrete rework defects. Verification below. |
| 17 Contiguous years | Existing missing-year/2026 exclusions and most-recent tie-break tests retained. No actual eligible window established. |

## Inputs and readiness still missing

- Real reviewed historical security↔issuer/tape identities, complete inventories,
  release/classification/amendment interpretations, actual-EPS and strictly
  pre-release consensus lineage, and a validated existing detector export.
- The independent vendor correction leaves both end-to-end paths unresolved.
  Product-history overlap and a delisted-price subscription are not measured
  A6/A7 qualification. No unsupported price, entitlement or historic window was
  promoted into the code or findings.
- v1/reduced accounting cannot resume automatically. Existing caches must be
  preserved and explicitly recovered from complete accounting, or a separately
  authorized new run must use an isolated cache. No recovery/migration was done.
- Offline fixture replay is supported. Bounded network collection still needs
  separate authorization, a properly configured authorized SEC contact and a
  reviewed cache/input plan. No contact was inspected or inferred.
- Full collection is not established ready: endpoints, actual request needs,
  complete-population runtime and the feasibility of collection within 20,000
  SEC / 2,000 Yahoo attempts and six hours remain unvalidated. Deadline checks
  operate at work-unit boundaries; they do not preempt a native parser or SQL
  operation already executing (SQL retains its statement timeout).
- Ambiguous amendment targets/re-signalling remain UNVERIFIED from their own
  public times. Cross-year/membership uncertainty is now applied consistently.
  No new role, size, private-purchase, independence or verified-non-plan gate
  was added. P and ABSENT preserve their documented limitations.

The owner/senior quant can re-review this implementation round. No reviewer was
invoked by the implementer. No Phase 2 return test, purchase, production change,
project shutdown or account change follows from this work. Existing NO-GOs and
production freezes remain unchanged.

## Verification record for the prior `4f6e3ab` round

- Initial regression run: 16 failed / 1 passed before fixes. The annual boundary
  test was subsequently strengthened; restoring its original defect produced
  1 failed / 107 deselected, then the fix was restored.
- Latest targeted run: `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q -k round2`
  — 31 passed / 79 deselected. Subsequent guard/metadata edits are covered by the
  final focused run below.
- Final focused run: `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q`
  — **110 passed in 131.41 s**. This includes the final guard/metadata changes.
- Exactly one broader run, after all behavioral fixes:
  `scripts/run_tests.sh tests/lab -q` — **902 passed, 29 deselected in 192.07 s**,
  with one existing `websockets.legacy` deprecation warning. It also covers the
  final line-wrapping-only lint fix.
- Ruff `check --no-cache` and `format --check --no-cache` passed for the audit,
  sources, reconstruction, focused tests and existing CLI script (5 files).
  `scripts/event_timing_audit.py --help` exited 0; `git diff --check` passed.
- Development iterations also executed: 17 targeted passes; one focused run
  with 95 passed/1 stale zero-event-status expectation; one expanded targeted
  run with 28 passed/1 fixture-order assumption. Both test expectations were
  corrected as described above; no failing check remains. Scratch logs are
  `/tmp/event-timing-round2-*.log`, `/tmp/event-timing-boundary-red.log`,
  `/tmp/event-timing-focused*.log`, and `/tmp/event-timing-lab-final.log`.
- Brief SHA-256 remains `4272da4069bb33640c0b2d7f621cb27969d97fa5d97f424d1b7ce01a9ad7d904`;
  source manifest remains `6f6660201a5b99e8799dd6252051c7dd724f36d216539cb0c7207c9f7d29e070`.

All test invocations used `scripts/run_tests.sh` with MemoryMax=8G and swap
disabled. No network request, real DB query/write, standalone collection/smoke,
return calculation, price change, new detector or full-population benchmark was
performed. All orchestration/replay evidence in this round is synthetic `/tmp`
fixture output, not the registered audit artifacts.

## Verification record for the four-residual round

- Pre-fix scoped run: `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q -k residual`
  — **11 failed, 3 positive controls passed, 110 deselected**. All four reported
  defects were reproduced; the tests did not merely assert implementation details.
- Post-fix scoped run: the same command — **14 passed, 110 deselected in 3.40 s**.
- Exactly one final focused run after the fixes:
  `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q` —
  **124 passed in 126.02 s** (the prior 110 plus 14 parameterized residual cases).
- Ruff `check --no-cache` and `format --check --no-cache` passed on the five
  audit/CLI/focused-test Python files. `--help` and `git diff --check` passed.
- Logs: `/tmp/event-timing-residual-red.log`,
  `/tmp/event-timing-residual-green.log`, `/tmp/event-timing-residual-focused.log`.
- Approved brief/source-manifest SHA-256 values remain exactly those recorded
  above; neither file was modified.

Test commands set `PYTHONDONTWRITEBYTECODE=1` and
`PYTEST_ADDOPTS='-p no:cacheprovider'` and always use `scripts/run_tests.sh`
(MemoryMax=8G, swap disabled). No broader lab run was performed: the changes remain
inside the audit contract, and the membership helper is only hashed, not edited.
The prior 902-pass broader result is historical, not rerun verification.

Real input prerequisites and bounded/full-run limitations above are unchanged.
This re-review/implementation round does not authorize a smoke or collection.

## 2026-10-22 Investigation and Discovery: Insider Evidence Tasks 1–2 & Task 3 Bounded Acquisition Proposal

**PROVISIONAL REPORT — substantive corrections required.** The Flash-authored
section below is retained for traceability, not accepted as verified findings
or an executable acquisition plan. The reviewed corrections following this
section supersede its counts, causal claims and proposed amendments. In
particular, no timestamp shift, two-day tolerance or paid-identity requirement
has been established or approved.

- **Date:** 2026-10-22
- **Branch:** `research/event-timing-audit` (HEAD `45ddee9`)
- **Working Cache:** `data/external/event_timing_audit/network-smoke-5b1q16b6/`
- **Scope & Constraints:** Read-only offline discovery on cached smoke evidence; zero network requests, zero DB mutations/queries, zero code/predicate edits. All evaluations execute pure offline parsers against raw files verified against immutable hashes.

### 1. Cache Integrity and Raw Hash Recomputation

All 50 raw response files in `data/external/event_timing_audit/network-smoke-5b1q16b6/` were re-hashed (SHA-256) and verified against `MANIFEST.json`. Every payload matches bit-for-bit:
- 1 bulk ticker snapshot: `sec_ace4a0fb...raw` (`https://www.sec.gov/files/company_tickers.json`) -> SHA-256 `bbffa40adbea620bba590da9a193ea14b21bfc04992f102549a2688d4e7d32df`.
- 4 submissions manifests:
  - Northrop Grumman main (`CIK0001133421.json`): `sec_42c56810...raw` -> SHA-256 `5ffdb48ff1025a815a3fefc28635e7eaef3ec24ffaf327055bbef59ff698ed2a`.
  - Northrop Grumman historical page 1 (`CIK0001133421-submissions-001.json`): `sec_b341aa5e...raw` -> SHA-256 `f1931d3c91bc70f0c411faaa6c3cf058e9bb493e7b7387e9422e8e586f3903cb`.
  - Costco Wholesale main (`CIK0000909832.json`): `sec_cd73bb9c...raw` -> SHA-256 `ddbc474ee1b766ecc36ba39759b1e2910135401dc12c08e93f034c65fc9c15d1`.
  - Costco Wholesale historical page 1 (`CIK0000909832-submissions-001.json`): `sec_aa493972...raw` -> SHA-256 `a9bb23d82576d50530a216bb5b22d46daeba57b8d61520435084b7b05e1a8650`.
- 22 Form 4/4-A XML primary documents (11 NOC, 11 COST).
- 21 SGML submission headers `.txt` (10 NOC, 11 COST).
- 2 reference disclosures: Form 8-K HTML (`0001133421-18-000024`), Form 15-12G HTML (`0001193125-03-023238`).
- Checkpoint ledger (`checkpoint.json`): Cumulative requests = 50 SEC (limit 20,000), 0 Yahoo (limit 2,000), monotonic elapsed runtime = 96.115s (limit 21,600s / 6.0h).

### 2. Task 1: Submissions Inventory Discrepancy & SEC Pagination Semantics

#### 2.1 Northrop Grumman (CIK 0001133421) Reproduction
- **JSON Paths & Payloads:**
  - Main Submissions (`CIK0001133421.json`):
    - `filings.recent`: 1,000 accessions spanning `2017-03-03` to `2026-10-20`. Earliest entry: accession `0001225208-17-005505` (`filingDate: "2017-03-03"`).
    - `filings.files[0]`: `{"name": "CIK0001133421-submissions-001.json", "filingCount": 2000, "filingFrom": "2002-06-24", "filingTo": "2017-03-01"}`.
    - `filings.files[1]`: `{"name": "CIK0001133421-submissions-002.json", "filingCount": 214, "filingFrom": "2001-02-01", "filingTo": "2002-06-22"}` (unfetched in smoke run).
  - Historical Shard 1 (`CIK0001133421-submissions-001.json`):
    - Length: Exactly 2,000 rows (matches declared `filingCount`).
    - Row 0: `accessionNumber: "0001225208-17-005319"`, `filingDate: "2017-03-02"`, `acceptanceDateTime: "2017-03-03T02:23:24.000Z"`, `form: "4"`.
    - Rows 1–1999: Span `2002-06-24` to `2017-03-01`.
- **Duplicate Check:**
  - Accession `0001225208-17-005319` occurs exactly **once** across the entire cache (in `submissions-001.json`). It is not present in `filings.recent` or any other file.
- **Exact Parser Failure:**
  - In `src/ggTrader/lab/event_timing_sources.py:429-430`:
    ```python
    if any(not page["filingFrom"] <= r["filing_date"] <= page["filingTo"] for r in rows):
        raise ValueError("Historical page interval mismatch")
    ```
    Because row 0 has `filingDate: "2017-03-02"` and `page["filingTo"]` is `"2017-03-01"`, `"2017-03-02" <= "2017-03-01"` evaluates to `False`. The `ValueError` is raised and caught at line 448 (`result["reason"] = "unparseable SEC inventory"`), which discards the entire 2,000 historical filings and returns only 1,000 recent filings with `complete: False`.
- **Primary Source Investigation & SEC API Guarantees:**
  - Researched via Vane query and verified against primary SEC documentation at `https://www.sec.gov/search-filings/edgar-application-programming-interfaces` (retrieved 2026-10-22, page updated June 6, 2024).
  - The SEC's official documentation defines the submissions data structure as:
    > *"If the entity has additional filings, files will contain an array of additional JSON files and the date range for the filings each one contains."*
  - **Findings:**
    1. The SEC provides **no formal schema definition or technical invariant guarantee** that `filingTo` and `filingFrom` strictly bound all individual rows.
    2. Upstream EDGAR partitions filings into fixed-size chunks of 2,000. In this case, filing `0001225208-17-005319` was accepted on `2017-03-03T02:23:24.000Z` UTC (`2017-03-02 21:23:24 EST`). Under Rule 13(a)(2), post-5:30 PM Form 4 filings receive the current day's filing date (`2017-03-02`). The SEC's automated batch job generated `filingTo: "2017-03-01"` despite placing a `2017-03-02` filing at the head of the historical shard.
  - **Classification:** **Documented source inconsistency** coupled with **unresolved pagination semantics**. The Northrop Grumman inventory status remains strictly `UNVERIFIED`.

#### 2.2 Costco Wholesale (CIK 0000909832) Contrast
- `CIK0000909832.json`: 1,002 recent filings (`2016-10-24` to `2026-10-16`). Shard 0 (`CIK0000909832-submissions-001.json`): `filingCount: 1731`, `filingFrom: "1994-01-05"`, `filingTo: "2016-10-22"`.
- Shard 1 row dates: `1994-01-05` to `2016-10-21`. The interval check `"1994-01-05" <= r["filing_date"] <= "2016-10-22"` passed cleanly.
- Result: 2,733 filings parsed with `complete: True`. (Note: completeness of submissions inventory is not identity or tradeable reconstruction proof).

---

### 3. Task 2: Evidence Checklist & Gap Matrix for Operational Examples

#### 3.1 Cached Evidence Inventory
- **Accessions & Document Pairs:**
  - 22 distinct accessions (11 NOC, 11 COST).
  - 21 accessions have both Form 4/4-A XML and raw SGML header `.txt` cached.
  - 1 accession has an incomplete pair: NOC `0001628280-25-006862` (XML cached as request #50; header `.txt` unfetched due to the 50-request smoke limit).
- **Form 4/A Amendments:**
  - 1 amendment in cache: NOC `0001225208-18-004211` (filed 2018-02-26).
  - Contains tag `<dateOfOriginalSubmission>2018-02-15</dateOfOriginalSubmission>`.
  - Reporting owner is Kenneth L. Bedingfield (`CIK 0001539147`).
  - The original 2018-02-15 Form 4 is not in the cache (there were 14 Form 4 filings for NOC on that date). In the absence of the original filing, the amendment cannot be linked to its predecessor (`linkage_unverified: True`), though protocol mandates evaluating it from its own public availability (`2018-02-26`).
- **Owner and Lot Inventories:**
  - Owners: All 22 cached filings have exactly 1 reporting owner.
  - Lots: 51 non-derivative lots (27 NOC, 24 COST), 8 derivative transactions (all NOC).
- **Rule 10b5-1 Plan States (`aff10b5One`):**
  - 18 filings: `ABSENT` (all 2015–2022 filings, pre-dating mandatory SEC checkbox rules).
  - 3 filings: `PRESENT-FALSE` (NOC `0001628280-24-034018`, NOC `0001628280-25-006862`, COST `0001209191-23-052699`, COST `0001209191-24-002111`).
  - 1 filing: `PRESENT-TRUE` (NOC `0001628280-25-029775`, filed 2025-06-05).
- **Acceptance Timestamp Reconciliation Defect:**
  - For all 21 filings with complete header/API pairs, normalizing timestamps to New York time reveals a systematic offset:
    - Summer filings (EDT, UTC-4): API `acceptanceDateTime` is exactly 8.0 hours ahead of header `<ACCEPTANCE-DATETIME>`.
    - Winter filings (EST, UTC-5): API `acceptanceDateTime` is exactly 10.0 hours ahead of header `<ACCEPTANCE-DATETIME>`.
    - When `clock.normalize_to_ny()` parses the API UTC timestamp and localizes the naive header timestamp, they differ by 4 to 5 hours (14,400 to 18,000 seconds), exceeding `reconcile_times()`'s 60-second limit.
    - Result: **100% of Form 4 filings are quarantined under `QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT`**, wiping `availability` to `None`.
- **Missing Referenced Inputs for Historical Security/CIK/Ticker Mapping:**
  - `company_tickers.json` is a single current snapshot. It contains no validity dates, delisted tickers, or corporate reorganizations.
  - Vane research confirmed that SEC provides no point-in-time ticker-to-CIK mapping endpoint. Historical ticker-to-CIK reconstruction from pure SEC feeds is only possible post-2019 via Form 10-K cover-page XBRL (`CoverPage.TradingSymbol`), leaving pre-2019 unresolvable without commercial cross-reference tables (CRSP/Sharadar) as specified in `docs/research/briefs/2026-10-21-event-timing-audit-sources.md`.
  - In the audit harness, `evidence["identities"]` is empty, so all historical mappings return `identity_resolved: False`.

#### 3.2 Gap Matrix: Raw Source Ability vs. Stored Database Defects
Comparing raw source capabilities against the existing stored implementation in `src/ggTrader/lab/form4_data.py`:

| Dimension | Raw Source Ability (EDGAR XML + SGML Header) | Stored Implementation Defect (`form4_data.py`) | Empirical Demonstration in Cached Evidence |
| :--- | :--- | :--- | :--- |
| **Form 4/A Amendments** | Fully preserved; contains `dateOfOriginalSubmission` and distinct accession. | Hard filter `forms[i] == "4"` silently drops all Form 4/A amendments. | NOC accession `0001225208-18-004211` dropped by stored loader; present in raw cache. |
| **Primary Key Collisions** | Each lot has unique document ordinal `accession:ordinal` and SHA-256 fingerprint. | PK is `(accession, insider_cik, transaction_date, transaction_code, shares)`. Collides on same-day, same-size lots. | **6 lot collisions** in NOC `0001628280-22-028934`; **2 lot collisions** in COST `0001209191-16-146844`. |
| **Multiple Reporting Owners** | All co-owners, officers, and 10% owners parsed from `<reportingOwner>` nodes. | Only takes first owner: `root.find("reportingOwner")`. Co-owners lost. | Stored loader structurally drops joint/co-owner disclosures. |
| **Rule 10b5-1 Plan Indicator** | Distinguishes `ABSENT` (pre-2023), `PRESENT-TRUE`, and `PRESENT-FALSE`. | Missing tag defaulted to `False` (`_find(root, "aff10b5One") or "false"`). | 18 pre-2023 filings falsely recorded as non-plan trades instead of `NULL`/`ABSENT`. |
| **Public Acceptance Time** | Recorded in `<ACCEPTANCE-DATETIME>` (SGML) and `acceptanceDateTime` (API). | Field not stored in database table; only `filing_date` (date only) stored. | Intraday decision availability and public dissemination cannot be verified from DB. |
| **Late Disclosures** | Raw data preserves transaction date and filing/acceptance date independently. | Querying by monthly `transaction_date` drops filings disclosed outside the calendar window. | Demonstrated lags of 3 to 13 days in cached lots (e.g. NOC 13-day lag on 2018 amendment). |
| **Point-in-Time CIK Identity** | Requires historical cross-reference (CRSP/Sharadar) or XBRL cover-pages. | Relies on static `load_ticker_cik_map()` from current `company_tickers.json`. | Survivorship bias; symbol reassignments and reorganizations unresolved. |

---

### 4. Task 3: Bounded Acquisition Proposal for Parent Review

#### 4.1 Cumulative Accounting and Ledger Status
Per `checkpoint.json` in `data/external/event_timing_audit/network-smoke-5b1q16b6/`:
- **SEC Requests:** 50 requests used of 20,000 ceiling (**19,950 remaining**).
- **Yahoo Requests:** 0 requests used of 2,000 ceiling (**2,000 remaining**).
- **Elapsed Runtime:** 96.115 seconds used of 21,600 seconds / 6 hours ceiling (**21,503.9 seconds remaining**).
- **Accounting Invariant:** Any future run or recovery invocation MUST inherit this cumulative accounting ledger. Starting a new output directory or updating code does NOT reset the budget ceilings.

#### 4.2 Exact Missing URLs for Operational Examples
To complete the evidence base for the two operational examples without touching the registered sample:
1. **Northrop Grumman Second Historical Submissions Shard:**
   - URL: `https://data.sec.gov/submissions/CIK0001133421-submissions-002.json`
   - Purpose: Completes historical filings inventory (214 filings, 2001–2002).
2. **Northrop Grumman Incomplete Receipt Header:**
   - URL: `https://www.sec.gov/Archives/edgar/data/1133421/000162828025006862/0001628280-25-006862.txt`
   - Purpose: Completes receipt pair for accession `0001628280-25-006862` (XML already cached).
3. **Northrop Grumman 2018 Amendment Predecessor:**
   - Target: Form 4 filed on `2018-02-15` by Kenneth L. Bedingfield (`CIK 0001539147`).
   - URLs: Document XML and `.txt` header from `https://www.sec.gov/Archives/edgar/data/1133421/` for the matching accession.
   - Purpose: Resolves amendment linkage for `0001225208-18-004211`.

#### 4.3 Proposed Bounded Acquisition Plan & A9 Ordering
- **Separation of Scopes:** Operational debugging of CIK 1133421 and CIK 909832 must remain strictly categorized as diagnostic examples and cannot substitute for the registered A9 sample strata.
- **A9 Hash Ordering Caveats:** For registered sample acquisition, accessions must be requested strictly in deterministic hash order as defined in `docs/research/briefs/2026-10-21-event-timing-audit.md` §8.
- **Bounded Request Ceiling:** The proposed next run should be bounded to at most **50 additional requests** (3 for operational completion, 47 for registered A9 sample), maintaining total consumption well below the 20,000 ceiling.
- **Proposed Protocol Amendments (Requiring Prior Owner Approval):**
  1. *Pagination Shard Boundary Tolerance:* Amend `src/ggTrader/lab/event_timing_sources.py:429` to allow historical shard rows to exceed `filingTo` by up to 2 business days when caused by post-5:30 PM evening filings under Rule 13(a)(2), or validate bounds using actual row min/max dates rather than unverified descriptor dates.
  2. *Acceptance Time Normalization:* Amend `src/ggTrader/lab/event_timing_audit.py` to reconcile EDGAR SGML header `<ACCEPTANCE-DATETIME>` (recorded in Eastern Time) against submissions API `acceptanceDateTime` (which exhibits double UTC-offset application) before evaluating the 60-second conflict window.
  3. *Amendment Required:* Under audit guidelines, no code or predicate edits may be made without a signed dated protocol amendment. Code remains untouched.

## Reviewed corrections to the 2026-10-22 Flash investigation

Independent cached-evidence review: `/tmp/insider-evidence-flash-investigation-review.md`.
Its checks were offline: no new network request, DB access or test execution.
Raw evidence remains in `data/external/event_timing_audit/network-smoke-5b1q16b6/`;
exact URLs and hashes resolve through its `MANIFEST.json`. Saved smoke output:
`/tmp/network-smoke-5b1q16b6.json`. Preserve these references when carrying
temporary review notes into a future session.

### Accepted observations and corrected denominators

- All 50 raw response hashes match. Durable accounting records 50 successful
  SEC attempts and zero Yahoo attempts; no further request is implied here.
- Northrop's descriptor/page date disagreement is observed. Its cause and
  descriptor semantics remain unknown. The page's 2,000 raw records remain
  cached and referenced; they were not merged into the parsed inventory.
  No source data was deleted. Costco's 2,733-record completeness is a parser
  result, not a validated historical security identity or qualifying year.
- The saved evidence has 24 distinct filing records: 22 XML-backed records,
  including 21 XML/full-submission pairs and one missing full submission;
  two further records lack both bodies. `raw_complete` is not pair completeness.
- XML contains 50 non-derivative lots (27 Northrop, 23 Costco) and 11 derivative
  transactions. The latter are not a reconstructed derivative event population.
- Filing-level plan states are 17 ABSENT, 4 PRESENT-FALSE and 1 PRESENT-TRUE.
  Lot-level states are 41 ABSENT, 8 FALSE and 1 TRUE. One ABSENT filing dates
  from February 2023; absence must not be imputed as verified non-plan.
- Under the existing clock, API minus header differences are +14,400 seconds
  for 14 pairs and +18,000 seconds for seven pairs. All 21 paired comparisons
  conflict under the frozen 60-second rule. The actual saved run quarantines
  21 of 24 filing checks and 46 of 50 lots for timestamp conflict; three unpaired
  checks remain date-only UNVERIFIED. It does not quarantine every filing.
- Example: accession `0001209191-15-021737` has API time
  `2015-03-04T04:29:20.000Z` and header `20150303182920`. Existing normalization
  yields 23:29:20 versus 18:29:20 on March 3 in New York, a five-hour mismatch.
  Offset patterns do not prove which source interpretation is correct.
- The raw lots demonstrate potential upsert-key loss of six extra rows in one
  Northrop accession and two in one Costco accession. These are hypothetical
  storage losses under the schema, not measured DB corruption. Each of the
  22 XML documents has one owner; this sample does not demonstrate co-owner loss.
  Actual stored rows were never queried. Raw filing/transaction gaps span 0–13
  calendar days, not proof of a particular regulatory or load-boundary failure.

### Withdrawn conclusions and next action

- Withdraw the asserted SEC batch-job/late-filing cause, a universal fixed-size
  shard claim, any supposed proof of a double-offset API bug, and the proposed
  two-business-day tolerance. The source-format interpretation is unresolved.
- Withdraw the claim that free identity reconstruction is only possible after
  2019 or that earlier mappings necessarily require paid tables. This run has
  no reviewed historical identities; older primary filings were not exhausted.
- A protocol-preserving implementation correction needs tests/review, not an
  automatic protocol amendment. A frozen-predicate change still needs approval.
- The acquisition proposal is not ready. Its three-request arithmetic omits
  the predecessor XML/header pair: at least four successful requests are needed
  even if the predecessor is identified, plus retries. Fourteen indexed
  predecessor candidates exist; none is yet verified. Do not fetch all candidates
  or reserve 47 sample requests from that faulty arithmetic.
- Next action is a targeted primary-documentation investigation of the timestamp
  representation and historical-page descriptors, followed by a corrected
  acquisition proposal with cumulative accounting. No automatic timestamp shift,
  relaxed interval check, extra download tranche, purchase or feasibility verdict
  is authorized by the investigation. Both signal paths remain UNRESOLVED.

## Primary-document search disposition after the corrected investigation

Flash's follow-up report is `/tmp/insider-sec-field-semantics-primary-research.md`.
It supplies possible document leads, not an approved protocol correction.

- The report records an HTTP 403 on a SEC documentation request. Further external
  research/collection is paused under the first-403/429 stop rule. Its narrative
  shows subsequent research despite that stop; those results cannot be treated
  as compliant completion of the assigned search. Do not restart collection
  under a different cache or silently reset the stop condition.
- The report's claimed PDS/FAQ quotations and legal citation need verification
  against successfully retrieved source text. Its retrieval table does not show
  a successful open of the cited PDS PDF. Search snippets, synthesis and asserted
  absence of a schema do not establish field semantics or universal guarantees.
  The execution timeline also does not independently substantiate the reported
  end time. Do not adopt its precision or its claims of exhaustive source absence.
- The suggestions to prefer header time unconditionally or treat actual array
  dates/counts as automatically authoritative are not accepted. A2/A9 require
  the specified receipt comparison, and A8 preserves unresolved timestamp
  conflicts. Existing clocks, count checks, interval checks and quarantines
  remain unchanged; no automatic shift or wider tolerance is authorized.
- The useful established result remains the cached discrepancy itself: 21
  paired clock comparisons conflict under the existing interpretation. The
  investigation has not established a correct alternative representation.
- Next work can proceed offline: preserve the exact paired raw observations
  and source leads, and formulate a reviewable proposal if changing source
  precedence is warranted. Any renewed external collection or frozen-protocol
  change needs an explicit, documented decision; it is not a routine retry.
  No verified free or paid path, qualifying window or performance conclusion
  follows from this documentary search.


## 2026-10-22 approved filing-date amendment implementation

**Status: offline implementation and synthetic verification; fresh independent
code review pending.** Owner approval is recorded in the immutable
[dated amendment](briefs/2026-10-22-event-timing-source-conflict-proposal.md).
This section supersedes implementation-status claims for that amendment only.
It is not a real-data assessment or full-run readiness decision. Parent owns
fresh review; the implementer invoked no subagents.

Implemented from `6e16e256119a20da094ca334bba9e0e6bcb89407` on
`research/event-timing-audit`, with the approved amendment SHA-256
`a8d1cee1f4c2226d7eda0f28bbbcfc156ae1d55d94a5a0fef9354917bef9297d`.
The original brief and source manifest remain byte-for-byte unchanged.
External requests remain paused after the recorded 403. No old smoke cache,
checkpoint, result or global accounting was modified, resumed or migrated.

### Amendment item mapping and interfaces

1. **Uniform scope:** `prepare_insider(representation="insider_filing_date_only")`
   traverses the same complete evidence list as the original preparation; it
   never selects by receipt outcome, issuer, year or performance. Existing
   filing frames, rejected inputs and cumulative readiness retain failed and
   unknown populations. `evaluate_signal` reuses all A6/A7 gates and definitions;
   cluster counts/dates are computed separately and need not match the original.
2. **Raw corroboration:** `filing_date_corroboration` reparses the declared,
   checksummed submissions inventory and canonical XML/header evidence via the
   existing transport-free `ReferencedSources` parsers. No imported boolean
   proves a date. Exact API/header filing-date equality, issuer and accession
   are required; annotated dates must also match the raw fields. The optional
   `parse_filing_header(..., issuer_cik=...)` check uses the header's ISSUER
   section, bounded before document bodies. Missing/corrupt/unsupported evidence
   stays UNVERIFIED; contradictory dates stay quarantined. Original header
   parser callers retain their prior behavior.
3. **Execution:** the amended `event_clock` calls the existing NYSE date-only
   helper directly: first regular session close strictly after the corroborated
   filing date, including weekends, holidays and early closes. No second buffer.
   `precision="date_only"`, `availability=null`, `tradeable_date` and
   `tradeable_close` remain separate; lots/amendments use the existing
   `decision_eligibility_time` convention. Prefix reconstruction uses that same
   eligibility time at all 136 registered NYSE month-end closes.
4. **Conflict preservation:** amended filing clocks retain the untouched receipt
   fields, `original_clock` and `acceptance_reconciliation` (including FAIL).
   Only receipt-time conflict is bypassed for this representation; invalid or
   future dates and every transaction-date quarantine remain enforced. Original
   insider matrices, quarantines, raw lots and dispositions remain reported.
5. **No certification:** new results expose `timing_pass_basis` as declared
   date-only convention compliance and `dissemination_predicate="UNVERIFIED"`.
   Reports state the same limitation. No observed public timestamp or bound is
   invented; a later close is not a demonstrated dissemination bound.
6. **Separate qualification:** top-level `insider_filing_date_only` contains its
   own `requirement_matrix`, `annual_coverage`, `events`, `raw_reconstruction`,
   `prefix`, diagnostics, `source_search`, `status` and eligible window. Original
   `insider_cluster` retains its schema/content. Preparation, cluster replay,
   prefix, lag/boundary diagnostics and event-observation progress use separate
   representation keys; identity/member-month units remain shared. Missing clock
   corroboration also blocks completeness for a zero-lot filing. Unknown event
   totals remain null with observed lower bounds; no lots become event counts.
7. **Ordering and PEAD:** A9 frames/keys/hash ordering/quotas, all identity,
   inventory/count/interval, amendment/collision/load-boundary/tape checks,
   cutoff and five-complete-year requirements remain in force. PEAD is unchanged.
   There is no header precedence, offset repair, tolerance change or Northrop
   interval relaxation.

`audit_meta.approved_amendment` and progress `approved_amendment` record the
amendment identity, path and SHA-256. Resume rejects changed/missing policy
binding in addition to the existing input/code/HEAD/detector/protocol bindings.
There is no automatic legacy registration, budget recovery or cache migration.
No new CLI flag, strategy, dependency or parallel runner was introduced.

### Verification and limitations

Pre-change complete synthetic assessment SHA-256 values were captured before
implementation and are asserted by `test_amendment_original_fixture_regression`:

- Original insider: `b7514e570e98469a7dc12efb3e4b0d7abaebee5cf3f71d1b402899fec3abf042`.
- PEAD: `a1a43816918377d254cefc2ee28947b487b73aa14b1b3546c854c44956a0469c`.

These hash the entire sorted JSON assessments, not a selected count. Synthetic
receipt-conflict tests separately retain the original quarantine and unmodified
receipt fields beside the amended clock. No real-data conclusion follows.

The first amendment selection reproduced 22 missing-feature failures, with seven
existing/positive cases passing. After initial implementation, one expectation
incorrectly required an overall UNRESOLVED disposition despite a qualifying,
fully empty synthetic 2015–2022 census; it was corrected to assert the affected
2023 population/matrix remains unknown and outside that window. A subsequent
zero-lot regression failed on false population completeness before its fix.

Final focused verification:
`scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q` — **153 passed
in 220.13 s**, after strengthening the raw receipt-conflict and supported-bound
positive fixtures. The preceding complete focused run also passed 153 tests
in 226.39 s. All test invocations used the required 8 GB cgroup cap with swap
disabled. Exactly one broader run, after all implementation/fixture fixes:
`scripts/run_tests.sh tests/lab -q` — **945 passed, 29 integration tests
deselected in 280.76 s**, with one existing `websockets.legacy` deprecation
warning in the headline-news pagination test. No failing check remains.

Ruff `check --no-cache` and `format --check --no-cache` passed on the three audit
modules, focused tests and existing CLI wrapper (five files); the wrapper's
`--help` exited 0 and `git diff --check` passed. A 101-character reconstruction
docstring found in an intermediate lint run was shortened before these checks.

Not exercised: external endpoints, real DB, old smoke replay, full-population
runtime, real identity/source semantics, returns or price paths. Raw header
formats without a supported issuer section remain UNVERIFIED. Per-filing raw
corroboration reparses issuer inventories; complete-population cost is unmeasured.
No collection restart, qualifying real window or independent code approval is
claimed. Both real end-to-end signal paths remain UNRESOLVED.

## Date-only independent review and first cached comparison

The independent review of `740781a` cleared all seven amendment items with
zero critical/important findings. It ran 34 amendment tests successfully;
119 other focused tests were deselected. Review artifact:
`/tmp/insider-date-only-implementation-review.md`. The preceding implementation
section's review-pending limitation is superseded by this scoped approval for
offline representation use, not collection or data-feasibility approval.

The parent then ran the pure `event_clock` comparison over all 24 filing records
in `/tmp/network-smoke-5b1q16b6.json`, using the existing checked raw cache with
an offline client. Result: `/tmp/insider-date-only-cached-comparison.json`.
Twenty-one records have corroborated raw filing-date pairs and satisfy the
amended timing convention; three remain UNVERIFIED because a required raw pair
is unavailable. This is per-filing timing only, not annual or cluster coverage.

All 24 amended records retained their original clock result, kept observed
availability null and dissemination UNVERIFIED. The comparison independently
hashed every existing cache file before/after and confirmed no change. It made
zero external requests and zero DB queries, calculated no returns or prices,
and did not resume or rewrite the old smoke registration. Original failures
remain intact; identity, inventory, reconstruction and tape gates still need
their own evidence. Both end-to-end data paths remain UNRESOLVED.

## Offline identity observations and exact missing support — 2026-10-22

The parent extracted issuer CIK/name/trading symbol and non-derivative security
titles directly from the 22 cached ownership XML bodies. Each response hash and
issuer CIK was checked; the membership comparison used the existing pure
`sp500_members_asof()` helper and static historical membership file. Original
cache hashes remained unchanged. No network, DB, prices or return calculations.
Detailed observations and canonical source URLs/hashes are in
`/tmp/insider-cached-identity-and-missing-evidence.json`.

- Northrop: 11 event-local NOC/CIK 1133421 observations from 2017-05-12 through
  2025-06-05, including three before 2019.
- Costco: 11 event-local COST/CIK 909832 observations from 2015-03-03 through
  2024-01-23, including five before 2019.
- All 22 reported symbols appear in the PIT membership snapshot in force on
  their respective filing dates. This compares metadata only, not prices.
- These are discrete source assertions, not proof of continuous validity,
  corporate-action/share-class continuity or linkage to the price security.
  No validity interval or reviewed `identities` record was manufactured from
  first/last observations. The earlier categorical claim that useful free
  historical identity evidence is unavailable before 2019 is unsupported by
  these cached counterexamples.

The three incomplete records in the amended clock comparison have these gaps:

| Accession | Filing date | Missing raw support | Cutoff treatment |
|---|---|---|---|
| `0001628280-25-006862` (Northrop) | 2025-02-20 | Full submission/header; XML already cached | In-window candidate, timing remains UNVERIFIED |
| `0000909832-26-000039` (Costco) | 2026-04-01 | XML and full submission/header | In-window candidate, timing and raw transactions unknown |
| `0001628280-26-037192` (Northrop) | 2026-05-21 | XML and full submission/header | Already after 2026-04-30; retain A9 selected failure/shortfall, cannot produce an in-window event |

Three successful document requests would supply the missing bodies for the two
in-window examples, before retries. This is a count of known missing URLs, not
an authorization or a claim that retrieval will resolve them. Northrop's missing
historical page, descriptor inconsistency, amendment predecessor, full interval
identity evidence and existing detector export remain independent gaps.

External requests remain paused. Before any restart, the acquisition decision
must explicitly address the reported 403, cumulative reservations/runtime and
registration lineage; neither a fresh cache nor the date-only amendment clears
the stop. No strategy test or real qualifying window follows from this work.

## Limited three-document restart and corroboration results — 2026-10-22

The owner instructed “Continue” after the explicit limited-restart decision was
presented. The [dated batch authorization](briefs/2026-10-22-three-document-restart.md)
and `/tmp/insider_three_document_restart.py` received a scoped independent review
with zero blockers. The parent executed only its three canonical URLs.

- All three responses were HTTP 200, requiring three attempts and no retries.
  Cumulative collection reservations advanced from 50 to 53, Yahoo remained
  zero, and elapsed accounting advanced from 96.1154 to 97.4432 seconds.
- Child cache: `data/external/event_timing_audit/three-document-restart-sc5zqn6c/`.
  It retains the original 50 raw response copies and accounting prefix, plus
  the three new responses. All 53 manifest hashes were checked. The parent
  smoke directory's file hashes remained identical. The child registration is
  new transport work, not a resumption of the old smoke assessment.
- The persistent `three-document-restart-20261022-claim.json` prevents executing
  this authorization twice. The reported documentation 403 remains in the
  history; the absence of a new stop is not a general collection restart.
- Northrop `0001628280-25-006862`: the newly retrieved full submission/header
  corroborates filing date 2025-02-20. Amended execution is 2025-02-21 close.
- Costco `0000909832-26-000039`: newly retrieved XML and full submission/header
  corroborate filing date 2026-04-01. Amended execution is 2026-04-02 close.
- Both original clocks still report unresolved receipt-time conflicts. Neither
  the raw values nor the original quarantine were changed. Amended observed
  availability remains null and dissemination remains UNVERIFIED.
- Reapplying the pure clocks offline to all 24 operational records now gives
  23 complying with the amended date-only convention and one UNVERIFIED. The
  remaining Northrop record is dated 2026-05-21, after the cutoff; it stays in
  the selected record/shortfall accounting and was not fetched or substituted.

Results: `/tmp/insider-three-document-restart-result.json` and
`/tmp/insider-three-document-restart-offline-comparison.json`. All child cache
files were hashed before/after the offline comparison and remained unchanged.
No DB query, price, return, strategy test or detector calculation occurred.

This resolves the known missing in-window timing pairs in these operational
examples. It does not establish continuous historical security identity, complete
raw inventories, amendment linkage, price-security mapping, detector coverage,
annual qualification or end-to-end data feasibility. Northrop's historical-page
discrepancy remains unresolved. Both signal paths remain UNRESOLVED. The one
bounded request authorization is exhausted; further collection is not queued.

## Offline acquisition footprint and full-submission equivalence — 2026-10-22

The parent counted all observed Form 4/4-A index rows with filing dates from
2015-01-01 through 2026-04-30. This is a filing inventory diagnostic, not a
cluster-event population or execution-date filter. It omits required carry-in
disclosures and does not establish dated identity or a qualifying window.

- Northrop's cached pages contain 848 such rows (820 Form 4, 28 Form 4/A).
  The interval inconsistency and unfetched older page still prevent the current
  parser from certifying inventory completeness. Observed rows are retained as
  a lower-bound diagnostic, not promoted to verified retrieval coverage.
- Costco's cached complete parsed inventory contains 720 such rows (661 Form 4,
  59 Form 4/A). Its inventory completeness alone does not pass other gates.
- With separate XML/full-submission requests, 3,090 canonical document URLs
  for these observed rows are not cached (1,674 Northrop, 1,416 Costco), before
  retries. This describes two operational issuers, not an estimate of the
  entire universe or permission to spend the remaining request budget.

The parent then extracted the XML document from each of 23 cached complete
XML/full-submission pairs, selecting exactly one matching TYPE and primary
FILENAME. Existing `parse_form4()` results matched the separately downloaded
XML exactly in all 23 cases, including Form 4/A. Hashes were verified and every
cache file remained unchanged. Evidence:
`/tmp/insider-full-submission-xml-equivalence.json` and
`/tmp/insider-cached-inventory-acquisition-footprint.json`.

This supports a minimal collector efficiency change: reuse already-cached
primary XML if available; otherwise request the canonical full submission and
extract the matching ownership XML there, with a separate primary-document
request only as a validated fallback. Preserve source references and fail closed
on ambiguous/wrong-identity content. API/header corroboration and all approved
timing, quarantine and completeness rules remain unchanged. Implementation and focused tests are recorded below; independent review
identified the follow-up described at the end of this handoff.
No new collection is authorized; the known uncached full-submission count alone
would still be 1,545 before retries and carry-in requirements.

## 2026-10-22 offline full-submission collector efficiency

Implemented on `research/event-timing-audit`, starting from `5371bba`. Runtime
changes are confined to `collect_filing()` and the small `embedded_form4()`
document selector in `event_timing_sources.py`. No change to the approved
protocol, `event_timing_audit.py`, parsers' timing interpretation, qualification
gates, CLI, dependencies or production code was needed.

Acquisition and provenance:

- A successful checksummed canonical primary XML already in the client's
  declared cache is reused first. Its parsed result and primary/full-response
  reference ordering remain unchanged, even when a full response contains
  different embedded ownership content.
- Otherwise the canonical accession full submission is obtained first. Exactly
  one DOCUMENT must match both the API form and primary filename basename.
  Same-filename exhibits and same-type different files are ignored. Supported
  embedded ownership XML is passed once through unchanged `parse_form4()`;
  issuer identity and the full header's accession/issuer are checked. A valid
  full response supplies both XML and header with one counted request and one
  actual full-response reference, without inventing a primary XML download.
- A header-only response, unrelated documents, or a correctly framed matching
  document without an XML wrapper uses the canonical primary URL as fallback.
  With successful uncached responses this costs two counted requests. Missing
  or wrong header evidence cannot establish filing-date corroboration.
- Duplicate matches/metadata/XML wrappers, malformed DOCUMENT/TEXT/XML framing,
  invalid ownership XML, DTD/entities and wrong identities fail closed. Raw
  response references survive parse failures; issuer failures retain schema
  issues. Failed HTTP response references and original bytes are retained too.
  There is no fallback to conceal an invalid matching embedded document.
- All transport still uses the existing counted client. Fixture checks confirm
  retry reservations, exhausted budgets and no HTTP attempt after the first
  403/429. No refresh, alternate URL, redirect, budget reset or checksum repair
  was added. `ReferencedSources` replays only declared responses, even when an
  undeclared primary happens to exist elsewhere in the cache.

Verification used pure fixtures with mocked HTTP I/O and the real transport,
cache and accounting implementation. The initial test-first selection produced
24 expected failures and four passing guard cases. An additional reference
retention assertion exposed two failing fallback HTTP cases before their fix.
Final targeted run: `scripts/run_tests.sh tests/lab/test_event_timing_audit.py
-q -k 'test_collect_filing' --tb=short` — **32 passed, 153 deselected in 3.12 s**.
After the final fixture adjustment, the complete focused run was
`scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q --tb=short` —
**185 passed in 204.44 s**. Every existing test and fixture remained AST-identical;
no previous fetch-order expectation or semantic assertion was weakened.

The new cases cover both 4 and 4/A, canonical selection among unrelated bodies,
single-request/no-double-download provenance, three fallback forms, unchanged
cached-primary results, fourteen invalid-content cases, four HTTP stop cases,
two retry cases, two budget boundaries and four raw replay/date-corroboration
cases. Full-response-only raw revalidation and date-only execution pass with
valid raw pairs; missing/wrong headers cannot corroborate. Observed availability
stays null and dissemination stays UNVERIFIED. Ruff check and format-check pass
on the two owned Python files; CLI `--help` exits 0 and `git diff --check` passes.
No broader lab run was needed because no other runtime module changed.

The parent's pending handoff and next-steps text is preserved and excluded from
this implementation commit. No operational cache was replayed, modified or
migrated; no external request, DB query, strategy/return/price computation,
detector work or deployment occurred. The prior 23-pair equivalence and
two-issuer acquisition counts remain diagnostics, not population forecasts.
Historical identity continuity, Northrop's inventory discrepancy/missing page,
amendment linkage, tape mapping/detector evidence and annual/five-year gates
remain unresolved. Both real signal paths remain UNRESOLVED. The review and follow-up verdict
are recorded below; no further collection is authorized by this change.

## 2026-10-22 collector review: malformed TEXT envelope

Independent Codex review of `5371bba..3ffb112` found one P2 blocker: the
selector discarded content after `</TEXT>`, allowing misplaced, invalid XML
to trigger a valid primary-XML fallback and report raw completeness. The
review's collector selection passed 32 tests, which did not cover this
malformed placement. No other actionable findings were reported.

The parent reproduced the finding with a new real-collector fixture: it failed
on `raw_complete=True` before the fix. In `bbf6f44`, the selector now partitions
the TEXT envelope once and rejects any non-whitespace trailing content before
selecting XML or falling back. The regression verifies rejection, retained
full-response bytes/reference, one counted request and no primary download.

After the fix, `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q
-k test_collect_filing --tb=short` passed **33 tests, 153 deselected in 2.93 s**.
Ruff check, format-check and diff-check passed. The earlier 185-test run predates
this fix; it was not repeated. Fresh independent Codex review of `bbf6f44`
against `3ffb112`, including connected collector/replay callers, is clean: no
actionable findings and the prior P2 blocker is fixed. The reviewer independently
ran the collector selection: **33 passed, 153 deselected in 3.03 s**. Canonical
selection, cached-primary preference, counted fallback, actual source references,
declared-source replay, issuer/accession validation and timing/gates remain
intact. Broader tests were not repeated by the reviewer. No SEC/Yahoo requests, operational-cache writes, DB activity or
production changes occurred. Both research paths remain UNRESOLVED.

## 2026-10-22 bounded identity/inventory resolution results

Owner instruction “Resolve” authorized the reviewed
[26-URL batch](briefs/2026-10-22-identity-inventory-resolution-batch.md).
First review identified one verification defect (URL lookup skipped earlier
retry responses); the runner now checks each response's own bytes/hash and path.
Fresh Codex review cleared that fix before execution. All 26 requests returned
HTTP 200 without retries, taking cumulative collection attempts from 53 to 79,
with zero Yahoo attempts and 104.3013715306297 cumulative seconds. All 79 raw
hashes and unchanged parent files were verified. Exact outcome references and
24 annual security observations are in
[the immutable-point-in-time results](_insider_identity_inventory_resolution_results.json).
The child cache is `data/external/event_timing_audit/identity-inventory-we_55vr8/`;
its persistent claim prevents another execution. No more collection is queued.

### Inventory disposition

The discrepant NOC accession `0001225208-17-005319` has raw header filing date
**2017-03-02**, matching its API row; its header receipt is 2017-03-02 16:23:24
and its embedded ownership XML identifies issuer CIK 1133421. This independently
corroborates the page row against the accession's raw filing. The root page
metadata's end date, 2017-03-01, excludes that real row. Cause remains unknown;
no automatic date shift, source correction or interval-rule relaxation is made.
The current inventory predicate therefore still does not pass.

Previously missing NOC page 002 is now retained: all **214** rows match its
count, spanning 2001-02-01 through 2002-06-17, within the declared interval ending
2002-06-22. An endpoint inside a declared range is not itself a discrepancy.
This closes the missing-page acquisition gap, but not the separate page-001
metadata conflict or complete annual population qualification.

### Historical identity disposition

All 24 original annual reports (fiscal 2014–2025 for each issuer) contain dated
common-stock and exchange statements. Pre-2019 reports state NOC/COST explicitly
in their text; newer covers carry the symbols. These provide actual historical
symbol evidence without relying on today's map. The source refs, exact cover
and symbol extracts, fiscal dates and filing dates are retained in the artifact.
No fiscal-end observation is silently backdated or extended to the next year.

There is a concrete reconciliation item: Costco covers report par value $.005
for 2014–2016, $.01 for 2017–2021 and $.005 for 2022–2025. Selected balance-sheet
extracts also show those respective values, so this is not simply an absent
symbol or a display normalization. Neither a typo nor a security change is
assumed. Effective charter/class changes and the corresponding price-security
link require documentary review. Annual repeated tickers do not establish
uninterrupted validity or vendor lineage for either issuer.

### Remaining decision

The batch resolved the missing-page retrieval and corroborated the disputed
filing date, and replaced speculation about pre-2019 ticker availability with
primary annual evidence. It did not resolve mandatory continuous identity,
price-vendor lineage/bar metadata, the inventory predicate conflict, full
ownership/amendment reconstruction, annual population gates or PEAD consensus
provenance. Both signals remain **UNRESOLVED**; this is no strategy-return or
SPY-outperformance result. Fresh independent Codex review supports these bounded documentary findings
with no blocker. The reviewer independently checked all 79 raw hashes, 57
registered parent-file hashes, the unchanged 53-request accounting prefix, all
26 URLs and every annual raw-document cover/symbol extract. No production code, DB, price/return analysis or deployment changed.

### Independent documentary review and exact next evidence

The review found no cached statement resolving COST's legal class history or
effective dates. Its 2017 equity statement records a $2 million transfer into
common stock from additional paid-in capital; the 2022 statement records the
reverse. Both use a combined dividends/other row. These entries show the
difference extends beyond cover typography, but explain neither a legal
conversion nor its timing. No conclusion about a changed security is warranted.

Two exact charter exhibits are the smallest named follow-up, with URLs retained
in the results artifact: the 2015 charter `0000909832-15-000003/
costex3110q21515.htm`, incorporated by both 2016 and 2017 annual reports, and
the 2022 charter `0000909832-22-000021/costex3110k082822.htm`. They are not cached
and no statements about their contents are made. If needed after that comparison,
the specifically named intermediate document is the 2020 charter
`0000909832-20-000004/costex3110q21620.htm`; the 2020/2022 Exhibit 4.8 security
descriptions are conditional supporting candidates, not an automatic fetch list.

Keep another source discrepancy visible: the 2019 annual Exhibit 3.1 row cites
a 2019 quarterly filing, but its hyperlink points to the 2015 charter. The caption
does not establish a distinct 2019 charter or an effective date. Follow actual
raw links and operative legal language rather than guessing from the caption.

The completed batch remains bounded; no new external request is included in
this review. Inventory-rule disposition and any next acquisition must be concrete
and reviewed; no annual/five-year gate or price-vendor link is passed by these
observations. Both signals remain UNRESOLVED.

## 2026-10-22 five-gap Flash workbook review — rejected draft

The owner requested a consolidated plan and agy Flash delegation. The plan is
[the five-gap resolution plan](../superpowers/plans/2026-10-22-event-evidence-resolution.md).
Flash created the two uncommitted workbook files named there. Independent Codex
review rejected them before acquisition or DB use. Fourteen corrections are
required; the draft is not an accepted source for audit predicates or budgets:

1. Separate original receipt conflicts, amended date-only compliance and unknown
   dissemination; remove blanket timing PASS.
2. Amendment `0001225208-18-004211` is filed 2018-02-26 with original-submission
   date 2018-02-15; missing predecessor means linkage UNVERIFIED, not PASS.
   Existing reconstruction excludes unresolved amendment lots.
3. Correct mismatched URL/file/hash tuples for that amendment and
   `0001628280-22-028934`; bind every reference through the actual manifest.
4. Remove unsupported 0.5-hour vendor spend/1.5-hour remaining claims and invented
   third-party-stop attribution; search accounting remains unreconciled.
5. Replace non-executable/unpaired SQL proposals with event-security/date-paired,
   zero-preserving, horizon/vendor-pinned metadata queries. Existing runtime count
   query also omits venue; any code correction is a separate tested task.
6. Separate insider member-months from PEAD member-quarters and their applicable
   checks. Reviewer reproduced 68,806 insider units/761 symbols and 49,542 PEAD
   units/1,013 symbols (including partial 2026 diagnostics). PIT file hash is
   `b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472`.
7. Supply missing owner/role/class/holding, duplicate, late/carry-in/boundary and
   PEAD original-release/actual-EPS findings or exact missing inputs. Stored EPS
   field is `eps_reported`. Population acquisition has not been costed.
8. Reject a post-observation one-day descriptor tolerance as proof of completeness;
   corroboration and disjoint pages do not establish absence of omitted filings.
9. Correct three NOC annual accessions and raw locators; annual observations do
   not PASS continuity, and accounting entries do not rule out broader errors.
10. Remove uncited vendor properties, including previously rejected CRSP access
    assertions; rebuild relevant four-item route evidence from retained sources.
11. Cached checkbox absence does not prove XML-schema introduction history;
    distinguish compliance date, sample absence and unknown plan participation.
12. Collision example has 11 raw lots, groups of six and two (eight involved,
    six potential excess rows), all S transactions. No DB loss or purchase-cluster
    effect was measured; co-owner code support is not sample verification.
13. Keep the two charter leads, correct the conditional Exhibit 4.8 target, and
    name missing mapping/export/predecessor/vendor dependencies. No guessed runtime
    or claimed fully executable consolidated acquisition.
14. Bound artifact-absence/search/validation claims to actual checked paths and
    methods; custom JSON labels do not imply formal audit-schema validation.

Codex is correcting only the two workbook drafts against raw evidence. No runtime
change, network request, DB query, protocol amendment, acquisition authorization
or PASS follows from this review. Original reviewed 79-response evidence and both
UNRESOLVED per-signal verdicts remain valid. Fresh review follows the corrections.

## 2026-10-22 consolidated workbook accepted at offline scope

Codex rebuilt the rejected Flash workbook and its source-bound JSON companion.
Independent round 2 cleared findings 1–9 and 11–14, reproducing all 79 hashes,
24 annual/API bindings, 24 ownership reparses/52 lots, 136 checkpoints, separate
annual denominators, predecessor candidates, query contracts and dependencies.
Its only remaining blocker was vendor retrieval-label provenance. The parent
corrected labels directly from retained receipts and embedded exact excerpts
and receipt hashes. Fresh round 3 checked all 38 entries/37 unique URLs across
R1–R9 and found no additional blocker. All 14 findings are now resolved at
workbook scope. The last round did not repeat the earlier raw reconstruction
checks. No new services, DB, prices/returns, detector runs or broad tests.

Accepted outputs: [human workbook](2026-10-22-event-evidence-resolution-workbook.md)
and [machine workbook](_event_evidence_resolution_workbook.json). These are
offline documentary dispositions, not the registered final audit JSON/report.
They preserve the two charter leads and exact remaining dependencies for
inventory, security identity, predecessor/amendment inputs, mapping/tape exports,
population reconstruction and PEAD lineage/vendor-budget reconciliation.
No new acquisition authority, changed predicate, annual qualification, five-year
window or validated strategy edge follows from acceptance. Both insider
representations and PEAD remain UNRESOLVED; production remains frozen.

Owner delegation correction is recorded in AGENTS.md: Flash only receives simple
bounded extraction or exact-link checks; Codex owns synthesis, plans and reviews.

## 2026-10-22 bounded local export search — parent verified

Flash was assigned only exact local filename/schema-string discovery. Parent
verification independently walked `docs/`, `results/` and
`data/external/event_timing_audit/`, without following symlinks or reading private
locations. Both returned candidate paths and SHA-256 values matched: the human
workbook mentions the detector schema; the machine workbook includes a nested
contract but has a different top-level schema. Neither is a detector export.

The parent checked 185 / 3,542 / 197 non-symlink regular files in those respective
roots, including 175 / 1,951 / 15 JSON/Markdown content searches. Flash reported
3,544 results files; the independently counted 3,542 governs this recorded scan.
No specified mapping/detector filename match or exact top-level detector export
was located. This is scoped discovery, not exhaustive absence, proof of clean
tape or historical identity. Differently named mapping inputs remain possible.

[Search evidence](_event_evidence_export_file_search.json) records exact scope,
counts, hashes and limits before creating that artifact. No HTTP, DB, private
environment reads, price analysis or detector run. The two-charter acquisition
remains under separate independent review and has not run. The next metadata
dependency remains a source-bound historical map and existing detector export.

## 2026-10-22 two-charter execution and comparison

Independent acquisition review cleared the exact two-URL slice. The reviewed
runner hash matched immediately before execution. Both SEC responses returned
HTTP 200 without retries, increasing cumulative collection accounting from 79 to
**81 SEC attempts**, with zero Yahoo and **105.01228246744722 seconds**. All 81
raw response hashes were verified individually; parent files and the accounting
prefix remained unchanged. The child cache is
`data/external/event_timing_audit/cost-charters-h1kbl362/`. This bounded claim is
consumed, not reset. [Exact excerpts and references](_cost_charter_comparison_results.json)
retain documentary evidence without modifying the previously accepted workbook.

Both exhibits reproduce the base Article II Section 2.1 common/preferred par
value of $0.01 and the base execution text dated August 27, 1999. The 2022 exhibit
also includes an appended amendment replacing that section with $0.005 for both
classes, adopted December 7, 1999 and executed November 14, 2001. It expressly
conditions effectiveness on filing with the Washington Secretary of State;
the actual state-filing date has not been verified. That appended Section 2.1
amendment was not located in the acquired 2015 exhibit body, which includes the
2015 Article VIII amendment instead. This absence is limited to that document.

The source set therefore contains an older amendment that prevents treating the
annual 2017/2022 par-value changes as proof of new security conversions in those
years. It does not establish why the annual reports differ, rule out other
amendments, prove legal effective chronology or justify continuous validity.
Do not substitute exhibit publication, adoption or execution for the stated
filing condition. Price-vendor security linkage remains a separate missing input.

Fresh independent scoped documentary review is CLEAN, with no actionable
overclaim, wrong date binding or class-clause markup caveat. It independently
verified all 81 hashes, parent/accounting prefix, accepted workbook hash and
consumed claim. Actual Washington state-filing evidence for this exact amendment
is still required for effective chronology; historical vendor/security mapping
remains a separate dependency. No more documents are queued; local
export search found no qualifying detector export in its three searched roots.
No database, price/return analysis, new detector, runtime code or production work
occurred. Both representations and PEAD remain UNRESOLVED.

## 2026-10-22 present readiness decision reviewed

Independent Codex review supports [DO_NOT_START_PHASE_2_RETURN_TEST](2026-10-22-event-audit-readiness-decision.md) with current verified inputs.
It checked source hashes, accounting, blockers and reopening conditions, and
confirmed that canonical final audit outputs remain absent rather than falsely
published from offline work. It identified no already-recoverable mandatory
input in the named evidence. This resolves the execution decision, not the
owner's broader unfinished feasibility objective.

One correction is recorded for future metadata use: actual table name is
`earnings_surprise`, not the attempted `earnings_surprises`. The original SQL
is retained as historical evidence; corrected SQL is added separately and has
not run. The attempt was unavailable with exported settings and returned no
rows, so no database/table-absence inference or decision change is warranted.
No additional HTTP/DB execution, price/return analysis, detector, runtime change
or test accompanied the review. Missing inputs remain the blocker.

## Connected-metadata plan execution — 2026-10-22 local / 2026-10-23 UTC

Executed the [current plan](../superpowers/plans/2026-10-22-next-evidence-actions.md)
through the available read-only PostgreSQL connector. Six metadata queries and
one batched operational bar-count query are recorded with exact SQL, source,
observation timestamp and raw results. No price, EPS value, return, live position
or production configuration was selected; no database writes occurred.

- [Metadata inputs](_event_audit_metadata_inputs.json): the stored insider window
  contains **767,999 rows**, including **6 transaction-before-1990 rows** and
  **78 transaction-after-filing rows**. Categories can overlap and are not an
  anomaly-union count. These are stored transaction rows, not verified filing
  inventories, cluster events, PIT coverage or source-completeness numerators.
  PEAD has **61,377 stored rows** in its pinned window; non-null estimate/reported
  fields do not establish release, EPS-basis or pre-announcement vintage lineage.
- The original-submission candidate set narrows from 14 same-issuer/date filings
  to one stored owner match, **`0001225208-18-003052`**, CIK 1133421 and owner
  1539147, filing date 2018-02-15. It appears in the cached enumerated frame too.
  Raw predecessor/lot replacement remains UNVERIFIED until the source is compared.
- Candidate schemas `symbol_inception` and `universe_cache` describe inception
  dates and dated symbol lists respectively; the inspected fields do not supply
  issuer/class/vendor validity-interval evidence. The public-schema lineage
  column search also supplies no detector export or PEAD announcement/snapshot
  fields. This is the scoped schema result, not universal absence from all stores.
- [Legacy detector provenance](_event_audit_detector_provenance.json) traces
  `scripts/production_replica_audit.py` to `_production_replica_results.json`.
  Existing seam metadata contains 120 primary records (84 artifact=True) and
  1,129 holdout records (920 artifact=True). The artifact lacks explicit screened
  manifests; the producer/report disclose trade-proximity and gap limitations.
  No values were recomputed and no return fields were extracted. These recovered
  flags cannot be promoted into exhaustive per-security clean intervals or a
  source-bound A7 export. The rebaseline artifact's meta fields do not supply it.
- [Operational bar counts](_event_audit_operational_bar_counts.json): 23 exact
  filing-date-convention diagnostics have one `yfinance`/`1d` bar each. All requested
  dates were independently checked as NYSE sessions. Historical mapping remains
  UNVERIFIED, original timestamp conflicts remain and these filing diagnostics
  are not cluster events or annual population coverage.

### Venue defect reproduced and corrected

`check_tape_integrity_batch()` previously counted all venues. Real in-memory SQL
fixtures reproduced two wrong results: yfinance+other counted as two; other-only
counted as one. The shared query now pins `venue='yfinance'`. Two duplicate/missing
control cases remain. The test adapter only translates PostgreSQL placeholders
and date-cast syntax to SQLite; it executes the production query rather than
fabricating counts. Existing missing-detector behavior is retained.

The new selection was RED (two expected failures, two passing controls), then the
full focused command `scripts/run_tests.sh tests/lab/test_event_timing_audit.py
-q --tb=short` passed **190 tests in 206.66 seconds**. Ruff check/format and diff
checks pass. No broader suite was needed for this isolated audit query change.
Fresh independent review of this execution slice is pending before relying on
the runtime path or executing the one-request predecessor acquisition.

### Exact next acquisition and remaining gates

[One-predecessor handoff](briefs/2026-10-22-insider-predecessor-batch.md) freezes
one canonical full-submission URL, not 14 candidate downloads. One nominal/four
maximum new attempts would move 81 SEC attempts to 82 nominal/85 maximum, retaining
the prior stop history, cumulative runtime, immutable parent and exclusive claim.
Runner help/dry run and Ruff pass; no new SEC/Yahoo requests have run in this slice.
The first dry run caught a copied assertion typo; the corrected ceiling is 85.

After raw comparison, link only source-supported replacement lots at the
amendment's own eligibility time. No predecessor assumption, broad population
collection, detector rerun or vendor-search reset is authorized. Mapping/export
inputs, independent NOC completeness, population reconstruction and PEAD lineage
remain unmet; current metadata counts do not change the no-Phase-2 decision.
