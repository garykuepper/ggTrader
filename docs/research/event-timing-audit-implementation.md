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
