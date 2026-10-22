# Event-timing audit implementation handoff — targeted review fixes

This is an implementation/re-review handoff, **not an audit report, collection
result, vendor approval or feasibility conclusion**. The starting commit was
`c964659` on `research/event-timing-audit`. The
[approved brief](briefs/2026-10-21-event-timing-audit.md), especially A1–A12,
and [source manifest](briefs/2026-10-21-event-timing-audit-sources.md) are unchanged.

The round follows all 14 findings in `/tmp/event-timing-rework-codex-review.md`
and the recovered original 17 in `/tmp/event-timing-original-code-review.md`.
The independent `/tmp/event-timing-vendor-evidence-review.md` supersedes the
unsupported vendor narrative. Neither vendor document is ingested as runtime
facts. No paid path, qualifying real window, price entitlement or guaranteed
SEC dissemination bound is established here.

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

## Disposition of the 14 rework findings

All 14 were confirmed against the code and fixed within this round's scope.
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
| 6 | Progress binds detector and module hashes plus original protocol/input hashes. It persists stored snapshots/query metadata, frames, selected keys/outcomes, failures and stage progress. Resume reuses stored observations. Regressions: `resume_binds_detector`, `resume_keeps_stored_snapshot`, plus the original full resume fixture. |
| 7 | Identical complete empty reconstructions pass; no minimum-activity gate. Empty tape/collision aggregation requires explicit proven emptiness plus complete inputs. Regression: `proven_no_cluster_prefix_passes`; the complete-census fixture also checks honest zero-event years. |
| 8 | Stored losses compare each `(lot_id, owner)` pair; lot-level summaries derive from those pairs. Regression: `coowner_missing_lot_is_counted`. Raw recoverability stays separate from Phase 2 storage repair. |
| 9 | Malformed owners/fingerprints are retained as unassignable, excluded from reconstruction and reported UNVERIFIED. Regressions: `malformed_owner_survives_preparation`, `malformed_raw_owner_is_reported_by_full_orchestration`. |
| 10 | CPU unit boundaries check deadlines and checkpoint preparation, units, replay disclosures, prefix months, diagnostic periods and event observations. Indexes replace per-filing/per-period stored scans and repeated identity/inventory lookups. Lag counts use vectorized session searches. Regressions: `expired_cpu_preparation_checkpoints_remaining`, `expired_runner_emits_partial_without_cpu_or_db_work`, `cpu_replay_resumes_completed_disclosures`, `prepared_pead_resumes_without_recomputing_clocks`. No full-population benchmark was run. |
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

## Verification record for this round

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
