# Prospective recorder implementation: independent review and fixes

Opus reviewed baseline `18b279e` through implementation `937e487`. Verdict:
SPEC NOT YET COMPLIANT / QUALITY APPROVE WITH FIXES. No Critical issues;
eight Important findings. It made no network requests or code edits.

Parent reproduced all eight with failing tests and corrected them directly:

| Finding | Correction and regression |
|---|---|
| I1: failed slot-20 check did not halt | Shared checkpoint calculation includes never-recorded stages; future decisions halt on failure. `test_review_failed_checkpoint_halts_slot21`; clean 20-session control also tested. |
| I2: retrospective calendar reclassification | Amendments refuse after 15:20 or after observations. `test_review_calendar_amendments_cannot_reclassify_an_observed_day`. |
| I3: bootstrap falsely late | Bootstrap uses its input cutoff, not the previous afternoon's archive deadline. `test_review_bootstrap_is_timely_against_its_input_cutoff`. |
| I4: activation ignored readiness | Live mode requires a manifest-bound readiness file with explicit checks; exact bytes are stored and hash-bound. Calendar file labeled operator-supplied, not certified official. `test_review_live_activation_requires_readiness_evidence`; mismatched evidence test. |
| I5: real issuer layout/identity failed | Add MM/DD/YYYY parsing, fixed SPY CUSIP selection, no-match refusal and unmapped-capital-gain reason. Real multi-fund issuer workbook test selects 136 SPY records without mixing funds. |
| I6: timestamps preceded slow work | Clock/halt probes precede completion checks; immutable publication-completion receipts invalidate late durable writes. `test_review_slow_halt_check_cannot_backdate_a_decision`, slow clock probe and slow durable-write tests. |
| I7: missing raw bytes undetected / crash temp fatal | Verify referenced payloads exist; hash only named evidence files and report strays separately. Missing-file, corrupt-evidence-report and crash-temporary tests. |
| I8: manifest edits mixed old/new bindings | Reopening refuses records bound to another manifest; verification checks every record binding. `test_review_verify_binds_old_records_to_current_manifest`. |

Additional boundary checks cover archive writes that finish after grace and GETs
whose pacing delay crosses cutoff. The runtime fingerprint now includes numpy,
pandas and statsmodels. The first remaining full session can bootstrap if slot 1
was legitimately cancelled. Malformed record fields become a controlled refusal.

Red evidence: `/tmp/ggtrader-observer-review-red.log` (12 failures),
`/tmp/ggtrader-observer-write-red.log`,
`/tmp/ggtrader-observer-archive-write-red.log`,
`/tmp/ggtrader-observer-integrity-red.log`. Final results are recorded in the
[build record](2026-10-25-intraday-observer-build.md).
No second Opus verdict is claimed; parent regression verification follows its review.

Remaining limits: real Alpaca access/delay, host clock synchronization and live
payload/budget behavior have not been preflighted. Readiness JSON is an
operator evidence record, not authentication or independent verification of its
assertions. Local timestamp evidence remains local. Optional revision requests
share the same 12-attempt budget and should not be scheduled ahead of required
captures. Code size was not addressed by a cosmetic split; actual end-of-year
runtime remains unmeasured. The initial implementation and raw-data trust boundary
are unchanged outside the documented fixes.

Task: `node:delegated-task:command%3Amcp%3A4f61f14c-07f3-415c-b8cf-f6ba9fc78619%3Adelegate-task%3Aintraday-observer-opus-code-review-2026-10-25-r1`.


## Final verification

Corrected code4ab01d1: 118 focused tests pass; final full capped suite1,857passed,
30deselected, one existing warning, exit0 in393.99s. Repository Ruff check and
format passed280files. Fresh rehearsal has no integrity problems. AgyFlash
independently verified code/test/report hashes and lint; its one-time full-log
read was pending, and parent verified the subsequent final summary directly.
No real-source preflight, activation, scheduler or order was performed.

Final mechanical check task:
`node:delegated-task:command%3Amcp%3A4f61f14c-07f3-415c-b8cf-f6ba9fc78619%3Adelegate-task%3Aintraday-observer-agy-flash-final-verification-2026-10-25-r1`.
