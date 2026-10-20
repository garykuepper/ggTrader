# Amendment: correct close-to-close event-return attribution

**Status: approved by Flynn's “please fix” instruction in this thread on
2026-10-19, after the registered FOMC semantics stop was reported.**

This amends the [original audit brief](2026-10-19-production-replica-audit.md)
and retains the [approved daily-bar scope](2026-10-19-production-replica-parity-amendment.md).
Both earlier briefs remain unchanged.

## Observed failure

The registered test generates an entry at session 4's close and an exit at
session 5's close. Signals are correct, but `in_position_mask` includes returns
for both sessions 4 and 5. Session 4's close-to-close return was earned before
the purchase and must not enter the held-return sample. The failure is preserved
in `logs/fomc_semantics_stop_20261019.log`. The holdout was intentionally stopped
during data loading; no holdout portfolio arms completed.

## Authorized correction and acceptance

Correct only the lab event-study helper's return attribution: for fills at the
signal close, include returns strictly after the entry session and through the
exit session, inclusive. Update the helper documentation and its existing tests;
retain the registered FOMC session-semantics assertion unchanged. Add a non-flat
price regression proving that pre-entry returns are excluded from the reported
event sample and mean. No strategy signals, fills, parameters, costs, windows,
thresholds or production code may change.

Files for this correction are `src/ggTrader/lab/event_study.py`,
`tests/lab/test_event_study.py`, `tests/lab/test_fomc_drift.py`, this amendment,
and `docs/changelog.md`. Review the diff and run tests only through
`scripts/run_tests.sh`, plus Ruff checks. Commit this correction separately.

The FOMC stop remains active until the unchanged registered test and affected
tests pass. Once verified, record the resolution without deleting the original
failure evidence, then resume the interrupted audit under its existing rules.
Primary replica metrics are unaffected because that simulator does not use this
event-study helper. Prior event-study statistics using the inclusive entry mask
must not be reused as corrected evidence; Task 8 must recompute them.

All other stop conditions, the live freeze, restricted imports, database rules,
review gates and no-deploy/no-push restrictions remain in force. Exact historical
live replication remains **UNVERIFIED** under the earlier approved amendment.
