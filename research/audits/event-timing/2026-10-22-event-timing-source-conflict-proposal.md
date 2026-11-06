# Decision proposal: insider source conflicts

**Status:** APPROVED by the owner on 2026-10-22 for offline implementation.
Prepared after the smoke data was inspected. This dated amendment changes the
registered insider representation only as specified below; it is not a
correction retroactively covered by the original approval. Implementation and
independent code review are pending. External requests remain paused.

**Independent review:** Codex found zero critical or important issues in the
proposal (`/tmp/insider-source-conflict-proposal-review.md`). This clears the
proposal for the subsequent owner decision; it did not itself authorize implementation or collection.

**Authority:** [Original audit](2026-10-21-event-timing-audit.md), especially
§8 A1, A2, A7, A8, A9 and A11. This approved amendment supersedes only the
conflict handling for the separate date-only representation described below;
all other original requirements remain binding.
[Execution plan](../../superpowers/plans/2026-10-21-insider-evidence-next-steps.md).

## What is established

The cached smoke has 21 API/header receipt pairs. Under the existing timezone
interpretation, 14 differ by four hours and seven by five hours. The existing
one-minute conflict check rejects all 21 pairs. This does not identify which
representation, source conversion or historical source vintage is wrong.

Northrop also has one historical-page row dated after its descriptor's end.
Raw bytes remain preserved, and the inventory remains incomplete. That issue
is separate from timestamp handling.

The follow-up research reported HTTP 403. External requests are paused.
Its source quotations and explanations have not been independently established
as authoritative evidence for changing either rule.

## Recommendation and immediate work under the current rules

Preserve the original result and quarantine. Do not shift timestamps, widen
the tolerance, ignore the page disagreement, or declare the raw route infeasible.
Complete the offline evidence record and assess an explicitly labelled date-only
alternative before deciding whether it is worth implementing. It is a different
historical representation and may imply different trade dates.

No new download or code change is needed to review this proposal. Historical
identity, complete inventories, amendment linkage and detector evidence remain
separate requirements; changing the clock cannot solve them.

## Approved alternative

The original A2 fallback covers date-only records. It does not expressly allow
discarding contradictory known receipt readings. The following would therefore
amend A2/A7/A8, rather than silently invoking the original fallback:

1. **Scope:** define a separate `insider_filing_date_only` representation for
   the entire registered insider population, not just records with a convenient
   timestamp conflict. Do not select the representation by performance, year,
   company, or which individual record passes.
2. **Corroboration:** require the API filing-date field and the raw submission
   header filing-date field to match exactly for the same issuer/accession.
   Missing or inconsistent date corroboration remains UNVERIFIED or quarantined
   as applicable. Do not substitute either source's acceptance-derived date for
   its filing-date field.
3. **Execution:** close of the first NYSE regular session strictly after that
   corroborated filing date. Preserve the existing holiday/early-close calendar;
   do not apply the 30-minute buffer again. Store `date_only` precision and the
   inferred execution time separately from observed timestamps.
4. **Conflicts:** retain both original receipt values and the original failed
   receipt-comparison result. The original timestamp representation remains
   quarantined. Only the separate date-only representation would be permitted
   to use corroborated filing dates despite an unresolved receipt-time conflict.
   All other A8 quarantines remain in force.
5. **No dissemination certification:** observed public dissemination and its
   intraday bound remain UNVERIFIED. A permitted date-only execution convention
   is not proof of an actual historical publication time. An owner decision
   would authorize that convention, not certify the disputed timestamps.
6. **Qualification:** if approved, assess all mandatory A6/A7 requirements
   separately for this representation. Retain the original matrix and report
   both dispositions. Never promote a diagnostic example into annual coverage
   or waive identity, inventories, reconstruction, tape or five-year gates.
   A timing PASS would mean compliance with the amended date-only convention;
   actual dissemination and its bound would remain UNVERIFIED.
7. **Ordering and scope:** all A9 sample frames, canonical keys, quotas and
   ordering remain fixed. The change applies only to insider availability;
   PEAD announcement/consensus lineage requirements are unchanged.

This is a slower date-based candidate, not an exact intraday reconstruction.
Delaying a trade may reduce same-day timing exposure; it does not establish
that every record was disseminated by the assumed execution time.

## Changes this proposal does not authorize

- No arithmetic repair to API timestamps, automatic preference for a header,
  wider clock tolerance, or Northrop page-boundary allowance.
- No restart after the reported HTTP 403. Any renewed external access needs its
  own reviewed decision, with retained stop history and cumulative request/runtime
  accounting; the already-authorized contact is not the missing input here.
- No new source route, vendor purchase, source-certification claim, return
  calculation, strategy implementation, live deployment or production change.

## Authorized offline implementation and checks

Reuse the existing `event_clock`, NYSE date-only helper, population accounting
and evidence-review schema. First pin tests for matching filing dates with
conflicting receipt times, missing/wrong-accession headers, conflicting filing
dates, weekend/holiday execution, all other quarantines, and unchanged original
timestamp output. Add a test proving the representation is applied uniformly
and reported independently of the original matrix. Use `scripts/run_tests.sh`;
no new framework or dependency is planned. A fresh reviewer checks the scoped
change before it is used for evidence conclusions.

New code/revision cannot resume the old smoke registration. Preserve its bytes,
outputs and accounting; any new registration must retain the global spent budget
and stop history. No automatic budget reset or cache migration is authorized.

## Decision record

- The owner replied “Approve” on 2026-10-22 to the explicit request to implement
  this date-only alternative offline. This authorizes the specified scoped
  implementation, fixture tests and independent review.
- Preserve the original timestamp result and quarantine alongside the separate
  amended representation. Both end-to-end data paths remain unresolved.
- No collection restart, return test, purchase, deployment or other change is
  authorized. The already-configured contact authorization does not clear the
  external-stop condition.
