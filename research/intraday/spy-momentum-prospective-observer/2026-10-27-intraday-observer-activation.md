# Intraday practice journal activated

Owner authorized activation with **“start”** after the plain-language explanation.
Installed and activated on **October 27, 2026, New York date**. The observation
registration was refreshed before initialization because the prior draft's date
had passed. The fixed rule and third-full-session start formula are unchanged.

| Milestone | Registered date |
|---|---|
| Anchor closing-minute capture | October 29, 2026 |
| First hypothetical observation | October 30, 2026 |
| Slot-20 operational checkpoint | November 30, 2026 |
| Slot-252 final observation | November 3, 2027 |
| Final next-session recovery opportunity | November 4, 2027, before 15:20 ET |

Four separate user timers are enabled and active: recovery 15:15, inputs 15:20,
decision 15:25, archive 16:20, all America/New_York. Calendar-aware dispatch skips
known non-sessions and dates before the anchor. Existing production paper cron,
Docker, broker configuration, database and account positions were unchanged.
The diary records hypothetical positions only and places no orders.

## Installation verification

- Exact detached release `280d4a0edc00f1b4b23ba7bbe8dd80432877b36f`, sealed read-only.
- Dedicated frozen runtime `366f9247c65e0a38`, with copied interpreter binary;
  recorded code/runtime/protocol/calendar hashes match the installed manifest.
- Owner-only 0600 persistent credentials provided to fetching services only;
  no secrets printed and the root `.env` was neither copied nor edited.
- Manifest initialized at the approved final state path, then checked against the
  refreshed proposal. Readiness binds its actual disk-byte hash, exact preflight
  copy, calendar, source delay and reviewed code commit.
- All eight actual installed units pass systemd syntax verification. All four
  timers are enabled/active with future scheduled runs.
- All four installed services were explicitly run once before the anchor. Each
  exited successfully with its expected no-op; no source GET budget records were
  created. This verifies installed paths, runtime, calendar dispatch and unit setup,
  not real data capture or hypothetical returns.
- Live-state report confirms activation and an intact ledger. Installation made
  zero source requests. First real anchor/input/decision/archive remain future.

[Machine-readable verification](_intraday_observer_activation_verification.json)
and [dated package](_intraday_observer_activation_proposal.json) preserve paths,
identities, approval, timers and checks. The prior unactivated proposal is retained
under the source artifact folder. Preparation evidence remains in the
[readiness report](2026-10-26-intraday-observer-readiness.md).

## What remains to check

On November 11, a read-only check of the live local state found the October 29
anchor archive and October 30 inputs, decision and archive completion records
(sequences 3, 5, 7 and 9). Their recorded publication completions were 16:20:01,
15:20:01, 15:25:00 and 16:20:01 New York time respectively; each reported `ok`
with no faults. The three fetched raw bodies matched their recorded SHA-256 hashes;
the first input referenced the anchor completion, and the decision referenced the
input completion. The installed verifier found 57 records, no integrity problems
and no stray raw files. These are local-clock and source-record checks, not independent
proof of source timeliness or a strategy performance result.

The November 11 scheduled inputs, decision and archive also recorded `ok`; all four
timers were active and the user manager showed no failed units at inspection. Hermes
supervision remains paused. Continue the runbook's daily local health check and
weekly/notice-driven calendar comparison; retain faults and never backfill a decision.
No external notification or automatic health monitor is installed.

On November 12, a read-only check of the Hermes job record confirmed that
supervision remains disabled and paused. The record says it was paused on
November 6 at 09:18 Pacific; it provides no pause reason, and its last recorded
status was `ok` with zero consecutive failures. That does not establish who
paused it or why. The four observer timers were still scheduled for November 12,
and the user service manager reported no failed units. The day's recording
window had not yet begun, so this is not the required after-close health check.
The Hermes user journal contains 30 gateway entries between 09:15 and 09:22
Pacific on November 6, but a bounded search for the job ID, `intraday`, or
`paus` found no matching entry. This search cannot determine who paused the
job or why; other records were not ruled out.
The installed verifier was also run locally before the November 12 session: it
reported 57 ledger records, no integrity problems and no stray raw files. That
does not verify a later recording or replace the after-close health check.

Local timestamps and hypothetical bar fills remain limited evidence. The dividend
comparison remains provisional. There is no profitability pass or automatic move
to broker orders. Disable only these four observer timers to stop recording while
preserving state; the operator runbook records the exact command.
