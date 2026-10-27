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

Verify the October 29 anchor archive and October 30 inputs, decision and archive
from actual scheduled jobs. Check their source references, durable completion
and integrity; retain any faults and never backfill a decision. Perform the
runbook's daily local health check and weekly/notice-driven calendar comparison.
No external notification or automatic health monitor is installed.

Local timestamps and hypothetical bar fills remain limited evidence. The dividend
comparison remains provisional. There is no profitability pass or automatic move
to broker orders. Disable only these four observer timers to stop recording while
preserving state; the operator runbook records the exact command.
