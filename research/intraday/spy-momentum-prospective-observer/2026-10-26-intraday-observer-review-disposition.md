# Intraday observer readiness review disposition

Opus reviewed the combined recorder and activation draft. It found four activation
blockers and three Important operational issues; the recorder's main timing,
calendar dispatch and final-session recovery checks held. This document records
parent fixes, not a second Opus approval or a profitability verdict.

## Findings and parent action

| Finding | Verified problem | Resolution |
|---|---|---|
| B1 | `init` printed a canonical-JSON hash while activation needs the actual file-byte hash; runbook required readiness before the manifest existed. | `init` now prints the disk-byte hash. Runbook separates initialization, inspection, evidence binding and activation. A CLI regression failed before the fix and passed after it; a separate integration test executes the actual runbook generator and activates only an isolated synthetic fixture. |
| B2 | User-manager environment would lose credentials on restart and expose them to unrelated user units. | Only the three fetching services reference a dedicated owner-only 0600 `EnvironmentFile`. Decide has no credential file. Provisioning extracts only the two existing variables after owner activation approval, without copying or printing the root `.env`. No credentials provisioned during preparation. |
| B3 | No command actually created the fixed release; templates could come from another checkout. | Explicit local clone, detach at exact commit and read-only seal; render templates from that release. Parent rehearsed cloning and CLI help in a temporary directory using the dedicated runtime. Persistent release remains activation-only. |
| B4 | Shared research environment could change underneath a year-long observer. | Independent 1.4 GB environment snapshot, with copied interpreter binary and no editable root-checkout hook. Complete installed-name/version list frozen; no downloads or new dependency installation. Capped tests support an explicit interpreter override, retaining the same 8 GB/no-swap boundary. Dedicated-runtime verification is recorded separately. |
| I1 | Missing/refused jobs can be absent from the ledger and therefore require local health inspection. | Daily post-16:30 operator procedure checks failed units, timers, integrity and journal against expected stage completion. It is explicitly a manual obligation, not an installed monitor. |
| I2 | No unexpected-closure/early-close maintenance procedure. | Weekly/notice-triggered bounded official-calendar comparison, counted in the existing daily request budget; preserve source and obtain owner approval for amendments before the affected day's 15:20 and any observation. Late notices remain faults. |
| I3 | Corruption made reporting raise; a resume could be recorded but could never clear current corruption. | Corruption report now returns unresolved identity/integrity diagnostics without return statistics. Resume explicitly refuses while evidence is corrupt. Exact preserved original bytes may restore integrity, then a future owner-approved resume can be recorded. Regression tests failed before and passed after the change. |

## Rulings

1. Keep 270-second recovery/input/archive and 45-second decision timeouts. Longer
   suggested values cross operational boundaries. Worst-case bootstrap retries may
   be killed and must remain faults; an anchor archive avoids bootstrap normally.
2. Do not fabricate a manifest hash in advance. The concrete package pins inputs,
   calendar, code, runtime and paths; actual readiness binds the real `init` output
   at the approved state path. Regenerate an unactivated registration if its date moves.
3. Approval cannot waive ongoing corruption or make unknown accounting trustworthy.
   A normal report with a manifest/code binding mismatch still refuses rather than
   trusting another identity. Preserve its error and bytes. If original evidence
   cannot be restored exactly, the study remains unresolved and positions stay halted.
4. Freeze a copy of the existing environment rather than install dependencies again.
   It isolates routine research upgrades but still depends on the host's system
   standard library and OS shared libraries; it is not a hermetic container.

## Minor limits retained or documented

- Year-sized descriptive reporting takes about 51 seconds. Run outside collection
  windows, not as a frequent status poll. No performance rewrite was needed.
- `atomic_checkpoint` and raw storage do not fsync parent directories; a power loss
  may lose a completion record. Missing evidence is treated conservatively.
- Optional revisions share the request budget; run outside 15:10–16:30 ET only.
- A final slot amended closed can make recovery of an older expired archive refuse;
  record the refusal without extending grace or adding a slot.
- Readiness remains operator attestation. Its payload now records the preflight
  hash and release commit; an exact preflight copy is retained with state.
- Avoid heavy research work during 15:10–16:30 ET. Measured timing is not a guarantee
  under future host stalls.
- Sampled SIP access and after-hours restriction checks do not measure live intraday
  delivery latency. Real fetching under installed systemd units and future DST
  firing remain first-run checks after activation approval.
- Issuer currency and blank capital-gain semantics remain unresolved. The comparison
  stays provisional; no complete benchmark or profitability pass is claimed.

No new independent re-review was commissioned for these small findings. Parent
used regressions, integration/syntax checks, the dedicated-runtime suite and a
read-only transient user-service clock probe. No source request beyond the
previous twelve was needed; no production code, account, service or schedule changed.
