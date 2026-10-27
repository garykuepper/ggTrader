# Intraday observer readiness

**Subsequent activation:** owner approved “start”; activated October 27, 2026.
See the [activation record](2026-10-27-intraday-observer-activation.md). The
preparation status and October 26 dates below describe the earlier reviewed package.

**Preparation is verified; the observer is off awaiting the separate owner activation
approval.** The package records hypothetical SPY positions and cannot place orders.
No credential file, live state, persistent release or daily timer has been installed.
A dedicated runtime snapshot and temporary release/clock probes were prepared and tested.

## What passed

Parent verified real-source samples with **12 of 20 allowed direct GET attempts**,
plus one Alpaca documentation-connector discovery. Requests disable automatic
redirects/retries. Exact original bodies, parameters, receipt times, counts and hashes
are preserved in [the request ledger](artifacts-2026-10-26-intraday-readiness/requests.json).
Flash mechanically confirmed these saved fields with no mismatches; parent independently
computed the schema/timestamp/hash checks. No strategy signals or returns were calculated
from the real sampled prices.

- October 23 and 26, 2026 each returned all 390 regular-session minute bars, no
  pagination and valid required 09:59, 15:30 and 15:59 timestamps. The preceding
  October 22 close and July 2 early-close anchor also passed.
- Five-minute-old SIP data was denied; a sixteen-minute-old query was accepted
  with an empty response. This agrees with the [official historical restriction](https://docs.alpaca.markets/us/docs/market-data-faq):
  the end parameter must be at least fifteen minutes old without a subscription.
  Register **900 seconds** as the access restriction. This after-hours test does
  not establish intraday publication latency or future completeness.
- The official paper calendar agrees with offline NYSE calendar 5.5.0 across the
  proposed span, early closes and post-endpoint recovery date.
- The host clock is NTP-synchronized and user lingering is enabled. A transient
  read-only user service also verified clock access with the dedicated runtime.
  It used no network, installed no persistent unit and executed no observer stage.
- Normal collection uses two single-page GETs per date. Recovery and bootstrap
  can add one each; three attempts for all four uses the entire twelve-request
  daily cap. Unusual pagination/retries can exhaust the unchanged cap and remain
  faults. Sampled successful requests took under one second.
- Largest sampled archive: 42,243 bytes, approximately 10.65 MB for 252 archives
  before ledger/inputs/overhead. The synthetic year-sized state measured about
  13 MB and 1,515 records. Five fresh-process final decisions took 1.90–3.50 seconds
  including imports and durable writes, against the sixty-second decision window.
  These measurements use an injected clock/transport, not actual timed execution.

## Implementation and verification

Sonnet added calendar-aware dispatch and fixed final-day recovery without creating
slot 253. Fault tests cover outages, interrupted jobs, calendar amendments, DST,
clock failures and checkpoint halts. Kimi drafted the standalone units and runbook.
Parent reviewed/integrated the changes and addressed Opus's activation findings.

Final code/package commit: **`280d4a0edc00f1b4b23ba7bbe8dd80432877b36f`**.

- Full capped suite under the dedicated runtime: **1,889 passed, 30 deselected,
  one existing deprecation warning, 441.45 seconds, exit 0**.
- One additional runbook-generator integration test passed after it was added:
  **151 unique observer tests** verified across the full suite and this test.
- The manifest-hash, corruption-report and ineffective-resume regressions were
  observed failing, then passed after the fixes.
- Repository-wide Ruff check and formatting pass. All eight rendered units pass
  `systemd-analyze verify --user`, and eight final Bash blocks pass `bash -n`.
- Temporary local clone/detach, read-only release seal and both CLI help paths were
  verified with the dedicated runtime. This did not install the persistent release.

[Machine-readable preflight](_intraday_observer_readiness_preflight.json),
[unit checks](_intraday_observer_unit_verification.json),
[runtime snapshot](_intraday_observer_runtime_snapshot.json),
[release rehearsal](_intraday_observer_release_verification.json),
[year-sized measurement](artifacts-2026-10-26-intraday-readiness/runtime_bench_252.json)
and [final full-suite log](artifacts-2026-10-26-intraday-readiness/full-tests-final.txt)
preserve the evidence. Earlier 1,887-test worktree results remain historical.

## What Opus found and what changed

Opus found four activation blockers: wrong printed manifest hash, readiness steps
in the wrong order, nonpersistent/shared-manager credentials, absent release creation
and a shared research interpreter (the hash/order issue was one combined finding).
Parent corrected them directly and added missing daily health and weekly calendar
procedures. The dedicated environment is an independent read-only copy of the
existing installed packages, with copied interpreter binary and no editable checkout
hook; no dependency was downloaded or newly installed.

The runbook now creates the exact local detached release, renders its own units,
provisions a private 0600 credential file only for fetching services, and separates
initialization, inspection, evidence binding and activation. The readiness payload
records its preflight hash and release commit, and preserves an exact preflight copy.
A broken ledger produces unresolved integrity diagnostics without return statistics;
ongoing corruption cannot be waived by resume. See the
[review disposition](2026-10-26-intraday-observer-review-disposition.md) for every
finding, ruling and retained limit. No second Opus approval is claimed.

## Concrete activation package

[The proposal](_intraday_observer_activation_proposal.json) fixes the code, runtime,
source calendar, delay, four daily jobs and proposed storage paths.
[The runbook](../../deploy/intraday_observer/OPERATOR_RUNBOOK.md) contains the exact
staged installation/init/readiness/activation/disable procedure.

For New York activation on **October 26, 2026**, the calendar gives:

| Milestone | Date |
|---|---|
| First observation, third full session after activation | October 29, 2026 |
| Operational checkpoint, slot 20 | November 25, 2026 |
| Final observation, slot 252 | November 2, 2027 |

If activation moves to another date, regenerate the unactivated dated package
using the approved third-full-session formula before any observation. Never roll
an active inception. Actual live manifest/readiness files are created only at the
approved final path after approval; no placeholder manifest hash is passed off as
real evidence. This is necessary because the manifest includes its creation time
and resolved final path.

Proposed jobs are 15:15 recovery, 15:20 inputs, 15:25 decision and 16:20 archive,
all **America/New_York**, independent of production. Timeouts remain 270/270/45/270
seconds. Known holidays and out-of-study dates are no-ops; early closes archive
anchors only. A missed decision cannot be backfilled. The daily local health check
is a manual operator obligation, not an automatic monitor or external alert.

## Limits retained

The issuer workbook has 43 parent-verified SPY distribution rows under its original
hash, but currency convention, blank capital-gain semantics and future completeness
remain unresolved. The comparison is provisional and cannot be claimed complete.
This does not prevent recording the primary rule, which does not use distributions.
Complete terminal comparison needs verified semantics and an issuer publication
covering the full future window.

Year-sized reporting takes 51 seconds; run it outside collection windows. Local
timestamps, bar-proxy fills, future data gaps/host stalls and OS-library dependencies
remain limits. Parent-directory fsync and the closed-final-slot recovery edge are
recorded in the review disposition. Actual fetching under installed services and
future DST firing remain first-run checks after approval. No profitability pass,
automatic broker route or amendment of the stock-candidate standard is implied.
