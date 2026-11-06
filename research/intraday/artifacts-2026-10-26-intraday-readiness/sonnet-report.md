# Sonnet report — Task 2 offline readiness (wrapper, faults, year-size runtime)

Scope kept: Tasks 1–3 offline only. No network, credentials/`.env`, DB, production imports, installs,
services, `deploy/` or docs touched. No commits. No subagents. r8 data/modules untouched.

## Files changed (all in my owned set)

| File | Change |
|---|---|
| `src/ggTrader/lab/intraday_observer.py` | `Observer.recover_target()` (defect fix); `run_recover` uses it; `open_observer()`, `run_job()`, `job_noop()`, `job_main()`, `JOB_STAGES`; `scripts/intraday_observer_job.py` added to `CODE_FILES` (pinned in the code hash) |
| `scripts/intraday_observer_job.py` (new, thin) | `main()` → `job_main()`; `--help`/arg errors never touch state/network |
| `tests/lab/test_intraday_observer.py` | +30 tests (118 → 148), all green |
| `.superpowers/sdd/…/runtime_bench.py` (new), `runtime_bench_252.json` (new), this report | measurement driver and its raw artifact |

`CODE_FILES` changed, so the code hash changes: build/hash the release **after** this code, before any `init`.

## Wrapper interface (for the Kimi units)

```
scripts/intraday_observer_job.py --stage recover|inputs|decide|archive --state-dir PATH
```
No other flags; there is deliberately **no `--session`**: the job acts only on the current New York
date, so a late/restarted timer can fault *today's* stage but can never decide a past session.
Output: one JSON object on stdout. Exit codes: **0** = evidence recorded (including a recorded fault)
or a silent no-op; **2** = refusal, nothing recorded (`error: <Type>: <msg>` on stderr, e.g. `TooEarly`,
`credentials_missing`, `NotActivated`, `grace_expired_use_revise`, `state_not_initialized`,
`BindingMismatch`). A recorded fault is not a failed unit by design; faults surface in `report`.

Dispatch is from the registered manifest calendar (with amendments applied) for today's NY date:

| Today is | recover | inputs | decide | archive |
|---|---|---|---|---|
| not in manifest (weekend, holiday, before anchor, after endpoint+1) | no-op `not_a_registered_session_day` | no-op | no-op | no-op |
| anchor session (slot none, full) | no-op `nothing_to_recover` | no-op `anchor_session` | no-op `anchor_session` | runs |
| registered full session (slot) | runs if prior has no canonical value, else no-op `nothing_to_recover` | runs | runs | runs |
| early close (slot none, or amended) | runs (recovers prior) | no-op `early_close` | no-op `early_close` | runs (closing minute 12:59) |
| amended `exchange_closed` | no-op `exchange_closed` | no-op | no-op | no-op |
| **first session after the final slot** | runs: recovers the final slot's archive (no slot 253) | no-op | no-op | no-op |

No-op: stdout `{"stage":…,"noop":true,"recorded":false,"reason":…}`; nothing is written (no claim,
no budget file, no record) and no credentials/transport are needed. `decide` never needs credentials.
Stage functions are unchanged, so late/early behaviour is the registered one: too early → exit 2,
nothing recorded; late `decide` → a `decision_window_missed` fault record with no side, never a
retroactive side.

## Defect found and fixed (RED → fix)

**Recovery of the final observation after the endpoint failed.** The first session after slot 252 is not
in the manifest, so `run_recover(day)` → `neighbours(day)` → `session_row(day)` raised
`unknown_session`. Pre-fix demonstration (old logic against a live observer state):
`PRE-FIX: ObserverError unknown_session:2027-11-05`. The last archive could therefore never use its
registered next-session grace. Fix at the shared cause: `Observer.recover_target(day)`; the post-endpoint
date resolves its prior session as the last non-closed registered session. Tests:
`test_job_recovers_the_final_session_on_the_next_session_without_a_slot_253`,
`test_recover_after_the_endpoint_captures_the_final_archive_directly`,
`test_job_final_recovery_expires_at_the_next_session_input_stage` (15:20:00 → refused
`grace_expired_use_revise`; 15:14:59 → `TooEarly`). Asserted: only the final slot's record is added, the
ledger holds no record for the post-endpoint date, `inputs --session <after>` still refuses
`unknown_session`, report still shows 252 registered slots.

I found no other demonstrated defect. Not fixed, only recorded: `report()` is O(sessions × records) (see
runtime), 51 s at year scale; correct, offline, outside every window.

## Fault / calendar rehearsal (tests, all pass)

`test_job_fault_rehearsal_keeps_every_fault_and_never_decides_retroactively` drives the wrapper on **every
calendar date** (weekends and Thanksgiving included) from the anchor through slot 22 with a fresh
`Observer` per stage and injected faults: outage (3×503 at slot 2 inputs), killed job (slot 3 archive never
runs → recovered late at slot 4 15:15), crashed claim before the decision (slot 4, reconciled), amended
`exchange_closed` (slot 5), amended `early_close` (slot 6), plus the registered 11-27 early close. Result
asserted: ledger verifies; exactly one decision per observed full session and none for the closed/early
days; every decision record is inside its own 15:25:00–15:26:00 window; no record dated outside
anchor/slots/early close; every fault retained in `report()` (`fetch_transient_exhausted`,
`inputs_unusable`, `archive_late`, late canonical flagged); failed month-one checkpoint is
`reached`/`conditions_met False`, and slot 21's decision is `halted_no_position` with the signal retained
(`signal_side=1`, `observer_side=0`, reason `checkpoint_failed`); the closed slot keeps its number
(slot counts: exchange_closed 1, early_close 1).

Other new tests: no-op matrix (3 non-session dates × 4 stages), NY-date (not UTC-date) dispatch, full session
through the wrapper, early-close archive/recovery only, early-close-day recovery of the prior session,
late `decide` at 15:40 is fault-only and next day cannot reach back, spring DST (2027-03-15: 09:59 ET =
13:59Z, 15:25 ET = 19:25Z, one second early refused), clock-probe failure (missing `timedatectl`,
timeout, non-zero exit → `unknown` → `clock_unknown` fault, no side, `timely_valid False`), `--help`/refusal
subprocess check with an empty env (no files created), code-hash binds the wrapper. Existing coverage
retained for outage/retry, delayed valuation, process restart, unexpected closure, fall DST.

## Host clock (read-only, 2026-10-26 19:15 PDT)

`timedatectl show -p NTP -p NTPSynchronized -p Timezone`: `NTP=yes`, `NTPSynchronized=yes`,
`Timezone=America/Los_Angeles`; `host_clock_status()` → `synchronized` in 8 ms. Failure behaviour: any
probe failure/odd output → `unknown`, recorded in each stage and a `clock_<status>` fault; a decision with
status ≠ `synchronized` writes no side. **The host zone is Pacific: timer `OnCalendar` must carry
`America/New_York` explicitly.**

## Runtime evidence — year-size synthetic state (measured, not estimated)

Driver: `.superpowers/sdd/2026-10-26-intraday-observer-readiness/runtime_bench.py`; raw artifact
`runtime_bench_252.json`. 252 registered sessions (activation 2026-11-23 → last slot 2027-12-02, including
the registered early closes) run through `run_job` for recover/inputs/decide/archive with a simulated
clock and a **fresh `Observer` per stage** (re-validates bindings, re-reads ledger), real durable writes
(fsync, atomic link) on the host's ext4 disk (same filesystem as `$HOME`), compact Alpaca-shaped JSON bars
(390-bar archives, `n`/`vw` fields). Host was loaded (load average ≈3.5 on 4 cores) — conservative.

| Item | Result |
|---|---|
| State before final decision | 1,515 records (2.9 MB), 506 raw payloads (9.7 MB), 255 budget files → ≈ **13 MB / 252 slots** (≈38.5 KB per archive payload) |
| **Final (slot-252) decision, in-process, warm** | **1.15–1.19 s** (3 runs) |
| **Final decision, cold interpreter + imports + Observer init + decision + durable write** | **1.90–3.50 s** (5 runs; 2.1 median) |
| Window | 60 s (15:25:00 ≤ t < 15:26:00) |
| Margin | ≥ 56 s measured (≥ 93%). Adding 1 s timer accuracy and the 5 s `timedatectl` timeout worst case ≈ 10 s total |
| Per-stage over the whole build (non-no-op) | decide median 0.58 s / p95 1.12 s / max 1.22 s; inputs median 0.38 / p95 0.67 / max 0.76; archive median 0.57 / p95 1.05 / max 1.38 |
| Growth | decide curve is linear in ledger size (0.02 s at slot 1 → ≈1.05–1.19 s at slot 252) ≈ 0.77 ms/record (extrapolation, not measured: a 10× fault-heavy ledger ≈ 12 s) |
| `verify_chain()` full year | 0.26 s (0 problems) |
| `report()` full year | **51 s** (quadratic: every row's `neighbours()` re-reads the ledger). Offline/operator only; do not put it in a timer window, and don't use it for a per-minute status poll |

Caveat: cold timing used the module's `run_job` in a fresh `python -c` with an injected clock, not the
script file under the real clock (the only difference is one `sys.path.insert`). The script itself was
verified for `--help`/refusals only. Real network latency/payload sizes are Task 1's measurement.

## Facts for the unit/runbook draft

- Credentials: environment `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` (inherited, read-only; never logged).
- Stage hard bounds in code (stop starting/retrying a GET at the cutoff; a GET started before it may run up
  to 30 s): recover ≤ 15:20:00 (+ ≤ 30 s in-flight); inputs ≤ 15:25:00 (+ ≤ 30 s); decide has **no
  network** and finishes in seconds; archive stops starting new GETs only at the next session's 15:20, but
  is bounded in practice by 12 shared attempts/date × ≤ 30 s + ≤ 30 s waits (a 390-bar archive is one
  page): ≈ 150 s single page worst case. Suggested `TimeoutStartSec`: recover 330, inputs 330, decide 60,
  archive 300. A killed run leaves only a claim, reconciled by the next attempt (never a second side).
- Normal GETs/date: recover ≤ 1, inputs 1 (+1 bootstrap on slot 1 if the anchor archive is missing),
  archive 1; frozen cap 12/date holds with retries (3 attempts × 4 fetching stage calls = 12 worst case).
- Use `Persistent=no`, no randomized delay and `AccuracySec=1s` (the systemd default accuracy is 1 min and
  would spend the decision window); `America/New_York` zone in `OnCalendar`.
- Frozen identities: the manifest stores the **absolute** protocol path and the code hash covers the three
  existing files plus the new wrapper, resolved relative to the release root (module `parents[3]`); the
  runtime hash binds python/requests/numpy/pandas/statsmodels/pandas_market_calendars versions. A release
  must be built (and the venv frozen) before `init`; any change after refuses with `BindingMismatch`.
- `--state-dir` must equal the resolved path recorded in the manifest.

## Test results

`scripts/run_tests.sh tests/lab/test_intraday_observer.py -q` → **148 passed** (78 s, 8 GB cap, no swap).
`ruff check` clean on `src`, the new script and the test file; `ruff format --check` clean. Full suite
not rerun (parent will after final code changes).

## Not done / left to the parent

Task 1 real-source checks; `deploy/` units and runbook (Kimi); live-mode readiness JSON; exact
calendar/manifest for the activation date (must be regenerated after these code changes). No Opus review
yet. Nothing activated.
