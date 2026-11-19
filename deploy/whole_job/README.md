# Whole-job broker: guest-tested candidate, NOT installed

**Status (2026-11-19): candidate only.** Nothing here is installed on Encom, no unit is enabled,
no user or group exists, and no memory limit is chosen. The unit files are templates
(`__BROKER_ROOT__`, `__JOB_CHECKOUT__`). It passed one run of a disposable Ubuntu 26.04 guest
(systemd 259.9, kernel 7.0.0-43, Python 3.14.4; results in `guest/`). It has not had its
required independent security review, and it does not by itself satisfy Task 4 of the memory
recovery plan (see "Not covered").

## What it is

| Piece | File | Runs as |
| --- | --- | --- |
| Broker | `scripts/whole_job_broker.py serve` in `ggtrader-job-broker.service` | root, `Type=notify` |
| Admission record | `scripts/whole_job_admission.py` in `/var/lib/ggtrader-job-broker` (0700) | broker |
| Job unit | `ggtrader-whole-job.service`, main process `scripts/whole_job_driver.py` | `ggtrader-job`, nondelegated |
| Operator requests | `ggtrader-job-operator@<request>.service`, `whole_job_broker.py request` | root oneshot |

Operator requests are the unit instance, never message content:
`@status`, `@start`, `@stop:<attempt>`, `@acknowledge:<attempt>`. A stop or acknowledge must
name the current attempt (from `@status`), so a delayed request cannot stop its replacement.

## Trust and identity

- **No caller-supplied paths, PIDs, cgroups or unit names.** They are constants in the broker
  and the unit files.
- **Per-message sender identity.** The broker sets `SO_PASSPIDFD` on each accepted socket and
  sends a nonce first, so every request carries a kernel-attached `SCM_PIDFD`. PID 1
  `GetUnitByPIDFD` must name the fixed unit; that unit must be nondelegated, its `MainPID` and
  `InvocationID` must match the pidfd, `/proc/<pid>/cgroup` must equal its `ControlGroup`, and
  the pidfd must still name the same PID after the lookups. A descendant writing on an
  inherited socket has its own pidfd and is refused. Operators must also be UID 0.
- **Job binding.** The record's job string binds a random attempt, the invocation, the cgroup
  id (`ControlGroupId`, checked against the opened directory inode) and the device/inode of
  `cgroup.kill` (write-only) and `cgroup.events` (read-only) opened `O_NOFOLLOW` from
  `/sys/fs/cgroup`; the boot id is stored beside it.
- **Exact descriptor store.** Both handles are sent with `FDSTORE=1`, `FDPOLL=0`, then a
  barrier. Because a full or unauthorized store can still report success, the broker reads
  `DumpUnitFileDescriptorStore` and admits the job only if the store holds exactly those two
  names with matching regular-file type, device, inode and access mode. Otherwise it kills
  that exact unadmitted cgroup, removes its entries and stays held. It forgets that
  invocation only once the cgroup is verified empty and the store is verified empty. If the
  kill fails, the cgroup does not empty or the drain is not verified, the job's identity is
  persisted as held and the pinned handles are kept (mode `held`): `@stop:<attempt>`, the
  cgroup emptying, or a broker restart that restores the stored pair clears it. If no identity
  can be recorded, the broker goes `unverifiable` and keeps the handles open.
- **No fork under the admission lock.** PID 1 queries (`busctl`) run only outside it.

## Sequencing

- **Start:** operator `@start` requires mode idle, a held and cleared record, an empty
  store and an inactive or failed job unit. The broker generates the attempt and starts the
  unit; the driver runs nothing until it registers and receives `go`. The record is written,
  then `go` is sent under the admission lock. Each phase (`core`, then `studies`, once each,
  in order) is admitted under the lock, with a broker deadline.
- **Stop** (operator, phase deadline, lost driver connection, driver report, or the pinned
  `cgroup.events` reporting empty): persist the hold, verify the handles, write
  `cgroup.kill`, wait up to 20 s for `populated 0` or removal (`ENODEV`), then clear the
  record and remove the stored handles. If the cgroup does not empty, the record stays held
  and start is refused. A termination the broker did not start is reported as "ended
  without a driver report; cause not determined", never as out-of-memory.
- **Broker restart:** admission starts held. With the same boot and an exact store, handles
  are restored (mode `held`): no further phase is admitted, and `@stop:<attempt>` or the
  cgroup emptying clears it. Any mismatch leaves mode `unverifiable`, which refuses start and
  stop and needs manual investigation.
- **Reboot:** nothing is enabled, so neither the broker nor the job starts. On a later broker
  start a record from another boot is `prior-boot`; only `@acknowledge:<attempt>` clears it.

## Guest evidence (2026-11-19, one run)

`guest/harness.sh` (with `guest/user-data`) ran inside a throwaway QEMU guest from the signed
Ubuntu 26.04 cloud image, no network; all 57 checks passed (`guest/2026-11-19-guest-results.txt`).
The host-refusal guard at the top of the saved harness was added after the run.

**Correction (2026-11-19, after an independent Sol review):** the `s1-outcome` check in that
run was vacuous: its command was always true, so its PASS proves nothing. 56 of the 57 checks
were real. The harness now asserts that `@status` reports `job ended; driver reported status
0` after a normal run, and `tests/scripts/test_whole_job_guest_harness.py` fails on any check
that cannot fail. The corrected harness has **not** been run in a guest yet. The broker and
runner fixes from the same review (below) are also untested in a guest. Covered in the run:

- Direct `systemctl start` of the job unit without an operator start: refused, no phase ran.
- Normal run: both phases in order, status 0 reported, record cleared, store empty, no job
  processes left. The fixture phase's `systemd-run --user --scope`, `systemd-run --scope` and
  write to the root `cgroup.procs` all failed.
- Operator stop mid-phase: stale attempt refused without a hold; real stop killed the cgroup
  (including a `setsid` grandchild), second phase never started, record cleared.
- `systemctl restart` of the broker with the driver frozen (`SIGSTOP`): handles restored by
  exact store match, job held, stop through the restored `cgroup.kill` emptied it.
- Broker `SIGKILL` with automatic restart: driver ended its phase on lost connection, broker
  restored handles and cleared as "without a driver report".
- Broker `stop` then `start` (`FileDescriptorStorePreserve=yes`): handles restored, stop worked.
- Root process outside the operator unit: refused.
- Guest reboot mid-phase: nothing resumed; `prior-boot`; start and a wrong-attempt
  acknowledge refused; the right acknowledge cleared; a new job then completed.

## Not tested or not covered

- **Not run on Encom** and not reviewed: a cross-vendor security review (Sol for a Claude
  build, per the project's delegation rules) is required before any install.
- In the guest: tampered or full descriptor stores, extra or duplicate entries, stop timeout
  with unkillable (D-state) processes, phase deadline expiry, a crash between the store push
  and the record write, PID reuse, descendant messages on an inherited socket, and the real
  `scripts/run_tests.sh` (a fixture script stood in for the phases). These are covered only by
  the fixture tests in `tests/scripts/test_whole_job_broker.py` and
  `tests/scripts/test_whole_job_driver.py`, which do not exercise systemd or the kernel.
- Only one fixed job at a time. T3 agents, SSH sessions, direct `pytest`, old worktree
  wrappers, cron and Docker are outside this boundary; this is not the broad pool, and it
  does not decide kernel OOM victims or oomd behaviour.
- No memory, swap or CPU limits are set; they need Task 2 measurements and Gary's decision.
- The dedicated user must be able to read the job checkout and its virtualenv; test data and
  `.env` exclusion under that user are untested.
- **Heavy-job lock and memory preflight.** The driver starts each phase with
  `GGT_TEST_NO_SCOPE=1`, not CI's `GGT_TEST_NO_CGROUP=1`: `scripts/run_tests.sh` skips only the
  nested `systemd-run` scope and still takes the shared lock in the git common directory and
  checks `MemAvailable`. In this mode it refuses (exit 75) instead of carrying on when it cannot
  find the git directory, cannot open the lock or cannot read `MemAvailable`. Under
  `ggtrader-job` this needs git to accept the checkout (ownership / `safe.directory`) and write
  access to the lock file. That privilege grant needs review and has not been tried. The
  preflight is sized from `GGT_TEST_MEM_MAX` (default 4G), not from the unit's cap, which is
  still unset. When the measured `MemoryMax` is chosen, the preflight value must match it.
  The real `run_tests.sh` has never run under the broker (the guest used a fixture).
- Wiring `@stop:<attempt>` into the recovery dispatcher (Task 3) and alert classification
  (Task 6) are not done.
- The admission store and driver depend on a trusted root-owned copy of `scripts/`; installing
  from a user-writable checkout would defeat the boundary.
