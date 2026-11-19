# Test-run safety (prevent a repeat of the 2026-11-18 out-of-memory incident) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible for two heavy ggTrader jobs (test runs, oracle drivers) to run at once, and make a hung job die in seconds or minutes instead of eating the machine.

**Architecture:** One shared helper, `scripts/_heavy_job.sh`, gives `scripts/run_tests.sh` and `scripts/run_capped.sh` a single-flight lock (shared by every worktree through ggTrader's git directory), a free-memory preflight and a wall-clock limit. The default test memory cap drops from 8G to 4G. A per-test timeout in the root `conftest.py` turns a hang that does not allocate into a test failure. Builder and orchestrator rules are written into the refactor plans and `CLAUDE.md`.

**Tech Stack:** bash (`flock`, `systemd-run --user --scope`, `/proc/meminfo`), pytest, Python `signal.setitimer`.

**Spec:** the incident of 2026-11-18, recorded in `/home/flynn/.claude/projects/-home-flynn-ggTrader/memory/feedback_one_test_run_at_a_time.md` and the existing rationale in the header of `scripts/run_tests.sh`. What happened: three builders started at 13:51 in one worktree and two `pytest` runs over `tests/paper/` grew without bound (a logger left at INFO consumed stubbed `time.time()` values, so the order-polling loop never reached its deadline; fixed in commit `a44fa90a`). Each run had its own 8 GiB cap, together they needed more than the machine had free, so neither was killed and Encom thrashed until it was power-cycled at 17:58.

## Scope fence (owner instruction, 2026-11-18)

Only how **ggTrader is tested and run** changes. Allowed: files in the ggTrader repository (`scripts/`, `conftest.py`, `tests/`, `pyproject.toml` markers, `CLAUDE.md`, `docs/`, `.gitignore`) and the instructions given to builders. **Not allowed:** system services or units, sysctl, swap or kernel settings, other projects, `~/.claude` settings or hooks, cron, Docker, or anything under `/etc`. The lock file lives in ggTrader's own git directory. `systemd-run --user --scope` is already what the scripts use; this plan only adds properties to the scope they already create.

## Global Constraints

- Work in a worktree of ggTrader; never push, merge or edit `main` without the owner's say-so.
- Run tests only through `scripts/run_tests.sh` (and long drivers through `scripts/run_capped.sh`), never a bare `pytest`. From the first task on, they take the lock.
- **One heavy job at a time while this plan is built.** No parallel builders edit the two runner scripts; no test run starts while another is running (check `ps -eo args | grep -E "pytest|run_tests|p5_wfo|p7_live"`).
- `GGT_TEST_NO_CGROUP=1` (CI) keeps today's behavior: no lock, no preflight, no time limit.
- New shell code follows the existing script style (`set -euo pipefail`, header comment as documentation, `--help` works).
- `ruff check .` and `ruff format --check .` stay clean; commits end with the attribution line from the session instructions.
- Verified facts this plan relies on (2026-11-18): `systemd-run --user --scope -p RuntimeMaxSec=3 sleep 30` is killed after 3 s with exit 143; `git rev-parse --path-format=absolute --git-common-dir` prints `/home/flynn/ggTrader/.git` from every worktree; the slowest single test in the recorded baseline takes 20.9 s and all 3961 tests together take 388 s; `/usr/bin/flock` exists.

## Review Focus

Failure modes the tasks' tests must pin, most likely first:

1. **Nested runs deadlock.** `tests/scripts/test_run_capped.py` and `tests/test_root_conftest.py` run these scripts from inside a pytest that `run_tests.sh` already started. A child that waits for the parent's lock waits for 30 minutes. Pinned by `test_nested_call_does_not_wait` and `test_dry_run_takes_no_lock`.
2. **Two runs at the same instant from two worktrees** (the incident). The second must wait, then run, never overlap. Pinned by `test_second_job_waits_then_refuses` and `test_two_jobs_never_overlap`.
3. **A killed holder.** If the first run is `kill -9`'d, the kernel releases the lock; a stale holder note must never block the next run. Pinned by `test_killed_holder_releases_the_lock`.
4. **Low free memory.** A run that would need more than is free must refuse at once, not wait and not start. Pinned by `test_low_memory_refuses`.
5. **A hang that does not allocate** (a spin loop) never reaches the memory cap. Pinned by the per-test timeout tests and the runtime limit test.

---

### Task S1: Single-flight lock, memory preflight and time limit for `run_capped.sh`

**Files:**
- Create: `scripts/_heavy_job.sh` (sourced, defines functions, no side effects when sourced)
- Modify: `scripts/run_capped.sh`
- Modify: `.gitignore` (add `/.ggtrader-heavy.lock*`, the fallback when no git directory exists)
- Test: `tests/scripts/test_heavy_job_guard.py`

**Interfaces:**
- Produces, in `scripts/_heavy_job.sh`:
  - `heavy_job_enter LABEL MEM_MAX` returns 0 when this process may start a heavy job and otherwise exits 75 (EX_TEMPFAIL). Order of work: (1) return 0 at once if `GGT_TEST_NO_CGROUP=1`, `GGT_HEAVY_LOCK_HELD=1` or `GGT_TEST_DRY_RUN=1`; (2) open the lock file on fd 9 and `flock -w "${GGT_HEAVY_WAIT:-1800}" 9`, printing `waiting for another heavy ggTrader job (<holder note>)` to stderr once; on timeout print `another heavy ggTrader job is still running after <N>s: <holder note>` and exit 75; (3) write the holder note (`pid=<pid> label=<LABEL> cap=<MEM_MAX> started=<UTC time>`) to `<lock>.holder` and remove it in an EXIT trap; (4) `export GGT_HEAVY_LOCK_HELD=1`; (5) preflight: read `MemAvailable` from `${GGT_MEMINFO:-/proc/meminfo}` and, if it is below `MEM_MAX` plus `${GGT_HEAVY_MARGIN:-4G}`, print `not enough free memory for a <MEM_MAX> job: <free> available, <needed> needed` and exit 75.
  - `heavy_job_lock_path` prints `${GGT_HEAVY_LOCK:-}` if set, else `<git common dir>/ggtrader-heavy.lock` from `git -C "$REPO" rev-parse --path-format=absolute --git-common-dir`, else `$REPO/.ggtrader-heavy.lock`.
  - `size_to_kb SIZE` converts `512M`, `4G` to kilobytes.
- Produces, in `run_capped.sh`: `GGT_RUN_MAX_SECONDS` (default 7200) passed as `-p RuntimeMaxSec=`; exit 143 prints `STOPPED AT THE TIME LIMIT (<N>s): a hang, not a long job; report it`.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests** in `tests/scripts/test_heavy_job_guard.py`, with `needs_systemd_scope` copied from `tests/scripts/test_run_capped.py`. Every test passes `GGT_HEAVY_LOCK=<tmp_path>/lock` so it never touches the real lock, and `GGT_HEAVY_WAIT=1` where it expects a refusal.
  - `test_second_job_waits_then_refuses`: a child holds the lock (`flock <lock> sleep 30`); `scripts/run_capped.sh true` exits 75 within 6 s and stderr contains `waiting for another heavy ggTrader job` and `still running after 1s`.
  - `test_killed_holder_releases_the_lock` (needs systemd): start the holder, `kill -9` it, then `scripts/run_capped.sh true` exits 0; the stale `<lock>.holder` file does not block it.
  - `test_nested_call_does_not_wait` (needs systemd): hold the lock, set `GGT_HEAVY_LOCK_HELD=1`, run `scripts/run_capped.sh true`: exit 0 in under 10 s.
  - `test_dry_run_takes_no_lock`: hold the lock, set `GGT_TEST_DRY_RUN=1`, run `scripts/run_tests.sh`: exit 0, no waiting message (this test is satisfied in Task S2; mark it with the task that makes it pass).
  - `test_low_memory_refuses`: `GGT_MEMINFO` points to a file containing `MemAvailable:    1048576 kB`; `GGT_RUN_MEM_MAX=6G`: exit 75, stderr contains `not enough free memory`.
  - `test_two_jobs_never_overlap` (needs systemd): launch two `run_capped.sh` jobs that each append `start`/`end` lines with timestamps to one file around a 2-second sleep; the intervals do not overlap.
  - `test_runtime_limit_stops_a_hang` (needs systemd): `GGT_RUN_MAX_SECONDS=2 scripts/run_capped.sh sleep 30` exits 143 within 10 s and prints `STOPPED AT THE TIME LIMIT`.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/scripts/test_heavy_job_guard.py`. Expected: FAIL (missing helper).
- [ ] **Step 3:** Implement `scripts/_heavy_job.sh` and source it from `run_capped.sh` (call `heavy_job_enter run_capped "$MEM_MAX"` before `systemd-run`; add `-p RuntimeMaxSec="$MAX_SECONDS"`; extend the `--help` text and header comment). The lock descriptor must stay open for the child's lifetime: do not `exec` or close fd 9.
- [ ] **Step 4:** Run the file again. Expected: all pass except the S2-marked test; also run `scripts/run_tests.sh tests/scripts/test_run_capped.py tests/scripts/test_scripts_have_help.py` and expect no regression.
- [ ] **Step 5: Commit** `scripts: single-flight lock, free-memory preflight and time limit for run_capped.sh`.

### Task S2: The same guard, a lower default cap and a time limit for `run_tests.sh`

**Files:**
- Modify: `scripts/run_tests.sh`
- Test: `tests/scripts/test_heavy_job_guard.py` (extend), `tests/test_root_conftest.py` (only if its dry-run expectations need the new message)

**Interfaces:**
- Consumes: `heavy_job_enter` from Task S1.
- Produces: `GGT_TEST_MEM_MAX` default `4G` (was `8G`); `GGT_TEST_MAX_SECONDS` default `1800`, applied as `-p RuntimeMaxSec=` to each of the (one or two) pytest scopes; exit 143 prints `STOPPED AT THE TIME LIMIT (<N>s): a hang, not a long suite; report it`. The lock is taken once at the top of the script and held across BOTH pytest runs of `--full` and `--studies`.

- [ ] **Step 1: Write the failing tests** (add to the file from S1): `test_run_tests_default_cap_is_4g` (`GGT_TEST_DRY_RUN=1 scripts/run_tests.sh` stderr contains `MemoryMax=4G`); `test_run_tests_lock_covers_both_halves` (needs systemd: start `scripts/run_tests.sh --studies --collect-only -q` with `GGT_HEAVY_LOCK=<tmp>/lock` as a subprocess and poll `flock -n <tmp>/lock true` every 0.2 s until it exits; the lock is observed held at least once and never free again before the script ends, so it is held across both pytest runs); `test_run_tests_refuses_when_busy` (lock held, `GGT_HEAVY_WAIT=1`: exit 75); `test_dry_run_takes_no_lock` from S1 now passes.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Source `_heavy_job.sh` in `run_tests.sh`; call `heavy_job_enter run_tests "$MEM_MAX"` after argument parsing and before the first `run_pytest`; change the default to `4G`; add `-p RuntimeMaxSec="${GGT_TEST_MAX_SECONDS:-1800}"` to the `systemd-run` call in `run_pytest`; map status 143 to the time-limit message next to the existing 137 message; update the header comment (cap, lock, limit, exit 75).
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/scripts/ tests/test_root_conftest.py`. Expected: PASS (the nested runner tests must not wait).
- [ ] **Step 5: Commit** `scripts: run_tests.sh takes the heavy-job lock, caps at 4G and stops a hang at 30 minutes`.

### Task S3: Per-test timeout

**Files:**
- Create: `tests/timeout_guard.py` (plain module like `tests/netguard.py`), `tests/timeout_probe/hang_probe.py` (one test, `def test_hangs(): while True: pass`; never collected by name)
- Modify: `conftest.py` (root), `pyproject.toml` (register marker `timeout_seconds(n)`)
- Test: `tests/test_timeout_guard.py`

**Interfaces:**
- Produces: `class TestTimedOut(Exception)`; `time_limit(seconds: float)` context manager built on `signal.setitimer(ITIMER_REAL)` that raises `TestTimedOut(f"test exceeded {seconds:g}s (a hang?)")` in the main thread, restores the previous handler and timer on exit, and does nothing for `seconds <= 0`, off the main thread or without `SIGALRM`. In the root conftest, an autouse fixture applies `time_limit(float(os.environ.get("GGT_TEST_TIMEOUT", "180")))`, or the first argument of a `@pytest.mark.timeout_seconds(n)` marker when present.

- [ ] **Step 1: Write the failing tests** in `tests/test_timeout_guard.py`: `test_spin_loop_is_interrupted` (`time_limit(0.5)` around `while True: pass` raises `TestTimedOut` within 3 s); `test_previous_handler_and_timer_restored`; `test_zero_disables`; `test_fixture_fails_a_hanging_test` (run `python -m pytest tests/timeout_probe/hang_probe.py -p no:cacheprovider` as a subprocess from the repo root with `GGT_TEST_TIMEOUT=1`: return code 1, output contains `exceeded 1s`, finishes within 20 s); `test_marker_overrides_the_default` (a probe function marked `timeout_seconds(1)` with the env default at 600 also fails in 1 s: add a second probe test selected with `-k`).
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement the module, the fixture and the marker registration. The fixture must not break `tests/netguard.py` or the transcript harness (`HarnessError` is a `BaseException` and passes through).
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/test_timeout_guard.py tests/paper/ tests/utils/test_logging_setup.py`. Expected: PASS. Then check that no existing test uses `SIGALRM`: `grep -rn "SIGALRM\|setitimer\|alarm(" tests research src --include=*.py` lists only the new module.
- [ ] **Step 5: Commit** `test: per-test timeout turns a hung test into a failure (default 180s, GGT_TEST_TIMEOUT)`.

### Task S4: Rules for builders and orchestrator, docs and the proof

**Files:**
- Modify: `CLAUDE.md` (section 3, "Running the test suite"), `docs/changelog.md`
- Modify: the three refactor plans in `docs/superpowers/plans/` (`2026-11-17-refactor-phase-0-2.md`, `2026-11-18-refactor-phase-3.md`, `2026-11-18-refactor-phase-4.md`): the "Heavy-job rules" block already inserted under their Global Constraints is updated to refer to the lock as built.

- [ ] **Step 1:** In `CLAUDE.md` section 3 add: tests and drivers run only through the two scripts; at most one heavy job runs at a time (they queue on a lock in the git directory, shared by all worktrees; `GGT_HEAVY_WAIT`, default 1800 s); the preflight refuses a job when free memory is below the cap plus 4G (exit 75); the default test cap is 4G, the time limit is 1800 s per pytest run (`GGT_TEST_MAX_SECONDS`) and 7200 s for `run_capped.sh` (`GGT_RUN_MAX_SECONDS`); exit 137 or 143 means a hang, not a big job; each test is stopped after 180 s (`GGT_TEST_TIMEOUT`, marker `timeout_seconds`). Add the 2026-11-18 incident to the existing incident paragraph.
- [ ] **Step 2: Builder rule for every delegated brief** (copy this block into the briefs): "Run tests only as `./offline.sh scripts/run_tests.sh <small targeted paths>`. Never run `tests/paper/`, a whole tier or an oracle driver unless the brief names it: the orchestrator runs those, one at a time. If you see `waiting for another heavy ggTrader job`, wait; never kill another run and never bypass the scripts. Exit 75 means busy or low memory, not a failure of your change. Exit 137 or 143 means a hang: stop and report."
- [ ] **Step 3: Proof, run one job at a time, small caps only.** (a) `GGT_TEST_MEM_MAX=512M scripts/run_tests.sh tests/timeout_probe/hang_probe.py -p no:cacheprovider` with `GGT_TEST_TIMEOUT` unset: the probe is stopped by the 180 s per-test timeout; to see it fast set `GGT_TEST_TIMEOUT=5`: expect a failure naming `exceeded 5s`. (b) A throwaway allocating loop (`python -c "x=[]; [x.append(b'x'*10**7) for _ in iter(int,1)]"` through `scripts/run_capped.sh` with `GGT_RUN_MEM_MAX=512M`): expect exit 137 and the existing cap message within seconds. (c) Start two `GGT_RUN_MAX_SECONDS=20 scripts/run_capped.sh sleep 8` from two worktrees (the refactor worktree and `/home/flynn/ggTrader-baseline`): the second prints `waiting for another heavy ggTrader job` and starts only after the first ends. (d) `GGT_HEAVY_MARGIN=100G scripts/run_tests.sh tests/utils/test_logging_setup.py`: exit 75 with `not enough free memory`. Record the four outputs in the commit message.
- [ ] **Step 4: Full gate at the new cap.** Run once, alone: `GGT_TEST_MEM_MAX=2G ./offline.sh scripts/run_tests.sh --full -rA --junitxml=<evidence>/safety-core.xml tests` and `./offline.sh scripts/run_tests.sh --studies -rA --junitxml=<evidence>/safety-studies.xml`, where `<evidence>` is `/home/flynn/ggTrader-refactor-evidence`, then `<evidence>/gate_compare.sh`-style comparison to the baseline files. Expected: both exit 0 under a 2G cap (half the new default, a quarter of the old one) and `compare_test_outcomes.py` exits 0. If either dies at 2G, record the peak and keep 4G; if it dies at 4G, stop and report.
- [ ] **Step 5:** Add a `docs/changelog.md` entry (what changed, why, the incident, the new knobs, that CI is unaffected) and commit `docs: test-run safety rules (lock, cap, time limits) after the 2026-11-18 out-of-memory incident`.

---

## Self-review

- **Spec coverage:** the incident has three causes and each has a layer: parallel runs (S1, S2 lock), caps that add up (S1, S2 preflight and the 4G default), hangs that run long or never allocate (S2 and S1 time limits, S3 per-test timeout); builders starting runs themselves (S4 rule, which the lock enforces mechanically).
- **Scope fence:** every file named is inside the ggTrader repository; no system setting is touched; the lock is in `.git`.
- **Type and name consistency:** `heavy_job_enter`, `GGT_HEAVY_LOCK`, `GGT_HEAVY_LOCK_HELD`, `GGT_HEAVY_WAIT`, `GGT_HEAVY_MARGIN`, `GGT_MEMINFO`, `GGT_TEST_MAX_SECONDS`, `GGT_RUN_MAX_SECONDS`, `GGT_TEST_TIMEOUT`, `timeout_seconds` and exit code 75 are used the same way in every task.
- **Proportion:** four tasks; the script bodies are left to the implementer, with the order of operations and every message string fixed.

## Execution roles (`CLAUDE.md` delegation preference)

Tasks are small, strictly serial (S2 needs S1's helper, S4's proof needs S1 to S3) and they edit the scripts every other job uses, so they run one builder at a time, with no other test run active. Builder: Codex Terra for S1 and S2 (shell), Luna or Sonnet for S3 and S4. Reviewers: Opus for Codex-built tasks and Sol for Claude-built ones; both review the whole branch before any merge. The owner reports quota before execution.
