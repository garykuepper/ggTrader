# Refactor completion II Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Finish every item the owner left at "default" on 2026-11-20: retire the eight private `PaperTrader` seam forwards (D3), add a runtime refusal of a non-durable ledger in a live run (D4), replace `run_wfo`'s string return with an exception (D5), enable the docstring rules (D6), do the three phase 6 tidy-ups (D7), then refresh the graph.

**Architecture:** Work in separate worktrees, one stream per file area, each stream gated and reviewed before it merges into local `main`. Behavior must not change except where a task says so by name (D4 and D5 are the two intended changes).

**Tech Stack:** Python 3.12, pytest through `scripts/run_tests.sh`, ruff 0.17.4, the gate scripts in `/home/flynn/ggTrader-refactor-evidence/`, `scripts/verify_pins.py`, git worktrees made with `/home/flynn/ggTrader-refactor-evidence/new_wt.sh DIR BRANCH BASE`.

**Spec:** [2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md); earlier plans [completion and cleanup](2026-11-20-refactor-completion-and-cleanup.md), [phase 3](2026-11-18-refactor-phase-3.md) (Task 29 is D5), [phase 4](2026-11-18-refactor-phase-4.md) (Task 45 is the persist transactions), [phase 5](2026-11-19-refactor-phase-5.md); sizing note `/home/flynn/ggTrader-refactor-evidence/p6/phase-6-sizing.md`.

## Owner decisions (2026-11-20, "yes I want all done")

D3 retire the eight seam forwards. D4 add the runtime refusal. D5 raise instead of returning a string. D6 enable `D100`-`D104`. D7 do all three tidy-ups, each with its own measured sub-plan first. Graph refresh at the very end. These replace the "default: skip/keep" rulings in the earlier plan. The trader stays stopped; **no deploy and no push without a new yes.**

## Global Constraints

- The 103 transcript goldens, `run_equivalence_digests.json`, `persist_sql.json` and the other `tests/paper/golden/**` files stay byte-identical in every task except where a task says a named golden changes and why. No golden is regenerated to make a test pass.
- Intended behavior changes are exactly two: D4 (a live run refuses a non-durable ledger) and D5 (`run_wfo` raises). Everything else is behavior-preserving.
- Tests only via `scripts/run_tests.sh`, one heavy job at a time (the repo lock queues others; never kill or bypass), `GGT_TEST_MEM_MAX=3G`; the p5 oracle needs the `6G` cap (measured peak 3.9 GB). A 137 or 143 exit is a hang: stop and report.
- Builders run only targeted tests; the orchestrator runs each stream's full gate (`gate_run.sh LABEL`, `gate_compare.sh LABEL`, with `WT=<worktree>`) and compares with `baseline-core.xml,baseline-studies-r9filtered.xml` (a stream that adds tests only adds nodes; removals must be named and justified).
- Hash-pinned files (`scripts/verify_pins.py`, CI counts at `.github/workflows/ci.yml:147`) are checked after each task: `./offline.sh .venv/bin/python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0`.
- Never `ruff --fix` under `src/ggTrader/paper/`. `ruff check .` and `ruff format --check .` pass. Engine boundary: only `src/ggTrader/engine/` imports vectorbt. `ohlcv.timestamp` stays naive UTC. Nothing under `data/` is deleted. No paid data.
- Frozen scripts never edited: `scripts/p5_wfo_validation.py`, `p7_live_golden.py`, `legacy_reference_*.py`, `verify_pins.py`, `verdict_register.py`, `study.py`, `vbt_latest.py`.
- Real-database tests (Task 6 only): only against a throwaway database whose name starts with `ggt_scratch_`, created and dropped by the test fixture itself, never against `ohlcv`, `paper_*` or any existing database; credentials come from the process environment of that one command, never printed or written to a file.
- Delegation: builders Claude Sonnet 5.5 (Haiku for bulk mechanical work); reviewers gpt-6.2-sol (reviews Claude builds), Claude Opus 5.5 and Claude Fable 5.2 (hard reviews; never reviews a Claude build alone for a trader change: Sol too). Astra only with the owner's approval. Each stream: builder → Sol review → fix loop (max 5 rounds) → gate → merge. Small wording fixes the orchestrator makes itself are recorded as rulings.

## Review Focus

1. A forwarding method is deleted but a test, harness or study still reaches it by attribute, and a test passes vacuously because the patch silently stops intercepting (Task 1): every former seam gets a test that proves the replacement hook is the thing that intercepts.
2. The non-durable-ledger refusal fires in a path that should run (dry runs, the read-only path, the tests' own fixtures) or fails to fire in the one that matters (live, `--live`): Task 2.
3. `run_wfo` now raises where a caller relied on a string (`isinstance(result, str)` in a frozen oracle driver, the blend and lab CLI): Task 3 lists every caller and keeps each one's observable output.
4. A docstring states something false about a function (docstrings are read as truth): Task 4 requires every docstring to be checked against the code by a reviewer.
5. A tidy-up changes an exit code, a printed line, or a SQL text that a systemd unit, a test fixture or the SQL golden pins (Tasks 5, 6, 7).

---

## Stream A (paper/ files, sequential): Tasks 1, 2, 7

### Task 1: Retire the eight seam forwards (D3)

**Files:** Modify `src/ggTrader/paper/trader.py` (remove `_submit_order`, `_poll_orders`, `_reconcile_order_intents`, `_reconcile_pending_orders`, `_flag_if_stale`, `_missed_exits`, `_compute_split_corrections`, `_accrue_dividends` and, if nothing else needs them, the seam plumbing in `seams.py`), the tests and harnesses that assign or patch them (find with `rg -n "_submit_order|_poll_orders|_reconcile_order_intents|_reconcile_pending_orders|_flag_if_stale|_missed_exits|_compute_split_corrections|_accrue_dividends" tests src scripts research`), `docs/architecture.md` (seam rules), `docs/changelog.md`.

**Interfaces:**
- Consumes: `RunHooks` and `TraderSeams` in `src/ggTrader/paper/seams.py`; the step objects (`OrderExecutor`, `CorporateActions`, ...) that now own the bodies.
- Produces: `PaperTrader` without the eight forwards; tests inject the same behavior through the step objects' constructor arguments or `TraderSeams` fields (name them in the report).

- [ ] **Step 1: Measure.** For each of the eight names list every reference (file:line, assignment/patch/call) in `src`, `tests`, `scripts`, `research/*/*/code`, and the frozen harness files that hold them (transcript harness, equivalence runner, phase 4 pins, seam-contract test). Save as `/home/flynn/ggTrader-refactor-evidence/p6/seam-references.md`. If a frozen/hash-pinned harness file must change, list it and why; the owner has approved editing them for this task only, and every golden and the 600 digests must stay byte-identical afterwards.
- [ ] **Step 2: Write a failing test per seam** (names `test_<seam>_is_replaced_by_<hook>`) proving the replacement injection point intercepts the call and that the old attribute no longer exists (`assert not hasattr(PaperTrader, "_submit_order")`).
- [ ] **Step 3: Migrate** each test off the old attribute, then delete the forwards. One commit per seam is fine; one commit if they are small.
- [ ] **Step 4: Verify.** `scripts/run_tests.sh tests/paper -q` (all pass), `git diff --stat -- tests/paper/golden` empty, `ruff check .`. Orchestrator then runs the stream gate.
- [ ] **Step 5: Docs.** Replace the "private seams stay" text in `docs/architecture.md` and add a changelog entry. Commit `refactor(paper): retire the eight private seam forwards`.

### Task 2: Refuse a non-durable ledger in a live run (D4)

**Files:** Modify `src/ggTrader/paper/ledger.py` (a class attribute `durable: bool` on every ledger: `PgPaperLedger` True, `ReadOnlyLedger` follows its inner ledger, `InMemoryLedger` and `SpyLedger` False), `src/ggTrader/paper/trader.py` (`PaperTrader.run`), `tests/paper/conftest.py` (an autouse fixture that lifts the refusal for tests), `tests/paper/test_trader_ledger.py`.

**Interfaces:**
- Produces: `class NonDurableLedgerError(RuntimeError)` in `paper/ledger.py`; `PaperTrader.run()` raises it before any broker or ledger call when the run is live (not dry-run) and `ledger.durable` is False, unless the module flag `ggTrader.paper.ledger.ALLOW_NON_DURABLE_LEDGER` is True (default False; the tests' conftest sets True per test and restores it).

- [ ] **Step 1: Write failing tests.** `test_live_run_refuses_in_memory_ledger` (flag False, live, `InMemoryLedger`: raises `NonDurableLedgerError`, broker and notifier untouched), `test_dry_run_allows_in_memory_ledger`, `test_live_run_accepts_pg_ledger`, `test_read_only_run_allows_in_memory_ledger`. Expect FAIL.
- [ ] **Step 2: Implement** the attribute, the exception, the flag and the check at the very top of `run()` (after nothing else is read). Keep the clock/env read order unchanged (transcript goldens prove it).
- [ ] **Step 3: Verify.** All paper tests pass with the conftest fixture; `tests/paper/golden` unchanged; the static production-import guard still holds. Commit `feat(paper): a live run refuses a non-durable ledger`. This is an intended behavior change: changelog entry says so.

### Task 7: Persist transactions (D7c) — after Tasks 1 and 2 merge

**Files:** Modify `src/ggTrader/paper/persist.py` (the 11 writers that open a connection and call `commit()` by hand), possibly `src/ggTrader/paper/ledger.py`; create `tests/paper/test_persist_real_db.py` (marker `integration`), `tests/paper/persist_scratch_db.py` (the throwaway-database fixture).

**Interfaces:**
- Produces: every writer uses `with engine.begin() as conn:` exactly as the design (5.2) says, with identical SQL text and bind parameters (the SQL golden `tests/paper/golden/persist_sql.json` pins 131 cases and must stay byte-identical).

- [ ] **Step 1: Build the real-database fixture first.** `scratch_db()` creates `ggt_scratch_<uuid8>` on the server named by the process environment, applies only the paper tables' DDL (find it in `persist.py`'s schema function), yields an engine, and drops the database in `finally`; it refuses any name not starting with `ggt_scratch_`. The test is marked `integration` and skipped when no server is reachable.
- [ ] **Step 2: Characterize the CURRENT writers** against the scratch database (write each writer's rows, then read them back with the reader functions; also a failure case: a second statement fails, assert what is and is not committed today). Run it on unchanged code: all pass. Save results.
- [ ] **Step 3: Convert** the 11 writers one at a time to `engine.begin()`, rerunning the SQL golden and the scratch tests after each. If a writer's partial-commit behavior differs on failure, stop and report instead of choosing.
- [ ] **Step 4: Verify** golden diff empty, `tests/paper` pass, scratch tests pass against the real server, scratch database dropped (`psql -l` shows none left). Commit `refactor(paper): persist writers use engine.begin()`.

## Stream B (research engine): Task 3

### Task 3: `run_wfo` raises instead of returning a string (D5)

**Files:** Modify `src/ggTrader/lab/wfo/run.py`, every caller found in Step 1, tests for them; read the earlier analysis in the phase 3 plan Task 29 (`docs/superpowers/plans/2026-11-18-refactor-phase-3.md` around line 1262-1342).

- [ ] **Step 1: List callers** (`rg -n "run_wfo" src scripts tests research/*/*/code`) with what each does with a string result; the oracle driver `scripts/p5_wfo_validation.py` (frozen) has `isinstance(result, str)` then `RuntimeError(result)`, unreachable on its windows. Save the table.
- [ ] **Step 2: Failing tests.** `test_run_wfo_raises_wfo_not_run_with_the_old_message` for each former string-return branch (exception type `WfoNotRun(RuntimeError)`, `str(exc)` equals the old string exactly), and one test per caller that its observable output (message, exit code) is unchanged.
- [ ] **Step 3: Implement** `class WfoNotRun(RuntimeError)` in `lab/wfo/` and raise it where the string was returned; update callers to catch it and reproduce their old output. Frozen scripts are not edited: with the exception, the oracle driver fails with the same text (state this in the changelog).
- [ ] **Step 4: Verify.** `tests/lab tests/engine -q`, oracle p5/p7 via the stream gate unchanged. Commit `refactor(lab): run_wfo raises WfoNotRun instead of returning a string`. Intended behavior change: say so in the changelog.

## Stream C (membership CLI): Task 5; Stream D (weekly job): Task 6

### Task 5: Membership CLI tidy-up (D7a)

**Files:** Modify `src/ggTrader/cli/cmd_membership.py`; tests under `tests/cli` or where the 217 existing tests live; sizing in `p6/phase-6-sizing-raw.md` section (a).

- [ ] **Step 1: Pin first.** Add the missing `apply-ddl` subcommand test and a table test of every subcommand's exit codes (0/1/2) and refusal messages (record today's values); run on unchanged code: pass.
- [ ] **Step 2: Merge** the eight repeated groups (six `connect()` blocks into one helper, three identical "REFUSED" exits into one) without changing any printed byte or exit code; the 26 `monkeypatch.setattr` sites that target module names keep working.
- [ ] **Step 3: Verify.** all membership CLI tests pass; the new table test still passes unchanged. Commit `refactor(cli): merge the repeated membership command blocks`.

### Task 6: Weekly job tidy-up (D7b)

**Files:** Modify `src/ggTrader/data/membership_weekly.py`; tests `tests/data/test_membership_weekly*.py` and the press tests.

- [ ] **Step 1: Pin first.** Add a golden-style test of `run_weekly`'s findings and message text for the paths with near-identical findings ("not fetched" three times, "structure" twice) on unchanged code.
- [ ] **Step 2: Merge** the near-identical finding builders into one helper; then split `run_weekly` (149 lines, McCabe 19) into ordered steps in the same module; the order of steps and the text of every message are unchanged.
- [ ] **Step 3: Verify.** tests pass; remove the file from the `ruff.toml` survivor list only if the rules no longer fire there (check, do not assume). Commit `refactor(data): the weekly membership job in ordered steps`.

## Stream E (docstrings, last): Task 4

### Task 4: Docstring rules `D100`-`D104` (D6) — after Streams A, B, C, D have merged

**Files:** `ruff.toml` (`select` gains `D100`, `D101`, `D102`, `D103`, `D104`; convention google; skip `D107`; scoped to `src/ggTrader/` only), every `src/ggTrader/**/*.py` with a hit (about 150).

- [ ] **Step 1: Measure** hits per package with the real config; split into batches by package (`cli`, `data`, `engine`, `lab`, `paper`, `utils`) for parallel Haiku writers, each in the same worktree but disjoint files.
- [ ] **Step 2: Write** one-line truthful docstrings (what the function does, from reading the body; no invented behavior). Pinned files (`verify_pins.py` list) get a per-file ignore instead of an edit unless the counts line is updated.
- [ ] **Step 3: Review.** A Sonnet or Opus reviewer reads every new docstring against its function and reports any that is false or misleading; fix them. Then enable the rules and `ruff check .`.
- [ ] **Step 4: Verify.** `ruff check . && ruff format --check .`, `verify_pins.py`, stream gate (docstring edits cannot change behavior but the gate proves it). Commit per package, then `lint: enable the docstring rules for src/ggTrader`.

## Finish

### Task 8: Final gate, graph, docs, hand-back

- [ ] **Step 1:** Merge each stream into local `main` in order A (Tasks 1, 2, 7), B, C, D, E with a gate on the merged tree after each merge that changed production code (`gate_run.sh <label>` with `WT=/home/flynn/ggTrader-merge`, then `gate_compare.sh`).
- [ ] **Step 2:** Update `docs/next_steps.md`, `docs/changelog.md` and `docs/architecture.md` (seams gone, durable-ledger refusal, `WfoNotRun`, docstring rules), and regenerate nothing else.
- [ ] **Step 3: Refresh the graph at the very end** (`/graphify` update on the merged tree in `/home/flynn/ggTrader-merge`; report node/edge counts before and after).
- [ ] **Step 4:** Remove merged worktrees and branches only after the plan's proof (`git branch --merged main`, clean status), update memory, and hand back with the full "Rulings I made" list. Do not push or deploy without a new yes.

---

## Self-review

- **Coverage:** D3 → Task 1, D4 → Task 2, D5 → Task 3, D6 → Task 4, D7 a/b/c → Tasks 5/6/7, graph → Task 8. Every Review Focus line has its tests in the owning task (1: Task 1 Step 2; 2: Task 2 Step 1; 3: Task 3 Steps 1-2; 4: Task 4 Step 3; 5: Tasks 5-7 Step 1).
- **Unmeasured at writing:** seam reference counts, the exact callers of `run_wfo`, the weekly job's golden shape, the persist writers' failure behavior, and the docstring hit counts per package. Each task starts with the measuring step; a task that finds the plan's assumption false reports instead of guessing.
- **Order and conflicts:** Stream A is sequential (Tasks 1, 2, 7 all touch `paper/`); Streams B, C, D touch disjoint files and run in parallel with A; Stream E waits because docstrings touch every file.
