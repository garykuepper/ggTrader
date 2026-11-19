# Refactor phases 0 to 2: what is left (revision 5, 2026-11-19)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 2 (annotation style and the first lint rules), prove phases 0 to 2 with the full gate and a whole-branch review, ship the one owner-approved behavior change that Phase 4 depends on (Telegram total shows both percentages), and lay out the road through phases 3 to 6.

**Architecture:** The safety nets (comparers, connection guard, 95 trader transcripts, loader characterization) and the heavy-job guard are built and green. What is left is a mechanical lint autofix that is applied but uncommitted, eight hand fixes plus the lint rules, the gates, a review, and one notifier change. Every behavior-neutral step must leave the transcripts, the oracle results and the test outcomes identical.

**Tech Stack:** Python 3.12 (uv `.venv`), pytest through `scripts/run_tests.sh`, ruff 0.17.4, typer, SQLAlchemy, the vectorbt Rust engine (read-only here).

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md). The earlier revisions of this plan, with the full text of Tasks 0 to 13, are in git: `git show 246200f7:docs/superpowers/plans/2026-11-17-refactor-phase-0-2.md`.

## Status (2026-11-19)

Branch `refactor/phase-0-2` at `/home/flynn/ggTrader-refactor`, HEAD `b562f6be`, created from `main` commit `8866f446`. Nothing is pushed or merged. Gates recorded: Phase 0 (core 3292, studies 1144) and Phase 1 (core 3298, studies 1144, oracles identical) passed; the Task 14 autofix gate (`t14a`) matched outcomes and differed from the baseline oracle only in `tape.path` (Task 14.0).

| Task | What | Commits |
|---|---|---|
| 0 | worktree, offline wrapper, baselines, comparers | `04f55df9` `13db1e28` `53cc9d82` `51fd5fec`, plan `40a5e3c3` |
| 1 | connection guard (sockets, lookups, psycopg2, curl_cffi, audit hook) | `aa85e6d3` `d6a14a3e` |
| 2, 3 | study-import surface test; import-direction rules | `ea161c89`; `04958996` `7d2c8fd5` |
| 4 | trader transcripts, 95 scenarios, mutation-checked | `54e7c0bc` `058296bf` `a270fc5a`, docs `246200f7` |
| 5 | `fetch_stock_ohlcv` characterization, 28 tests | `faa5e2ce` `68f7e7ad` |
| 6 to 9 | dead untradable branch; unused parameter; docs; wipe flags | `ea1a762f`; `9dd26f26`; `c509d7c1`; `641c7014` |
| 10 to 13 | cached engine home; paper constants; scripts; logging | `f9b7a093`; `2c4a0dcd`; `05cbb12c`; `a44fa90a` |
| test-run safety (own plan) | heavy-job lock, 4G cap, time limits, per-test timeout | `ecc93843` to `b562f6be` |
| **14** | **annotation autofix applied, uncommitted; lint rules and hand fixes not started** | **this plan** |

What changed since revision 4 and why it matters here:
- **Evidence moved** to `/home/flynn/ggTrader-refactor-evidence/` (a reboot cleared `/tmp`). The baselines were re-recorded there from the untouched base commit (worktree `/home/flynn/ggTrader-baseline`).
- **`run_tests.sh --full` writes two JUnit files that overwrite each other**, so every gate keeps one file per half and the comparer takes both.
- **Two parallel runs once exhausted the host.** The cause was a hang: the `ggt` logging callback left INFO on and paper tests that stub `time.time()` starved (fixed in `a44fa90a`). The scripts now serialize heavy jobs (see Global Constraints).
- **Transcripts grew from 36 to 95 scenarios** after two independent reviews found regressions that would have passed.

## Scope of what is left

**In this plan:** Task 14.0 to 14.4 (finish Phase 2 and review phases 0 to 2), 14.5 (merge to `main`, owner-gated) and 14.6 (the Telegram change). **Then, in their own plans:** Phase 3 (`2026-11-18-refactor-phase-3.md`, tasks 15 to 29) and Phase 4 (`2026-11-18-refactor-phase-4.md`, tasks 30 to 45) are written and Sol-reviewed; Phase 5 and the optional Phase 6 need plans (see "The road to done").

**Not done, on purpose:** edits to `research/`; the oracle and standard-library-only scripts (`p5_wfo_validation.py`, `p7_live_golden.py`, `legacy_reference_runs.py`, `legacy_reference_compare.py`, `verify_pins.py`, `verdict_register.py`, `study.py`, `vbt_latest.py`); deleting the unused database writers (owner decision O3, keep); anything that pushes, deploys or touches cron.

## Owner decisions (recorded)

- **O1** delete the dead untradable branch: done (`ea1a762f`). **O2** wipe flags: done (`641c7014`). **O3** keep the unused writers.
- **O4 (2026-11-18): the Telegram total shows both percentages**, against the day's starting value and against starting capital: Task 14.6.
- **O5** execution approved 2026-11-18. **Merging to `main` and pushing need a separate say-so** (Task 14.5).

## Global Constraints

`E` below is `/home/flynn/ggTrader-refactor-evidence`.

- **Branch and worktree.** Work on `refactor/phase-0-2` in `/home/flynn/ggTrader-refactor`. Every command uses the worktree's own `.venv/bin/python` and `.venv/bin/ruff` by absolute path and runs through `./offline.sh` (an untracked wrapper at the worktree root, listed in the shared `info/exclude`). Never commit to `main`, push or merge without the owner's say-so. Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- **Offline by construction.** No `.env`; `./offline.sh` runs `env -i HOME PATH XDG_RUNTIME_DIR LANG=C.UTF-8 TZ=UTC GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02 "$@"` (`XDG_RUNTIME_DIR` is needed by `systemd-run`; it is not a credential). Tests must never reach the database, Alpaca, Telegram, yfinance, Tiingo or the network; never `ggt paper --live`, never cron.
- **Heavy jobs run one at a time** (enforced by `scripts/_heavy_job.sh`, merged in `b562f6be`): a pytest run or an oracle driver takes a lock in the git directory; a second one waits up to 1800 s then exits 75; a job exits 75 when free memory is below its cap plus 4G. Defaults: test cap 4G, 1800 s per pytest run, 180 s per test. Exit 137 or 143 means a hang, never a big suite: stop and report, never raise the cap. Delegated builders run only small targeted paths and never `tests/paper/`, a whole tier or an oracle driver unless their brief says so.
- **Tests only through `scripts/run_tests.sh`** (never a bare `pytest`). Test outcomes are gated with two files per run, one per half:
  `./offline.sh scripts/run_tests.sh --full -rA --junitxml=$E/<label>-core.xml tests` and `./offline.sh scripts/run_tests.sh --studies -rA --junitxml=$E/<label>-studies.xml`, compared with `scripts/compare_test_outcomes.py $E/baseline-core.xml,$E/baseline-studies.xml $E/<label>-core.xml,$E/<label>-studies.xml --must-pass tests/paper/test_live_golden.py::test_golden_has_120_sessions tests/paper/test_live_golden.py::test_rebuilt_signal_runner_reproduces_the_live_golden`. Rules: every baseline pass still passes; every non-pass equals the baseline's; the live-golden tests pass rather than skip.
- **The oracle check** (both reference drivers through `scripts/run_capped.sh` and `./offline.sh`, about 100 s) is compared with `scripts/compare_evidence_json.py $E/baseline $E/<label>`. Both drivers exit 0, all 13 p5 results have `"ok": true`, p7 reports 120 sessions with 0 different, and the comparison is empty. The baseline's 16 of 26 `DIFFERENCES - classify` p5 reports are the accepted state and must stay exactly as they are.
- **The gate scripts** (`E=/home/flynn/ggTrader-refactor-evidence`): `$E/gate_run.sh LABEL` runs the oracles then both test halves (about 9 minutes, alone), `$E/gate_compare.sh LABEL` compares to the baselines and writes `$E/LABEL-compare-status.txt`. Baselines: `$E/baseline/`, `baseline-core.xml` (2811 passed, 1 skipped, 5 xfailed), `baseline-studies.xml` (1144 passed). Re-record only if `BASE` changes, with `$E/run_baseline.sh`.
- **Pins:** `scripts/verify_pins.py --expect <counts in .github/workflows/ci.yml>` exits 0. The counts become `MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` when `cached_yfinance_loader.py` is edited (Task 14.1; the CI line is already changed in the working tree).
- **Fixed rules:** only `src/ggTrader/engine/` imports vectorbt; nothing in `src/` imports a study; `ohlcv.timestamp` is naive UTC (no tz-aware write); `ruff check .` and `ruff format --check .` stay clean; new code has a module docstring, type hints and Google docstrings.

## Review Focus

Failure modes of the remaining work, most likely first; each is pinned by the named task.

1. **An autofix or lint edit removes a compatibility re-export or a side-effect import** (names studies import; the imports the transcript harness finds by `__module__`). Pinned by the study-surface test, `verify_pins.py` and the 95 transcripts in Tasks 14.1 and 14.3.
2. **typer stops parsing a rewritten annotation** (`list[str]`, `X | None`). Pinned by `test_every_ggt_command_help_exits_zero` (Task 14.1).
3. **A debug log line in an `except` block changes behavior** (it raises, or the exception is no longer swallowed). Pinned by one caplog test per site plus the unchanged `poll_clock_failure` golden (Task 14.2).
4. **The evidence comparer ignores too much or too little.** `tape.path` must be ignored and `tape.sha256` must not be. Pinned by two cases in Task 14.0.
5. **The Telegram change misreports a total when a base is missing** (no day-start snapshot, no starting capital, a non-positive base). Pinned by the four formatting cases in Task 14.6.
6. **Two heavy jobs overlap or one hangs.** Pinned by the guard, not by this plan's tests: if a run prints exit 75, wait; if 137 or 143, stop.

---

### Task 14.0: The evidence comparer ignores the checkout path of the tape

The re-recorded baseline was produced in `/home/flynn/ggTrader-baseline`, so `tape.path` (`SnapshotInfo.as_dict`, `src/ggTrader/data/snapshot.py:54`) differs in all 13 p5 results while `tape.sha256` is identical. That is a location, not a result.

**Files:**
- Modify: `scripts/compare_evidence_json.py` (the `VOLATILE` literal; no module-level call, which `tests/scripts/test_scripts_have_help.py` forbids)
- Test: `tests/scripts/test_compare_evidence_json.py`

**Interfaces:** Produces: `compare(before, after)` ignores the path `("tape", "path")` in every JSON file and still reports any other `tape` difference.

- [ ] **Step 1: Write the failing cases** in the existing `test_json_values` parametrization: `({"tape": {"path": "/a/t.parquet", "sha256": "s"}}, {"tape": {"path": "/b/t.parquet", "sha256": "s"}}, True)` and `({"tape": {"path": "/a/t.parquet", "sha256": "s1"}}, {"tape": {"path": "/a/t.parquet", "sha256": "s2"}}, False)`.
- [ ] **Step 2:** `./offline.sh scripts/run_tests.sh tests/scripts/test_compare_evidence_json.py -q`. Expected: the first new case FAILS (the second already passes).
- [ ] **Step 3:** Add `("tape", "path")` to the `VOLATILE` set literal with the comment `# src/ggTrader/data/snapshot.py:54 (where the tape file sits in this checkout; its sha256 stays compared)`.
- [ ] **Step 4:** Run the file and `tests/scripts/test_scripts_have_help.py`. Expected: PASS. Then `./offline.sh .venv/bin/python scripts/compare_evidence_json.py $E/baseline $E/t14a; echo $?`. Expected: no output, exit 0 (the earlier autofix gate result, now judged correctly).
- [ ] **Step 5: Commit** `test: the evidence comparer ignores the checkout path of the tape, not its hash`.

### Task 14.1: Commit the annotation autofix

State on entry: 30 `src/` and `scripts/` files are modified by `ruff check --select UP006,UP035,UP045,F401,I --fix` plus `ruff format`, `.github/workflows/ci.yml` carries the new pin counts, and the only removed imports are `typing` names. Nothing is committed.

**Files:**
- Modify: the 30 files already changed in the working tree (`git diff --name-only`), `.github/workflows/ci.yml`
- Test: `tests/cli/test_cli_help_parses.py` (new)

**Interfaces:** Produces: `test_every_ggt_command_help_exits_zero`, a parametrized test over `["--help"]`, `["paper", "--help"]`, `["lab", "--help"]`, `["db", "--help"]`, `["db", "truncate", "--help"]`, `["db", "clean", "--help"]`, run through `typer.testing.CliRunner` on `ggTrader.cli.main.app`.

- [ ] **Step 1:** Write the test above; assert `result.exit_code == 0` and the command name in the output.
- [ ] **Step 2:** Run it. Expected: PASS on the autofixed tree (it guards the autofix; it already passed by hand: `ggt --help`, `paper`, `db truncate`, `lab`, `db` all exited 0).
- [ ] **Step 3:** Confirm the diff is only what is expected. (Checked 2026-11-19: `scripts/make_tape_snapshot.py`, which the legacy reference runner executes by path, and `scripts/engine_parity_evidence.py` both have `from __future__ import annotations`, and `scripts/legacy_env.sh` builds the legacy environment with Python 3.12, so their modern annotations are safe.) `git diff --name-only | grep -v -E '^(src|scripts)/|^\.github/workflows/ci\.yml$|^docs/superpowers/plans/'` prints nothing; `git diff -U0 | grep -E '^-(from|import) ' | grep -v typing` prints nothing; `./offline.sh .venv/bin/python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` exits 0.
- [ ] **Step 4: The gate.** Alone: `$E/gate_run.sh t14a2`, then `$E/gate_compare.sh t14a2; cat $E/t14a2-compare-status.txt`. Expected: `oracle compare exit=0` and `outcome compare exit=0`; the core log ends `passed` with no failures and the 95 transcript goldens unchanged (`git status --short tests/paper/golden` empty).
- [ ] **Step 5: Commit** (add the 30 files, the CI line and the new test; not the plan file): `style: modern type annotations (ruff UP006/UP035/UP045 autofix)`.

### Task 14.2: The eight hand fixes and the lint rules

Today's hits for `UP006,UP035,UP045,B,S110` on `src/` and the non-excluded scripts are exactly eight; every other hit is in `tests/` (37) or in an excluded script (10) and stays ignored.

**Files:**
- Modify: `ruff.toml`; `src/ggTrader/cli/cmd_db.py:105`, `src/ggTrader/data/bars.py:127`, `src/ggTrader/data/live/cached_yfinance_loader.py:240` and `:279`, `src/ggTrader/lab/persist.py:114`, `src/ggTrader/paper/overlay.py:116`, `src/ggTrader/paper/trader.py:434` and `:1388`
- Test: `tests/paper/test_swallowed_exceptions_are_logged.py`, `tests/lab/test_persist_hypertable_fallback.py`, `tests/data/test_cached_loader_close_failure.py` (new)

**Interfaces:** Produces: three caplog tests, one per `except ...: pass` site, each asserting that a DEBUG record on that module's logger contains the exception's class name and that the surrounding behavior is unchanged.

- [ ] **Step 1 (RED): add the rules.** In `ruff.toml` append `"UP006"`, `"UP035"`, `"UP045"`, `"B"`, `"S110"` to `select`, and add per-file ignores of those five rules for `"research/**"`, `"tests/**"` and each of the eight excluded scripts (`scripts/p5_wfo_validation.py`, `scripts/p7_live_golden.py`, `scripts/legacy_reference_runs.py`, `scripts/legacy_reference_compare.py`, `scripts/verify_pins.py`, `scripts/verdict_register.py`, `scripts/study.py`, `scripts/vbt_latest.py`), each with a comment saying why (frozen evidence, or a script the oracle or the legacy runner executes). Run `.venv/bin/ruff check .`. Expected: FAIL with exactly the eight errors above (three B905, two B007, three S110).
- [ ] **Step 2: write the three logging tests, run them, watch them fail.**
  - `test_poll_clock_failure_is_logged_at_debug`: run the existing `poll_clock_failure` scenario through `run_scenario` (see `tests/paper/test_trader_transcripts.py` for how scenarios are looked up) under `caplog.at_level(logging.DEBUG, logger="ggTrader.paper.trader")`; assert a record's message contains the raised exception's class name, and that the run's result is the one the golden records.
  - `test_plain_postgres_fallback_is_logged_at_debug`: `init_schema` with `get_engine` replaced by a fake whose `execute` raises `RuntimeError` for the `create_hypertable` statement; assert the DEBUG record on `ggTrader.lab.persist` and that `init_schema` returns normally.
  - `test_connection_close_failure_is_logged_at_debug`: `CachedYFinanceLoader._cache_to_db` with a fake connection whose `close` raises; assert the DEBUG record on the loader's `self.logger` and that the call returns normally. Expected for all three: FAIL (no record).
- [ ] **Step 3: fix the eight.** `zip(...)` at the three flagged sites gets `strict=False` (today's behavior; existing `strict=True` calls stay). `label` (`overlay.py:116`) and `reason` (`trader.py:1388`) become `_label` and `_reason`. The three `except Exception: pass` become `except Exception as exc:` followed by `_log.debug("<what was skipped>: %s", type(exc).__name__)`: `trader.py` already has `_log`; `lab/persist.py` gains `import logging` and `_log = logging.getLogger(__name__)`; the loader uses its existing `self.logger`.
- [ ] **Step 4:** `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` Expected: both clean. `./offline.sh scripts/run_tests.sh tests/paper/ tests/lab/ tests/data/ tests/utils/ -q`. Expected: PASS; the goldens are unchanged.
- [ ] **Step 5: Commit** `style: enable bugbear and try-except-pass rules for src and scripts`.

### Task 14.3: The Phase 2 gate

- [ ] **Step 1:** Alone: `$E/gate_run.sh phase2` then `$E/gate_compare.sh phase2`. Expected: both compare lines exit 0.
- [ ] **Step 2:** `./offline.sh .venv/bin/python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` exits 0; `ruff check .` and `ruff format --check .` clean; `git status --short` shows only intended files.
- [ ] **Step 3:** Add the changelog entry to `docs/changelog.md` (phases 2: engine cache home, paper constants, scripts, logging, annotation style and lint rules, pin counts) and update the design document's status line to "phases 0 to 2 built and gated". **Commit** `docs: changelog and status for refactor phases 0-2`.

### Task 14.4: Whole-branch review of phases 0 to 2

- [ ] **Step 1:** Build the package: `git diff 8866f446..HEAD -- . ':(exclude)tests/paper/golden'` (the 95 goldens are reviewed through their scenarios and mutation table instead). Give two fresh reviewers the package, this plan, the design and the Review Focus above: Codex Sol (reviews Claude-built work) and Claude Opus (reviews Codex-built work). Both are read-only and run only small targeted tests, one at a time.
- [ ] **Step 2:** Re-grade by effect. Critical and Important findings get one fix pass, each fix with a test that failed first; minors are listed for the owner. If code changed, rerun Task 14.3's gate with label `phase2b`.
- [ ] **Step 3:** Report to the owner: what changed, the gate numbers, the rulings, the deferred minors and the quota used.

### Task 14.5: Merge to `main` (only on the owner's say-so)

- [ ] **Step 1:** Do not start without an explicit yes. The main checkout is on branch `research/intraday-r9-screen` with uncommitted files, so merge in a separate worktree: `git worktree add /home/flynn/ggTrader-merge main`, then `git -C /home/flynn/ggTrader-merge merge --no-ff refactor/phase-0-2`.
- [ ] **Step 2:** In that worktree run the merge gate (it needs its own `.venv`, the frozen tape copy and `offline.sh`): both test halves compared with the baselines, `ruff`, `verify_pins.py`. Expected: all exit 0.
- [ ] **Step 3:** Stop there. **Pushing `main` is a separate decision:** a push triggers `.github/workflows/docker-build.yml`, which publishes the container image; the trader is off and nothing deploys by itself, but the owner decides.

### Task 14.6: Telegram total shows both percentages (owner-approved 2026-11-18; a deliberate behavior change)

`TelegramNotifier.daily_summary` (`src/ggTrader/paper/notifier.py:~151`) divides total profit by the day's starting value only. Starting capital already exists in the trader: `get_earliest_snapshot()` (`src/ggTrader/paper/persist.py:~515`) read at `trader.py:~1480`, where `total_pnl = corrected_value - starting_capital`. This must land before Phase 4, whose Task 36 fixes the notifier's signature.

**Files:**
- Modify: `src/ggTrader/paper/notifier.py` (`daily_summary`), `src/ggTrader/paper/trader.py` (its one call site)
- Modify: `tests/paper/golden/trader_transcripts/*.json` (regenerated; only the `Total P&L` text lines change)
- Test: `tests/paper/test_notifier.py` (extend)

**Interfaces:** Produces: `TelegramNotifier.daily_summary(..., starting_capital: float | None = None)` (a new optional keyword after the existing parameters; `DryRunNotifier` inherits it). The total line becomes
`Total P&L: $+1,234.56 (+1.23% of day start, +0.98% of starting capital) — realized: ..., unrealized: ...`.

- [ ] **Step 1: Write the failing formatting tests** (names and assertions): `test_total_line_shows_both_percentages` (day start 100000, starting capital 126000, total 1234.56: `(+1.23% of day start, +0.98% of starting capital)`); `test_total_line_without_day_start_shows_only_capital` (`day_start=None`: `(+0.98% of starting capital)`); `test_total_line_with_a_non_positive_starting_capital` (`starting_capital=0` or negative: the capital part reads `n/a`); `test_total_line_without_starting_capital_keeps_todays_text` (`starting_capital=None`: exactly today's `(+1.23%)` form, so an old caller is unchanged).
- [ ] **Step 2:** `./offline.sh scripts/run_tests.sh tests/paper/test_notifier.py -q`. Expected: the four new tests FAIL.
- [ ] **Step 3:** Implement the keyword and the line; pass `starting_capital=starting_capital` at the trader's call site.
- [ ] **Step 4: Regenerate the goldens and review them:** `./offline.sh env GGT_REGEN_TRANSCRIPTS=1 scripts/run_tests.sh tests/paper/test_trader_transcripts.py` (it fails with "regenerated, review and rerun"), then rerun without the variable. Expected: PASS. `git diff --stat tests/paper/golden` lists only goldens that reach the summary, and `git diff -U0 tests/paper/golden | grep '^[-+] ' | grep -v 'Total P&L'` shows only the recorded `starting_capital` argument and nothing else.
- [ ] **Step 5: Mutation check** (temporary, restored from a backup): swap the two bases in the formula; at least one new test and one golden fail.
- [ ] **Step 6:** The Phase 2 gate again (label `telegram`), `ruff`, then **commit** `feat(paper): Telegram total shows the percentage against day start and against starting capital`, with a changelog line.

---

## The road to done

| Phase | Plan | State | Needs before it starts |
|---|---|---|---|
| 2 finish, review | this plan, 14.0 to 14.4 | next | nothing |
| merge to `main` | 14.5 | owner-gated | the Phase 2 gate and review |
| Telegram change | 14.6 | approved, unbuilt | Phase 2 gate; must precede Phase 4 |
| 3 engine and WFO | `2026-11-18-refactor-phase-3.md` (tasks 15 to 29) | written, Sol-reviewed, fixes applied | Phase 2 gate; its Task 29 (string return to exception) defaults to skip |
| 4 paper ports and records | `2026-11-18-refactor-phase-4.md` (tasks 30 to 45) | written, Sol-reviewed, revised | Task 14.6; the owner's answer on its optional Task 45 and on quota |
| 5 split `PaperTrader._run` | to be written | not planned | Phase 4 done; see below |
| 6 optional tidy-ups | to be written, only if wanted | not planned | owner decision |

**Order:** 14.0 to 14.4, then (on the owner's yes) 14.5, then 14.6, then Phase 3 (its engine and WFO chains can be edited in parallel worktrees; their test runs queue on the lock), then Phase 4, then write and review the Phase 5 plan, then Phase 5. Report quota after each phase so the owner can stop early.

**What the Phase 5 plan must start from** (this is the part of the design that carries the most risk: an 829-line live-order method becomes about ten objects):
- A first task that adds transcript scenarios for the branches that two reviews found unexercised and that Phase 5 moves: persisted split-state cleanup and re-verification and the split and dividend persistence failures (`trader.py` about `:514-694`), the prior-snapshot and invalid-final-equity paths, notice-failure handlers, and everything in `run_paper_trading` (`:1584-1605`, including the margin guard and the dry-run notifier wiring). Each gets a golden and a mutation that fails it before any code moves.
- The Phase 4 hand-off (ledger injection, `RunResult`, `TraderConfig`, typed records) as its entry condition, and the narrowed claim that Phase 4 gives ledger-independent testing, not a patch-free whole-trader test.
- One step object per task (`PreflightGate`, `PortfolioView`, `RiskAssessor`, the pure planning functions, `OrderExecutor` with `DryRunExecutor`, `CorporateActions`, `RunReporter`), the order of calls preserved exactly, goldens byte-identical at every task, and a seeded property test of the new planning functions against the old method before it is deleted.

## Execution roles (`CLAUDE.md` delegation preference of 2026-11-08)

- **Tasks 14.0 to 14.4 and 14.6** are small, strictly serial and touch shared files, so they run natively (one builder, the orchestrator) with the two whole-branch reviewers at 14.4. Sol reviews Claude-built work and Opus reviews Codex-built work; neither reviews its own family.
- **Phases 3 and 4** use fresh builders per task (Codex Terra or Luna, or Claude Sonnet) with the heavy-job rules above; the orchestrator re-verifies each diff, reruns each task's tests and commits.
- **Quota:** the owner reports headroom before Phase 3 and again before Phase 4; usage is reported after each phase.
