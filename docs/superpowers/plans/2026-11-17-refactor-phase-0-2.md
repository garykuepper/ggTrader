# Refactor phases 0 to 2: what is left (revision 6, 2026-11-19)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 2 (annotation style and the first lint rules), prove phases 0 to 2 with the full gate and a whole-branch review, ship the one owner-approved behavior change that Phase 4 depends on (Telegram total shows both percentages, built before the merge so the merged branch and Phase 4's base are the same), and lay out the road through phases 3 to 6.

**Architecture:** The safety nets (comparers, connection guard, 95 trader transcripts, loader characterization) and the heavy-job guard are built and green. What is left is a mechanical lint autofix that is applied but uncommitted, eight hand fixes plus the lint rules, the gates, a review, and one notifier change. Every behavior-neutral step must leave the transcripts, the oracle results and the test outcomes identical.

**Tech Stack:** Python 3.12 (uv `.venv`), pytest through `scripts/run_tests.sh`, ruff 0.17.4, typer, SQLAlchemy, the vectorbt Rust engine (read-only here).

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md). The earlier revisions of this plan, with the full text of Tasks 0 to 13, are in git: `git show 246200f7:docs/superpowers/plans/2026-11-17-refactor-phase-0-2.md`.

## Status (2026-11-19)

Branch `refactor/phase-0-2` at `/home/flynn/ggTrader-refactor`, HEAD `b0fb839c` (the last code commit is `b562f6be`; the two after it are plan documents), created from `main` commit `8866f446`. Nothing is pushed or merged. Historical gate figures: Phase 0 (core 3292, studies 1144) and Phase 1 (core 3298, studies 1144, oracles identical) passed; the safety branch's final gate was core 3349, studies 1144; the Task 14 autofix gate (`t14a`) matched outcomes and differed from the baseline oracle only in `tape.path` (Task 14.0).

| Task | What | Commits |
|---|---|---|
| 0 | worktree, offline wrapper, baselines, comparers | `04f55df9` `13db1e28` `53cc9d82` `51fd5fec`, plan `40a5e3c3` |
| 1 | connection guard (sockets, lookups, psycopg2, curl_cffi, audit hook) | `aa85e6d3` `d6a14a3e` |
| 2, 3 | study-import surface test; import-direction rules | `ea161c89`; `04958996` `7d2c8fd5` |
| 4 | trader transcripts, 95 scenarios, mutation-checked | `54e7c0bc` `058296bf` `a270fc5a`, docs `246200f7` |
| 5 | `fetch_stock_ohlcv` characterization, 28 test cases (26 functions) | `faa5e2ce` `68f7e7ad` |
| 6 to 9 | dead untradable branch; unused parameter; docs; wipe flags | `ea1a762f`; `9dd26f26`; `c509d7c1`; `641c7014` |
| 10 to 13 | cached engine home; paper constants; scripts; logging | `f9b7a093`; `2c4a0dcd`; `05cbb12c`; `a44fa90a` |
| test-run safety (own plan) | heavy-job lock, 4G cap, time limits, per-test timeout | `ecc93843` to `b562f6be` |
| **14** | **annotation autofix applied, uncommitted; lint rules and hand fixes not started** | **this plan** |

What changed since revision 4 and why it matters here:
- **Evidence moved** to `/home/flynn/ggTrader-refactor-evidence/` (a reboot cleared `/tmp`). The baselines were re-recorded there from the untouched base commit (worktree `/home/flynn/ggTrader-baseline`).
- **`run_tests.sh --full` writes two JUnit files that overwrite each other**, so every gate keeps one file per half and the comparer takes both.
- **Two parallel runs once exhausted the host.** The cause was a hang: the `ggt` logging callback left INFO on and paper tests that stub `time.time()` starved (fixed in `a44fa90a`). The scripts now serialize heavy jobs (see Global Constraints).
- **Transcripts grew from 36 to 95 scenarios** after two independent reviews found regressions that would have passed.
- **The gate wrappers do not fail closed** (found by the review of revision 5): they record exit codes in files and finish with a successful `echo`. Task 14.0 fixes that before any gate result is trusted.

## Scope of what is left

**In this plan:** Task 14.0 to 14.4 (gate tooling, finish Phase 2, review phases 0 to 2), 14.5 (the Telegram change) and 14.6 (merge to `main`, owner-gated). **Then, in their own plans:** Phase 3 (`2026-11-18-refactor-phase-3.md`, tasks 15 to 29) and Phase 4 (`2026-11-18-refactor-phase-4.md`, tasks 30 to 45) are written and Sol-reviewed; Phase 5 and the optional Phase 6 need plans (see "The road to done").

**Not done, on purpose:** edits to `research/`; the oracle and standard-library-only scripts (`p5_wfo_validation.py`, `p7_live_golden.py`, `legacy_reference_runs.py`, `legacy_reference_compare.py`, `verify_pins.py`, `verdict_register.py`, `study.py`, `vbt_latest.py`); deleting the unused database writers (owner decision O3, keep); anything that pushes, deploys or touches cron.

## Owner decisions (recorded)

- **O1** delete the dead untradable branch: done (`ea1a762f`). **O2** wipe flags: done (`641c7014`). **O3** keep the unused writers.
- **O4 (2026-11-18): the Telegram total shows both percentages**, against the prior snapshot's value (what the code calls `day_start`: the portfolio value before today's change, see below) and against starting capital: Task 14.5.
- **O5** execution approved 2026-11-18 (the design's section 8 still says "not yet": Task 14.3 corrects it). **Merging to `main` and pushing need a separate say-so** (Task 14.6).

## Global Constraints

`E` below is `/home/flynn/ggTrader-refactor-evidence`.

- **Branch and worktree.** Work on `refactor/phase-0-2` in `/home/flynn/ggTrader-refactor`. Every command uses the worktree's own `.venv/bin/python` and `.venv/bin/ruff` by absolute path (never a bare `ruff`) and runs through `./offline.sh` (an untracked wrapper at the worktree root, listed in the shared `info/exclude`). Never commit to `main`, push or merge without the owner's say-so. Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- **Offline by construction.** `./offline.sh` scrubs the environment (`env -i HOME PATH XDG_RUNTIME_DIR LANG=C.UTF-8 TZ=UTC GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02 "$@"`; `XDG_RUNTIME_DIR` is needed by `systemd-run` and is not a credential), so no database, broker, Telegram or Tiingo credentials reach a process, and `utils/config.py` raises without database settings. **That is not a network block.** The network boundary is the missing credentials plus the in-process guards (`tests/netguard.py`, applied to every paper test and the loader characterization; the root `conftest.py` makes `load_dotenv` a no-op). Never copy `.env` into a worktree; never `ggt paper --live`, never cron.
- **Heavy jobs run one at a time** (enforced by `scripts/_heavy_job.sh`, merged in `b562f6be`): a pytest run or an oracle driver takes a lock in the git directory; a second one waits up to 1800 s then exits 75; a job exits 75 when free memory is below its cap plus 4G. Defaults: test cap 4G, 1800 s per pytest run, 180 s per test. Exit 137 or 143: stop and investigate (a hang is the usual cause, not a big suite); never raise the cap or the limit to make a run pass. Delegated builders run only small targeted paths and never `tests/paper/`, a whole tier or an oracle driver unless their brief says so.
- **Tests only through `scripts/run_tests.sh`** (never a bare `pytest`). Test outcomes are gated with two files per run, one per half:
  `./offline.sh scripts/run_tests.sh --full -rA --junitxml=$E/<label>-core.xml tests` and `./offline.sh scripts/run_tests.sh --studies -rA --junitxml=$E/<label>-studies.xml`, compared by `scripts/compare_test_outcomes.py $E/baseline-core.xml,$E/baseline-studies.xml $E/<label>-core.xml,$E/<label>-studies.xml --must-pass <the two live-golden nodes> --must-pass-file $E/safety-nodes.txt`. Rules: every baseline pass still passes; every non-pass equals the baseline's; the live-golden tests pass rather than skip; **and every node in `safety-nodes.txt` is present and passing.** The baseline (2811 core tests) predates the 95 transcripts and the guard tests, so without that list a deleted safety net would not show as a regression. `safety-nodes.txt` holds the passing nodes of the safety-net test files (Task 14.0 builds it; Task 14.5 adds the new scenario).
- **The oracle check** (both reference drivers through `scripts/run_capped.sh` and `./offline.sh`, about 100 s) is compared with `scripts/compare_evidence_json.py $E/baseline $E/<label>`. Both drivers exit 0, all 13 p5 results have `"ok": true`, p7 reports 120 sessions with 0 different, and the comparison is empty (file by file: every `_p5_compare_*` report, equity CSV and p7 field equals the baseline's). In the baseline all 13 of 13 p5 comparison reports say `DIFFERENCES - classify`: the accepted state (the older committed evidence had 16 of 26), and it must stay exactly as it is.
- **The gate scripts** in `$E` (tooling outside the repository, rewritten fail-closed in Task 14.0): `gate_run.sh LABEL` runs the oracles then both test halves (about 9 minutes, alone) and `gate_compare.sh LABEL` compares them to the baselines; both take the tested checkout from `WT` (default `/home/flynn/ggTrader-refactor`), refuse a label that already has results, stop at the first child that exits non-zero, and exit non-zero themselves. A gate passes only when all six statuses are 0: p5, p7, core, studies, oracle comparison, outcome comparison. Baselines: `$E/baseline/`, `baseline-core.xml` (2811 passed, 1 skipped, 5 xfailed), `baseline-studies.xml` (1144 passed). Re-record only if `BASE` changes, with `$E/run_baseline.sh`.
- **Pins:** `scripts/verify_pins.py --expect <counts in .github/workflows/ci.yml>` exits 0. The counts become `MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` when `cached_yfinance_loader.py` is edited (Task 14.1; the CI line is already changed in the working tree).
- **Fixed rules:** only `src/ggTrader/engine/` imports vectorbt; nothing in `src/` imports a study; `ohlcv.timestamp` is naive UTC (no tz-aware write); `ruff check .` and `ruff format --check .` stay clean; new code has a module docstring, type hints and Google docstrings.

## Review Focus

Failure modes of the remaining work, most likely first; each is pinned by the named task.

1. **A gate reports green when a step failed or a safety net vanished** (wrappers that swallow exit codes, a reused label with a stale XML, a deleted transcript test). Pinned by Task 14.0 (fail-closed scripts, `--must-pass-file`).
2. **An autofix or lint edit removes a compatibility re-export or a side-effect import** (names studies import; the imports the transcript harness finds by `__module__`). Pinned by the study-surface test, `verify_pins.py` and the 95 transcripts in Tasks 14.1 and 14.3.
3. **typer stops parsing a rewritten annotation** (`list[str]`, `X | None`). Pinned by `test_selected_ggt_commands_print_help` (Task 14.1).
4. **A debug log line in an `except` block changes behavior** (it raises, or the exception is no longer swallowed). Pinned by one caplog test per site plus the unchanged `poll_clock_failure` golden (Task 14.2).
5. **The evidence comparer ignores too much.** `tape.path` is ignored only in `_p5_*` results and `tape.sha256` always counts. Pinned by Task 14.0.
6. **The Telegram change misreports a total or mislabels its base**, and the existing goldens cannot see a swapped denominator (only 2 of them print a number, and both round to the same percentage on either base). Pinned by the formatter tests and the new `total_pnl_two_bases` scenario (Task 14.5).

---

### Task 14.0: Comparer fixes and fail-closed gate tooling

The re-recorded baseline was produced in `/home/flynn/ggTrader-baseline`, so `tape.path` (`SnapshotInfo.as_dict`, `src/ggTrader/data/snapshot.py:54`) differs in all 13 p5 results while `tape.sha256` is identical: a location, not a result. Separately, the review of revision 5 found that the gate wrappers cannot fail, and that the outcome gate cannot see a deleted Phase 0 test.

**Files:**
- Modify: `scripts/compare_evidence_json.py` (`_ignored`; no module-level call, which `tests/scripts/test_scripts_have_help.py` forbids), `scripts/compare_test_outcomes.py` (`--must-pass-file`)
- Test: `tests/scripts/test_compare_evidence_json.py`, `tests/scripts/test_compare_test_outcomes.py`
- Rewrite, outside the repository: `$E/gate_run.sh`, `$E/gate_oracles.sh`, `$E/gate_tests.sh`, `$E/gate_compare.sh`; create `$E/safety-nodes.txt`

**Interfaces:** Produces: `compare(before, after)` ignores the path `("tape", "path")` in files named `_p5_*.json` only (mirroring `P5_VOLATILE`); `compare_test_outcomes.py ... --must-pass-file FILE` (one node id per line; blank lines and `#` comments ignored; each node must be present and passed in the new run), combinable with `--must-pass`.

- [ ] **Step 1: Write the failing tests.** `test_tape_path_is_ignored_only_in_p5_result_files` (two folders; `_p5_run.json` with `tape.path` `/a/t.parquet` against `/b/t.parquet` and equal `sha256`: no difference; the same pair in `x.json`: a difference); `test_tape_hash_difference_is_reported_in_p5_results` (`sha256` `s1` against `s2` in `_p5_run.json`: a difference); `test_must_pass_file_requires_every_listed_node` (a file listing two nodes, one skipped in the new XML: exit 1 and the node named; all passing: exit 0; comments and blank lines ignored).
- [ ] **Step 2:** `./offline.sh scripts/run_tests.sh tests/scripts/test_compare_evidence_json.py tests/scripts/test_compare_test_outcomes.py -q`. Expected: the three new tests FAIL.
- [ ] **Step 3:** Implement both changes; the comment on the ignore cites `src/ggTrader/data/snapshot.py:54`.
- [ ] **Step 4:** Run those two files and `tests/scripts/test_scripts_have_help.py`. Expected: PASS. Then `./offline.sh .venv/bin/python scripts/compare_evidence_json.py $E/baseline $E/t14a; echo $?`. Expected: no output, exit 0 (the earlier autofix gate result, now judged correctly). **Commit** `test: comparers ignore the tape's checkout path in p5 results and take a must-pass node file`.
- [ ] **Step 5: Rewrite the four gate scripts fail-closed** (`set -u`; `WT="${WT:-/home/flynn/ggTrader-refactor}"` instead of a hard-coded `cd`; refuse a label with any existing `$E/LABEL-*` result; stop at the first child with a non-zero status and exit with it; `gate_compare.sh` exits non-zero if either comparison does; the status files keep every code). Verify without running a gate: `$E/gate_run.sh t14a` exits non-zero and starts nothing (label reused); `$E/gate_compare.sh no-such-label` exits non-zero.
- [ ] **Step 6: Build `$E/safety-nodes.txt`** from the clean safety gate (`$E/safety-core.xml`): every passed node in `tests/paper/test_trader_transcripts.py`, `tests/lab/test_fetch_stock_ohlcv_chain.py`, `tests/test_netguard.py`, `tests/utils/test_study_surface.py`, `tests/utils/test_import_rules.py`, `tests/scripts/test_heavy_job_guard.py`, `tests/test_timeout_guard.py`, `tests/scripts/test_compare_evidence_json.py` and `tests/scripts/test_compare_test_outcomes.py`, using `compare_test_outcomes.outcomes`. Expected: the file holds exactly 95 `test_transcript_matches_golden[...]` nodes and 28 loader characterization nodes, and `compare_test_outcomes.py` accepts it against `$E/safety-core.xml`.

### Task 14.1: Commit the annotation autofix

State on entry: 30 `src/` and `scripts/` files are modified by `ruff check --select UP006,UP035,UP045,F401,I --fix` plus `ruff format`, `.github/workflows/ci.yml` carries the new pin counts, and the only removed imports are `typing` names. Nothing is committed.

**Files:**
- Modify: the 30 files already changed in the working tree (`git diff --name-only`), `.github/workflows/ci.yml`
- Test: `tests/scripts/test_cli_help_parses.py` (new)

**Interfaces:** Produces: `test_selected_ggt_commands_print_help`, parametrized over `["--help"]`, `["paper", "--help"]`, `["lab", "--help"]`, `["db", "--help"]`, `["db", "truncate", "--help"]`, `["db", "clean", "--help"]`, run through `typer.testing.CliRunner` on `ggTrader.cli.main.app` (it covers these six invocations, not every database subcommand).

- [ ] **Step 1:** Write the test; assert `result.exit_code == 0` and `ggt` (the root command's name), or the subcommand's name, in the output (`lab` and `db` hand `--help` to argparse, which prints it and exits 0).
- [ ] **Step 2:** Run it. Expected: PASS on the autofixed tree (it guards the autofix; by hand `ggt --help`, `paper`, `db truncate`, `lab` and `db` all exited 0).
- [ ] **Step 3:** Confirm the diff is only what is expected. (Checked 2026-11-19: `scripts/make_tape_snapshot.py`, which the legacy reference runner executes by path, and `scripts/engine_parity_evidence.py` both have `from __future__ import annotations`, and `scripts/legacy_env.sh` builds the legacy environment with Python 3.12, so their modern annotations are safe.) `git diff --name-only | grep -v -E '^(src|scripts)/|^\.github/workflows/ci\.yml$'` prints nothing; `git diff -U0 | grep -E '^-(from|import) ' | grep -v typing` prints nothing; `./offline.sh .venv/bin/python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` exits 0.
- [ ] **Step 4: The gate.** Alone: `$E/gate_run.sh t14a2` then `$E/gate_compare.sh t14a2`. Expected: both exit 0 and all six statuses in `$E/t14a2-*status.txt` are 0; the core log ends `passed` with no failures; `git status --short tests/paper/golden` is empty.
- [ ] **Step 5: Commit** the 30 files, the CI line and the new test: `style: modern type annotations (ruff UP006/UP035/UP045 autofix)`.

### Task 14.2: The eight hand fixes and the lint rules

Today's hits for `UP006,UP035,UP045,B,S110` on `src/` and the non-excluded scripts are exactly eight; every other hit is in `tests/` (39), in `research/` (about 100) or in an excluded script (10) and stays ignored.

**Files:**
- Modify: `ruff.toml`; `src/ggTrader/cli/cmd_db.py:105`, `src/ggTrader/data/bars.py:127`, `src/ggTrader/data/live/cached_yfinance_loader.py:240` and `:279`, `src/ggTrader/lab/persist.py:114`, `src/ggTrader/paper/overlay.py:116`, `src/ggTrader/paper/trader.py:434` and `:1388`
- Test: `tests/paper/test_swallowed_exceptions_are_logged.py`, `tests/lab/test_persist_hypertable_fallback.py`, `tests/data/test_cached_loader_close_failure.py` (new)

**Interfaces:** Produces: three caplog tests, one per `except ...: pass` site, each asserting a DEBUG record on the right logger whose message contains the exception's class name, and that the surrounding behavior is unchanged.

- [ ] **Step 1 (RED): add the rules.** In `ruff.toml` append `"UP006"`, `"UP035"`, `"UP045"`, `"B"`, `"S110"` to `select`. Per-file ignores of those five rules go on `"research/**"`, `"tests/**"` and the eight excluded scripts (`scripts/p5_wfo_validation.py`, `scripts/p7_live_golden.py`, `scripts/legacy_reference_runs.py`, `scripts/legacy_reference_compare.py`, `scripts/verify_pins.py`, `scripts/verdict_register.py`, `scripts/study.py`, `scripts/vbt_latest.py`), each with a comment saying why. **`scripts/legacy_reference_runs.py` already has an entry (`["TID251"]`): extend that list in place and keep `TID251`** (a second key makes the TOML invalid, and dropping it breaks `tests/engine/test_boundary.py`); every other path is a new key (distinct overlapping patterns union). Run `.venv/bin/ruff check .`. Expected: FAIL with exactly the eight errors above (three B905, two B007, three S110).
- [ ] **Step 2: write the three logging tests, run them, watch them fail.**
  - `test_poll_clock_failure_is_logged_at_debug`: run the existing `poll_clock_failure` scenario through `run_scenario` (see `tests/paper/test_trader_transcripts.py` for how scenarios are looked up) under `caplog.at_level(logging.DEBUG, logger="ggTrader.paper.trader")`; assert a record's message contains the raised exception's class name and that the result is the one the golden records.
  - `test_plain_postgres_fallback_is_logged_at_debug`: replace `ggTrader.lab.persist.get_engine` with a fake whose `begin()` is a context manager yielding a fake connection; that connection's `execute(stmt)` raises `RuntimeError` when `"create_hypertable"` is in `str(stmt)` and records the other statements. Assert the DEBUG record on `ggTrader.lab.persist` and that `init_schema()` returns normally.
  - `test_connection_close_failure_is_logged_at_debug`: `CachedYFinanceLoader._cache_to_db` with a non-empty valid frame and a fake connection whose write path succeeds and whose `close` raises (an empty frame returns before the close seam). The loader inherits its logger from `YFinanceDataLoader`: use `caplog.at_level(logging.DEBUG, logger=loader.logger.name)` and assert on that name. Expected for all three: FAIL (no record).
- [ ] **Step 3: fix the eight.** `zip(...)` at the three flagged sites gets `strict=False` (today's behavior; existing `strict=True` calls stay). `label` (`overlay.py:116`) and `reason` (`trader.py:1388`) become `_label` and `_reason`. The three `except Exception: pass` become `except Exception as exc:` followed by a DEBUG call with `type(exc).__name__` and a few words on what was skipped: `trader.py` already has `_log`; `lab/persist.py` gains `import logging` and `_log = logging.getLogger(__name__)`; the loader uses `self.logger`.
- [ ] **Step 4:** `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` Expected: both clean. `./offline.sh scripts/run_tests.sh tests/paper/ tests/lab/ tests/data/ tests/utils/ -q`. Expected: PASS; the goldens are unchanged.
- [ ] **Step 5: Commit** `style: enable bugbear and try-except-pass rules for src and scripts`.

### Task 14.3: The Phase 2 gate and the durable record

- [ ] **Step 1:** Alone: `$E/gate_run.sh phase2` then `$E/gate_compare.sh phase2`. Expected: both exit 0 and all six statuses are 0.
- [ ] **Step 2:** `./offline.sh .venv/bin/python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` exits 0; `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` clean; `git status --short` shows only intended files.
- [ ] **Step 3: documents.** Add the changelog entry to `docs/changelog.md` (phase 2: engine cache home, paper constants, scripts, logging, annotation style, lint rules, pin counts, the fail-closed gates). Update the design: the status line to "phases 0 to 2 built and gated", and section 8's O5 to "execution approved 2026-11-18". The Phase 3 and Phase 4 plans exist only as untracked files in the main checkout: copy `2026-11-18-refactor-phase-3.md` and `2026-11-18-refactor-phase-4.md` into this worktree's `docs/superpowers/plans/` so they travel with the branch. **Commit** `docs: changelog and status for refactor phases 0-2; carry the phase 3 and 4 plans`.

### Task 14.4: Whole-branch review of phases 0 to 2

- [ ] **Step 1:** Build the package: `git diff 8866f446..HEAD -- . ':(exclude)tests/paper/golden'`, and give both reviewers read access to `tests/paper/golden/` and its scenario list too (an exclusion hides what a reviewer needs to judge). Two fresh reviewers get the package, this plan, the design and the Review Focus above: Codex Sol (reviews Claude-built work) and Claude Opus (reviews Codex-built work). Both are read-only and run only small targeted tests, one at a time.
- [ ] **Step 2:** Re-grade by effect. Critical and Important findings get one fix pass, each fix with a test that failed first; minors are listed for the owner. If code changed, rerun Task 14.3's gate with label `phase2b`.
- [ ] **Step 3:** Report to the owner: what changed, the gate numbers, the rulings, the deferred minors and the quota used.

### Task 14.5: Telegram total shows both percentages (owner-approved 2026-11-18; a deliberate behavior change)

`TelegramNotifier.daily_summary` (`src/ggTrader/paper/notifier.py:151`) divides total profit by `day_start`, which the method derives as `portfolio_value - daily_pnl`; its daily line is labelled "Change since prior snapshot", so the base is the **prior snapshot's value** (it can be several days old), not a literal day start. Starting capital already exists in the trader: `get_earliest_snapshot()` (`src/ggTrader/paper/persist.py:515`) read at `trader.py:1480`, where `total_pnl = corrected_value - starting_capital` (and is `None` when the starting capital is falsy). This lands before the merge and before Phase 4, whose Task 36 fixes the notifier's signature.

**Files:**
- Modify: `src/ggTrader/paper/notifier.py` (`TelegramNotifier.daily_summary`), `src/ggTrader/paper/trader.py` (its one call site, `:1496`), `tests/paper/transcript/scenarios.py` (one new scenario), `tests/paper/test_trader_transcripts.py` (the scenario count and one behavior assertion), `tests/paper/golden/trader_transcripts/*.json`
- Test: `tests/paper/test_notifier.py` (extend)

**Interfaces:** Produces: `TelegramNotifier.daily_summary(..., starting_capital: float | None = None)`, a new keyword-only-in-practice parameter appended after the existing ones, sanitized with `_pnl_number` like the other money values. `DryRunNotifier.daily_summary` and the harness's `RecordingNotifier.daily_summary` are variadic wrappers that already accept it. The total line:
`Total P&L: $+1,234.56 (+1.23% of prior snapshot, +0.98% of starting capital) — realized: ..., unrealized: ...`.

- [ ] **Step 1: Write the failing formatter tests** (inputs are `portfolio_value` and `daily_pnl`, from which the method derives the base; `portfolio_value=101234.56, daily_pnl=1234.56` gives a base of 100000, `total_pnl=1234.56`):
  - `test_total_line_shows_both_percentages` (`starting_capital=126000`): `(+1.23% of prior snapshot, +0.98% of starting capital)`.
  - `test_total_line_without_a_prior_snapshot_shows_only_capital` (`daily_pnl=None`): `(+0.98% of starting capital)`.
  - `test_total_line_with_a_non_positive_starting_capital` (`starting_capital=0`, and `-5`): `(+1.23% of prior snapshot, n/a of starting capital)`. At unit level only: through the trader a falsy capital makes `total_pnl` `None` and the total line is not printed.
  - `test_total_line_keeps_todays_text_for_an_old_caller` (the keyword **omitted**; and, separately, an explicit `None` and `float("nan")`): exactly today's `(+1.23%)`. The omitted-keyword case already passes before the change; the explicit `None` case is the one that fails first.
- [ ] **Step 2:** `./offline.sh scripts/run_tests.sh tests/paper/test_notifier.py -q`. Expected: the tests that pass the keyword FAIL (unexpected keyword); the omitted-keyword one passes.
- [ ] **Step 3: A scenario that can see the denominator.** Only 2 of the 95 goldens print a numeric total and both round to the same percentage on either base, so swapping the bases would pass every existing golden. Add `total_pnl_two_bases` to `scenarios.py`: a copy of `plain_buys_and_sells` whose `get_earliest_snapshot` returns a starting capital that differs enough from the prior snapshot's value that the two rendered percentages differ at two decimals. Its docstring states the two expected percentages; `test_golden_shows_scenario_behavior` asserts both substrings; the scenario count becomes 96 (`EXPECTED_SCENARIO_COUNT`).
- [ ] **Step 4:** Implement the keyword and the line; pass `starting_capital=starting_capital` at the trader's call site.
- [ ] **Step 5: Regenerate the goldens and check them exactly.** `./offline.sh env GGT_REGEN_TRANSCRIPTS=1 scripts/run_tests.sh tests/paper/test_trader_transcripts.py` (it fails with "regenerated, review and rerun"), then rerun without the variable. Expected: PASS. Every golden that records `notifier.daily_summary` (73 of them) gains `starting_capital` in its kwargs, and only the 2 with a numeric total plus the new one change their message text. Prove nothing else moved with a normalization script run over every changed golden: load the old version (`git show HEAD:<path>`) and the new one, remove `starting_capital` from each `notifier.daily_summary` call's kwargs, rewrite the new text `\(([+-]\d+\.\d+)% of prior snapshot, [^)]*\)` back to `(\1%)`, and require the two JSON trees to be equal (the new scenario has no old version and is read by hand). A line-level `grep -v 'Total P&L'` filter is not enough: each golden keeps the whole Telegram message on one line.
- [ ] **Step 6: Mutation check** (temporary, restored from a backup): swap the two bases in the formula. Expected: the first formatter test and the `total_pnl_two_bases` golden both fail. Rebuild `$E/safety-nodes.txt` (the node count rises by the new scenario's cases).
- [ ] **Step 7:** One fresh Sol review of this diff (small; Claude-built). Then the gate with label `telegram` (all six statuses 0), `ruff`, and **commit** `feat(paper): Telegram total shows the percentage against the prior snapshot and against starting capital`, with a changelog line. The Phase 4 plan's counts of 95 goldens become 96 (its hash records and scenario-count checks): update it before Phase 4 starts.

### Task 14.6: Merge to `main` (only on the owner's say-so)

The merge candidate is the head of `refactor/phase-0-2` after Task 14.5: record its SHA and test that exact checkout.

- [ ] **Step 1:** Do not start without an explicit yes. The main checkout is on branch `research/intraday-r9-screen` with uncommitted files, and no worktree has `main`, so merge in a separate worktree: `git worktree add /home/flynn/ggTrader-merge main`, then `git -C /home/flynn/ggTrader-merge merge --no-ff refactor/phase-0-2`.
- [ ] **Step 2:** The merge worktree needs its own `.venv` (`uv sync --frozen --extra test --extra dev`), a copy of the frozen price tape, and `offline.sh`. Run the gate against it: `WT=/home/flynn/ggTrader-merge $E/gate_run.sh merge`, then `WT=/home/flynn/ggTrader-merge $E/gate_compare.sh merge`, plus `ruff` and `verify_pins.py` there. Expected: all six statuses 0, all other checks exit 0.
- [ ] **Step 3:** Stop there. **Pushing `main` is a separate decision:** a push to `main` runs CI and `.github/workflows/docker-build.yml`, which smoke-tests and publishes the container image to GHCR (it has no deploy step); the trader is off and nothing deploys by itself, but the owner decides.

---

## The road to done

| Phase | Plan | State | Needs before it starts |
|---|---|---|---|
| gate tooling, finish Phase 2, review | this plan, 14.0 to 14.4 | next | nothing |
| Telegram change | 14.5 | approved, unbuilt | the Phase 2 gate; precedes the merge and Phase 4 |
| merge to `main` | 14.6 | owner-gated | the review, 14.5 and a say-so |
| 3 engine and WFO | `2026-11-18-refactor-phase-3.md` (tasks 15 to 29) | written, Sol-reviewed, fixes applied | the Phase 2 gate; the owner's go and quota (only tasks 15 to 19 if quota is unknown); its Task 29 (string return to exception) defaults to skip |
| 4 paper ports and records | `2026-11-18-refactor-phase-4.md` (tasks 30 to 45) | written, Sol-reviewed, revised | Task 14.5 (its goldens become 96); the owner's go, quota and answer on its optional Task 45 (default skip) |
| 5 split `PaperTrader._run` | to be written | not planned | Phase 4 done; see below |
| 6 optional tidy-ups | to be written, only if wanted | not planned | owner decision |

Phases 3 and 4 each branch from `refactor/phase-0-2` (Phase 4 does not need Phase 3 merged); running them one after the other is the plan, not a dependency. **Order:** 14.0 to 14.4, then 14.5, then (on the owner's yes) 14.6, then Phase 3 (its engine and WFO chains can be edited in parallel worktrees; their test runs queue on the lock), then Phase 4, then write and review the Phase 5 plan, then Phase 5. Report quota after each phase so the owner can stop early.

**What the Phase 5 plan must start from** (this is the part of the design that carries the most risk: an 822-line live-order method, `trader.py:716-1537`, becomes about ten objects):
- A first task that adds transcript scenarios for the branches that two reviews found unexercised and that Phase 5 moves: persisted split-state cleanup and re-verification and the split and dividend persistence failures (`trader.py` about `:514-694`), the prior-snapshot and invalid-final-equity paths, notice-failure handlers, and everything in `run_paper_trading` (`:1539-1583`, including the margin guard and the dry-run notifier wiring). Each gets a golden and a mutation that fails it before any code moves.
- The Phase 4 hand-off (ledger injection, `RunResult`, `TraderConfig`, typed records) as its entry condition, and the narrowed claim that Phase 4 gives ledger-independent testing, not a patch-free whole-trader test.
- The design's largest Phase 5 item: **migrating the 4,220-line `tests/paper/test_trader.py` to the in-memory ledger, then deleting the `ModuleLedger` adapter and `_run_legacy`**, kept valid in the meantime by the adapter.
- One step object per task (`PreflightGate`, `PortfolioView`, `RiskAssessor`, the pure planning functions, `OrderExecutor` with `DryRunExecutor`, `CorporateActions`, `RunReporter`), the order of calls preserved exactly, goldens byte-identical at every task, and a seeded property test of the new planning functions against the old method before it is deleted.

## Execution roles (`CLAUDE.md` delegation preference of 2026-11-08)

- **Tasks 14.0 to 14.5** are small, strictly serial and touch shared files, so they run natively (one builder, the orchestrator) with the two whole-branch reviewers at 14.4 and one Sol review of 14.5. Sol reviews Claude-built work and Opus reviews Codex-built work; neither reviews its own family.
- **Phases 3 and 4** use fresh builders per task (Codex Terra or Luna, or Claude Sonnet) with the heavy-job rules above; the orchestrator re-verifies each diff, reruns each task's tests and commits.
- **Quota:** the owner reports headroom before Phase 3 and again before Phase 4; usage is reported after each phase.
