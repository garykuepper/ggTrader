# Finish the Membership Build, Merge, and Clean Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and merge the finished S&P 500 membership-database branch to `main`, then remove confirmed dead code, fix the leftover minor issues and stale docs, and refresh the knowledge graph, without ever restarting or deploying the live trader.

**Architecture:** Three gated stages. Stage 1 verifies branch `feature/membership-db` at its final commit exactly as GitHub CI will see it (no database), then merges and pushes it. Stage 2 is a cleanup slice on top of the merged `main`, built by a delegated builder and reviewed by an independent model. Stage 3 refreshes the graph and notes. Every stage ends with the same verification: tests with the database variables removed, the reference backtests, the file pins, lint.

**Tech Stack:** Python 3.12, uv, pytest through `scripts/run_tests.sh`, ruff, PostgreSQL/TimescaleDB (schema `universe`), vulture (run through `uvx`, never added to the project), graphify, GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-11-07-membership-database-design.md` (Appendix C overrides earlier sections). Related: `docs/superpowers/plans/2026-11-07-study-isolation/` (finished).

## Global Constraints

- The live trader stays OFF. Never run `ggt paper` in any mode, never run `docker compose pull`, `up`, `restart` or rebuild the `ggtrader_live` container, never touch cron or any installed systemd unit. Pushing to `main` only rebuilds the published image; nothing pulls it.
- The intraday observer is untouched: release `280d4a0edc00f1b4b23ba7bbe8dd80432877b36f`, its `ggtrader-intraday-observer-*` unit files (sha256 of the unit-file hashes `7e50dbfc6345ff91`) and its 4 timers.
- Reference results must not move: core17 Sharpe 0.3905, CAGR 3.50%, gates 1/17; dsr0 Sharpe 0.6374, CAGR 3.41%, gates 17/17; both `oos_equity.csv` files `cmp`-identical to `research/programme/vectorbt-engine/p5-evidence/guarded/`; live replay 120/120 on `fixture:sp500_v1_2026-06-02`.
- `scripts/verify_pins.py` must print `MATCH=1 MOVED-MATCH=4 HISTORY-MATCH=17 FAIL=0`.
- Real database facts after the build: `universe.loads` 3, `universe.membership` 1,293, `public.ohlcv` 295,251,075 rows. Read-only commands only unless a task says otherwise; never print or copy credentials or connection strings.
- GitHub CI has no database: run every full-tier check with `env -u POSTGRES_CONNECTION_STRING -u DB_HOST -u DB_USER -u DB_PASS -u DB_NAME -u DB_PORT -u GGT_TEST_DB_URL -u GGT_MEMBERSHIP_DB_URL -u GGT_MEMBERSHIP` in front of `scripts/run_tests.sh`.
- Tests only through `scripts/run_tests.sh` (8 GB memory cap). The fast tier takes about 100 s; the full tier about 12 minutes (tests half then the study half).
- Stop background work only with the TaskStop tool or `kill` of a PID whose full command line was read first; never `pkill -f` with text that appears in the same command, never kill a parent PID found with `ps -o ppid` (see memory note "Never kill by guessed PID").
- Ruff `E/F/W/I`, line length 100; never `--no-verify`; commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- No real Telegram message from any test or dry run; no private account records in delegate prompts.
- The owner's delegation rule (2026-11-08): Claude orchestrates and re-verifies every slice itself; Codex `gpt-6.1-terra` / `gpt-6-luna` may build; `gpt-6.1-sol` may review. Call `orchestrator_capabilities` once per thread before the first `delegate_task`.

## Review Focus

- **A deleted name that is used dynamically** (a `getattr`, a string in a YAML/JSON/TOML file, a CLI entry point, a pytest fixture): the mention search must cover every tracked file type, and the full tier must still pass.
- **Tests that quietly need the database:** they pass on this machine and fail on CI. The with-variables-removed run catches them.
- **The cleanup touching a hash-pinned or moved study file:** `verify_pins.py` and `research/CODEMOVED.csv` must still agree.
- **A doc that states today's gate status or counts:** such text goes stale in a day; describe behavior, not a snapshot.
- **Pushing triggers an image rebuild:** confirm nothing in the repo or host auto-pulls it onto the running trader container before every push.

---

### Task 1: Verify the membership branch at its final commit

**Files:**
- Read only: branch `feature/membership-db` in `/home/flynn/ggTrader-deps`
- Scratch output: `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/final_v/`

**Interfaces:**
- Consumes: the branch HEAD. The fix agent's last code commit is `6d46f67f` (`docs: changelog for the second review round`); the only commit after it adds this plan file. Worktree clean.
- Produces: a pass/fail verdict for Task 2; scratch logs `full.txt`, `o.log`.

- [ ] **Step 1: Confirm the four second-round fixes are present.** Run `git log --oneline ee83e37d..HEAD` and check the subjects: mismatch closes the gate (`5e8f7c36`), xlsx column and cell caps (`657d2cd2`), bounded commits API read (`bed4c4f2`), normalizer version not a no-op (`075ebaf5`), plus the changelog commit. Expected: five commits.
- [ ] **Step 2: Run the full tier with the database variables removed.** `env -u ... scripts/run_tests.sh --full -q > final_v/full.txt` (expected: tests half and study half both end `passed`, zero failed; study half is 777; the earlier totals were fast 1,845, full 2,184 + 777, so the new total is at least that).
- [ ] **Step 3: Run the reference backtests.** Adapt `scratchpad/m7v/o_run.sh` (output folder `final_v/o`; it already omits the finished-refactor allow-list step) and run it. Expected: `core17_ensemble_sp500 FIELD-EQUAL ... 1/17`, `core17_ensemble_sp500_dsr0 FIELD-EQUAL ... 17/17`, both CSV `cmp` steps and the replay (120/120) exit 0, fresh-clone `ruff check` and `ruff format --check` pass.
- [ ] **Step 4: Check pins, lint and the database read-only.** `scripts/verify_pins.py` prints the pin line above; `ggt db membership verify --load-id 1`, `2`, `3` (no options) each end `OK`; `ggt db membership status` shows `live gate would PASS`; table counts match Global Constraints.
- [ ] **Step 5: Decide.** All steps green: continue to Task 2. Anything red: stop and report the first failing command and its output; do not merge.

### Task 2: Merge to `main`, push, watch CI

**Files:**
- Modify: `main` of `/home/flynn/ggTrader` (merge commit only)

**Interfaces:**
- Consumes: Task 1 green.
- Produces: `origin/main` containing the membership build; the CI run id for Task 3.

- [ ] **Step 1: Pre-push safety.** Confirm `git status` clean in `/home/flynn/ggTrader`, `git rev-parse main origin/main` equal before the merge, `crontab -l | grep -v '^#' | grep -c paper_trade` prints 0, and `docker ps` shows no auto-update container (watchtower).
- [ ] **Step 2: Merge.** `git merge --no-ff feature/membership-db -F <message file>` (message: "merge: versioned S&P 500 membership database, reader, live gate, weekly job"; `git merge` cannot read the message from stdin). Re-run `scripts/verify_pins.py` and `ruff check .` in the main checkout.
- [ ] **Step 3: Push.** `git push`, then wait for the CI run on the new head sha (jobs `test`, `studies`, `full` and the Docker build). Expected: all success. Failed job: read its log, fix on a new branch, do not force-push.
- [ ] **Step 4: Do not deploy.** Never run `docker compose pull`/`up`; the trader image on the host stays as is.

### Task 3: Cleanup slice A: remove confirmed dead code

**Files:**
- Modify/Delete (names, not line numbers; lines shift after the merge): `src/ggTrader/data/core/constants.py` (`interval_map`, `QUOTES`; possibly the whole module), `src/ggTrader/data/core/stock_constants.py` (`MARKET_OPEN_MINUTE`, `MARKET_CLOSE_MINUTE`, `MARKET_TIMEZONE`), `src/ggTrader/data/historical/timescaledb_loader.py` (`get_consistent_movers`), `src/ggTrader/engine/__init__.py` (`SETTINGS`), `src/ggTrader/lab/google_trends_data.py` (`QueryFn`), `src/ggTrader/engine/_compat.py` (module with no non-test importer)
- Candidates to judge one by one (used only by tests): `engine/windows.py:fold_positions`, `engine/config.py:SETTINGS_DOC`, `lab/persist.py:plan_done/write_plan/read_plan/start_sweep`, `lab/strategies/contract.py:used_values`, `lab/strategies/voter_strategies.py` classes `EmaCross`, `BbReversion`, `RsiReversion`, `MacdDivergence`, `VolumeBbReversion`, `utils/fred_data.py:cache_series`, `utils/manifest.py:file_record/verify_frozen`, `utils/results.py:read_result_json`
- Keep regardless: `utils/study.py:load_driver` and the other documented study tooling (`AGENTS.md` names them), anything a registry, CLI decorator or pytest fixture reaches dynamically
- Test: the existing suites; delete a test only together with the code it alone covers

**Interfaces:**
- Consumes: merged `main` (Task 2).
- Produces: a branch `cleanup/dead-code` with one commit per removed group and a report listing each candidate as removed or kept with its reason.

- [ ] **Step 1: Regenerate the candidate list.** In a scratch clone of `main`: `uvx --quiet vulture src --min-confidence 60`, then for every name search `git ls-files '*.py' '*.md' '*.toml' '*.yml' '*.yaml' '*.sh' '*.json'` for other mentions (word-boundary match, excluding the definition line). Zero other mentions means certainly dead; mentions only in `tests/` or `research/**/tests/` means test-only. Expected on `654265da`: 103 candidates, 8 with zero mentions (the names in Files above).
- [ ] **Step 2: Remove the zero-mention names** in small commits per file; when `data/core/constants.py` ends with no importer, delete the module and fix `AGENTS.md` rule 11 (it names `SYMBOL_MAPPING` there) in the same commit. Run `scripts/run_tests.sh --fast -q` after each commit.
- [ ] **Step 3: Judge the test-only candidates** with this rule: remove (with their tests) when not named in `AGENTS.md`, `docs/` or a skill as supported API and no study code uses them; otherwise keep and record the reason. Never remove a table-creating path in `lab/persist.py` (database objects are never deleted).
- [ ] **Step 4: Verify** with Task 1's steps 2-4 on the cleanup branch (full tier with database variables removed, reference backtests, pins, lint). Expected: identical reference results; total test count lower by exactly the tests removed in Step 3.

### Task 4: Cleanup slice B: minor fixes and stale docs

**Files:**
- Modify: `AGENTS.md` (symlinked as `CLAUDE.md`): delegation section (Codex models per the owner's 2026-11-08 message, Claude stays orchestrator and verifier), section 1 data layer and section 2 universe/data-access bullets (the S&P 500 list now comes from `MembershipSource`; the frozen file is the vintage-1 fixture), the CLI list (`ggt db membership ...`, `ggt db load-sp500`)
- Modify: `docs/architecture.md` (remove the nonexistent `lab_periods` row; add the `universe` schema and the single reader), `docs/next_steps.md` (state that the trader is off since 2026-11-06 and what the membership build left open), `.claude/skills/research-report-review/SKILL.md` (the old `max_leverage 2.0` wording), `scripts/run_tests.sh` (its closing message lists old marker names)
- Check, change only if pins stay intact: `research/audits/event-timing/code/form4_backfill.py` and `earnings_surprise_backfill.py` (they skip `bootstrap`)
- Test: `tests/scripts/test_codemoved.py`, docs sweep tests, the full tier

**Interfaces:**
- Consumes: Task 3's branch (or `main` if Task 3 is skipped).
- Produces: the same branch with doc and script fixes; a changelog entry in `docs/changelog.md`.

- [ ] **Step 1: Make each doc edit** so it describes behavior and not a dated snapshot; every path it names must exist (check with `ls`).
- [ ] **Step 2: Backfill drivers.** Add `bootstrap(__file__)` per `docs/superpowers/plans/2026-11-07-study-isolation/PLAN_V2.md` section 3 only if `scripts/verify_pins.py` still prints the pin line and the study tests pass; otherwise leave them and note why.
- [ ] **Step 3: Update `docs/changelog.md`** with one entry (what was removed and fixed, why, reference results unchanged).
- [ ] **Step 4: Verify** as in Task 3 Step 4.

### Task 5: Independent review, merge, push

**Files:**
- Read only: the cleanup branch diff

**Interfaces:**
- Consumes: Tasks 3 and 4 verified.
- Produces: `origin/main` with the cleanup; CI green.

- [ ] **Step 1: Delegate a read-only review** to `gpt-6.1-sol` (`role: review`, reasoning effort high; same constraints as the membership review: no tests, no network, no edits, no credentials printed). It checks that every removed name is truly unused (grep evidence), no pinned or moved study file changed, no test was weakened, and the doc statements are true.
- [ ] **Step 2: Fix confirmed findings** with a test that fails before and passes after; re-verify.
- [ ] **Step 3: Merge and push** as in Task 2 Steps 1-4 (merge commit, push, CI green, no deploy).

### Task 6: Refresh the graph and the notes

**Files:**
- Modify: `graphify-out/` (git-ignored), memory notes under `/home/flynn/.claude/projects/-home-flynn-ggTrader/memory/`

**Interfaces:**
- Consumes: merged `main`.
- Produces: an updated graph and notes.

- [ ] **Step 1: Run the graph update** (`detect_incremental` on `/home/flynn/ggTrader`; code files are free; docs changed since the last build go through one Haiku extraction agent; prune deleted paths; if the shrink guard refuses, check that no existing file lost its nodes before forcing).
- [ ] **Step 2: Update the notes:** mark the membership build done in `project_study_isolation_state_2026-11-07.md` (or a new `project_membership_db_2026-11-08.md`), record the incident lesson already saved, and keep `MEMORY.md` to one line per note.

### Task 7: Owner-gated follow-ups (do not execute without the owner's explicit word)

- Enable the weekly timer (`deploy/membership_weekly/`): only after three clean dry runs (done once, 2026-11-08), a restore drill (done), and the owner's go; the trader's gate expires 45 days after the last SPY match (about 2026-12-24) without it.
- A second free source of index changes before the trader ever returns, and the spec's open checks (Appendix C6).
- Returning the trader to service, the AVB holding, the SPY sell order: owner decisions, not for the agent to raise or act on.
- The dev database password is a guessable word: change it if the database port is reachable beyond this machine.

---

## Self-review

- **Spec coverage:** the spec's remaining items map to Tasks 1-2 (merge) and 7 (timer, second source); the owner's goal (merge and push, fix bugs, refactor, remove dead code) maps to Tasks 1-5.
- **Step scan:** every verification step names its command and expected output; no step has an unnamed body.
- **Type consistency:** names used (`MembershipSource`, `verify_pins.py`, `run_tests.sh --full -q`) match the code.
- **Proportion:** seven tasks for two stages of work; the plan names files and decisions, not code.
