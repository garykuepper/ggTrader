# Refactor completion and clean-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land refactor phases 3 to 5 on `main` with the trader's behavior unchanged, finish the
lint and tidy-up work the design left open, and remove the branches, worktrees, stale files and
stale text the refactor left behind.

**Architecture:** Three parts, in order. Part A closes one behavior gap found in phase 5, merges the
stacked branch `refactor/phase-5` (phases 3, 4 and 5) into `main`, and proves nothing changed with
the gate from phases 0 to 5. Part B finishes the lint rules and the optional phase 6 tidy-ups on
fresh branches off `main`. Part C audits the docs and removes what is no longer needed.

**Tech Stack:** Python 3.11+, pytest through `scripts/run_tests.sh`, ruff, the gate scripts in
`/home/flynn/ggTrader-refactor-evidence/`, `scripts/verify_pins.py`, git worktrees.

**Spec:** [2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md) (sections 4, 5.4,
6, 7, 8). Phase plans already executed: [0-2](2026-11-17-refactor-phase-0-2.md),
[3](2026-11-18-refactor-phase-3.md), [4](2026-11-18-refactor-phase-4.md),
[5](2026-11-19-refactor-phase-5.md).

## State at writing (2026-11-20)

- `main` is at `8282d233` (pushed). It holds phases 0 to 2 and the test-run safety work.
- `refactor/phase-5` is at `c6dbd1f1`, 131 commits ahead of its merge base `1c7a3138`. It holds
  phases 3, 4 and 5 stacked (`refactor/phase-3` and `refactor/phase-4` are its ancestors).
- Phase 5 gate at `3475f405`: 4,967 core tests, 1,144 study tests, oracle and outcome comparisons
  passed, 103 transcript goldens and 600 seed digests unchanged. Opus found no wiring blocker. Sol
  conditionally approved; the gate record it asked for was added afterwards and **Sol has not
  re-read it**.
- Known gap: non-mapping weights from a signal source now behave differently from `1c7a3138`
  (`docs/changelog.md`, 2026-11-20 entry, last bullet).
- Not measured here: how many tests the r9 study (merged to `main` after the merge base) adds to the
  study tier. Task 2 counts it.
- The trader stays stopped and cron stays off throughout. Deploying is **not** part of this plan.

## Global Constraints

- Nothing the trader trades, no Telegram text, no SQL statement and no research number may change,
  except through a change the owner approves by name (design section 1).
- Run tests only through `scripts/run_tests.sh`, one run at a time, `GGT_TEST_MEM_MAX=3G`; the p5
  oracle driver needs the literal `6G` cap and is recorded as approved only by the owner's word
  (`owner-approved-6G.txt`, first line `cap=6G`). A 137 or 143 exit is a hang: stop and report, do not
  raise the cap.
- Every command that touches the gate runs offline: `./offline.sh` (scrubbed environment, no `.env`,
  no database address) in a worktree with no `.env`.
- Engine boundary: only `src/ggTrader/engine/` imports vectorbt. `ohlcv.timestamp` is naive UTC.
  Nothing under `data/` is deleted. No paid data. No recommendation to retire the trader, the sleeve or
  the project.
- Frozen files stay frozen: `scripts/p5_wfo_validation.py`, `p7_live_golden.py`,
  `legacy_reference_runs.py`, `legacy_reference_compare.py`, `verify_pins.py`, `verdict_register.py`,
  `study.py`, `vbt_latest.py`, the transcript goldens and the digest file. Any edit to a pinned source
  file updates the counts in `.github/workflows/ci.yml:147` in the same commit and passes
  `python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0`
  (or the new counts).
- Lint: `ruff check .` and `ruff format --check .` pass. Never run `ruff --fix` on `src/ggTrader/paper/`.
- No push without the owner's word. A merge into local `main` is allowed once its gate passes.
- Delegation (`CLAUDE.md`): builders Terra, Luna, Sonnet, Haiku; reviewers Sol and Opus; a reviewer
  never reviews its own family's build (Codex output goes to Opus or Fable, Claude output goes to Sol);
  Task 3 uses Sol and Opus on the whole branch. **Owner instruction 2026-11-20: Claude quota is ample
  and Fable (`claude-fable-5-2`) may be used for hard, complex reviews.** This plan uses it in Tasks 2
  (conflict check), 3 (whole-branch review, in addition to Sol and Opus) and 6 (survivor list); it is
  not used for builds or bulk reads. `CLAUDE.md` does not name Fable yet; the owner decides whether it
  is added. The orchestrator (Sonnet) reads every diff and reruns every check itself. Haiku does bulk
  reads, and a stronger model checks each figure it reports. Report usage after each part. Keep
  account records out of delegate prompts.

## Owner decisions (defaults apply only if the owner says "use the defaults")

| # | Question | Default |
|---|---|---|
| D1 | Merge phases 3 to 5 into local `main` after the gate and both reviews pass? | Yes. Pushing is a second, separate yes. |
| D2 | Restore the old behavior for malformed (non-mapping) weights (Task 1)? | Yes: it is a behavior change the refactor made without approval. |
| D3 | Retire the eight private `PaperTrader` seam forwards? | Keep. |
| D4 | Add a runtime refusal of `InMemoryLedger` in a live run? | No; keep the static import guard. |
| D5 | Replace `run_wfo`'s string return with an exception (phase 3, Task 29)? | Skip. |
| D6 | Docstring rules `D100`-`D104` (about 150 hits)? | Defer; enable annotations only. |
| D7 | Which Phase 6 candidates (Task 7) go ahead? | None until Task 7 sizes them. `verdict_register.py` stays frozen. |
| D8 | Evidence folder `~/ggTrader-refactor-evidence` (341 files, 79 MB)? | Copy to `/media/thesix/ggTrader-data/refactor-evidence/` with hashes, keep the original 30 days. |

## Review Focus

Failure modes the spec implies that no existing task's tests cover; each has its test in the task named.

1. Malformed weights (`[]`, a list of pairs, `None`) reach the planners and let exits or a sweep sell
   through that the base code stopped: Task 1.
2. The merge with `main` changes a gated artifact silently (a conflict resolved in `ruff.toml`, `ci.yml`
   or a changelog hides a rule or a pin count): Task 2 re-runs the whole gate on the merged tree, not
   on the branch.
3. A lint rule's autofix changes behavior (an unused-argument removal drops a parameter callers pass
   by keyword; `G004` changes a logged message; `zip(strict=...)`): Task 5 compares rendered log
   messages and the study-surface test before and after.
4. Cleanup deletes something still referenced (a branch with unique commits, a worktree with
   uncommitted work, the baseline worktree the gate compares against, a doc cited by a report): Task 9
   proves each target is merged or identical before removal and removes nothing else.
5. The docs keep naming retired things (`ModuleLedger`, `_run_legacy`, the frozen oracle copy of
   `_run`, `lab/simulate.py`) and mislead the next session: Task 8 greps for each name.

---

## Part A: land phases 3 to 5

### Task 1: Restore the old behavior for malformed weights (D2)

**Files:**
- Modify: `src/ggTrader/paper/models.py:206-211` (`StrategySignals.__post_init__` copies weights with
  `dict(...)`, which accepts `[]` and lists of pairs) and, if the base behavior needs it,
  `src/ggTrader/paper/planning.py:35`
- Test: `tests/paper/test_run_rules.py` (named tests below)

**Interfaces:**
- Consumes: `StrategySignals(..., weights: Mapping[str, float], scale: float)`
  (`models.py`); the base behavior is read from `git show 1c7a3138:src/ggTrader/paper/trader.py`
  (weights are used at base lines 801, 1158-1165 and 1491 only).
- Produces: no new names. After this task the trader raises the same exception type, at the same
  point in the run, as `1c7a3138` for each malformed input.

- [ ] **Step 1: Characterize the base.** In a scratch worktree at `1c7a3138`
  (`git worktree add --detach /home/flynn/ggTrader-p5base 1c7a3138`), run the run-rule harness with
  three weights values (`[]`, `[("sp500", 1.0)]`, `None`) and record, for each, the exception type
  and the list of orders submitted before it. Save the three records in
  `/home/flynn/ggTrader-refactor-evidence/followup/weights-base.json`.
- [ ] **Step 2: Write the failing tests** in `tests/paper/test_run_rules.py`:
  `test_empty_sequence_weights_match_base`, `test_pair_sequence_weights_match_base`,
  `test_none_weights_match_base`. Each asserts the exception type and the submitted-order list equal
  the Step 1 record. Run `scripts/run_tests.sh tests/paper/test_run_rules.py -k weights`. Expected:
  the empty-sequence and pair-sequence tests FAIL on the branch.
- [ ] **Step 3: Implement** the smallest change that reproduces the base: reject a non-`Mapping`
  `weights` in `StrategySignals.__post_init__` with the type and at the stage the base used (if the
  base failed later, at the first `weights.get`/`.items()` use, make the new check raise there
  instead). Choose by the Step 1 record, not by preference.
- [ ] **Step 4: Run** `scripts/run_tests.sh tests/paper -q` (expect all pass) and the 103 transcript
  goldens plus the 600-digest check (part of `tests/paper`; expect unchanged, no golden edited).
- [ ] **Step 5: Edit `docs/changelog.md` and `docs/architecture.md`** to delete the "non-mapping
  weights can behave differently" caveat and say the behavior is restored. Commit:
  `fix(paper): malformed weights fail as they did before the phase 5 split`. Builder Terra, reviewer Opus.

### Task 2: Integrate `main` and rerun the full gate on the merged tree

**Files:**
- Modify: whatever the merge conflicts in (expected: `docs/changelog.md`, `.gitignore`, possibly
  `docs/architecture.md`, `research/RESEARCH_SNAPSHOT.md`)
- Create: `/home/flynn/ggTrader-refactor-evidence/p6/` (logs; outside the repo)

**Interfaces:**
- Consumes: gate scripts `gate_run.sh <label>` (runs `gate_oracles.sh` then `gate_tests.sh`) and
  `gate_compare.sh`; baselines in `/home/flynn/ggTrader-refactor-evidence/baseline*`. A label is
  used once.
- Produces: a merge commit on `refactor/phase-5` whose tree passed the gate; files
  `p6/final-compare-status.txt` with `outcome compare exit=0` and `oracle compare exit=0`.

- [ ] **Step 1: Merge** `main` into `refactor/phase-5` in `/home/flynn/ggTrader-refactor`
  (`git merge main`). Resolve conflicts keeping both sides' entries (newest changelog entry first).
  Verify `git diff --check` is empty and `.github/workflows/ci.yml` and `ruff.toml` differ from the
  branch head only if `main` changed them (`git diff c6dbd1f1 HEAD -- ruff.toml .github`).
- [ ] **Step 2: Count what the merge added to the study tier**: `./offline.sh scripts/run_tests.sh
  --studies --collect-only -q | tail -3`. Record the new total next to the 1,144 baseline; the
  difference must be exactly the r9 study's 222 tests (and any other test `main` gained). Anything
  else is a finding.
- [ ] **Step 3: Run** `/home/flynn/ggTrader-refactor-evidence/gate_run.sh p6final` offline, one run at a
  time. Then `gate_compare.sh`. Expected: both compare statuses exit 0; full tier at least 4,967
  passed with the same one skip and five expected failures; study tier passes with only the
  added tests as additions.
- [ ] **Step 4: Run** `ruff check .`, `ruff format --check .`, and
  `python scripts/verify_pins.py --expect MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0`. Expected: clean.
- [ ] **Step 5: Record** the results in `docs/changelog.md` (one paragraph, "integration gate on the
  merged tree") and commit `docs: record the merged-tree gate`. Builder Sonnet; the orchestrator
  reruns Step 3's compare itself. Before Step 3, Fable reads the conflict resolutions (`git diff c6dbd1f1 HEAD`
  limited to the files that conflicted) and confirms no rule, pin count or entry was dropped.

### Task 3: Whole-branch review by Sol, Opus and Fable

**Files:**
- Create: `/home/flynn/ggTrader-refactor-evidence/p6/review-sol.md`, `review-opus.md`, `review-fable.md`

**Interfaces:**
- Consumes: the merged branch head from Task 2 and the claims list below; nothing else (fresh
  contexts, no builder summaries).
- Produces: two review files with each finding classified blocker or nit.

- [ ] **Step 0: Split the work so the three reads are not copies of each other.** Sol: the behavior
  claims (a) and (c) below against the transcripts and digests. Opus: production wiring and the gate
  evidence, claims (b), (d), (e). Fable: the hardest read, a fresh adversarial pass over the whole diff
  for anything the other two framing prompts would not surface (state shared between steps, clock reads,
  exception types, ordering of broker and database calls), told only what the code is for, not what
  the builders concluded. Each writes its own file; none sees another's.
- [ ] **Step 1: Brief each reviewer** with only: the diff `main...HEAD` (phases 3 to 5), the design
  sections 4 to 6, the gate logs, and these claims to test: (a) orders, SQL, Telegram text and clock
  reads unchanged; (b) the 21 mutation forms are caught by tests that reach the code; (c) Task 1
  restores the base weights behavior; (d) the supplement to the Phase 5 gate record (the part Sol never
  re-read, `docs/architecture.md` "Phase 5 verification") matches the logs; (e) the phase 3 and phase 4
  verification claims match their logs in the evidence folder. If phase 3 or 4 never had both
  reviews, say so and review them here.
- [ ] **Step 2: Fix** every blocker with its own test and commit; re-run Task 2 Step 3 only if production
  code changed. Nits go to the Task 8 doc audit or are listed as accepted.
- [ ] **Step 3: Record** reviewer, verdict and any residual item in `docs/changelog.md`. Commit
  `docs: record the final reviews of phases 3 to 5`.

### Task 4: Merge into local `main` (D1)

**Files:** none new.

- [ ] **Step 1: Confirm** D1 is answered yes and `git -C /home/flynn/ggTrader-refactor status --short` is empty.
- [ ] **Step 2: Merge** in `/home/flynn/ggTrader-merge` (the worktree holding `main`; it is on
  `docs/snapshot-2026-11-20` now, so `git switch main` there first and keep the snapshot branch
  untouched): `git merge --no-ff refactor/phase-5 -m "Merge refactor phases 3 to 5: engine helpers, paper interfaces, run split"`.
  Expected: no conflicts, because Task 2 already merged `main`.
- [ ] **Step 3: Verify** `git diff refactor/phase-5 HEAD --stat` prints nothing, then run
  `./offline.sh scripts/run_tests.sh -q` (fast core). Expected: pass.
- [ ] **Step 4: Stop.** Report the merge. Do **not** push; ask the owner. CI cannot run the real-tape
  oracles, so the local gate of Task 2 is the evidence.

## Part B: finish the open design items (new branches off `main`, one worktree each)

### Task 5: Lint rules with small hit counts

**Files:**
- Modify: `ruff.toml` (the `select` list and `per-file-ignores`), the `src/ggTrader/` files the hits
  name
- Test: existing suites plus `tests/utils/test_import_rules.py` unchanged

Hits counted 2026-11-20 with `ruff check src/ggTrader --isolated --select <rule>` (before honoring
per-file ignores, so approximate): `ARG001`/`ARG002` 11, `G004` 7, `DTZ` 1, `ANN001`/`ANN201` 30,
`T201` 170.

**Interfaces:** Produces: `select` gains `ARG001`, `ARG002`, `G004`, `DTZ`, `ANN001`, `ANN201`, `T201`
for `src/ggTrader/` only (negated glob in `per-file-ignores`, as the paper rules already do); CLI
modules (`src/ggTrader/cli/**`) are exempt from `T201`; the tape writer gets one documented `DTZ`
exception (design 5.4).

- [ ] **Step 1: Capture rendered logs before.** Run the fast core and `--studies` once; keep the
  pass lists (`gate_tests.sh` JUnit) as the "before".
- [ ] **Step 2: For each rule in turn** (one commit per rule, `ARG` first): list its hits, fix by hand
  (never `--fix` in `paper/`), add the rule to `select` with the per-file ignore, and run
  `ruff check .`. `ARG`: before removing a parameter, `grep -rn` for keyword callers in `src`, `tests`,
  `scripts` and `research/*/*/code`; a parameter a study passes by keyword stays and is documented.
  `G004`: convert to `%s` style; the rendered message must be identical (the log comparison in the
  transcript goldens pins logger, level and rendered text). `T201`: replace `print` by `logging` only
  where the output is not user-facing CLI output; list the survivors per file.
- [ ] **Step 3: After each rule** run `scripts/run_tests.sh --full` (serialized) and the study-surface test
  (`tests/utils/`, every `ggTrader` name a study imports still resolves). Expected: same pass list as
  Step 1, no golden changed.
- [ ] **Step 4: If a pinned file changed** (`cached_yfinance_loader.py` is the `MATCH` pin), update the
  count line in `.github/workflows/ci.yml:147` in the same commit and run `verify_pins.py`.
- [ ] **Step 5: Changelog entry** per rule set. Builder Terra for `ARG`/`G004`/`DTZ`/`ANN`; Luna for
  `T201`; reviewer Opus on the whole set.

### Task 6: Complexity and argument rules outside `paper/`

**Files:** `ruff.toml`, the files the hits name.

Hits (approximate, same method): `PLR0913` 68, `C901`/`PLR0912`/`PLR0915` 60 across `src/ggTrader/`.

- [ ] **Step 1: Measure honestly.** With the real config, `ruff check src/ggTrader --select C901,PLR0912,PLR0915,PLR0913
  --config ruff.toml` after removing the negated-glob ignore on a scratch copy; list hits per file.
- [ ] **Step 2: Decide per hit** whether it is a survivor (list it in `per-file-ignores` with its reason, as the
  paper survivors are) or a real split. A split is allowed only for a pure function covered by an
  existing test; otherwise list it. **No slice of `lab/wfo/run.py` or `engine/portfolio.py` is
  reshaped here**; they were handled in phase 3.
- [ ] **Step 3: Enable** the four rules for all of `src/ggTrader/` with the survivors listed; the
  `!src/ggTrader/paper/**` negated glob is removed. Run `ruff check .`, then the gate of Task 2 Step 3
  only if a production file changed beyond ignore-list edits.
- [ ] **Step 4: Changelog entry** with the survivor count. Builder Sonnet; reviewer Sol; Fable checks
  the survivor list for any entry that hides a real defect rather than a deliberate shape.

### Task 7: Size the phase 6 tidy-ups (read-only), then build what the owner picks (D7)

**Files:** Create `docs/superpowers/plans/2026-11-20-refactor-phase-6-sizing.md`.

Candidates from design section 7: membership CLI dispatch (`src/ggTrader/cli`), weekly-job helpers
(`src/ggTrader/data/membership_weekly.py`, 925 lines), persist transactions (phase 4 skipped
Task 45), `verdict_register` validator class (**excluded**: the design freezes that script).

- [ ] **Step 1: Haiku reads** each candidate's file in full and reports, with line numbers: its size,
  duplicated blocks, the tests that cover it, and any frozen-name importers (`grep -rn` over `research/*/*/code`
  and `tests`). The orchestrator checks every figure against the file.
- [ ] **Step 2: Write the sizing note**: per candidate, the change, the proof (existing tests, or a
  characterization test first), the risk, and a recommendation in plain words. Stop and ask the
  owner which to build. Default: none.
- [ ] **Step 3: For each approved candidate**, write its own plan (`superpowers:writing-plans`) before
  any code. This plan does not authorize the code.

## Part C: clean-up

### Task 8: Documentation audit and retired-name sweep

**Files:** `CLAUDE.md`, `AGENTS.md`, `docs/architecture.md`, `docs/cli_reference.md`, `docs/next_steps.md`,
`docs/where-we-left-it-2026-11-12.md` (move to `docs/archive/`), `docs/changelog.md`.

- [ ] **Step 1: Grep** each retired name over `docs/`, `README.md`, `CLAUDE.md`, `AGENTS.md`, `src/`, `tests/`,
  `scripts/`: `ModuleLedger`, `_run_legacy`, `RunHalted`, `lab/simulate.py`, `lab/replica.py`,
  `fetch_stock_ohlcv` (wrapper stays), `trader.py:` line references, and the old trader size
  (1,600 lines, 826-line `_run`). Command: `rg -n "<name>" docs README.md CLAUDE.md AGENTS.md src tests scripts`.
- [ ] **Step 2: Fix** each hit: current name, or "retired in phase 5" in changelog history only.
  `CLAUDE.md` section 1 must list the new `paper/` modules (`preflight.py`, `portfolio.py`,
  `risk_assessor.py`, `planning.py`, `order_flow.py`, `execution.py`, `corporate_actions.py`,
  `reporter.py`, `seams.py`, `run_helpers.py`, `ledger.py`, `ports.py`, `models.py`, `config.py`,
  `constants.py`) and the test-run rules from `scripts/run_tests.sh` (4G cap, heavy-job lock).
- [ ] **Step 3: `docs/cli_reference.md`**: confirm `ggt db truncate --yes-wipe-ohlcv` and `ggt db clean --yes`
  are documented, with exit code 2 without the flag.
- [ ] **Step 4: `docs/next_steps.md`** gets a status row for the refactor (what is merged, that the trader
  stays stopped, what remains owner-gated: D3 to D5, D7, deploy). Run `rg` again; expected no hits
  outside changelog history. Commit `docs: audit after the refactor`. Builder Sonnet; reviewer Sol.

### Task 9: Remove what the refactor left behind

**Files:** none edited; git branches, worktrees and untracked files only. **Remove nothing without the
proof step passing for that exact item.**

- [ ] **Step 1: Inventory** (read-only): `git worktree list`, `git branch -vv`, `git branch --merged main`,
  `git status --short` in each worktree, `du -sh` of each.
- [ ] **Step 2: Proof per item.**
  - Branch: removable only if `git branch --merged main` lists it, or `git diff main...<branch> --stat`
    is empty. Candidates: `refactor/phase-0-2`, `refactor/phase-3`, `refactor/phase-4`,
    `refactor/phase-5` (after Task 4), `chore/test-run-safety` (shown as not merged today: check
    its commits against `main` with `git cherry main chore/test-run-safety`; if they appear on `main`
    under other hashes, it is removable, otherwise keep and report).
  - Worktree: removable only if its branch is removable and its status is clean. Candidates:
    `/home/flynn/ggTrader-refactor`, `/home/flynn/ggTrader-safety`. **Keep** `/home/flynn/ggTrader-baseline`
    until the 30-day evidence hold of D8 ends. **Never touch** the `~/.t3/worktrees/*` and
    `.claude/worktrees/*` entries (owned by T3 Code and the agent tool); list them for the owner.
  - Untracked duplicates in `/home/flynn/ggTrader` (`docs/superpowers/plans/2026-11-17-refactor-phase-0-2.md`,
    `2026-11-18-refactor-phase-3.md`, `...phase-4.md`, `2026-11-18-test-run-safety.md`,
    `docs/superpowers/specs/2026-11-17-refactor-design.md`): removable only if `cmp` against the file
    at the same path on `main` (`git show main:<path>`) is identical. The `gglibrary-*.json`,
    `graphify-*` and `library-search-review.json` files are not refactor leftovers: leave them and list them.
  - Worktree scratch from Task 1 (`/home/flynn/ggTrader-p5base`): remove with `git worktree remove`.
  - Backup branches (`backup/*`): keep; list them.
- [ ] **Step 3: Show the owner the list** of proven-removable items and wait for a yes (deleting
  branches and worktrees is not undone by this plan). Then `git worktree remove <path>` and
  `git branch -d <name>` (lowercase `-d`, which refuses an unmerged branch).
- [ ] **Step 4: Evidence folder (D8)**: `cp -a` to `/media/thesix/ggTrader-data/refactor-evidence/`,
  `sha256sum -c` against a manifest written before the copy, and add the path to the memory entry.
  Delete the original only after 30 days and only on the owner's word.

### Task 10: Snapshot, graph, memory and hand-back

**Files:** `research/RESEARCH_SNAPSHOT.md` and the two prompt templates (in the open snapshot task on
`docs/snapshot-2026-11-20`), `graphify-out/` (regenerable, git-ignored), memory entries.

- [ ] **Step 1: Finish the full snapshot regeneration** already started on
  `docs/snapshot-2026-11-20` (it must record the paper-trader changes: halt policy, order recovery,
  invalid-value halt in `RiskGuard`, the new module layout, and corrected sizing citations
  `engine/portfolio.py` instead of `lab/simulate.py`).
- [ ] **Step 2: Refresh the graphify graph** (`/graphify` on the merged tree) so module names match.
- [ ] **Step 3: Memory:** update `project_refactor_state_2026-11-18.md` to "merged to local main
  <date>, pushed <yes/no>", and note D3 to D5 and D7 outcomes.
- [ ] **Step 4: Hand back** with: heads and merge hashes, gate results, the anchors (`_run` 826 to 38
  lines; 239 persist string patches to 0), what was removed in Task 9, what is still owner-gated, usage
  after each part. **Say plainly that nothing was deployed and the trader is still off.**

---

## Self-review

- **Spec coverage:** design section 7 phases 3 to 5 land in Tasks 1 to 4; section 5.4's later lint rules
  are Tasks 5 and 6 (docstrings deferred by D6); phase 6 is Task 7; section 8 decisions 1 to 5 were
  recorded earlier and the new ones are D1 to D8. Deployment and the cron are outside the design's scope
  and excluded here.
- **Gaps stated rather than hidden:** the exact old behavior of malformed weights is read from the base
  in Task 1 Step 1, not assumed; the study-tier delta of the merge is measured in Task 2 Step 2; hit
  counts in Tasks 5 and 6 are approximate and re-measured in their first step.
- **Names:** `StrategySignals`, `gate_run.sh`, `gate_compare.sh`, `offline.sh`, `verify_pins.py` are used
  as they exist in the branch and the evidence folder. Task 7's candidate list comes from design
  section 7 and has not been inspected yet.
