# Project Pause and Clean Slate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pause development and research with everything committed, the human-readable documents current and scannable, and a new one-page research summary that rates each strategy category as a range in basis points (bps) against the market.

**Architecture:** Three kinds of work in strict order, one heavy job at a time. (1) Snapshot, then commit the verdict-register worktree's uncommitted work on its own branch in three bisectable commits. (2) Build the summary report from the committed result files. (3) Refresh the readable documents, tidy merged worktrees and branches, write the "where we left it" note. Every builder's output is checked by the other model family (`crossover`): Claude builds -> Codex Sol checks; Codex Terra builds -> Claude Opus checks. The lead (Claude Sonnet) re-verifies every slice itself.

**Tech Stack:** git, Python 3.12 (`/home/flynn/ggTrader/.venv`), `scripts/run_tests.sh`, `scripts/check_reading_edition.py`, Markdown, T3 `delegate_task` (providers `codex`, `claudeAgent`).

**Spec:** the description approved in chat on 2026-11-12 (no separate spec file). Its parts are in "Global Constraints" and "Decisions" below.

**Where steps run:** `FW` = `/home/flynn/.t3/worktrees/ggTrader/feature-verdict-registe-fe5dae64178abc0b` (branch `feature/verdict-register`); `MC` = `/home/flynn/ggTrader` (branch `main`). Each step names its checkout; run commands with `git -C <path>` or `cd` into it, never in the other one.

## Global Constraints

- The live trader stays off; nothing is deployed (no `docker compose pull/up`, no `ggt paper --live`, no cron or systemd edits). The weekly membership timer and the four intraday-observer timers are not touched.
- Nothing is merged to `main` and nothing is pushed, including the local-only commit `783c7267` on `main`. Everything unreviewed stays on `feature/verdict-register`.
- `data/` is never edited, moved or deleted, and no step may delete a directory or worktree that contains ignored files under `data/` or a `.venv`. No `.env`, connection string, credential, account balance or position size is read into a prompt, a document or a commit.
- The section "Owner instructions (2026-10-20)" in `CLAUDE.md` (and `AGENTS.md`, if present) stays byte-identical: `git diff -- CLAUDE.md AGENTS.md` shows no change inside that section before every commit.
- Tests only through `scripts/run_tests.sh` (8 GB cap). Run one heavy job at a time. Never use `--no-verify` (the pre-commit hook at `MC/.githooks` runs ruff). Stop background work only with TaskStop or by a PID whose full command line was read first; never `kill $(ps -o ppid ...)` and never `pkill -f` with text that is also in your own command.
- No history rewriting: a needed fix is a new commit, never `rebase` or `--amend`.
- No retirement, shutdown or account-type recommendations. Taxable account only. Free data only.
- Report ratings are ranges in bps (100 bps = 1 percentage point a year), never pass/fail or GO/NO-GO. Numbers dated before 2026-11-06 carry the label "legacy engine, not re-run".
- Delegate prompts carry no private account records. The plan is the smaller phased version (9 subagents, see D7; usage headroom is unknown); the lead reports and pauses after each phase so the owner can stop.

## Decisions made by the lead (flagged, owner may override)

- D1. The finished state lives on `feature/verdict-register`. `MC` gets one local, unpushed commit: this plan plus a short pointer in `docs/next_steps.md`. The branch rewrites `docs/next_steps.md`, so a later merge of the branch will conflict there; the final note says so.
- D2. The report lives at `research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md` (programme-level document next to `RESEARCH_GUIDE.md`).
- D3. Band method (approximate, and labelled so). The result JSON stores `comparison.candidate.annualized_return` and `comparison.benchmark.annualized_return` as compound annual returns written as fractions (0.1523 means 15.23%), so `candidate_pct = annualized_return * 100`; it also gives `mean_excess_t` (a 12-lag Newey-West t on the mean monthly excess return), not the annualized mean excess. So: `gap_bps = (candidate_pct - benchmark_pct) * 100`; `SE_bps = |gap_bps / mean_excess_t|` (undefined when t is 0, missing or non-numeric: then no band, the row says "no uncertainty measure"); `range = gap_bps +/- 2 * SE_bps`. The band is labelled "about 95% for a single test; approximate; not adjusted for the number of screens tried". The primary variant of every screen is `results.full` (1% annual cost, full window); `full_stress`, `since_1994` and, where present (a top-level key in some files), `spy_overlap` appear as extra columns, chosen before looking at them. A t near zero gives a very wide band by construction (reversal: t -0.65 on a gap of about -204 bps gives roughly +/- 630 bps); the report says so in one line under the table.
- D4. Roll-up per category: the span from the lowest `range_low_bps` to the highest `range_high_bps` among rows with a band, the median of the point estimates, and the count of rows without a band. The +240 bps after-tax tie line is a reference for a short-term book; the rows are before tax and the table says the two are not directly comparable.
- D5. A 2.4 MB PDF (`research/programme/benchmark-objectives/sources/irs-publication-550-current.pdf`, a cited source document) is committed with the research commit and listed in the final note.
- D6. Worktrees and branches are removed only after the candidate list is shown to the owner. By default none that holds ignored files is removed: `/home/flynn/ggTrader-vbt12` (holds the only copy of the frozen price tape and a `.venv`) is never a removal candidate, even with an owner yes, and every `backup/*` branch stays.

- D7. The plan uses 9 subagents, one more than the 6-8 the owner approved, because Sol checks Haiku's CSV (Codex checks Claude's work) in addition to reviewing document group A (a new Codex task each time, never a message to a running one). To stay at 8, drop Task 4 step 2's Sol check; the lead's own recomputation then is the only check of the CSV. That 8-agent option gives up the cross-family check of the CSV (Claude output would then be checked only by Claude models, against the delegation rule), so the owner should choose knowingly; the alternative that keeps every pair cross-family is to merge the two Opus document reviews (the report in Task 4 and group B in Task 5) into one task. The owner may choose.

## Review Focus

1. A range reads as stronger than the data (broad-market proxy not SPY; screens overlap; pre-tax; approximate band). Task 4 step 3 requires each row to name its benchmark, basis and variant, and step 4's reviewer checks wording.
2. A commit that only passes together with the others (`utils/json_safe.py` is imported by `utils/io.py`, `utils/manifest.py`, `utils/results.py`, `lab/wfo/run.py`, `paper/trader.py`, `paper/notifier.py`). Task 2 step 4 tests each commit in a detached scratch worktree.
3. The engine commit changes reference results (`engine/portfolio.py` +185 lines). Task 3 step 1 compares the reference numbers to the recorded ones with an exact command.
4. Two documents quote different counts or dates (register counts 261/264/269 already disagree). Task 5 step 4 prints every number that has no source.
5. The other session edits the worktree while it is committed, so counts drift. Task 1 treats drift as a stop condition.
6. A secret or a large file is committed by accident. Task 1 step 4 scans both new files and the diff of modified files; Task 2 step 5 lists committed paths.

---

### Task 1: Safety snapshot, quiet check, scans (lead, in `FW`)

**Files:** none in the repo. Writes only to the scratchpad directory.
**Interfaces:** Produces: `SNAP` = `<scratchpad>/pause-snapshot-2026-11-12/` holding `worktree-diff.patch`, `untracked.tar`, `SHA256SUMS`, `state.txt`, `exec_summary_corrected.md`.

- [ ] **Step 1: Quiet check.** `find . -path ./.git -prune -o -type f -mmin -10 -print | head`. Expected: no output. If files changed in the last 10 minutes, stop and ask the owner whether the other session is still working.
- [ ] **Step 2: Record state.** Write to `state.txt`: `git status -sb`, `git rev-parse HEAD`, `git diff --name-only | wc -l`, `git ls-files --others --exclude-standard | wc -l`. Expected today: HEAD `73d4a4f4`, 54 modified, 188 untracked. The counts recorded here are the reference for later steps; if they differ from these numbers, state the difference to the owner before continuing.
- [ ] **Step 3: Back up.** `git diff > worktree-diff.patch`; `git ls-files --others --exclude-standard -z | xargs -0 tar -cf untracked.tar`; `sha256sum worktree-diff.patch untracked.tar > SHA256SUMS`. Expected: `sha256sum -c SHA256SUMS` prints OK twice and `tar -tf untracked.tar | wc -l` equals the untracked count in `state.txt`.
- [ ] **Step 4: Secret and size scan.** On the untracked list: `grep -Ei '(^|/)\.env|secret|credential|\.pem$|\.key$'`; on the contents of untracked and of `git diff`: `grep -E 'postgres(ql)?://[^ ]*:[^ ]*@|api[_-]?key *[:=] *[A-Za-z0-9]{16,}'`; and list untracked files over 1 MB. Expected: no secret hit; exactly one large file (the IRS PDF, D5). Any hit is excluded from the commits and reported.
- [ ] **Step 5: Save the corrected executive summary.** The lead writes the corrected executive summary of 2026-11-12 (the chat version, not the earlier draft) to `SNAP/exec_summary_corrected.md`. Task 5 reads it from there.

### Task 2: Three bisectable commits on `feature/verdict-register` (lead, in `FW`)

**Files (commit groups, by path; together they cover every modified and untracked path with no overlap, checked in step 1):**
- C-engine: `src/ggTrader/engine/`, `src/ggTrader/lab/`, `src/ggTrader/utils/` (including the new `json_safe.py`), `tests/engine/`, `tests/lab/`, `tests/utils/`.
- C-paper: `src/ggTrader/paper/`, `src/ggTrader/cli/main.py`, `tests/paper/`.
- C-research: everything else (`docs/`, `research/` modified and untracked, `tests/scripts/test_research_layout.py`).
**Interfaces:** Consumes: `SNAP`. Produces: three commits whose subjects start `engine:`, `paper:`, `research:`; tip hash `TIP2`.

- [ ] **Step 1: Prove the groups partition the changes.** Build the three path lists with `git status --porcelain`; expected: every path appears in exactly one list and the union equals the file count in `state.txt`.
- [ ] **Step 2: Commit C-engine.** `ruff check` and `ruff format --check` on its `.py` files with `MC/.venv/bin/ruff`; `git add` its paths; `git diff --cached --stat`; `scripts/run_tests.sh --fast -q tests/engine tests/lab tests/utils`. Expected: all pass. Commit `engine: intraday stop and time-stop inputs, safe JSON hashing` with the Claude attribution trailer.
- [ ] **Step 3: Commit C-paper, then C-research.** Same procedure. C-paper: tests `tests/paper`, subject `paper: durable order intents, halt-policy proposal (undeployed), clearer output`, body "Proposed, undeployed; needs owner approval and Sol + Opus review before any merge". C-research: test `tests/scripts/test_research_layout.py`, subject `research: verdict register, eight stock screens, input pilot, readiness notes`.
- [ ] **Step 4: Prove each commit stands alone.** For `HEAD~2`, `HEAD~1` and `HEAD`, one at a time: `git worktree add --detach <scratchpad>/bisect-N <hash>`, then `GGT_PYTHON=/home/flynn/ggTrader/.venv/bin/python scripts/run_tests.sh --fast -q` inside it, then `git worktree remove --force <scratchpad>/bisect-N`. `--force` is used for these three scratch paths only and nowhere else in this plan. Finish with `git worktree prune` and `git worktree list`: expected, the same list as before the step. Expected: all three pass. A failure means the grouping is wrong; report it and stop.
- [ ] **Step 5: Prove nothing is left and nothing leaked.** `git status --short` prints nothing; `git diff --name-only HEAD~3 | grep -E '^data/|(^|/)\.env'` prints nothing; the owner-instructions section is unchanged (`git diff HEAD~3 -- CLAUDE.md`).
- [ ] **Step 6: Full tier once on the tip.** `scripts/run_tests.sh --full -q` in `FW`. Run it with `GGT_SNAPSHOT=/home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet` exported so the real-tape tests can run, and record the passed/skipped counts and the skip reasons (the runner already passes `-rs`). Expected: 0 failed; every remaining skip has a stated reason. Record the counts for the final note.

### Task 3: Reference check and cross-family review of the engine and paper commits

**Interfaces:** Consumes: `TIP2`. Produces: `<scratchpad>/review-engine-paper.md` and the reference-check result.

- [ ] **Step 1: Reference numbers (lead).** Create a detached scratch worktree of `TIP2` as in Task 2 step 4 (call it `REF`). The drivers read only the frozen tape and the fixture list, no database. The only copy of the tape is in `/home/flynn/ggTrader-vbt12`, so point at it read-only (it is hash-checked against the manifest). From `REF`, under the same memory cap as the test runner (`systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- <command>`):
  `export GGT_SNAPSHOT=/home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet`
  `PYTHONPATH=src /home/flynn/ggTrader/.venv/bin/python scripts/p5_wfo_validation.py --runs core17_ensemble_sp500,core17_ensemble_sp500_dsr0 --out-dir <scratchpad>/ref-vintage1 --force`
  `PYTHONPATH=src /home/flynn/ggTrader/.venv/bin/python scripts/p7_live_golden.py --out <scratchpad>/ref-vintage1/p7_live_golden.json`
  (both default to the frozen fixture `sp500_v1_2026-06-02`; `PYTHONPATH=src` makes the scratch copy of `src` the one imported, not the editable install). Expected: core17 Sharpe 0.3905, CAGR 3.501%, gates 1/17; dsr0 Sharpe 0.6374, CAGR 3.406%, gates 17/17; live replay 120/120. If the tape hash check fails, stop and tell the owner; do not copy or edit anything under `data/`. Any numeric difference is a finding against C-engine, not something to fix here. Remove `REF` afterwards as in Task 2 step 4.
- [ ] **Step 2: Opus review.** The two commits were written by the Codex session (thread `c6d68101-fc81-4573-97da-f51867cc36cf`, runs 334-560). `delegate_task` to `claudeAgent` model `claude-opus-5-5`, role `review`, isolation `shared`: review `git show` of the engine and paper commits for (a) behaviour that changes reference results, (b) the halt-policy proposal and its tests, (c) tests that pass vacuously (it may mutate a copy under `/tmp`, never a file inside the repo). Output ranked findings under 600 words. The prompt ends "Analysis only. Do NOT edit, create, or delete any files inside the repository."
- [ ] **Step 3: Record, do not fix.** Findings go to the final note's "waiting on review" list. No code changes in this pause unless the owner asks.

### Task 4: Research summary report (Haiku extracts, Terra drafts, Sol and Opus check, lead verifies; in `FW`)

**Files:**
- Create: `research/programme/research-summary/_inputs/category_inputs_2026-11-12.csv` (owned by the Haiku agent only)
- Create: `research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md` (owned by the Terra agent only)
- Test: `tests/scripts/test_research_layout.py`; lint `scripts/check_reading_edition.py`.

**Interfaces:** Produces: CSV columns `category,strategy,benchmark,variant,window,candidate_pct,benchmark_pct,gap_bps,mean_excess_t,range_low_bps,range_high_bps,basis,source_file,source_line,engine_label`. The report table reads only from this file. The eight stock screens are the seven `research/stock-signals/feasibility/_*_screen_*.json` files plus `_short_term_reversal_20261111.json`.

- [ ] **Step 1: Haiku extraction.** `delegate_task` to `claudeAgent` `claude-haiku-5-5`; it may write only the CSV path above. Rows: one per strategy, from `research/RESEARCH_SNAPSHOT.md`, `research/VERDICT_REGISTER.md`, the eight screen JSONs (variant `full`), `research/audits/root-cause-2026-11-04/2026-11-05-executive-summary.md`, and the century-leverage, cross-asset, intraday and event-timing reports. Exact numbers with `source_file:line`; for the JSON files use `results.full.comparison.<candidate|benchmark>.annualized_return * 100` for the two `_pct` columns; leave `mean_excess_t` and the range columns empty where the source has no t; never estimate.
- [ ] **Step 2: Lead verification and Sol check.** Lead recomputes `gap_bps` and the range per D3 for every row in a throwaway script outside the repo and spot-checks 15 rows against the raw JSON. Sol (`codex`, `gpt-6.1-sol`, read-only) then checks every CSV figure against its `source_file:line`. Expected: 0 mismatches beyond rounding; fix the CSV, not the report.
- [ ] **Step 3: Terra draft (Codex builds).** `delegate_task` to `codex` `gpt-6.1-terra`, owning only the `.md`: page 1 is a table of categories (what was tried, number of strategies, D4 roll-up, a one-line plain read, the +240 bps after-tax tie line with the "not directly comparable" note); then one short paragraph per category. No pass/fail, GO/NO-GO or "winner" words; every row names benchmark, variant and basis ("after 1% annual cost, before tax" or the row's real basis); legacy-engine numbers carry the label; terms are defined on first use (bps, t-stat, benchmark); the band label from D3 appears under the table.
- [ ] **Step 4: Lint and Opus review.** `scripts/check_reading_edition.py` on the report (expected exit 0), then Opus (`claudeAgent`, `claude-opus-5-5`, read-only) checks each range against the CSV and flags wording that implies a verdict, a missing benchmark or basis, or a claim stronger than the range supports. The lead applies accepted edits.
- [ ] **Step 5: Commit.** `scripts/run_tests.sh --fast -q tests/scripts/test_research_layout.py`, then commit `research: strategy categories as alpha ranges in bps` on `feature/verdict-register`.

### Task 5: Refresh the readable documents (Sonnet and Terra build, Sol and Opus review; in `FW`)

**Files (modify):** `docs/next_steps.md`, `docs/working_task_list.md` (group A: Sonnet); `README.md`, `research/programme/RESEARCH_GUIDE.md`, `docs/roadmap.md` (group B: Terra).
**Interfaces:** Consumes: the branch tip, the Task 4 report path, `SNAP/exec_summary_corrected.md`.

- [ ] **Step 1: Group A (Claude Sonnet builds).** `delegate_task` to `claudeAgent` `claude-sonnet-5-5`, owning group A only: each document gets a "Status at a glance" table (done, waiting on the owner, waiting on review, parked), short sections, plain words, dates and links; the trader is described as stopped with no ask to restart; each open owner decision is listed once. The prompt forbids edits outside its two files.
- [ ] **Step 2: Group B (Codex Terra builds).** `delegate_task` to `codex` `gpt-6.1-terra`, owning group B only: README gets a 10-line "where things stand" block and a link to the new report; RESEARCH_GUIDE gets the category ranges in two sentences and a link; the roadmap's status table is updated from `research/RESEARCH_SNAPSHOT.md` and the register without new claims.
- [ ] **Step 3: Cross-family reviews.** Sol (`codex`, `gpt-6.1-sol`) reviews group A; Opus (`claudeAgent`, `claude-opus-5-5`) reviews group B. Each checks that every number, date and count appears in a named source, nothing contradicts another refreshed document, and the style is scannable. Read-only.
- [ ] **Step 4: Lead cross-check.** A throwaway script lists every number with a unit (`%`, `bps`, `pp`, a count followed by a noun) in the five files and prints those found in no file under `research/` or in `docs/changelog.md`. Expected: each printed number is explained or removed.
- [ ] **Step 5: Lint and commit.** `scripts/check_reading_edition.py` on files it supports; `scripts/run_tests.sh --fast -q tests/scripts`; commit `docs: refresh next steps, task list, README, research guide and roadmap`.

### Task 6: Tidy, pointer on main, final note (lead)

**Files:**
- Create: `docs/where-we-left-it-2026-11-12.md` in `FW` (committed on `feature/verdict-register`).
- Modify: `docs/next_steps.md` in `MC`; add this plan file in `MC`.

- [ ] **Step 1: Inventory and owner check.** `git worktree list`, `git branch -vv`; for each branch `git merge-base --is-ancestor <tip> main`; for each worktree `git -C <path> status --short --ignored | head`. Candidates are only branches already merged into `main` whose worktree has no ignored `data/` or `.venv`. Show the candidate list to the owner (D6) and remove nothing without a yes. Expected: `feature/verdict-register`, `research/momentum-retest`, every `backup/*` branch and `/home/flynn/ggTrader-vbt12` remain.
- [ ] **Step 2: Final note.** One page in plain words and tables: where each piece of work lives (branch, commit), waiting on the owner (Nasdaq inquiry unsent, AVB inquiry unsent, halt-policy proposal, push of `783c7267`), waiting on review (engine, paper, expected-growth results), the test counts and reference-check result, the large PDF, the `next_steps` merge conflict (D1), the untracked `research/programme/benchmark-objectives/` in the main checkout (tracked on the branch, so a later merge stops with "untracked files would be overwritten" until it is moved aside), and the first three things to do on return. Commit it on the branch.
- [ ] **Step 3: Pointer on main.** In `MC`, add this plan and 6 lines to `docs/next_steps.md` (branch name, note path, "nothing merged or pushed"). Commit locally; do not push. Expected: `git status -sb` shows `ahead 2`.
- [ ] **Step 4: Final verification.** In `FW`, `git status --short` prints nothing. In `MC`, `git status --short` prints only paths that were already untracked before this work and are not part of it (today: `docs/superpowers/plans/2026-11-10-verdict-register-and-framework.md`, `docs/superpowers/specs/2026-11-10-research-verdict-framework-design.md`, `research/programme/benchmark-objectives/` if present, `data/external/event_timing_audit/`); the final note lists them for the owner and none is touched. `systemctl --user is-enabled ggtrader-membership-weekly.timer` prints `enabled`; `docker ps` shows the same `ggtrader_live` container; the owner-instructions section is unchanged. Report to the owner with the usage each phase consumed.

---

## Self-review

- **Coverage:** report (Task 4), docs refresh (Task 5), commit everything (Task 2), cross-family subagents (Tasks 3-5), tidy and clean slate (Task 6), safety (Task 1).
- **Agents used (9):** Haiku extraction, Sol CSV check, Terra report draft, Opus report review, Sonnet docs A, Terra docs B, Sol review of A, Opus review of B, Opus review of engine and paper commits.
- **Not done on purpose:** no merge, no push, no deploy, no change to the halt policy, no code fixes from the reviews, no deletion of unmerged branches or of any worktree holding ignored data.
