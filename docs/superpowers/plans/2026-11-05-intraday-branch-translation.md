# Intraday Branch Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a new branch `translate/intraday-stage0a` that carries the unmerged intraday research branch (`research/intraday-stage0a`, 33 commits ahead of an older `main`) on top of the current `main`, with its 74 report files moved into the `research/` layout and every reference rewritten, so it can be merged later. Nothing is merged, pushed or moved on the original branch or its in-use worktree.

**Architecture:** Follow `docs/superpowers/plans/2026-11-05-research-reorg-branch-recipe.md` (the recipe; its facts were scanned on 2026-11-05 and are re-checked in Task 1). Work in a new branch and worktree created from the original tip, merge `main` into it (never rebase: the branch cites its own commit hashes as evidence), extend the reorganization tool with the three missing pieces the recipe names (E1 pin three briefs, E2 `--also-map`, E3 branch-mode verify), then move, rewrite, hand-edit and verify.

**Tech Stack:** git worktrees, Python 3 (`scripts/research_reorg.py`), pytest, ruff.

**Spec:** the recipe above plus the owner's go-ahead of 2026-11-05 ("2 yes"). The owner said he does not follow the four blocking choices (D1 to D4 in the recipe), so the recipe's recommended defaults are used and explained in plain words at the end. Everything stays reversible: the branch is unmerged and unpushed.

## Global Constraints

- Never edit, check out in, rebase or move `research/intraday-stage0a` or its worktree (`/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a`, tip `58919c09`). Never touch the other worktrees (`t3-b1d80938`, `t3-ebb52a96`) or `research/momentum-retest` (`d0e30056`; the recipe says do not translate it).
- The live observer runs from the pinned release `280d4a0`; do not restart or modify it, its timers or its release directory.
- Frozen bytes: files the recipe lists as frozen (49) and the three pinned briefs (E1) move byte-for-byte and never go through a rewrite scope. The 78 frozen and 12 pinned files already on `main` stay untouched.
- Rulings (recipe defaults): D1 apply the path-only code edits and record them in the changelog; the recorded validation (`code_sha256` f0b85690...) is never reused. D2 the three hash-quoted briefs are pinned. D3 keep the feasibility brief as it is on `main` (new hash) and annotate the old and new hash in the records; no change to the pinned set on `main`. D4 three new folders: `research/intraday/spy-momentum-prospective-observer` (17 files) and two `artifacts-*` packages beside the study folders.
- Another thread's untracked `research/programme/prompt-kit-review/` is not ours: never touch it; use `verify --allow-new 'programme/prompt-kit-review/*'` where the gate needs it.
- Co-Authored-By trailers; specific paths staged; `scripts/run_tests.sh` for tests; ruff clean; no push, no merge to `main`.
- Models: Sonnet or Haiku implement; Opus reviews every task and does the final review.

## Review Focus

1. A merge-conflict resolution that silently drops branch or `main` content (`docs/next_steps.md`, `docs/changelog.md`, `docs/architecture.md`, `README.md`, `docs/cli_reference.md`).
2. A pinned brief or frozen artifact whose bytes change (the r8 protocol hash must still verify: `intraday_momentum.protocol_sha256()` returns `be395d95...`).
3. Code edits beyond the nine path literals (path-only), or a changed code hash the owner was not told about.
4. Tool extensions that change behaviour for the 174 existing rows.
5. The original branch, its worktree and the observer release changed in any way.

---

### Task 1: Re-scan, isolate and merge `main`

**Files:** none edited by hand except conflict resolutions. Create branch `translate/intraday-stage0a` and a new worktree for it.

**Interfaces:** Produces a clean merge of `main` into a copy of the original tip, in a new worktree at `/home/flynn/.t3/worktrees/ggTrader/translate-intraday-stage0a`, plus a scan report.

- [ ] **Step 1:** Re-run the recipe's read-only scan (section 1 and the conflict table): tips still `58919c09` and `d0e30056`, merge-base with `main`, `git diff --name-status main...research/intraday-stage0a -- docs/research` (expect 74 added, 0 modified), the 9 code literals. If any figure differs, stop and report.
- [ ] **Step 2:** Back up and isolate: `git branch backup/intraday-stage0a-pre-translate 58919c09`; `git worktree add -b translate/intraday-stage0a <new path> 58919c09`. Confirm the original worktree's `git status` and tip unchanged afterwards.
- [ ] **Step 3:** In the new worktree run `git merge main` (never rebase). Resolve conflicts by the recipe table (`docs/next_steps.md`: keep `main`, re-add the intraday item; `docs/changelog.md`, `docs/architecture.md`, `README.md`, `docs/cli_reference.md`: keep both sides' content). Write each resolution and its reason into the task report. Run `scripts/run_tests.sh tests/lab/test_intraday_probe.py tests/lab/test_intraday_momentum.py tests/lab/test_intraday_momentum_data.py tests/lab/test_intraday_observer.py -q` and `ruff check .`.
- [ ] **Step 4:** Commit the merge (the merge commit itself). Opus review: conflicts resolved without loss (compare both parents' hunks), original branch and worktree untouched (tip, status, `git worktree list`), tests green.

### Task 2: Tool extensions E1 to E3 (TDD)

**Files:** Modify `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py` (in the new worktree).

**Interfaces:** Produces: E1 `PINNED_PATHS` gains the three intraday briefs (pinned count becomes 15; confirm the exact three paths from the recipe's sources: the r8 protocol brief, the prospective observation brief, the Opus verification manifest); E2 `rewrite --also-map CSV` (extra mapping rows that are rewritten TO but never made targets); E3 `verify --stage <moved|rewritten|final> --branch-mode` which skips the whole-tree count checks that assume exactly the 174 rows. Existing behaviour without the new flags is unchanged.

- [ ] **Step 1:** Failing tests: `test_pinned_paths_include_the_three_intraday_briefs`, `test_rewrite_also_map_resolves_links_to_already_moved_files`, `test_rewrite_also_map_rows_are_never_targets`, `test_verify_branch_mode_skips_whole_tree_counts`, `test_verify_without_branch_mode_unchanged`.
- [ ] **Step 2:** Implement; all existing tests stay green; ruff clean; `--help` documents the flags.
- [ ] **Step 3:** Commit `feat(scripts): add branch mode to the research tool`. Opus review: no behaviour change for the 174 existing rows (run the existing gates unchanged), the three pinned paths exactly right, tests non-vacuous.

### Task 3: Move, rewrite, hand edits, records

**Files:** the 74 intraday report files under `docs/research/`, the five code/test files with the nine literals, the Markdown hand edits listed in the recipe, `research/MOVED.csv`, `research/README.md`, `research/RESEARCH_INDEX.md`, `docs/changelog.md`, a branch-only manifest and ledger under `docs/superpowers/plans/` (`2026-11-05-intraday-translation-manifest.csv`, `...-ledger.json`), the verification record.

**Interfaces:** Consumes the CSV `docs/superpowers/plans/2026-11-05-research-reorg-branch-additions.csv` (74 intraday rows; momentum rows are ignored) and the tool of Task 2.

- [ ] **Step 1:** Build the branch manifest from the 74 rows (current_path = branch_path, folder = parent of proposed_path, kind = report, result, brief or artifact); plain `manifest-check` passes (it needs no branch mode: docs/research holds exactly the 74 files); build the ledger (sha256 of all 74 before the move).
- [ ] **Step 2:** `move --dry-run` then `move`; commit move-only (pure renames at 100%). Append the 74 pairs to `research/MOVED.csv` (keep its format and order; regenerate with `moved-csv` from the union manifest if simpler).
- [ ] **Step 3:** Rewrite each scope with `--dry-run --report` first, then for real: moved Markdown (with `--also-map` of the reorg manifest), fixed Markdown (`--exclude deploy/intraday_observer`), code. Hand edits exactly as the recipe section 4 lists (nine code literals; `docs/cli_reference.md:116`; two plans; the changelog directory citation; wrap long lines with `ruff format`, introduce a `STUDY_DIR` constant where E501 remains). `deploy/intraday_observer/OPERATOR_RUNBOOK.md` is excluded (the active observer runs the pinned release).
- [ ] **Step 4:** Records: index entries for the three new folders in `research/README.md` and `research/RESEARCH_INDEX.md` (verdict wording elsewhere byte-identical); a changelog entry that states the path-only code edits and that the recorded r8 validation is not reusable (D1); an annotation of the 2026-10-24 brief's old hash `51162768...` and new hash `33c00494...` (D3); the three pinned briefs listed as byte-for-byte (D2); the verification record gets a section for this translation with every gate result.
- [ ] **Step 5 (gates):** `git ls-files docs/research` empty; `verify --branch-mode` stages exit 0 on the branch manifest and ledger (74 rows, the 49 frozen and 3 pinned files matching their ledger), **and** the union gate passes (branch mode only checks the manifest's own rows, so it is blind to the 185 files from `main`): union manifest = `main`'s 174 rows plus the 74 branch rows; union ledger = the `files` lists of `main`'s ledger and the branch ledger concatenated (`json` load, `files = a["files"] + b["files"]`, write to a scratch file); union rewrite manifest = `main`'s `files` entries plus the branch's listed edits (the moved-markdown report's `changed` files); then `scripts/research_reorg.py verify --stage final --manifest <union> --ledger <union ledger> --rewrite-manifest <union rewrite> --require-new README.md --require-new MOVED.csv --require-new audits/root-cause-2026-11-04/README.md --require-new audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md --require-new audits/root-cause-2026-11-04/reference-sheet.md --require-new audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md --require-new audits/root-cause-2026-11-04/2026-11-05-executive-summary.md`, **without** `--branch-mode`, prints `OK stage=final 248 manifest files ... 259 tracked files under research/` with the pinned count derived as 15 (add `--allow-new` for any new file the records step puts under `research/`, and say so in the record); `intraday_momentum.protocol_sha256()` returns `be395d95...`; focused intraday tests, `tests/scripts`, the full suite via `scripts/run_tests.sh`; ruff; `check_lens_report --paths-only` on the moved reports (failures classified as pre-existing or move-caused); the original branch tip is still `58919c09` and the original worktree unchanged; the observer release directory and timers untouched.
- [ ] **Step 6:** Commits: `refactor: move intraday research into research/ (move only)`, `refactor: point intraday references at research/`, `docs: record the intraday translation`. Opus review of each as it lands (pure renames; pinned and frozen bytes; code edits path-only; hand edits; records).

### Task 4: Final review and hand-back

- [ ] **Step 1:** Opus whole-branch review of `translate/intraday-stage0a` against `main` and against the original tip (what the translation changed versus what the branch always carried).
- [ ] **Step 2:** One fix wave and a scoped re-review.
- [ ] **Step 3:** Report to the owner in plain words: what translation means, what was chosen for D1 to D4 and why, how to inspect the result, and the three ways to proceed (merge later, merge never, rework). Leave `translate/intraday-stage0a` unmerged and unpushed; write the memory note.

## Self-review

- Spec coverage: re-scan and isolation (Task 1), tool pieces the recipe names (Task 2), move, rewrite, hand edits, records and gates (Task 3), review and hand-back (Task 4). D1 to D4 are ruled in Global Constraints. Momentum branch is deliberately not translated.
- Consistency: path names (`translate/intraday-stage0a`, the manifest and ledger names, the 74 and 15 counts) match across tasks.
- Proportion: the recipe carries the detailed edits; this plan names the decisions, the files, the tests and the gates.
