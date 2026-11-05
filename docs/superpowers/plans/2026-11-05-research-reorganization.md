# Research Reorganization and Review Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every file under `docs/research/` to a new top-level `research/` in the approved layout without breaking evidence, tests, the path checker or the Docker image, and add the README files and one consolidated direction-review report.

**Architecture:** One small tested tool, `scripts/research_reorg.py`, does the mechanical work from a reconciled manifest (check, ledger, move, rewrite, verify), so every step is reviewable and repeatable. Work happens on a new branch in this order: manifest, ledger, consumer prep, move-only commit, scripted reference rewrite plus reviewed manual fixes, verification gates, then the new documents. Frozen evidence moves byte-for-byte and is never rewritten.

**Tech Stack:** Python 3.12 (repo `.venv`), pytest via `scripts/run_tests.sh` only, ruff, git (`git mv`), CSV/JSON, Markdown.

**Spec:** `docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` (approved 2026-11-05, all section 9 defaults accepted, section 14 calls apply). Draft mapping: `docs/superpowers/specs/2026-11-05-research-reorganization-mapping-draft.csv`.

## Global Constraints

- Depth: at most two folder levels below `research/` (category, then one folder per study); all of a study's files directly inside it. The five `artifacts-*` packages keep their names and contents as the study folder.
- Names: every existing basename kept except exactly three briefs that gain `-brief`: `2026-10-23-century-leverage.md`, `2026-09-28-cross-asset-sleeve.md`, `2026-09-30-static-fractional-leverage.md`.
- Byte-frozen set: the twelve hash-pinned paths (spec section 3), every `_*.json`, `_*.csv`, `.txt`, `.py` and `.tgz` file, and the contents of the five `artifacts-*` packages. Rewritten set: every other Markdown file. Each rewrite pass emits a manifest of changed files; byte-identity checks compare everything not in it.
- Never execute a script that has not been read; never run a script only to see `--help`; no database writes; no network; the live paper trader, `ggtrader_live`, the intraday observer timers and its worktree, `data/` and `src/` behavior are untouched.
- Tests only via `scripts/run_tests.sh`; `ruff check .` and `ruff format --check .` must pass; no formatter change to frozen evidence.
- Do not touch `research/intraday-stage0a` or `research/momentum-retest` (recipe only, Task 13). No push, no merge to `main`; commit on the work branch locally.
- No scientific verdict, number or conclusion in any report changes. New content is limited to the READMEs, index entries, `MOVED.csv`, the consolidated report and the two pointer lines.
- Standing owner rules: no retirement or shutdown advice, taxable account only, free data only. Models: Claude only (Sonnet drafts and implements, Opus for judgment and the consolidated report, Haiku for bulk checks, a stronger model checks Haiku).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Uncovered failure modes most likely to bite, each pinned to a test in the owning task:
1. A moved Markdown file with a relative link that leaves the research tree (`../roadmap.md`, `../../superpowers/...`) must resolve from the new parent (Task 4 test `test_relative_link_outside_tree_recomputed_from_new_parent`).
2. A backticked directory or glob citation (`docs/research/briefs/`, `docs/research/_lens*.json`) must be reported as unresolved, never silently mapped (Task 4 `test_directory_and_glob_citations_are_unresolved`).
3. A pinned or machine file rewritten or reformatted by accident must fail the gate (Task 5 `test_verify_fails_when_frozen_file_bytes_change`).
4. Path built from pieces (`Path("docs") / "research"`, f-strings) must be flagged for manual fixing (Task 4 `test_split_path_construction_is_flagged`).
5. The path checker must reject a missing `research/...` path, and the old-to-new lookup must never rescue it (Task 3 `test_missing_research_path_fails_and_moved_csv_is_not_consulted`).

---

### Task 1: Work branch, tool skeleton, reconciled manifest

**Files:**
- Create: `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`
- Create: `docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv`

**Interfaces:**
- Produces: `ManifestRow(current_path: str, proposed_path: str, kind: str, notes: str, frozen: bool, pinned: bool)`; `PINNED_PATHS: tuple[str, ...]` (the twelve, with the `docs/research/` prefix); `load_manifest(path: Path) -> list[ManifestRow]`; `check_manifest(rows: list[ManifestRow], tracked: set[str]) -> list[str]` (problems, empty means pass); `main(argv: list[str] | None = None) -> int` with subcommands added by later tasks. `frozen` is true for pinned rows, every extension in `{.json,.csv,.txt,.py,.tgz}`, and anything inside an `artifacts-*` folder.

- [ ] **Step 1:** `git switch -c reorg/research-top-level` from `main`. Confirm `git status` shows only the pre-existing untracked `data/external/event_timing_audit/`.
- [ ] **Step 2: Write failing tests** in `tests/scripts/test_research_reorg.py`: `test_check_manifest_passes_clean_fixture`; `test_check_manifest_flags_unmapped_tracked_file`; `test_check_manifest_flags_duplicate_destination`; `test_check_manifest_flags_depth_over_two` (a proposed `research/a/b/c/file.md` fails); `test_check_manifest_flags_unlisted_basename_change` (only the three `-brief` renames allowed); `test_frozen_flag_rules` (json/csv/py/tgz, artifacts-*, pinned → frozen; plain `.md` → not).
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/scripts/test_research_reorg.py -q`. Expected: FAIL (module not found).
- [ ] **Step 4:** Implement the interfaces above in `scripts/research_reorg.py` (argparse, `main()`, `if __name__ == "__main__":` guard, module docstring, type hints, constants at top, line length 100).
- [ ] **Step 5:** Run the tests. Expected: all PASS; `.venv/bin/ruff check` and `ruff format --check` clean on both files.
- [ ] **Step 6:** Add subcommand `manifest-check --manifest PATH` (loads the CSV, lists `git ls-files docs/research`, prints problems, exit 1 on any). Run it on the draft mapping. For every problem (files added since the draft, wrong counts), fix by writing `docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv` as the reconciled manifest (same columns as the draft; the draft CSV is not edited). Expected final: `manifest-check` prints `OK 174 rows` (or the reconciled count) and exits 0.
- [ ] **Step 7:** Commit: `feat(scripts): add research reorganization tool and reconciled manifest`.

### Task 2: Pre-move provenance ledger

**Files:**
- Modify: `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`
- Create: `docs/superpowers/plans/2026-11-05-research-reorg-ledger.json`

**Interfaces:**
- Consumes: `ManifestRow`, `load_manifest`.
- Produces: `sha256_file(path: Path) -> str`; `build_ledger(rows: list[ManifestRow], root: Path, git_head: str) -> dict` with keys `source_revision`, `files` (list of `{current_path, proposed_path, sha256, size, frozen, pinned}`), `inbound_references` (count of `docs/research` string occurrences per consuming file outside the research tree, from a read-only text scan), `existing_dead_links` (relative Markdown links in research files that already do not resolve); subcommand `ledger --manifest PATH --out PATH`.

- [ ] **Step 1: Write failing tests:** `test_sha256_matches_hashlib`; `test_ledger_has_entry_per_manifest_row_with_digest`; `test_ledger_records_pinned_flag_for_twelve_paths`; `test_ledger_lists_preexisting_dead_relative_links`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** implement; run; expect PASS.
- [ ] **Step 4:** Run `python scripts/research_reorg.py ledger --manifest docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv --out docs/superpowers/plans/2026-11-05-research-reorg-ledger.json` (the tool is read and tested; read-only). Verify the twelve pinned paths show `pinned: true` and the file count equals the manifest count.
- [ ] **Step 5:** Commit: `feat(scripts): record pre-move provenance ledger for research reorganization`.

### Task 3: Consumer prep (Docker ignore, path checker)

**Files:**
- Modify: `.dockerignore`, `scripts/check_lens_report.py:26`, `tests/scripts/test_check_lens_report.py`

**Interfaces:**
- Consumes: existing `REPO_PATH_RE`, `check_report(path, repo_root, paths_only=False)`.
- Produces: `REPO_PATH_RE` accepting the prefix `research` (alongside `docs|src|scripts|tests|results`).

- [ ] **Step 1: Write failing tests** in `tests/scripts/test_check_lens_report.py`: `test_research_path_that_exists_passes`; `test_missing_research_path_fails_and_moved_csv_is_not_consulted` (create a `research/MOVED.csv` mapping the missing path to something real; assert the report still fails).
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/scripts/test_check_lens_report.py -q`; expect the new tests FAIL.
- [ ] **Step 3:** Add `research` to the prefix group in `REPO_PATH_RE`. Append `research/` to `.dockerignore` (keep the existing entries and the `!pyproject.toml` exception last).
- [ ] **Step 4:** Run tests; expect all PASS. `ruff check`/`format --check` clean.
- [ ] **Step 5:** Commit: `fix: check research/ citations and keep research out of the Docker image`.

### Task 4: Reference rewriter (tested, dry-run only)

**Files:**
- Modify: `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`

**Interfaces:**
- Consumes: manifest rows (old path to new path map).
- Produces: `Unresolved(file: str, line: int, text: str, reason: str)`; `rewrite_markdown(text: str, old_file: str, new_file: str, mapping: dict[str, str]) -> tuple[str, list[Unresolved]]` (rewrites backticked `docs/research/X` where X maps, preserving a `:line` or `:a-b` suffix; rewrites `](relative)` links by resolving against the old file's directory then recomputing relative to the new directory, keeping `#anchors`; leaves bare filenames, URLs and code fences untouched; reports every other `docs/research` mention as unresolved); `rewrite_code(text: str, mapping: dict[str, str]) -> tuple[str, list[Unresolved]]` (exact path replace; flags directory prefixes, globs, f-strings and `Path(...)/"research"` constructions); subcommand `rewrite --manifest PATH --scope moved-markdown|fixed-markdown|code [--dry-run] --report PATH`. Scopes: `moved-markdown` = non-frozen `.md` files under the new `research/` (old location from the manifest); `fixed-markdown` = `.md` files elsewhere (`docs/**`, excluding `docs/research`); `code` = `scripts/`, `src/`, `tests/`, `.claude/`, `AGENTS.md`. Frozen files are never opened for writing. Every scope honors `--exclude PATH` (repeatable) and always excludes this reorganization's own spec, mapping draft, plan, manifest, ledger, rewrite manifest, verification and recipe files, because they describe the old paths on purpose.

- [ ] **Step 1: Write failing tests:** `test_backticked_path_rewritten_and_line_suffix_kept`; `test_relative_link_outside_tree_recomputed_from_new_parent`; `test_relative_link_inside_tree_follows_mapping`; `test_directory_and_glob_citations_are_unresolved`; `test_bare_filename_and_url_untouched`; `test_split_path_construction_is_flagged`; `test_frozen_file_is_never_written`; `test_report_lists_every_changed_file`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** implement; run; expect PASS; ruff clean.
- [ ] **Step 4:** Run each scope with `--dry-run --report` into the scratch directory and read the reports. Record the unresolved counts per scope in the commit message body. (The draft counts to expect, to be recounted: about 26 scripts, 9 src files, 8 tests, 25 docs files, 6 skill files.)
- [ ] **Step 5:** Commit: `feat(scripts): add reference rewriter for research reorganization (dry-run verified)`.

### Task 5: Move-only commit and the first gate

**Files:**
- Modify: `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`; moves all 174 files per the manifest.

**Interfaces:**
- Produces: `apply_moves(rows: list[ManifestRow], root: Path, dry_run: bool) -> list[tuple[str, str]]` (creates destination folders, `git mv`, never overwrites, refuses a destination that exists); `verify_moved(rows, ledger, root) -> list[str]`; subcommands `move --manifest PATH [--dry-run]` and `verify --stage moved --manifest PATH --ledger PATH`.

- [ ] **Step 1: Write failing tests** (temporary git repo fixture via `tmp_path` and `git init`): `test_apply_moves_preserves_history_and_content`; `test_apply_moves_refuses_existing_destination`; `test_verify_fails_when_frozen_file_bytes_change`; `test_verify_passes_after_clean_move`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** implement; run; expect PASS.
- [ ] **Step 4:** `move --dry-run` on the real manifest; read the plan output; then run `move` for real. Apply the three `-brief` renames from the manifest.
- [ ] **Step 5 (gate):** `verify --stage moved` must report: file count equal, every destination sha256 equal to the ledger for all 174 files, no path deeper than two levels below `research/`, nothing left under `docs/research/`. Expected: exit 0.
- [ ] **Step 6:** Commit with only the moves staged: `refactor: move docs/research to top-level research/ (move only)`. Confirm with `git diff -M --stat HEAD~1` that git detects renames.

### Task 6: Rewrite code consumers: src and tests

**Files:**
- Modify: `src/ggTrader/lab/event_timing_audit.py` (lines about 47-51: the three brief paths), `src/ggTrader/lab/wfo.py`, `src/ggTrader/lab/data.py`, `src/ggTrader/lab/strategies/{pairs_stat_arb,leveraged_trend,cross_asset_sleeve,insider_cluster}.py`, `src/ggTrader/paper/{cash_sweep,signal_runner}.py` (comment-only), and the 8 test files that mention `docs/research` (`tests/lab/test_momentum_screen.py`, `test_wfo.py`, `test_data.py`, `test_ensemble.py`, `test_event_timing_audit.py`, `tests/data/test_cached_yfinance_loader.py`, `tests/paper/test_signal_runner.py`, `tests/scripts/test_leveraged_rotation_research.py`).

**Interfaces:**
- Consumes: `rewrite --scope code` and its report.
- Produces: no behavior change; path constants and comments point at `research/...`.

- [ ] **Step 1:** Run `rewrite --scope code --dry-run` restricted to these files and read the unresolved list. Apply with `rewrite` for the exact-path hits; fix every unresolved hit by hand to the exact destination from the manifest. For `event_timing_audit.py`, update the three brief constants only; do not change hashing logic, resume logic or anything else.
- [ ] **Step 2:** For the paper files, change comments only; confirm `git diff` shows no code lines changed in `src/ggTrader/paper/`.
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/lab tests/data tests/paper tests/scripts -q`. Expected: all PASS (event-audit tests that read briefs now read the new paths).
- [ ] **Step 4:** `ruff check .` and `ruff format --check .` clean.
- [ ] **Step 5:** Commit: `refactor: point src and tests at research/ (paths only)`.

### Task 7: Rewrite code consumers: driver scripts

**Files:**
- Modify: the 21 driver scripts listed in the dry-run report, namely `wfo_fold_attribution.py`, `factor_french_screen.py`, `position_sizing_regimes.py`, `run_position_sizing_wfo_scheduled.sh`, `ic_kelly_rebaseline_wfo.py`, `static_leverage_aftertax.py`, `core_spy_blend.py`, `shadow_cross_asset.py`, `audit_reruns_wfo.py`, `wfo_policy_synthetic.py`, `pairs_stat_arb_research.py`, `month_end_treasury_wfo.py`, `lens2_build.py`, `replica_parity_check.py`, `leveraged_trend_research.py`, `position_sizing_wfo.py`, `anchor_fix_reproduction_wfo.py`, `live_indicator_warmup.py`, `net_issuance_screen.py`, `production_replica_audit.py`, `cross_asset_sleeve_wfo.py`, `momentum_french_screen.py`; plus `scripts/guard_snapshot.sh` only if it names a research path.

**Interfaces:**
- Consumes: manifest map; each script's own outputs and inputs.
- Produces: exact destination defaults for each output file and each input brief (outputs now belong to different study folders, so a prefix substitution is wrong); `lens2_build.py` and `wfo_fold_attribution.py` get explicit per-artifact locations in place of their single research-root constants; shell `--out` arguments updated in `run_position_sizing_wfo_scheduled.sh`.

- [ ] **Step 1:** For each script, read it fully first (no execution). From the manifest decide the destination of every file it reads or writes; edit the constants, CLI defaults, help text and docstrings accordingly. Keep behavior otherwise identical.
- [ ] **Step 2:** Add or extend a test per changed script that asserts its default output and input paths exist as manifest destinations (file or parent folder), without running the script's computation (import-only or argument-parser-only; if a script runs on import, do not import it: assert on the source text instead). Name: `test_default_paths_point_into_research_tree`.
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/scripts -q`; expect PASS. `ruff check .` / `ruff format --check .` clean (do not format frozen `.py` artifacts under `research/`).
- [ ] **Step 4:** Reproduction check for Lens 2: run `python scripts/lens2_build.py --out-json <scratch>/lens2.json --md-dir <scratch>/md` after reading it to confirm it writes only to those paths, then compare the `classifications` list (id, class, rules-out size, verdict type) with the committed `research/audits/root-cause-2026-11-04/_lens2_sensitivity.json`. Expected: identical on those fields. Do not regenerate the committed JSON.
- [ ] **Step 5:** Commit: `refactor: point research drivers at per-study research/ locations`.

### Task 8: Rewrite instructions, skills and fixed docs

**Files:**
- Modify: `AGENTS.md` (section 5 "Where docs live", the research-documents list, the changelog rule path references), the 5 tracked skill files `.claude/skills/{daily-trader-check,research-prompts,research-report-review,research-snapshot,trader-performance-review}/SKILL.md` and `.claude/skills/research-report-review/scripts/flatten_results.py`, and the `docs/**` files outside the research tree that mention `docs/research` (about 25, including older `docs/superpowers/plans/*` and `specs/*`, `docs/next_steps.md`, `docs/working_task_list.md`, `docs/roadmap.md`, `docs/changelog.md`; excluding this reorganization's own spec and plan files).

**Interfaces:**
- Consumes: `rewrite --scope fixed-markdown` and `--scope code` (skills, `AGENTS.md`).
- Produces: `AGENTS.md` states that research reports live in the top-level `research/` (study folders, ordered placement test in spec section 5, dated filenames), and that `docs/` holds architecture, CLI, changelog, roadmap, next steps, working task list and `superpowers/` only.

- [ ] **Step 1:** Run both scopes (`--dry-run` first). For `docs/changelog.md`, rewrite only path strings, never entries. Read each unresolved hit and edit by hand.
- [ ] **Step 2:** Rewrite `AGENTS.md` section 5: replace the `docs/research` placement rule and the "Where docs live" paragraph with the new rule; keep every other rule verbatim. In the skills, update owned paths; do not touch generated copies (`*-workspace` folders are ignored).
- [ ] **Step 3:** `.venv/bin/python scripts/check_lens_report.py --paths-only docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` and on this plan; spec/plan placeholder paths may fail by design, so list them and confirm they are only the placeholders in spec section 4 (not a real citation).
- [ ] **Step 4:** Commit: `docs: point instructions, skills and docs at research/`.

### Task 9: Rewrite moved Markdown (mutable set)

**Files:**
- Modify: every non-frozen `.md` under `research/` (about 96 files).
- Create: `docs/superpowers/plans/2026-11-05-research-reorg-rewrite-manifest.json` (the changed-files manifest emitted by the tool).

**Interfaces:**
- Consumes: `rewrite --scope moved-markdown`.
- Produces: the rewrite manifest that defines "explicitly listed reference-only edits".

- [ ] **Step 1:** Run `rewrite --scope moved-markdown --dry-run --report <scratch>/md.json`; read the unresolved list. Expected classes: bare filenames, historical citations in prose, directory citations, citations of the Argon file's nonexistent paths (these must stay as written; do not "fix" invented paths).
- [ ] **Step 2:** Run for real. Hand-edit only unresolved hits that are real citations of existing files; leave the rest and list them in the report as known survivors.
- [ ] **Step 3:** For each moved report run `.venv/bin/python scripts/check_lens_report.py --paths-only <file>`; for the nine lens reports run the full checker. Every failure is either fixed (a rewrite miss) or recorded as a pre-existing defect (compare against the ledger's `existing_dead_links`). Expected: no new failures.
- [ ] **Step 4:** `verify --stage rewritten` (add: compares every file's sha256 to the ledger; any difference must be in the rewrite manifest; frozen and pinned files must match exactly). Expected: exit 0.
- [ ] **Step 5:** Commit: `docs(research): rewrite mutable report references for the new layout`.

### Task 10: Full verification gate

**Files:**
- Create: `docs/superpowers/plans/2026-11-05-research-reorg-verification.md`

- [ ] **Step 1:** Run `verify --stage final` (extend `verify`: file count, no unmapped, no duplicate, depth, pinned digests, nothing under `docs/research`). Expected: exit 0.
- [ ] **Step 2:** `grep -rIn "docs/research" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=results --exclude-dir=data .` and categorize every survivor in the verification document as frozen provenance, historical citation, spec/plan text, or known prior defect. Expected: no uncategorized survivor.
- [ ] **Step 3:** Run `scripts/guard_snapshot.sh` to a scratch file and diff against `research/audits/root-cause-2026-11-04/_investigation_guard_baseline_2026-11-04.txt`. Expected: only the `main:` hash line differs, or nothing.
- [ ] **Step 4:** Docker content check by inspection: `git ls-files research | wc -l` and confirm `.dockerignore` contains `research/` and keeps `!pyproject.toml`; do not build.
- [ ] **Step 5:** Run the full suite `scripts/run_tests.sh -q`; `ruff check .`; `ruff format --check .`. Expected: all green.
- [ ] **Step 6:** Write the verification document with each gate's command and result; commit: `docs: record research reorganization verification`.

### Task 11: New navigation documents

**Files:**
- Create: `research/MOVED.csv`, `research/README.md`, `research/audits/root-cause-2026-11-04/README.md`
- Modify: `research/RESEARCH_INDEX.md`

**Interfaces:**
- Consumes: manifest; spec sections 8.1, 8.2, 8.5.
- Produces: `moved-csv --manifest PATH --out research/MOVED.csv` (columns `old_path,new_path,renamed`).

- [ ] **Step 1: Failing test** `test_moved_csv_covers_every_manifest_row_and_marks_three_renames`; implement `moved-csv`; PASS; generate `research/MOVED.csv`.
- [ ] **Step 2:** Write `research/README.md` (what `research/` is; the nine folders and the question each answers; the ordered placement test; a "current review" link; where `docs/` content lives now). Plain language, under one page.
- [ ] **Step 3:** Write `research/audits/root-cause-2026-11-04/README.md` per spec 8.2: three-step reading route (synthesis, consolidated direction review, independent challenge), then a table grouped as summary, lenses, reviews, evidence, one line each; states Phase 2 re-runs were not performed and carries the Argon unverified warning.
- [ ] **Step 4:** Add one entry per study folder to `research/RESEARCH_INDEX.md` (question, main report, brief, evidence, status, date, limitations, aliases such as A23 or PEAD); do not change any verdict wording.
- [ ] **Step 5:** `check_lens_report.py --paths-only` on each new file; expected exit 0. Fresh Sonnet reviewer reads the three documents as the owner would and reports unclear passages; fix.
- [ ] **Step 6:** Commit: `docs(research): add README, MOVED.csv and index entries`.

### Task 12: Consolidated direction review and pointers

**Files:**
- Create: `research/audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md`
- Modify: top of `.../2026-11-05-senior-quant-direction-review.md` and `.../2026-11-05-argon-synthesis-review.md` (one dated pointer line each; update the Argon banner's verification path)

**Interfaces:**
- Consumes: the Opus review, the Argon review, the Argon verification, spec 8.3.

- [ ] **Step 1 (Opus, one drafting task):** Write the consolidated report, about 1,500 words, with the spec 8.3 content: sources and status, side-by-side table by topic (Opus, Argon, verification disposition, status agree/differ/Argon-only/dropped), consensus, disagreements with settling evidence, corrections, at most three ranked options (each: question, cheapest step, prerequisites, falsifier, stop gate), one de-duplicated owner-decision list with defaults, a "what to ignore" list. Use only what the Opus review states and what the verification marks as surviving; mark anything else "not verified". Cite a source line for each substantive statement. No retirement or shutdown advice; keep the owner's no-price-table-writes decision and the freeze.
- [ ] **Step 2:** Add the two pointer lines (`> Reading copy: see 2026-11-05-direction-reviews-consolidated.md`).
- [ ] **Step 3:** `check_lens_report.py --paths-only` on the report; expected exit 0.
- [ ] **Step 4 (fresh Opus reviewer):** traceability check: every claim maps to a source line; nothing added, softened or dropped silently; Argon-unverified items not presented as verified. Fix findings in one round.
- [ ] **Step 5:** Commit: `docs(research): add consolidated direction review`.

### Task 13: Branch translation recipe (recipe only)

**Files:**
- Create: `docs/superpowers/plans/2026-11-05-research-reorg-branch-recipe.md`, `docs/superpowers/plans/2026-11-05-research-reorg-branch-additions.csv`

- [ ] **Step 1:** Read-only: `git diff --name-status main...research/intraday-stage0a -- docs/research` and the same for `research/momentum-retest`. Do not check out, merge or modify either branch or any worktree.
- [ ] **Step 2:** For each added or modified path, assign a destination by the spec's ordered test (intraday additions go under `research/intraday/spy-momentum-feasibility/` or a new intraday study folder; list any needing a new folder) and mark code edits that cite `docs/research` (for example `intraday_probe.py`, `intraday_momentum.py`, `scripts/intraday_momentum_research.py`) with the exact rewrite needed.
- [ ] **Step 3:** Write the recipe (steps to apply the manifest rewrite to a branch tip: rebase point, `research_reorg.py` invocations, verification gates) and state clearly that it is applied only on the owner's separate go-ahead. Commit: `docs: add branch translation recipe for research reorganization`.

### Task 14: Final review and handoff

- [ ] **Step 1:** Run the whole-branch review (most capable model) using the branch package; triage deferred findings; one fix wave.
- [ ] **Step 2:** `git log --oneline main..reorg/research-top-level` and `git diff -M --stat main...HEAD | tail -3`; confirm no change under `src/` other than path constants and comments in the files named in Task 6.
- [ ] **Step 3:** Report to the owner: what changed, the verification results, the survivors list, and the three open items (merge to `main`, push, branch translation). Merge and push only on the owner's word; pushing `main` also publishes a Docker image.

---

## Self-Review

- **Spec coverage:** layout and ordered test (Tasks 1, 5, 11); frozen set and byte identity (Tasks 1, 2, 5, 9, 10); three renames (Tasks 1, 5); consumers: Docker (3), checker (3), src and tests (6), scripts (7), skills, `AGENTS.md` and docs (8); rewrite scopes and manifest (4, 9); MOVED.csv, READMEs and index (11); consolidated report and pointers (12); branch recipe (13); verification and sequence (10, 14). Decision 6 (no stub) is satisfied because nothing is left at `docs/research/` (Task 5 gate).
- **Type consistency:** `ManifestRow`, `load_manifest`, `check_manifest`, `build_ledger`, `apply_moves`, `rewrite_markdown`, `rewrite_code`, `verify_moved` are defined once (Tasks 1 to 5) and only consumed later; subcommand names `manifest-check`, `ledger`, `move`, `rewrite`, `verify`, `moved-csv` match across tasks.
- **Proportion:** the plan states decisions, signatures, test names and gates; the tool bodies and the per-script path decisions are left to the implementer with exact acceptance checks.
