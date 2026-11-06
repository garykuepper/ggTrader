# Research Readability, Sub-folders and Clean-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the root-cause reports readable by an engineer who is not a quant (plain words, defined terms, descriptive names instead of lens/phase numbers, Cyrus voice), organize the root-cause folder into sub-folders, finish the optional clean-up items from the reorganization, and merge to local `main`.

**Architecture:** Three phases after a clean-state gate. Phase B closes the open clean-up items (small, independent). Phase C moves the 33 root-cause files into four sub-folders with a new tested retarget tool (byte-identical `git mv`, then reference rewrite, then gates). Phase D writes the reading editions on top of the final layout: a reference sheet, a lint tool that enforces the owner's writing rules, a plain-English findings report, a revised consolidated direction review, and a rewritten folder README. Originals stay untouched as sources of record.

**Tech Stack:** Python 3 (`scripts/research_reorg.py` style: argparse, dataclasses, pytest, ruff), git, Markdown.

**Spec:** none separate. The owner's message of 2026-11-05 is the spec, restated under "Requirements" below. Rulings made without a spec are provisional until the owner confirms this plan. Builds on `docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` (layout, ordered placement test) and `docs/superpowers/plans/2026-11-05-research-reorg-verification.md` (hash drift register).

## Requirements (owner, 2026-11-05)

1. Abbreviations get plain definitions (example: DSR, "deflated Sharpe ratio", is opaque).
2. Do not refer to lenses, phases, parts or gates by number. Each gets a short descriptive name and a reference sheet like the abbreviation list.
3. Write the reading reports in Cyrus voice (fun, easier to follow). The reader is an engineer, not a quant.
4. Organize `research/audits/root-cause-2026-11-04/` into sub-folders (by lens, by general material).
5. The owner will read only: the final findings of the root-cause investigation, and the recommendations of the two frontier-model reviews (Opus and Argon). Everything else is reference.
6. Also: do the still-optional clean-up items, then merge. First make sure the repo is in a clean state.
7. Subagent development: Sonnet or Haiku implement, Opus reviews every task and does one final review of the whole branch.

## Global Constraints

- Frozen bytes: the 78 frozen files (12 hash-pinned, every `_*.json`/`_*.csv`/`.txt`/`.py`/`.tgz`, the `artifacts-*` packages) are never edited, only moved with `git mv`. Never Edit/Write a file under `research/` that the manifest marks frozen.
- Originals are sources of record and are NOT rewritten: the synthesis, challenge, Phase 1 checkpoint, all lens reports, the Opus review, the Argon review (carries an UNVERIFIED banner) and its verification. Only path/link tokens inside them change in Task 8 (listed edits).
- Never delete `data/`; never touch the untracked `data/external/event_timing_audit/`; never touch the other worktrees (`research-intraday-stage0a`, `t3-*`).
- Production paper trader stays frozen. No changes under `src/ggTrader/paper/`.
- No retirement/shutdown recommendations. Taxable account only, no IRA advice. Free data only. Owner decisions kept: no price-table (`ohlcv`) writes, Phase 2 re-runs = none.
- Cyrus-voice documents: no em dash (U+2014) and no semicolon outside code spans and code fences. Facts first in each section, at most one or two dry lines per section, none inside tables or citations. Numbers real and traceable. A NO-GO is stated as plainly as in the source.
- Plain-English rule: every technical term is defined in plain words at first use and in the reference sheet. No lens, phase, part or gate numbers in prose (the reference sheet maps names to file names once).
- Commits: `.venv/bin/ruff check .` and `ruff format --check .` clean; run tests only via `scripts/run_tests.sh`; stage specific paths only; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (or Haiku/Opus for the model that wrote it).
- Push only on the owner's word (a push to `main` builds a Docker image). Merge locally at the end (fast-forward).
- Models: Sonnet or Haiku implement. Opus reviews every task (spec and quality) and does the final whole-branch review. Haiku output is spot-checked by the Opus reviewer before reliance.
- Quota: unknown to the agent (owner said Claude Max, plenty, on 2026-11-04). Run phase by phase: after each phase report usage and pause point; the owner may stop early.

## Decisions to confirm at plan review (defaults will be used if not changed)

| # | Decision | Default |
|---|---|---|
| D1 | Originals untouched; new reading editions written beside them | yes |
| D2 | Sub-folders of the root-cause folder | `findings/`, `lenses/`, `reviews/`, `records/` (see Task 6) |
| D3 | One level deeper than the spec's depth rule, for this folder only | allowed by a spec addendum |
| D4 | Descriptive names | table in Task 9 |
| D5 | Hash drift: restore bytes of `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md`; annotate (do not restore) `scripts/production_replica_audit.py` | as stated |
| D6 | Stray research-like docs in `docs/` go to `docs/archive/` | yes |
| D7 | Intraday branch translation | NOT in this plan (needs its own go-ahead and tool extensions) |
| D8 | The reading set is exactly three documents plus the sheet | findings, consolidated direction review, folder README, reference sheet |

## File Structure

New or changed (final paths after Task 8):

- `research/audits/root-cause-2026-11-04/README.md` (rewritten, stays at folder root)
- `research/audits/root-cause-2026-11-04/reference-sheet.md` (new): abbreviations, check names, phase names, other labels
- `research/audits/root-cause-2026-11-04/findings/` : synthesis, challenge, Phase 1 checkpoint, Phase 0 trust audit, plus new `2026-11-05-findings-plain-english.md`
- `research/audits/root-cause-2026-11-04/lenses/` : 10 lens reports (`2026-11-04-lens*.md`) and 10 `_lens*.json`
- `research/audits/root-cause-2026-11-04/reviews/` : Opus review, Argon review, verification, consolidated direction review (revised in place)
- `research/audits/root-cause-2026-11-04/records/` : investigation inventory (md, json), ledger, synthesis inventory map, guard baseline
- `scripts/research_retarget.py` + `tests/scripts/test_research_retarget.py` (new tool for the second move)
- `scripts/check_reading_edition.py` + `tests/scripts/test_check_reading_edition.py` (new lint)
- Modified: `scripts/research_reorg.py` (depth exception), `scripts/lens2_build.py`, `tests/scripts/test_driver_default_paths.py`, `ruff.toml`, `docs/changelog.md`, the reorg manifest/rewrite manifest/`MOVED.csv`/verification doc.

## Review Focus

1. A rewritten relative link resolves, but to the wrong file (two files named `README.md`; a link from a moved file to an unmoved one that is now one level deeper). Test in Task 7.
2. A frozen file edited, or a moved file whose bytes changed. Gate in Tasks 7 and 8.
3. Links alive before the sub-folder move and dead after, especially in frozen Markdown (cannot be fixed, must be listed). Delta check in Task 8.
4. A reading edition quotes a number or verdict that drifts from the source, or turns a NO-GO or a caveat into a joke. Lint in Task 11, fidelity review in Tasks 12 and 13.
5. Banned punctuation or lens/phase numbers re-enter through quoted source text. Lint rules in Task 11.

---

## Phase A: Clean state

### Task 1: Clean-state gate and branch

**Files:** none modified. Create branch `research/readability-and-subfolders` from `main`. Create the SDD workspace for this plan.

**Interfaces:** Produces a recorded clean-state result in the new ledger and a branch for all later tasks.

- [ ] **Step 1:** Record and check (read-only): `git branch --show-current` is `main`; `git status --short` shows only `?? data/external/event_timing_audit/`; `git stash list` empty; `git worktree list` shows exactly four worktrees (this one plus the three belonging to other branches, which are left alone); `git log origin/main..main --oneline | wc -l` (expected 33, unpushed, noted not pushed); `git status --short --ignored research | head` shows no stray ignored files; `find research -name __pycache__ -o -name '*.pyc' | head` is empty (delete only if found under `research/` and untracked).
- [ ] **Step 2:** Run `scripts/run_tests.sh -q` (expect 1633 passed, 30 deselected), `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, and `verify --stage final` with the four `--require-new` arguments (expect exit 0, 178 tracked files). Any failure stops the plan and is reported.
- [ ] **Step 3:** `git switch -c research/readability-and-subfolders`. Run this skill's `scripts/sdd-workspace` on this plan file and write the ledger header `# SDD ledger — plan: docs/superpowers/plans/2026-11-05-research-readability-and-subfolders.md` plus a "Clean state" line with the results. No commit.

## Phase B: Clean-up items

### Task 2: Hash drift: restore one file, annotate one

**Files:** Modify `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` (restore bytes), `docs/superpowers/plans/2026-11-05-research-reorg-verification.md` (register status), `scripts/research_reorg.py` and its tests only if the restored file would otherwise be rewritten again (it is not in any rewrite scope today: confirm).

**Interfaces:** Produces a restored file whose sha256 equals the digest recorded in `research/audits/event-timing/_event_evidence_resolution_workbook.json` (`spec_hashes`, line ~8636).

- [ ] **Step 1:** Failing test `test_restored_plan_matches_recorded_hash` in `tests/scripts/test_driver_default_paths.py` (or a new `tests/scripts/test_recorded_hashes.py`): reads the workbook JSON, finds the recorded sha256 for that plan path, asserts `sha256(file)` equals it. Run: FAIL.
- [ ] **Step 2:** `git show 42990c74:docs/superpowers/plans/2026-10-22-event-evidence-resolution.md > <same path>`; test PASSES. The restored file mentions `docs/research` again: it joins the "spec/plan text" survivors.
- [ ] **Step 3:** Update the verification doc's Hash drift register: row 1 "restored (D5)", row 2 (`scripts/production_replica_audit.py`) "annotated, not restored: restoring would point three constants back at `docs/research`; old bytes recoverable from `42990c74`". Recompute the survivor counts and `verify --stage final` exit 0.
- [ ] **Step 4:** Commit `docs: restore recorded-hash plan file and annotate script drift`.

### Task 3: Exclude frozen `.py` artifacts from ruff

**Files:** Modify `ruff.toml`; Test `tests/scripts/test_ruff_excludes_research.py` (new).

- [ ] **Step 1:** Failing test `test_ruff_force_excludes_research_python_files`: for each tracked `research/**/*.py`, run `.venv/bin/ruff check --force-exclude <file>` and `ruff format --check --force-exclude <file>` via `subprocess` and assert exit 0 with no diagnostics when the file is excluded; plus assert `ruff.toml` contains `force-exclude = true` and an `extend-exclude` entry covering `research`. FAIL first.
- [ ] **Step 2:** Add `extend-exclude = ["research"]` and `force-exclude = true` to `ruff.toml` (keep `line-length = 100` and rules `E`/`F`/`W`/`I`). Verify `.claude/hooks/ruff-fix.sh` and `.githooks/pre-commit` still format ordinary files (read both scripts first; do not edit them unless they bypass `--force-exclude`).
- [ ] **Step 3:** `.venv/bin/ruff check .` and `ruff format --check .` pass; the frozen `.py` sha256 values equal the ledger; commit `chore: keep ruff away from frozen research artifacts`.

### Task 4: Move the two stray research documents to the archive

**Files:** `git mv docs/quant_strategy_research_report.md docs/archive/` and `git mv docs/2026-07-06_strategy_recommendations.md docs/archive/`; modify references in `docs/archive/next-steps-history-2026-11-03.md`, `docs/superpowers/plans/2026-07-06-overnight-gap-reversion-strategy.md`, `docs/superpowers/plans/2026-07-07-weight-based-wfo-and-idio-vol.md` (path tokens only).

- [ ] **Step 1:** Confirm both are dated, point-in-time documents not referenced as live guidance by `docs/next_steps.md`, `docs/roadmap.md`, `AGENTS.md`, README or any skill (`git grep -n <name>`); if one is live guidance, leave it in place and report.
- [ ] **Step 2:** Move, update each reference by path token, check relative links in the moved files still resolve (they sit one level deeper than before: fix `../` prefixes by hand), run `scripts/run_tests.sh tests/scripts -q`, commit `docs: archive two point-in-time strategy documents`.

### Task 5: Fix forward-looking output paths in two older plans

**Files:** `docs/superpowers/plans/2026-10-20-momentum-retest.md` (lines ~500, 501, 545), `docs/superpowers/plans/2026-11-04-root-cause-investigation.md` (line ~247).

- [ ] **Step 1:** Replace the `docs/research/...` output paths with `research/stock-signals/momentum/...` (momentum phase 2 outputs) and `research/audits/root-cause-2026-11-04/` (Phase 2 re-run outputs). Path tokens only; no other text changes.
- [ ] **Step 2:** `git grep -n "docs/research" -- <the two files>`: only unmapped historical placeholders remain; commit `docs: point two older plans at the research tree`.

## Phase C: Sub-folders

### Task 6: Spec addendum and depth exception in the gen-1 tool

**Files:** Modify `docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` (append an addendum), `scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`, `AGENTS.md` section 5 (one sentence), `research/README.md` (one sentence).

**Interfaces:** Produces `verify --stage final --allow-deeper PREFIX` (repeatable; PREFIX relative to `research/`; default `audits/root-cause-2026-11-04`). `DEFAULT_ALLOW_NEW` becomes: `README.md`, `MOVED.csv`, `audits/root-cause-2026-11-04/README.md`, `audits/root-cause-2026-11-04/reference-sheet.md`, `audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md`, `audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md`.

- [ ] **Step 1:** Failing tests `test_verify_final_allows_one_extra_level_under_allow_deeper_prefix` (a path `audits/root-cause-2026-11-04/lenses/x.md` passes with the prefix, fails without) and `test_verify_final_still_rejects_two_extra_levels` and `test_verify_final_default_allow_new_lists_reading_documents`.
- [ ] **Step 2:** Implement in `_verify_tree`/`verify_final`: depth limit is 4 components, or 5 when the path starts with an allowed-deeper prefix. Tests pass; all 86 existing tests pass.
- [ ] **Step 3:** Spec addendum "Sub-folders for a large study folder": a study folder with more than about 30 files may group its files one level deeper into sub-folders named for what they are (not for a number); only `audits/root-cause-2026-11-04/` uses it today. AGENTS.md section 5 and `research/README.md` get one sentence each.
- [ ] **Step 4:** Commit `feat(scripts): allow one extra level for large study folders`.

### Task 7: Retarget tool for the second move (TDD)

**Files:** Create `scripts/research_retarget.py`, `tests/scripts/test_research_retarget.py`.

**Interfaces:**
- `@dataclass(frozen=True) class MoveRow: old: str; new: str` and `load_moves(path: Path) -> list[MoveRow]` (CSV header `old_path,new_path`, repo-relative POSIX).
- `check_moves(rows: list[MoveRow], tracked: set[str], allow_deeper: tuple[str, ...]) -> list[str]`: problems when an old path is not tracked, a new path exists or is duplicated, the basename changed, or the depth rule is broken.
- `apply_moves(rows, root: Path, dry_run: bool) -> list[tuple[str, str]]` using `git mv` with a preflight; refuses on any problem.
- `rewrite_text(text: str, file_old: str, file_new: str, mapping: dict[str, str]) -> tuple[str, list[Unresolved]]`: rewrites (a) exact repo-path tokens (backticked or plain, with optional `:N` or `:N-M` suffix) that equal a mapping key; (b) relative Markdown link targets: resolve against `file_old`'s directory, map, re-express relative to `file_new`'s directory, preserve `#anchor` and `:line` suffixes; links to unmoved files from a moved file are recomputed too; reports unresolved items instead of guessing.
- `run_rewrite(root, rows, dry_run: bool, report: Path | None) -> dict`: scope = every tracked text file except frozen paths (use `research_reorg.load_manifest` and `is_frozen`/`_frozen_paths` on the current manifest), the reorg's own files (`_is_reorg_own_file`) and this plan, its ledger and its moves CSV; writes atomically; emits a rewrite-manifest JSON (`{"files": [...], "hand_edited": [...]}`) that lists every changed file by its final path.
- `verify_moved_tree(rows, before: dict[str, str], rewrite_files: set[str], root: Path) -> list[str]`: every moved file's sha256 equals `before[old]` unless it is a non-frozen `.md` listed in `rewrite_files`; every frozen file unchanged; no other file differs from HEAD unless listed.
- CLI: `check`, `move [--dry-run]`, `rewrite [--dry-run] [--report PATH]`, `snapshot --out PATH` (records `before` sha256 values), `verify --before PATH --rewrite-manifest PATH`, `update-manifest --manifest PATH` (rewrites `proposed_path` and `folder` of moved rows in the gen-1 manifest CSV so the gen-1 `verify --stage final` and `moved-csv` stay the single source of truth; also updates the gen-1 rewrite manifest `files` entries for moved paths).

- [ ] **Step 1:** Decision procedure: write the failing test `test_rewrite_text_reuses_or_matches_gen1_link_logic` using a small mapping under `research/`; try `research_reorg.rewrite_markdown`; import it if it passes, otherwise implement locally. Do not change gen-1 behavior (its 86 tests stay green).
- [ ] **Step 2:** Failing tests (each asserts exact output): `test_token_with_line_suffix_is_rewritten`, `test_relative_link_from_moved_file_to_unmoved_file_gains_one_level`, `test_relative_link_between_two_moved_files_in_different_subfolders`, `test_two_readme_files_do_not_collide` (link to `README.md` in the same folder vs the research root), `test_anchor_preserved`, `test_fenced_code_mentions_are_reported_not_rewritten`, `test_frozen_paths_never_in_rewrite_scope`, `test_check_moves_rejects_basename_change`, `test_check_moves_rejects_depth_over_limit`, `test_apply_moves_dry_run_changes_nothing`, `test_verify_moved_tree_flags_changed_moved_file`, `test_verify_moved_tree_flags_unlisted_edit`, `test_update_manifest_rewrites_proposed_paths`. Fixtures: temp git repos.
- [ ] **Step 3:** Implement to green. `.venv/bin/ruff check .` / `ruff format --check .` clean. Every subcommand has `--help`.
- [ ] **Step 4:** Commit `feat(scripts): add research_retarget for sub-folder moves`.

### Task 8: Apply the move, rewrite references, update the manifest

**Files:** Create `docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv`; move 33 files inside `research/audits/root-cause-2026-11-04/`; modify references everywhere the old paths appear (`scripts/lens2_build.py` `ARTIFACT_LOCATIONS`, `tests/scripts/test_driver_default_paths.py`, `docs/**`, `research/**.md` mutable, `AGENTS.md` if cited); regenerate `research/MOVED.csv`; update the gen-1 manifest and rewrite manifest.

**Interfaces:** Moves CSV rule (basenames unchanged):
`findings/` = `2026-11-04-root-cause-investigation.md`, `-root-cause-challenge.md`, `-phase1-checkpoint.md`, `-phase0-trust-audit.md`; `lenses/` = every `2026-11-04-lens*.md` (10) and `_lens*.json` (10); `reviews/` = `2026-11-05-senior-quant-direction-review.md`, `-argon-synthesis-review.md`, `-argon-review-verification.md`, `-direction-reviews-consolidated.md`; `records/` = `2026-11-04-investigation-inventory.md`, `-investigation-ledger.md`, `_investigation_inventory.json`, `_synthesis_inventory_map.json`, `_investigation_guard_baseline_2026-11-04.txt`; `README.md` stays.

- [ ] **Step 1:** Generate the moves CSV from that rule; `research_retarget.py check` exit 0; `snapshot --out <scratch>/before.json`.
- [ ] **Step 2:** `move`, commit alone (`refactor: group root-cause files into sub-folders (move only)`) so git records pure renames.
- [ ] **Step 3:** `rewrite --dry-run --report`, read every unresolved item, then `rewrite`; hand-fix unresolved real citations in mutable files only; update `ARTIFACT_LOCATIONS` and any test expecting old paths; `update-manifest`; regenerate `MOVED.csv` with `research_reorg.py moved-csv`; commit `refactor: point references at the root-cause sub-folders`.
- [ ] **Step 4 (gates, all must pass before the commit above):** `research_retarget.py verify` exit 0; `research_reorg.py verify --stage final` with the updated manifest and `--require-new` for the four paths that exist at this point (`README.md`, `MOVED.csv`, `audits/root-cause-2026-11-04/README.md`, `audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md`; the sheet and the findings are required in Task 14) and exit 0; `scripts/run_tests.sh -q` green; ruff clean; `check_lens_report.py --paths-only` over `research/**/*.md` fails on exactly the previously recorded 12 files (compare the failing set) and the full checker passes on every lens report; dead-link delta: write a link checker over `README.md`, `AGENTS.md`, `docs/**`, `research/**/*.md` and list links that resolved at the previous commit and do not now. Expected: none in mutable files; frozen files listed by name (Review Focus 3).
- [ ] **Step 5:** Reproduction check for the Lens 2 table into scratch only (as in the reorg): `python scripts/lens2_build.py --out-json <scratch>/l2.json --md-dir <scratch>/md`; classifications identical to `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` (114/114); never write the committed JSON.

## Phase D: Reading editions

### Task 9: Inventory of terms and names (Haiku)

**Files:** Create `.superpowers/sdd/<workspace>/term-inventory.json` (scratch, not committed).

**Interfaces:** Produces a JSON list `{term, kind: abbreviation|check|phase|gate|part|other, first_source_file, first_source_line, source_definition}` covering every abbreviation and every numbered label used in the reading sources: synthesis, challenge, Phase 1 checkpoint, the nine check reports, both model reviews, verification, current consolidated review.

Proposed descriptive names (the sheet and every reading document use these, never the numbers): Lens 1 = **Backtest Cleanliness check**, Lens 2 = **Verdict Trust check**, Lens 3 = **Paper-versus-Backtest check**, Lens 4 = **Ceiling check** (can the target be reached at all), Lens 5 = **Coverage check** (what was never tested), Lens 6 = **Code Correctness check**, Lens 7 = **Report Accuracy check**, Lens 8 = **Crypto check**, Lens 9 = **Test Power check** (could the tests even see an effect). Phase 0 = **Early Trust Audit**, Phase 1 = **Main Evidence Round**, Phase 2 = **Optional Re-run List** (29 items, not run). Part A / B0 and Gate A / Gate B get names derived from the sources by Task 10 and approved by the Opus reviewer.

- [ ] **Step 1:** Scan with `grep`/Python (read-only); every row needs the source file and line of its first use and the source's own definition or "none given".
- [ ] **Step 2:** Self-check the inventory is complete: count distinct all-caps tokens of 2+ letters in the reading sources and confirm each appears in the JSON or is on an explicit ignore list (file extensions, units, month names). Report the numbers.
- [ ] **Step 3:** No commit. The Opus reviewer samples at least 25 rows against the sources and checks for omissions (Haiku output is not relied on unchecked).

### Task 10: Reference sheet (Sonnet)

**Files:** Create `research/audits/root-cause-2026-11-04/reference-sheet.md`.

**Interfaces:** Consumes the term inventory. Produces the vocabulary every later document must use.

- [ ] **Step 1:** Write the sheet in Cyrus voice (read `~/.hermes/SOUL.md` and the cyrus-voice skill first; the persona file is canonical). Sections: "How to read these reports" (3 lines), "Abbreviations" (term, plain meaning in at most 30 words that uses no other undefined abbreviation, and one line on why it matters here; example target for DSR: what a Sharpe ratio is, then how DSR marks it down for the number of ideas tried), "The nine checks" (name, the question in one line, file name, one-line verdict), "The three rounds" (Early Trust Audit, Main Evidence Round, Optional Re-run List), "Other labels" (checkpoints, test stages, gate names, owner-decision names such as the no-writes rule). The numbers appear only in the file-name column.
- [ ] **Step 2:** Run `scripts/check_reading_edition.py` (Task 11) on it once that exists; until then check punctuation by `grep -n "—\|;"` outside code spans.
- [ ] **Step 3:** Commit `docs(research): add root-cause reference sheet`. The Opus reviewer verifies each definition against the source, checks that no definition is wrong in a way that matters (for example Sharpe, Holm, MDE, WFO, OOS), and approves the derived gate/part names.

### Task 11: Reading-edition lint (Sonnet, TDD)

**Files:** Create `scripts/check_reading_edition.py`, `tests/scripts/test_check_reading_edition.py`.

**Interfaces:** `check_file(path: Path, sheet_terms: set[str], source_texts: dict[str, str], max_words: int | None) -> list[str]` returns problems. Rules: (1) no U+2014 and no `;` outside code spans and fences; (2) no `Lens \d`, `Phase \d`, `Part [AB]`, `\bB0\b`, `Gate [AB]`, `P2-\d+` in prose or tables (allowed in the reference sheet's file-name column and inside code spans/paths); (3) every all-caps token of 2+ letters appears in `sheet_terms` or on the ignore list; (4) every number with a decimal point or `%` or 4+ digits appears in at least one source text (numeric traceability); (5) every backticked repo path exists (delegates to `check_lens_report.py --paths-only` logic); (6) optional word cap. CLI: `check_reading_edition.py FILE... --sheet reference-sheet.md --sources FILE... [--max-words N]`, exit 1 on any problem, `--help` works.

- [ ] **Step 1:** Failing tests: `test_flags_em_dash_and_semicolon_outside_code`, `test_allows_em_dash_inside_code_span`, `test_flags_lens_number_in_prose`, `test_allows_lens_number_in_file_name_column`, `test_flags_undefined_abbreviation`, `test_flags_number_absent_from_sources`, `test_word_cap`.
- [ ] **Step 2:** Implement to green; ruff clean; commit `feat(scripts): add reading-edition lint`.

### Task 12: Findings in plain English (Sonnet)

**Files:** Create `research/audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md` (new in the `verify --stage final` allow-list).

**Interfaces:** Consumes the synthesis (sections 1 to 3, 7 and 8 mainly), the challenge's resolutions, the Phase 1 checkpoint's limits, the reference sheet. Target 1,800 to 2,600 words (the synthesis is about 11,700 words).

- [ ] **Step 1:** Outline first, with every section mapped to source sections, saved in the report file of the task (not committed).
- [ ] **Step 2:** Write in Cyrus voice for an engineer: what was asked (beat the benchmark by 2x in raw dollars), what was found (ranked causes, each with the evidence strength in plain words and the number that matters), what could not be settled and why, what the checks cannot see, the owner decisions already made and the ones still open. Each section ends with `Source:` and the synthesis section it comes from. Use engineering analogies only where they do not change meaning, and label an analogy as one.
- [ ] **Step 3:** `check_reading_edition.py` passes with `--sources` set to the synthesis, challenge, checkpoint, nine check reports and model reviews; `check_lens_report.py --paths-only` exit 0; word count within the cap.
- [ ] **Step 4:** Commit `docs(research): add plain-English findings`. Opus review rubric: every verdict, number and caveat in the findings maps to a source line; no NO-GO or limit is softened or made funny; nothing added that the sources do not say; unverified material is labelled; no retirement, IRA, paid-data, live-trader, cache-write or Phase 2 language.

### Task 13: Consolidated direction review, revised (Sonnet)

**Files:** Modify `research/audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md` in place (already in the rewrite manifest scope; add it to `hand_edited`).

**Interfaces:** Keeps the required elements of spec 8.3 and the owner's reading goal: what each frontier model recommends, where they agree and differ, which Argon claims survived verification, at most three ranked options (question, cheapest step, prerequisites, falsifier, stop gate), one owner-decision list with defaults, and a "what to ignore" list.

- [ ] **Step 1:** Rewrite in Cyrus voice for the engineer reader: open with "what each reviewer recommends in two sentences each", then the options and the decision list, then the side-by-side table and the corrections. Replace every number-label with its descriptive name; keep real numbers and `file:line` citations to the originals (the originals are not edited, so their line numbers are stable). Move the old abbreviation block into the reference sheet and point to it; keep a short inline definition at first use of each term.
- [ ] **Step 2:** Lint and path checks as in Task 12, with `--sources` = both reviews, the verification, the synthesis.
- [ ] **Step 3:** Commit `docs(research): rewrite consolidated direction review for the engineer reader`. Opus review rubric as in Task 12 plus: the Argon unverified items stay labelled; the recommendations of each reviewer are reproduced at their original strength; the ranking order and the Phase 2 and no-writes decisions are unchanged unless the source says otherwise.

### Task 14: Folder README, index, pointers, changelog, verification record (Sonnet)

**Files:** Rewrite `research/audits/root-cause-2026-11-04/README.md`; modify `research/RESEARCH_INDEX.md` (the root-cause row and Study folder directory entry), `research/README.md` (link), `docs/changelog.md`, `docs/superpowers/plans/2026-11-05-research-reorg-verification.md` (new section "Sub-folder and readability pass").

- [ ] **Step 1:** README in Cyrus voice: a three-step reading route by name (plain-English findings, consolidated direction review, reference sheet), then a table of the four sub-folders and every file with a one-line description using descriptive names; states plainly that the Optional Re-run List was not run and carries the Argon unverified warning.
- [ ] **Step 2:** Index and research README updates (verdict wording of existing rows byte-for-byte unchanged); changelog entry for this plan's work (what moved, new documents, the lint, the ruff exclusion, the hash-drift restore); verification doc section recording every gate result with commands.
- [ ] **Step 3:** `verify --stage final` with all six `--require-new` paths exit 0; lint and path checks on README; full suite and ruff green. Commit `docs(research): rewrite root-cause README and record the sub-folder pass`.

## Phase E: Final review and merge

### Task 15: Whole-branch review, one fix wave, local merge

- [ ] **Step 1:** Dispatch the final reviewer (Opus) over `git diff main...HEAD` using the review package. Required checks: frozen bytes 78/78 and pinned 12/12 against the ledger; pure renames for the 33 moves; rewrite edits only path tokens (token-diff checker); no change under `src/`, `data/`, `results/`; Docker context unchanged (`.dockerignore` contains `research/`); hash drift register accurate; reading editions meet the lint, voice and fidelity rubrics; new dead links; full suite and ruff.
- [ ] **Step 2:** One fix wave for Critical and Important findings and cheap safe minors; list the rest for the owner.
- [ ] **Step 3:** Use superpowers:finishing-a-development-branch. On the owner's word (this plan includes the merge; present the menu anyway): fast-forward `main`, run the full suite on the merged tree, delete the feature branch. Do not push. Write the project memory update. Final message lists "Rulings I made".

## Self-review

- **Spec coverage:** requirement 1 (Tasks 9, 10), 2 (9, 10, 11, 12, 13), 3 (10, 12, 13, 14), 4 (6, 7, 8), 5 (12, 13, 14), 6 (1 to 5, 15), 7 (every task and the Global Constraints). Gaps: none found. Deliberately out: intraday branch translation (D7), rewriting the originals (D1).
- **Consistency:** moved files total 33 (4 findings, 20 lenses, 4 reviews, 5 records); `DEFAULT_ALLOW_NEW` in Task 6 matches the new files created in Tasks 10, 12, 13, 14.
- **Proportion:** tool bodies are left to implementers; test names and signatures are given.
