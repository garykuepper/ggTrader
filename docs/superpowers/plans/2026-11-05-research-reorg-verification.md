# Research reorganization: verification record

Date: 2026-11-05. Branch `reorg/research-top-level`. Tool commit `70d37120` (`verify --stage final`), tightened by fix round 1 (the commit `fix(scripts): tighten verify final gate and verification doc`). Results below were taken on the tree at `70d37120` (Tasks 1-9 applied) and, for the gate, tests and ruff, re-taken after fix round 1.

This record is **re-run in Task 14**. Tasks 11 and 12 (not Task 13, which creates nothing under `research/`) add the research README, `MOVED.csv`, the consolidated direction-review report and the investigation README. Those four files are the default allow-list of `verify --stage final` (`DEFAULT_ALLOW_NEW` in `scripts/research_reorg.py`); any other new file needs `--allow-new <pattern>` (fnmatch, relative to `research/`, repeatable). Until Tasks 11/12 run, the allow-list entries simply do not exist yet, which the allow-list accepts. To make their presence mandatory, Task 14 re-runs the final gate with:

```
--require-new README.md --require-new MOVED.csv \
--require-new audits/root-cause-2026-11-04/README.md \
--require-new audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md
```

`--require-new` fails with `required new file missing` (not on disk) or `required new file not tracked` (on disk, not in the git index).

Paths used below: `M` = `docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv`, `L` = `docs/superpowers/plans/2026-11-05-research-reorg-ledger.json`, `R` = `docs/superpowers/plans/2026-11-05-research-reorg-rewrite-manifest.json`.

## Summary

| Gate | Command | Result |
|---|---|---|
| verify final | `.venv/bin/python scripts/research_reorg.py verify --stage final --manifest $M --ledger $L --rewrite-manifest $R` | exit 0: `OK stage=final 174 manifest files, 68 listed edits, 174 tracked files under research/` |
| verify rewritten | same with `--stage rewritten` | exit 0 at `70d37120`: `OK stage=rewritten 174 files checked, 68 listed edits`. **Exit 1 by design after Tasks 11/12** (5 problems: it does not accept the allow-list of new files; section 7) |
| verify moved | same with `--stage moved` (no rewrite manifest) | exit 1 by design after Task 9: 68 `sha256 differs from ledger` lines, one per rewritten Markdown file, followed by one `NOTE` line (the exit code is unchanged). After Tasks 11/12 it is **73 problems** (the 68, plus 1 count problem and 4 unexpected new files; section 7). `moved` is only valid before the rewrite. |
| frozen digests | independent: `git show HEAD:<path>` piped to sha256 per ledger entry | 78/78 frozen match, 12/12 pinned match; 106/174 of all files match, the other 68 are exactly the 68 listed edits (174 - 106) |
| survivors | `git grep -I -n "docs/research"` (tracked files) | 2,670 hits in 58 files before this record was written (2,679 in 59 with its first version), 0 uncategorized; shifts when this record is re-committed (section 2). Task 14 re-run at `138b18ea`: 2,952 hits in 62 files (section 7) |
| guard snapshot | `scripts/guard_snapshot.sh <scratch>` vs baseline | only the `main:` line differs |
| Docker | inspection | `research/` and `*.md` excluded, `!pyproject.toml` kept; the 174 tracked files under `research/` stay out of the image context |
| full suite | `scripts/run_tests.sh -q` (at `70d37120`) | 1627 passed, 30 deselected, 1 warning, 294 s, no exit 137 |
| fix round 1 | `scripts/run_tests.sh tests/scripts -q` | 151 passed |
| ruff | `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | all checks passed; 285 files already formatted |
| checker | `scripts/check_lens_report.py --paths-only` on 102 tracked `research/**/*.md` | 12 files fail, all classified, none uncategorized |

## 1. The `verify --stage final` gate

Implementation: `verify_final()` in `scripts/research_reorg.py`, built test-first (ruling R2). It runs everything `verify --stage rewritten` runs (sha256 of all 174 files against the ledger with only listed Markdown edits allowed; frozen and pinned files must match exactly even if listed; every destination exists; leftovers under `docs/research` flagged) and adds:

- no duplicate proposed path in the manifest;
- every manifest destination is tracked (`git ls-files research`);
- tracked files under `research/` = manifest destinations plus files matching the allow-list; anything else is `unexpected tracked file`;
- no tracked file deeper than `research/<category>/<study>/<file>` (4 path components). The five `artifacts-*` packages are study folders by spec section 4, so they share the same limit; the manifest has 174 rows (4 at depth 2, 2 at depth 3, 168 at depth 4) and none deeper;
- exactly 12 pinned rows, each marked pinned in the ledger and equal to its ledger digest (`pinned digest differs from ledger`);
- the `docs/research` folder is gone, not only untracked;
- optional `--require-new PATH` (repeatable, relative to `research/`): each must exist and be tracked.

`verify --stage moved` is only valid before the rewrite step (the CLI help says so); when it fails and the rewrite manifest exists at its default path, it prints one `NOTE` line saying to use `--stage rewritten` or `final` (exit code unchanged). The unreachable "stage not implemented" branch and `IMPLEMENTED_STAGES` were removed.

Every failure prints `PROBLEM <reason>: <path>` and the run exits 1.

Tests (16 new in `tests/scripts/test_research_reorg.py`, 86 pass in the file): `test_verify_final_passes_clean_tree_and_allows_new_files`, `test_verify_final_flags_docs_research_leftover`, `test_verify_final_flags_empty_docs_research_directory`, `test_verify_final_flags_missing_destination`, `test_verify_final_flags_duplicate_proposed_path`, `test_verify_final_flags_tracked_file_too_deep`, `test_verify_final_accepts_artifact_package_as_study_folder`, `test_verify_final_flags_pinned_digest_change`, `test_verify_final_flags_wrong_pinned_count`, `test_verify_final_flags_unlisted_change_and_untracked_extra`, `test_main_verify_final_cli`, and from fix round 1 `test_verify_final_flags_destination_on_disk_but_untracked`, `test_verify_final_flags_ledger_entry_not_marked_pinned`, `test_verify_final_require_new_flags_missing_and_untracked`, `test_main_verify_final_require_new_cli`, `test_main_verify_moved_failure_notes_existing_rewrite_manifest`.

## 2. Survivors of `docs/research`

Scope: **tracked files only**. Command (exact, run from the repo root; it does not depend on `.gitignore` handling or on shell aliases):

```
git grep -I -n "docs/research"
```

The interactive shell's `grep` is a wrapper that skips gitignored files, so a plain `grep -r` does not reproduce a fixed count; with GNU `command grep -rIn` (excluding `.git`, `.venv`, `results`, `data`, `.superpowers` and cache directories) the scan finds 3,014 hits in 118 files. The extra 335 hits in 59 files are gitignored scratch and out of scope: 311 hits in 47 files under `.claude/skills/*-workspace/` (skill eval scratch, five workspace folders), 5 hits in 3 files under `logs/`, and 19 hits in 9 files under `.remember/`. No tracked file is in the 335.

Counts:

- **Before this record was written (tool commit `70d37120`): 2,670 hits in 58 files, uncategorized 0.** The tables below use these numbers.
- At `6143691e` (this record's first version) `git grep` gave 2,679 hits in 59 files: the 2,670/58 plus the record's own 9 hits.
- This record names the old path throughout, so it is itself spec/plan text and every re-run after it is committed counts its own hits in the spec/plan category (the count changes with every edit of this document). Task 14's re-run will therefore read higher by the document's own hits, with 0 uncategorized.

Classification used a throwaway script (path and line rules) over the `git grep` output, then spot-checks.

| Category | Hits | Files | What it is |
|---|---:|---:|---|
| Frozen provenance | 1,794 | 27 | Byte-frozen `research/` files (JSON artifacts and three frozen Markdown files) that name the old paths they recorded; verified byte-identical to the ledger |
| Historical citation | 225 | 18 | Mutable `research/` Markdown quoting a recorded command, ledger entry, table mirror, dissolved folder or an invented path |
| Spec/plan text | 651 | 13 | This reorganization's own spec, plan, manifest, ledger, mapping draft, tool and tests, plus 5 older plans with never-created or placeholder paths |
| Known prior defect | 0 | 0 | none left: the template placeholders and the lens7 citation were fixed in 348e959d (section 5) |

Frozen provenance, by file (hits): `_investigation_inventory.json` 459, `_research_process_audit_ledger.json` 397, `_lens2_sensitivity.json` 272, `_lens9_power.json` 239, `_lens7_recompute.json` 172, `_lens5_coverage_matrix.json` 109, `_family_scope_extraction.json` 39, `_free_data_feasibility_matrix.json` 20, `_event_evidence_resolution_workbook.json` 14, `_lens4_ceiling.json` 13, `2026-10-21-event-timing-audit.md` 12, `_event_audit_readiness_decision.json` 10, `_event_source_search_ledger.json` 8, `_lens1_tape_state.json` 7; and 13 files with 1-3 hits each: `event-timing-audit-implementation.md` 3, `_production_replica_results.json` 3, `_wfo_fold_attribution_results.json` 3, the perf-review artifact Markdown 2, `_lens3_snapshot_gap.json` 2, `_synthesis_inventory_map.json` 2, `_event_evidence_export_file_search.json` 2, and one each in `_lens6_verdicts.json`, `_lens6_replays.json`, `_event_audit_detector_provenance.json`, `_net_issuance_screen_results.json`, `_momentum_french_screen_results.json`, `_factor_screen_results.json`.

Historical citation, by subtype:

| Subtype | Hits | File(s) |
|---|---:|---|
| lens2 table mirror of the frozen `_lens2_*.json` (184 table lines plus 1 recorded gate command) | 185 | `2026-11-04-lens2-verdict-audit.md` |
| Argon invented paths (files the Argon review cited that never existed) | 16 | `2026-11-05-argon-synthesis-review.md` 11, `2026-11-05-argon-review-verification.md` 5 |
| ledger, inventory, checkpoint and challenge entries naming the dissolved folder | 10 | `investigation-ledger.md` 7, `root-cause-challenge.md` 1, `investigation-inventory.md` 1, `phase1-checkpoint.md` 1 |
| recorded gate or generation commands | 7 | lens1, lens4, lens5, lens6, lens7, lens9 reports, `static-fractional-leverage.md` |
| planned output paths never created (intraday, off-branch) | 3 | `2026-10-24-intraday-feasibility.md` 2, `2026-10-25-intraday-stage0a-contract.md` 1 |
| `briefs/` directory citations and an off-branch worktree path | 2 | `lens5-coverage-gaps.md` |
| recorded command or quoted code comment | 2 | `2026-10-23-insider-acceptance-time-amendment.md`, `2026-09-28-live-indicator-warmup.md` |

Spec/plan text, by subtype:

| Subtype | Hits | File(s) |
|---|---:|---|
| this reorganization (ledger 175, manifest 174, mapping draft 174, tool tests 64, tool 26, plan 16, spec 9, `test_driver_default_paths.py` 3) | 641 | 8 files |
| older plans with never-created files, globs or placeholders | 10 | `2026-10-20-momentum-retest.md` 3, `2026-09-11-paper-trading-remediation.md` 3, `2026-11-04-root-cause-investigation.md` 2, `2026-11-02-research-followups-master.md` 1, `2026-11-02-research-reliability-next-steps.md` 1 |

No survivor remains in `README.md`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, `src/`, any driver under `scripts/` other than the tool itself, or any `docs/` file outside `docs/superpowers/`.

## 3. Guard snapshot

`scripts/guard_snapshot.sh` was read in full first: it runs only `git rev-parse/log/status`, `systemctl --user is-enabled/cat` and `docker ps -a --filter`; no network, no database access, no orders, nothing started, stopped or edited. Run to a scratch file and diffed against `research/audits/root-cause-2026-11-04/_investigation_guard_baseline_2026-11-04.txt`:

```
19c19
< main: 1e419c27ceb956c66806e0a654264daf6f7defa0
---
> main: 42990c74a569aeb3b9ff1092036209e1bc045ce3
```

Only the `main:` hash differs, as expected. `1e419c27` is an ancestor of `42990c74` (40 commits behind); this branch has not touched main. Observer worktree HEAD `58919c09`, the four timers (enabled, unit hashes) and the `ggtrader_live` container line (`ghcr.io/garykuepper/ggtrader:latest running`) are identical to the baseline, and `git status --short data/` still shows only the untracked `data/external/event_timing_audit/`.

## 4. Docker content

By inspection, no build. `.dockerignore` contains `research/`, `docs/`, `data/`, `results/`, `*.md`, then `!pyproject.toml`. The Dockerfile copies `pyproject.toml` and then `COPY . .`, so the 174 tracked files under `research/` (`git ls-files research | wc -l` = 174) are outside the build context and the image content is unchanged by this reorganization. `git ls-files docs/research | wc -l` = 0 and `docs/research` does not exist.

## 5. Checker results

Command, per tracked Markdown file (102 files): `.venv/bin/python scripts/check_lens_report.py --paths-only <file>`. 12 fail, none uncategorized. The checker prints no line numbers; lines below come from Task 9's report. "Move-caused" means the cited `docs/research/...` file existed in the ledger (revision `e4fca023`), so it passed before the move.

| File | Failing citation(s) | Class |
|---|---|---|
| `research/WEB_RESEARCH_CANDIDATES.md` | `research/<category>/<study>/<YYYY-MM-DD>-<slug>.md` | placeholder (forward-looking template text) |
| `research/prompts/local-implementation-prompt-TEMPLATE.md` | same placeholder; line 68 banned phrase "retire the sleeve" | placeholder; banned phrase pre-existing |
| `research/allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve-brief.md` | same placeholder | placeholder |
| `research/allocation/month-end-treasury/2026-09-24-month-end-treasury-sleeve.md` | same placeholder | placeholder |
| `research/allocation/static-leverage/2026-09-30-static-fractional-leverage-brief.md` | same placeholder | placeholder |
| `research/audits/artifacts-2026-09-09-perf-review/ggtrader_paper_review_20260909.md` (frozen) | `docs/research/_rebaseline_corrected_tape_20260822.json` (lines 82, 114) | caused by the move, unfixable (frozen) |
| `research/audits/event-timing/2026-10-21-event-timing-audit.md` (frozen, pinned) | 7 move-caused: `2026-10-20-momentum-french-screen-nogo.md`, `2026-10-21-factor-screen-nogo.md`, `2026-07-17-pead-nogo.md`, `2026-07-19-insider-cluster-buy-nogo.md`, `2026-07-25-strategy-implementation-audit.md`, `briefs/2026-10-20-momentum-retest.md`, `briefs/2026-10-21-event-timing-audit-sources.md` (lines 25, 26, 35, 36, 42, 69, 73, 222). 3 pre-existing: `src/ggTrader/lab/strategies/pead.py:24,27` (the checker does not parse a comma line list), `_event_timing_audit.json`, `2026-10-21-event-timing-audit-report.md` (never existed) | move-caused (frozen) plus pre-existing |
| `research/audits/event-timing/event-timing-audit-implementation.md` (frozen, pinned) | `docs/research/briefs/2026-10-21-event-timing-audit-sources.md` and `.../2026-10-21-event-timing-audit.md` (lines 405, 447) | caused by the move, unfixable (frozen) |
| same file | `research/event-timing-audit` (lines 6, 323, 559, 805) | branch-name false positive, pre-existing |
| `research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md` | `src/ggTrader/lab/retail_attention.py` | pre-existing |
| `research/audits/root-cause-2026-11-04/2026-11-05-argon-synthesis-review.md` | 7 invented paths (`research/a06_production_replica.py`, `src/ggtrader/core/tax_lots.py`, five `docs/research/2026-11-04-lens*.md` that never existed) | pre-existing, kept as written |
| `research/intraday/spy-momentum-feasibility/2026-10-24-intraday-feasibility.md` | `_intraday_probe.json`, `_intraday_momentum_results.json` (never created) | pre-existing |
| `research/intraday/spy-momentum-feasibility/2026-10-25-intraday-stage0a-contract.md` | `_intraday_probe.json`, `intraday_probe.py`, `intraday_coverage_probe.py`, `test_intraday_probe.py` (never created) | pre-existing |

Totals: 5 placeholder files (the placeholder is this reorganization's own forward-looking text), 3 frozen files with move-caused failures, 1 branch-name false-positive set, 4 files with pre-existing failures only. The frozen event-timing and perf-review files cannot be edited (byte-pinned or inside an `artifacts-*` package) and the checker was not changed. The ledger's one `existing_dead_links` entry (`2026-10-22-event-evidence-resolution-workbook.md:175`, `/tmp/event-timing-vendor-evidence-review.md`, a pinned file) is unchanged and not flagged by `--paths-only`. The nine lens reports pass the full checker as of Task 9 (commit 348e959d); no lens report has changed since.

## 6. Known limits

- `verify_rewritten` (and so `final`) enforces "explicitly listed", not "reference-only": it proves that no unlisted file changed, not that a listed file changed only in its paths. Reference-only was proven separately by the independent token checker at the Task 9 review (1,254 tokens, 0 unexplained).
- The lens2 table keeps about 155 old-path mentions (Task 9's count of unresolved prose mentions; a current count is 275 mentions in 184 table lines, 276 in 185 lines file-wide) as a mirror of the frozen `_lens2_*.json`, which cannot change, so the table is deliberately left as written.
- Argon's invented paths are retained: that review is kept as written with its unverified-claims banner.
- Recorded gate commands, ledger entries and `briefs/` directory citations stay as written (ruling R14); the dissolved folder has no successor path.
- `verify --stage moved` is only valid right after the move step; after the rewrite it correctly fails with the 68 listed Markdown differences (73 problems once the four Task 11/12 files exist, section 7).
- The allow-list matches by pattern, so an unexpected file that happens to match an allowed pattern would pass; the defaults are exact paths.
- The `ruff.toml` exclusion for frozen `.py` artifacts under `research/` was left unchanged (ruling R8d, out of spec scope); a repo-wide `ruff format .` could rewrite them. Optional owner follow-up. Today `ruff format --check .` passes (285 files already formatted).
- Survivor counts in section 2 shift whenever this document, the tool tests or any other file with old-path text is committed; the figures are for the stated commits.
- Results reflect commit `70d37120` (gate, tests and ruff also re-run after fix round 1); Task 14 repeats `verify --stage final` with the `--require-new` flags above once Tasks 11 and 12 have added their files, plus the survivors grep (`git grep`), the checker table and the guard diff.

## 7. Task 14 re-run (HEAD 138b18ea)

Re-run on the tree at `138b18ea` (Tasks 1 to 13 applied), after the whole-branch review. Every figure below was recomputed on that tree, not copied from sections 1 to 6. `M`, `L`, `R` are as defined above.

### 7.1 Gate results

| Gate | Command | Result |
|---|---|---|
| verify final | `verify --stage final --manifest $M --ledger $L --rewrite-manifest $R --require-new README.md --require-new MOVED.csv --require-new audits/root-cause-2026-11-04/README.md --require-new audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md` | exit 0: `OK stage=final 174 manifest files, 68 listed edits, 178 tracked files under research/` (174 + the 4 allow-listed new files) |
| verify rewritten | same without `--require-new`, `--stage rewritten` | exit 1, 5 problems: `178 files under research/ (excluding 0 allowed new) but 174 manifest rows` and 4 `unexpected file under research/` (`MOVED.csv`, `README.md`, the consolidated review, the root-cause `README.md`). By design: this stage has no allow-list. It was valid only until Tasks 11/12 added files; the summary-table row above is corrected. |
| verify moved | same with `--stage moved` (no rewrite manifest) | exit 1, 73 problems: 68 `sha256 differs from ledger` + the same 1 count problem + the same 4 unexpected files; the `NOTE` line names `--stage rewritten` or `final`. The earlier table said "exactly 68"; that held only before Tasks 11/12. |
| checker | `scripts/check_lens_report.py --paths-only` on each of the 105 tracked `research/**/*.md` | 12 fail, the same 12 files as the table in section 5. Section 5 covered 102 files; the 3 added since (`research/README.md`, the root-cause `README.md` and the consolidated review) all pass. `README.md` at the repo root passes; `AGENTS.md` fails on one citation, the placeholder `research/<category>/<study>/` (forward-looking template text in the "Where docs live" rule; the same class as the five placeholder files in section 5). |
| full suite | `scripts/run_tests.sh -q` | 1633 passed, 30 deselected, 1 warning, 290 s, exit 0 (section 1 recorded 1627 at `70d37120`) |
| scripts tests | `scripts/run_tests.sh tests/scripts -q` | 152 passed (151 after fix round 1) |
| ruff | `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | all checks passed; 285 files already formatted |

### 7.2 Survivors of `docs/research`

`git grep -I -n "docs/research"` at `138b18ea`, before this section was written: **2,952 hits in 62 files**, no live survivor (none in `README.md`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, `src/`, any driver under `scripts/` other than the tool, or any `docs/` file outside `docs/superpowers/`). It is the section 2 figure of 2,670 hits in 58 files plus:

| Added | Hits | Files | Category |
|---|---:|---:|---|
| `research/MOVED.csv` (old path column, one row per moved file) | 174 | 1 | new category, navigation resolver: by design, it exists to map old paths to new ones |
| `docs/superpowers/plans/2026-11-05-research-reorg-branch-additions.csv` | 82 | 1 | spec/plan text (branch translation data) |
| `docs/superpowers/plans/2026-11-05-research-reorg-branch-recipe.md` | 16 | 1 | spec/plan text (branch translation recipe) |
| this document | 10 | 1 | spec/plan text (counted again whenever this section is edited; it now also contains this section's own mentions) |
| **Total added** | **282** | **4** | 2,670 + 282 = 2,952; 58 + 4 = 62 |

Only `MOVED.csv` is a new category; the others fall in the existing spec/plan category. The Task 11/12 files that name the old path in prose (`research/README.md`, the investigation README, the consolidated review, `RESEARCH_INDEX.md`) contribute no `docs/research` hits.

### 7.3 Relative links that stopped working in frozen files

A link check over every byte-frozen `.md` file (each relative link resolved against the file's old directory at `42990c74` and its new directory at `138b18ea`; directory links count as live if any file sits under them) found **19 relative links in 4 frozen event-timing Markdown files that resolved before the move and are dead now**. By design (spec decision 9.1: frozen bytes are not edited, even to keep a link alive):

| File (frozen, under `research/audits/event-timing/`) | Dead links |
|---|---:|
| `event-timing-audit-implementation.md` | 9 |
| `2026-10-21-event-timing-audit.md` | 5 |
| `2026-10-22-event-evidence-resolution-workbook.md` | 4 |
| `2026-10-22-event-timing-source-conflict-proposal.md` | 1 |

### 7.4 Other consequences of the move that are left as written

- **Line numbers in the two 2026-11-05 reviews shifted by 2.** The pointer line (`> Reading copy (2026-11-05): ...`) plus one blank line were added at the top of `2026-11-05-argon-synthesis-review.md` and `2026-11-05-senior-quant-direction-review.md`. The Argon verification (`2026-11-05-argon-review-verification.md`) quotes Argon line numbers in the frame of the Argon file before its banner, so those numbers were already 5 lines low (for example the six `lens6-live-vs-backtest-parity` citations are at lines 49, 50, 51, 70, 72, 107 in the verification and at 54, 55, 56, 75, 77, 112 in the file at `42990c74`). Today they sit at 56, 57, 58, 77, 79, 114, so the **total offset is now 7 (5 banner + 2 pointer)**, not 9. The verification file was not renumbered.
- **A historical statement now reads false.** `argon-review-verification.md` line 29 says "(there is no top-level research/ directory)" about Argon's `research/a06_production_replica.py`. At the time it was true; there is now a top-level `research/`. The file is left as is (the real file is `scripts/production_replica_audit.py`, as the same line says).
- **`2026-11-04-investigation-inventory.md` must not be re-rendered.** Line 5 says the page is rendered from `_investigation_inventory.json` by a script ("edit the JSON, not this table"). The JSON is frozen and still holds the old `docs/research` paths, so re-rendering would write the old paths back into the page. Treat the page as hand-maintained from now on.
- **`event_timing_audit.py` changed its own code hash.** The path edits change the sha256 of `src/ggTrader/lab/event_timing_audit.py` (old and new values in the register, 7.5), so resume checkpoints made before the move will not resume. Accepted per spec decision 9.7 and ruling R10.
- **The tool emits no rewrite manifest.** Spec section 14.1 says the `rewrite` stage should write the manifest of edited files; `scripts/research_reorg.py` does not. The manifest `R` was hand-built from the git list of changed files and then checked by the gate (`verify_rewritten`). This is a deviation from the spec; the owner may accept it (ruling R14 parked it).

### 7.5 Hash drift register

A reorganization that edits file contents changes their sha256, and some frozen evidence records those hashes. Scope of the sweep (done by the whole-branch review, ruling R18, not re-run in full here): every full 64-hex token recorded in tracked files (1,099 distinct tokens) checked against the file bytes before and after, over the tracked files at `HEAD` and at `main`, branches `research/intraday-stage0a` and `research/momentum-retest`, two `t3/*` branches, and the untracked `data/external/event_timing_audit/`; plus 8, 10, 12 and 16-character prefixes and git blob SHA-1s. Mentions in the ledger and the branch recipe were excluded (they record the hashes on purpose). Three file:line facts in the first three rows were re-checked for this record against `git show 42990c74:<path> | sha256sum` and `sha256sum` of the file at `HEAD`, and by reading the cited lines.

| Referent | Old -> new sha256 | Where recorded | Runtime-checked? | Handling options |
|---|---|---|---|---|
| `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` | `9cc5c1a4630e01d703a7241fdfd64d6f726935982212d297a49e0e5d37aa841e` -> `ef8ff7af2473a7f29725cba7262b9f8a432ea2c2bcbc45c0942cafaa8c3d7d5d` | `research/audits/event-timing/_event_evidence_resolution_workbook.json:24` (input sources) and `:8636` (`spec_hashes`); pinned, frozen file | No. Nothing in `src/`, `scripts/`, `tests/` or `.claude/` reads `spec_hashes` (`git grep` finds the key only in that JSON) | **Restored (decision D5), 2026-11-05.** `git show 42990c74:<path>` written back, so the sha256 is again `9cc5c163...` at both recorded locations; `tests/scripts/test_recorded_hashes.py::test_restored_plan_matches_recorded_hash` pins it. The plan is closed and historical and its 8 edited lines were navigation only; its 5 old-path mentions are now spec/plan survivors (7.6). The file is in the always-excluded set of `scripts/research_reorg.py` (`REWRITE_EXCLUDED_PATHS`), because the `fixed-markdown` scope would otherwise rewrite 6 of those mentions again. |
| `scripts/production_replica_audit.py` | `9d179ca40e0631c539aba409921002a3ca6edc611ecb200b8d4f6a034584b053` -> `7e0ac01fe321216b25aaa8e8a3626471941309bb438cc2dff10340990ea6eb5e` | `research/audits/event-timing/_event_audit_detector_provenance.json:6284` (`producer.sha256`); frozen | No | **Annotated, not restored (decision D5), 2026-11-05.** Restoring would point `DEFAULT_OUT`, `DEFAULT_CURVES` and `AMENDMENT` back at the old research folder; the 3 changed lines are path-only, and the old bytes are recoverable from `42990c74`. The recorded digest stays false by design; nothing reads it at run time. |
| `research/intraday/spy-momentum-feasibility/2026-10-24-intraday-feasibility.md` | `51162768ea497c242f0c199baf7e2a0f79a62a85071c8f6ff85fd06b670eaec3` -> `33c00494472115c75205690d331d684050d876584f959c5dc7f0a395924ccd07` | Only on the unmerged branch `research/intraday-stage0a` (tip `58919c09`): `docs/research/_intraday_probe.json:25` (`document_hashes.strategy_brief`) | No. `src/ggTrader/lab/intraday_probe.py:499` (on that branch) only writes the hash | Decided at branch translation time (recipe decision D3). |
| `src/ggTrader/lab/event_timing_audit.py` (its own code hash) | `d4bf67436385986ca9d03e5eb840a310b5d84edd6c8f2538192a4294d6b09622` -> `ed40ef1cd77290a1c4457e285d994389f7e6269d3f18a8d2f55b30a5bf58b48b` | Not recorded in any tracked file | Only on `--resume`: the run compares `code_sha256` with the checkpoint (`event_timing_audit.py` line 1619; the hash set is built at line 1545) | Accepted (spec decision 9.7, ruling R10). The one untracked checkpoint found (`data/external/event_timing_audit/network-smoke-5b1q16b6/smoke_progress.json`) already carried a different hash before the move, so it was not resumable either. |

The path edits behind the first two rows are in the diff `42990c74..HEAD` (before the 2026-11-05 restore of the first row, section 7.6): the plan had 8 changed lines, `production_replica_audit.py` 3, `event_timing_audit.py` 10. The register lists no other recorded hash whose referent the reorganization changed.

### 7.6 Hash drift restore (2026-11-05, readability plan Task 2)

Dated change record. Register rows 1 and 2 in 7.5 now read "restored (decision D5)" and "annotated, not restored". Recomputed on the working tree before this section was written, with `git grep -I -n "docs/research"` over tracked files.

- **Restore.** `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` is byte-identical to `42990c74` again (`git diff 42990c74 -- <file>` is empty). Its sha256 equals `9cc5c1a4630e01d703a7241fdfd64d6f726935982212d297a49e0e5d37aa841e`, the digest recorded at both places in `research/audits/event-timing/_event_evidence_resolution_workbook.json` (the input-sources entry and `spec_hashes`). New test: `tests/scripts/test_recorded_hashes.py`.
- **Rewrite scope.** A dry run of `research_reorg.py rewrite --scope fixed-markdown` found the restored file in scope (6 replacements, 2 unresolved), so it was added to the always-excluded set (`REWRITE_EXCLUDED_PATHS`, applied by `_is_rewrite_excluded`) with `test_recorded_hash_plan_is_never_rewritten`. After that, the dry runs of `fixed-markdown`, `code` and `moved-markdown` report 0 changed files. The expectation that no rewrite scope touched this file was wrong for `fixed-markdown`.
- **Survivor counts.** At `fd84580d` (the commit before this change) `git grep` finds 2,965 hits in 65 files. This change adds 5 hits in the restored plan (1 file) and 1 hit in `tests/scripts/test_research_reorg.py` (the new test's fixture text): **2,971 hits in 66 files**, 0 uncategorized.

| Category | Hits | Files | Change |
|---|---:|---:|---|
| Frozen provenance | 1,794 | 27 | none |
| Historical citation | 225 | 18 | none |
| Navigation resolver (`research/MOVED.csv`) | 174 | 1 | none |
| Spec/plan text: this reorganization (ledger 175, manifest 174, mapping draft 174, branch additions 82, tool tests 65, tool 26, plan 16, branch recipe 16, this record 16, spec 9, `test_driver_default_paths.py` 3) | 756 | 11 | tool tests 64 to 65 (+1); the other rows as in 7.2 and the commits since |
| Spec/plan text: older plans (restored plan 5, momentum retest 3, paper-trading remediation 3, root-cause investigation 2, followups master 1, reliability next steps 1) | 15 | 6 | restored plan +5, +1 file (was 10 hits in 5 files) |
| Spec/plan text: the readability-and-sub-folders plan | 4 | 1 | added after section 7.2, not in it |
| Descriptive mention of the move (`README.md` 1, `docs/changelog.md` 2) | 3 | 2 | added by Task 13 and later commits, after section 7.2; they say where the folder went |
| **Total** | **2,971** | **66** | 2,193 under `research/` (46 files) + 778 elsewhere (20 files) |

Reconciliation with 7.2 (2,952 in 62 files at `138b18ea`): +6 hits in this record (10 to 16), +4 in the readability plan, +3 in `README.md` and `docs/changelog.md` = 2,965 in 65 files at `fd84580d`; this change then adds 6 more. Two sentences earlier in this record are therefore out of date: section 2's "No survivor remains in `README.md` ... or any `docs/` file outside `docs/superpowers/`" no longer holds for `README.md` line 120 and `docs/changelog.md` lines 6 and 9. Those three hits are deliberate (they tell a reader where the folder went), not live paths. This record's own hit count shifts again whenever it is edited.
