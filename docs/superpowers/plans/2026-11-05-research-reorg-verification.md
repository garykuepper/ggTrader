# Research reorganization: verification record

Date: 2026-11-05. Branch `reorg/research-top-level`. Tool commit `70d37120` (`verify --stage final`), tightened by fix round 1 (the commit `fix(scripts): tighten verify final gate and verification doc`). Results below were taken on the tree at `70d37120` (Tasks 1-9 applied) and, for the gate, tests and ruff, re-taken after fix round 1.

This record is **re-run in Task 14**. Tasks 11 and 12 (not Task 13, which creates nothing under `research/`) add the research README, `MOVED.csv`, the consolidated direction-review report and the investigation README. Those four files are the default allow-list of `verify --stage final` (`DEFAULT_ALLOW_NEW` in `scripts/research_reorg.py`); any other new file needs `--allow-new <pattern>` (fnmatch, relative to `research/`, repeatable). Until Tasks 11/12 run, the allow-list entries simply do not exist yet, which the allow-list accepts. To make their presence mandatory, Task 14 re-runs the final gate with:

```
--require-new README.md --require-new MOVED.csv \
--require-new audits/root-cause-2026-11-04/README.md \
--require-new audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md
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

No survivor remains in `AGENTS.md`, `CLAUDE.md`, `.claude/`, `src/`, or any driver under `scripts/` other than the tool itself. As of `fd84580d`, three deliberate descriptive mentions of the move exist outside `docs/superpowers/`: `README.md` line 120 and `docs/changelog.md` lines 6 and 9. They tell a reader where the folder went and are not live paths (see 7.6). At the time of this section (`138b18ea`), none existed.

## 3. Guard snapshot

`scripts/guard_snapshot.sh` was read in full first: it runs only `git rev-parse/log/status`, `systemctl --user is-enabled/cat` and `docker ps -a --filter`; no network, no database access, no orders, nothing started, stopped or edited. Run to a scratch file and diffed against `research/audits/root-cause-2026-11-04/records/_investigation_guard_baseline_2026-11-04.txt`:

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
| `research/audits/root-cause-2026-11-04/reviews/2026-11-05-argon-synthesis-review.md` | 7 invented paths (`research/a06_production_replica.py`, `src/ggtrader/core/tax_lots.py`, five `docs/research/2026-11-04-lens*.md` that never existed) | pre-existing, kept as written |
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
| verify final | `verify --stage final --manifest $M --ledger $L --rewrite-manifest $R --require-new README.md --require-new MOVED.csv --require-new audits/root-cause-2026-11-04/README.md --require-new audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md` | exit 0: `OK stage=final 174 manifest files, 68 listed edits, 178 tracked files under research/` (174 + the 4 allow-listed new files) |
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
| `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` | `9cc5c1a4630e01d703a7241fdfd64d6f726935982212d297a49e0e5d37aa841e` -> `ef8ff7af2473a7f29725cba7262b9f8a432ea2c2bcbc45c0942cafaa8c3d7d5d` | `research/audits/event-timing/_event_evidence_resolution_workbook.json:23-24` (input sources) and `:8636` (`spec_hashes`); pinned, frozen file | No run-time check. Nothing in `src/`, `scripts/` or `.claude/` reads `spec_hashes`. Since 2026-11-05 the test `tests/scripts/test_recorded_hashes.py` reads it and pins the digest. | **Restored (decision D5), 2026-11-05.** `git show 42990c74:<path>` written back, so the sha256 is again `9cc5c163...` at both recorded locations; `tests/scripts/test_recorded_hashes.py::test_restored_plan_matches_recorded_hash` pins it. The plan is closed and historical and its 8 edited lines were navigation only; its 5 old-path mentions are now spec/plan survivors (7.6). The file is in the always-excluded set of `scripts/research_reorg.py` (`REWRITE_EXCLUDED_PATHS`), because the `fixed-markdown` scope would otherwise rewrite 6 of those mentions again. |
| `scripts/production_replica_audit.py` | `9d179ca40e0631c539aba409921002a3ca6edc611ecb200b8d4f6a034584b053` -> `7e0ac01fe321216b25aaa8e8a3626471941309bb438cc2dff10340990ea6eb5e` | `research/audits/event-timing/_event_audit_detector_provenance.json:6284` (`producer.sha256`); frozen | No | **Annotated, not restored (decision D5), 2026-11-05.** Restoring would point `DEFAULT_OUT`, `DEFAULT_CURVES` and `AMENDMENT` back at the old research folder; the 3 changed lines are path-only, and the old bytes are recoverable from `42990c74`. The recorded digest stays false by design; nothing reads it at run time. |
| `research/intraday/spy-momentum-feasibility/2026-10-24-intraday-feasibility.md` | `51162768ea497c242f0c199baf7e2a0f79a62a85071c8f6ff85fd06b670eaec3` -> `33c00494472115c75205690d331d684050d876584f959c5dc7f0a395924ccd07` | Only on the unmerged branch `research/intraday-stage0a` (tip `58919c09`): `docs/research/_intraday_probe.json:25` (`document_hashes.strategy_brief`) | No. `src/ggTrader/lab/intraday_probe.py:499` (on that branch) only writes the hash | Decided at branch translation time (recipe decision D3). |
| `src/ggTrader/lab/event_timing_audit.py` (its own code hash) | `d4bf67436385986ca9d03e5eb840a310b5d84edd6c8f2538192a4294d6b09622` -> `ed40ef1cd77290a1c4457e285d994389f7e6269d3f18a8d2f55b30a5bf58b48b` | Not recorded in any tracked file | Only on `--resume`: the run compares `code_sha256` with the checkpoint (`event_timing_audit.py` line 1619; the hash set is built at line 1545) | Accepted (spec decision 9.7, ruling R10). The one untracked checkpoint found (`data/external/event_timing_audit/network-smoke-5b1q16b6/smoke_progress.json`) already carried a different hash before the move, so it was not resumable either. |

The path edits behind the first two rows are in the diff `42990c74..HEAD` (before the 2026-11-05 restore of the first row, section 7.6): the plan had 8 changed lines, `production_replica_audit.py` 3, `event_timing_audit.py` 10. The register lists no other recorded hash whose referent the reorganization changed.

### 7.6 Hash drift restore (2026-11-05, readability plan Task 2)

Dated change record. Register rows 1 and 2 in 7.5 now read "restored (decision D5)" and "annotated, not restored". Recomputed on the working tree before this section was written, with `git grep -I -n "docs/research"` over tracked files.

- **Restore.** `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` is byte-identical to `42990c74` again (`git diff 42990c74 -- <file>` is empty). Its sha256 equals `9cc5c1a4630e01d703a7241fdfd64d6f726935982212d297a49e0e5d37aa841e`, the digest recorded at both places in `research/audits/event-timing/_event_evidence_resolution_workbook.json` (the input-sources entry at `:23-24` and `spec_hashes`). New test: `tests/scripts/test_recorded_hashes.py`.
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

Reconciliation with 7.2 (2,952 in 62 files at `138b18ea`): +6 hits in this record (10 to 16), +4 in the readability plan, +3 in `README.md` and `docs/changelog.md` = 2,965 in 65 files at `fd84580d`; this change then adds 6 more. Section 2's no-survivor sentence was amended in place to name these three hits, which are deliberate (they tell a reader where the folder went) and are not live paths. This record's own hit count shifts again whenever it is edited.

## 8. Sub-folder and readability pass

Dated 2026-11-05, readability plan Task 14 (`docs/superpowers/plans/2026-11-05-research-readability-and-subfolders.md`), branch `research/readability-and-subfolders`. Every figure was recomputed on the tree at `645c2bf3` (the README and index commit) plus the uncommitted edits of this task, not copied from the task reports. `M`, `L`, `R` are as defined above. `SP` is the session scratch directory for the link scripts and the clone used for the Task 8 check.

Stale paths in this record were corrected in place: the `--require-new` path of the consolidated review (the command at the top and the verify final row in 7.1) now names `reviews/`, the guard baseline in section 3 names `records/`, and the Argon review in the section 5 table names `reviews/`. The same path-only edit was made in the reorganization plan (lines 138, 178 and 203) and the design spec (line 130). Eight lines changed in the three files, and nothing else in them.

### 8.1 Gate results

| Gate | Command | Result |
|---|---|---|
| verify final | `.venv/bin/python scripts/research_reorg.py verify --stage final --manifest $M --ledger $L --rewrite-manifest $R --require-new README.md --require-new MOVED.csv --require-new audits/root-cause-2026-11-04/README.md --require-new audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md --require-new audits/root-cause-2026-11-04/reference-sheet.md --require-new audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md` | exit 0: `OK stage=final 174 manifest files, 68 listed edits, 180 tracked files under research/` (174 + the 6 allow-listed new files) |
| verify rewritten and moved | same command with `--stage rewritten` and with `--stage moved` | **Obsolete, read `--stage final` instead.** Neither has an allow-list for new files. `rewritten`: exit 1, 7 problems (the count line, `180 files under research/ (excluding 0 allowed new) but 174 manifest rows`, plus 6 `unexpected file under research/`). `moved`: exit 1, 75 problems (68 `sha256 differs from ledger`, the same count line, the same 6 unexpected files). The ledger and manifests now carry the sub-folder paths, so no problem line is about layout. |
| retarget verify | `.venv/bin/python scripts/research_retarget.py verify --moves docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv --root <clone at fca2ac82> --before $SP/t8-before.json --base f4b3ba1e^ --rewrite-manifest $SP/t8-rewrite-manifest.json --manifest <clone>/$M` plus `--allow` for 8 paths | exit 0: `OK 33 moved files, 26 rewritten files`. Run on a local clone at `fca2ac82` (the Task 8 tree), because the live tree has moved on. The 8 allowed paths are the 5 bookkeeping files named in the Task 8 report (manifest, ledger, rewrite manifest, moves CSV, `research/MOVED.csv`) and the 3 hand-edited files (`scripts/research_reorg.py`, `tests/scripts/test_research_reorg.py`, `tests/scripts/test_research_retarget.py`). Without the 3 hand-edited paths the run exits 1 with 3 problems. |
| move and rewrite counts | `git diff -M --name-status f4b3ba1e^ f4b3ba1e`; `research_retarget.py rewrite --moves ... --root <clone at f4b3ba1e> --dry-run --report $SP/t14-rewrite-dry.json` | 33 renames, all `R100` (lenses 20, records 5, findings 4, reviews 4). Rewrite dry run on the move-only tree: 26 files, 559 replacements, 0 unresolved, 126 excluded of which 79 frozen. Same as Task 8. |
| frozen and pinned digests | independent script: for each of the 174 ledger rows, `git show HEAD:<proposed_path> \| sha256sum` against the ledger sha256 | 78/78 frozen match, 12/12 pinned match, 106/174 of all files match (the other 68 are the 68 listed Markdown edits of the first reorganization). `HEAD` is `645c2bf3`. |
| lint, four reading documents | `.venv/bin/python scripts/check_reading_edition.py <file> --sheet research/audits/root-cause-2026-11-04/reference-sheet.md --sources <every .md under findings/ lenses/ reviews/ records/> research/WEB_RESEARCH_CANDIDATES.md research/programme/evidence-standard/2026-10-23-forward-paper-standard.md --max-words <cap>` | all four exit 0. `README.md`: 586 counted words, cap 1400. `findings/2026-11-05-findings-plain-english.md`: 2,878, cap 3000 (ruling P5). `reviews/2026-11-05-direction-reviews-consolidated.md`: 3,478, cap 3500 (ruling P6). `reference-sheet.md`: 542, cap 100000 (no cap in practice). The count is the lint's own measure of prose, which leaves out tables, code spans and fences. Each file excludes itself from its sources. |
| checker | `scripts/check_lens_report.py --paths-only <file>` on each of the 107 tracked `research/**/*.md` | 12 fail, the same 12 files as section 5 (5 placeholder files, 3 frozen move-caused, the `research/event-timing-audit` branch-name false positive, 4 pre-existing): `WEB_RESEARCH_CANDIDATES.md`, the three allocation briefs and reports (`cross-asset-sleeve-brief`, `month-end-treasury-sleeve`, `static-fractional-leverage-brief`), `ggtrader_paper_review_20260909.md`, the comprehensive audit, `2026-10-21-event-timing-audit.md`, `event-timing-audit-implementation.md`, the Argon synthesis review (now under `reviews/`), `2026-10-24-intraday-feasibility.md`, `2026-10-25-intraday-stage0a-contract.md` and `local-implementation-prompt-TEMPLATE.md`. The 2 files added since section 7 (the reference sheet and the plain-English findings) pass. So do the rewritten README, the rewritten consolidated review, `research/README.md` and `RESEARCH_INDEX.md`. |
| full suite | `scripts/run_tests.sh -q` | 1731 passed, 30 deselected, 1 warning, 292.87 s, exit 0 (1694 at Task 8, 1633 at section 7). No exit 137. |
| ruff | `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | `All checks passed!`; 287 files already formatted (285 at section 7). |
| working tree | `git status --short` | only the pre-existing untracked `data/external/event_timing_audit/` |

`research_retarget.py verify` counts every untracked file as a change, so on the live checkout it needs one `--allow` per file under `data/external/event_timing_audit/` (370 files). That is parked for the final fix wave. The clone avoids it.

### 8.2 Relative links, base `2678a493` against this tree

Scripts: `SP/linkcheck.py` (every `research/...` token, inline link, reference definition and HTML `href` in tracked Markdown, resolved against the file's directory) and `SP/linkdelta.py`. Occurrences: 2,367 at the base, 2,434 now.

- **Frozen files: 0 newly broken links.** The 19 dead links in 4 frozen event-timing files (section 7.3) are unchanged.
- **First-reorganization documents: 6 links fixed.** Six tokens (this record's guard baseline and Argon review paths, plan lines 138, 178 and 203, spec line 130) were the only newly broken links the sub-folder move caused. They resolve again.
- **Net dead occurrences, counted per file** (directory tokens ending in `/` left out, because the script cannot resolve them): 216 at the base, 225 before this section was written. The difference is 9 and it is all in four older or new plan files. This section itself names branch names that the link script reads as paths, so it adds dead tokens to this record. The count above is for the tree before this section, and it shifts whenever this record is edited. By file:

| File | Base to now | Why |
|---|---|---|
| `docs/superpowers/plans/2026-10-20-momentum-retest.md` | 1 to 5 | Task 5 retargeted four output paths (`_momentum_phase2_results.json` and `_momentum_phase2_curves.csv`, in a step and a command block) to the new folder. They are planned outputs that were never created, so they were dead before and are dead now under a new name. The old text did not count as a `research/` token. The one at the base is the branch name `research/momentum-retest`. |
| `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` | 2 to 3 | The file was restored to its original bytes (7.6), so its three relative links to the old folder are dead again. The two dead output tokens it had at the base no longer match the `research/` pattern. |
| `docs/superpowers/plans/2026-11-04-root-cause-investigation.md` | 0 to 2 | Line 247 names two template output paths (`.../2026-11-04-phase2-rerun-<slug>.md` and `.../_phase2_...`) for work that was not run. Task 5 gave them the folder's current prefix. |
| `docs/superpowers/plans/2026-11-05-research-readability-and-subfolders.md` | 0 to 2 | New file. Both are the branch name `research/readability-and-subfolders`. |

### 8.3 Survivors of the old research folder name

`git grep -I -n "docs/research"` over tracked files at `645c2bf3` plus the uncommitted edits, before this section was written: **2,974 hits in 66 files** (7.6: 2,971 in 66). No live survivor. The change is +7 in `tests/scripts/test_research_retarget.py` (new, fixture text for the second-move tool), -3 in `docs/superpowers/plans/2026-10-20-momentum-retest.md` (Task 5) and -1 in `docs/superpowers/plans/2026-11-04-root-cause-investigation.md` (Task 5).

| Category | Hits | Files | Change since 7.6 |
|---|---:|---:|---|
| Frozen provenance | 1,794 | 27 | none |
| Historical citation | 225 | 18 | none |
| Navigation resolver (`research/MOVED.csv`) | 174 | 1 | none |
| Spec/plan text: this reorganization (ledger 175, manifest 174, mapping draft 174, branch additions 82, tool tests 65, the second-move tool tests 7, tool 26, plan 16, branch recipe 16, this record 16, spec 9, `test_driver_default_paths.py` 3) | 763 | 12 | +7 hits and +1 file (the second-move tool tests) |
| Spec/plan text: older plans (restored plan 5, paper-trading remediation 3, root-cause investigation 1, followups master 1, reliability next steps 1) | 11 | 5 | momentum retest plan out (-3, -1 file), root-cause plan -1 |
| Spec/plan text: the readability-and-sub-folders plan | 4 | 1 | none |
| Descriptive mention of the move (`README.md` 1, `docs/changelog.md` 2) | 3 | 2 | none |
| **Total** | **2,974** | **66** | 2,193 under `research/` (46 files) + 781 elsewhere (20 files) |

The new changelog entry avoids the literal old path and adds no hit. This section adds one (the `git grep` command above), so with it the tree holds 2,975 hits in 66 files and this record's own count is 17. The table counts the tree before this section, as 7.2 and 7.6 did, and this record's own count shifts whenever it is edited.

## 9. Executive summary pass

Dated 2026-11-05, visual-reports plan Task 5 (`docs/superpowers/plans/2026-11-05-research-visual-reports.md`), the visual-reports branch. Figures were recomputed on the tree at the executive summary commits (`70b20f24` and its fidelity fix) plus the uncommitted edits of this task (the folder README, the index, the research README, the changelog and this record). The comparison base for links is `7e66b5ba`, the tip of `main` when the branch was cut. `SP` is the session scratch directory. The gate figures below were re-run after the final fix wave (summary wording and headings, the folder `README.md` figures table, the options chart, two record corrections), so the counts are those of that tree.

| Gate | Command | Result |
|---|---|---|
| verify final | `.venv/bin/python scripts/research_reorg.py verify --stage final --manifest $M --ledger $L --rewrite-manifest $R` with seven `--require-new` paths: `README.md`, `MOVED.csv`, the folder `README.md`, the consolidated review, the reference sheet, the plain-English findings and `audits/root-cause-2026-11-04/2026-11-05-executive-summary.md` | exit 0: `OK stage=final 174 manifest files, 68 listed edits, 185 tracked files under research/` (174 plus 11 allow-listed new files: the 7 paths above and the 4 figures) |
| figure traceability | `scripts/run_tests.sh -q tests/scripts/test_root_cause_figures.py` | 16 passed |
| PNG regeneration | `.venv/bin/python scripts/root_cause_figures.py --only options-at-a-glance`, run twice (second run into a scratch folder) | `options-at-a-glance.png` re-rendered for the Option 2 stop-gate wording and byte-identical on the second render. The other three PNGs are unchanged (same md5 as before). |
| lint, scan rules on | `.venv/bin/python scripts/check_reading_edition.py <file> --sheet research/audits/root-cause-2026-11-04/reference-sheet.md --sources <every .md under findings/ lenses/ reviews/ records/> research/WEB_RESEARCH_CANDIDATES.md research/programme/evidence-standard/2026-10-23-forward-paper-standard.md --max-words <cap>` | all exit 0. Summary: 1,179 counted words, cap 1200 (ruling V9). The first 191 are the one-minute layer. The Momentum bullet is dropped to fit. Folder `README.md`: 605, cap 1400. Findings: 3,167, cap 3300. Consolidated review: 4,087, cap 4200. Reference sheet (`--no-scan-rules`, no standard in the sources): 542, cap 9000. The summary's images and image links resolve from its folder. |
| paths only | `scripts/check_lens_report.py --paths-only <file>` | exit 0 for the summary, the folder `README.md`, `research/README.md` and `research/RESEARCH_INDEX.md`. `docs/changelog.md` exits 1 with 41 dead historical paths, the same 41 at `HEAD` before this task (the new bullet adds none). |
| full suite | `scripts/run_tests.sh -q` | 1770 passed, 30 deselected, 1 warning, 323.22 s, exit 0 (1731 at section 8) |
| ruff | `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | `All checks passed!`; 289 files already formatted |
| frozen and pinned digests | for each of the 174 ledger rows, `git show HEAD:<proposed_path> \| sha256sum` against the ledger sha256 | 78/78 frozen match, 12/12 pinned match, 106/174 of all files match (the other 68 are the listed Markdown edits of the first reorganization), as in section 8 |
| working tree | `git status --short` | only the pre-existing untracked `data/external/event_timing_audit/` |

### 9.1 Relative links, base `7e66b5ba` against this tree

Scripts: `SP/linkcheck.py` and `SP/linkdelta.py`, read before use, run with an identity move map because nothing moved. Occurrences: 2,451 at the base, 2,476 on the tree before this section and 2,481 with it. Index-matched delta (final run, with this section): 0 links broken in mutable files, 0 in frozen files, 0 newly resolving. Net dead occurrences, counted per file with directory tokens left out: 264 at the base and 264 now, and the set of dead (file, token) pairs is identical. The 25 added occurrences before this section all resolve: 8 in the summary, 8 in the visual-reports plan, 2 in the index, 2 in the folder `README.md`, 1 in the research `README.md`, 3 in the findings and 1 in the consolidated review (the last two from earlier tasks). This section adds 5 resolving occurrences to this record. It names no dead path, so the dead count does not move.

## 10. Intraday branch translation

Dated 2026-11-05 to 2026-11-06. Branch `translate/intraday-stage0a` (worktree `/home/flynn/.t3/worktrees/ggTrader/translate-intraday-stage0a`), plan `docs/superpowers/plans/2026-11-05-intraday-branch-translation.md`, recipe `docs/superpowers/plans/2026-11-05-research-reorg-branch-recipe.md`. The branch was made from the unmerged intraday research branch (tip `58919c09`) and `main` (`6774b4d3`) was merged into it (merge commit `fc93fa0c`, with `merge.directoryRenames=false`). Nothing is merged or pushed. In plain words: 74 intraday report files were moved from the old `docs` research folder into `research/intraday/`, every reference to them was updated, and the checks below show that nothing that must stay byte-for-byte changed.

Paths used below: `BM` = `docs/superpowers/plans/2026-11-05-intraday-translation-manifest.csv`, `BL` = `docs/superpowers/plans/2026-11-05-intraday-translation-ledger.json`, `BR` = `docs/superpowers/plans/2026-11-05-intraday-translation-rewrite-manifest.json`. `PY` = `/home/flynn/ggTrader/.venv/bin/python` (the worktree has a `.venv` link that is never staged). `UM`, `UL`, `UR` = the union manifest, union ledger and union rewrite manifest (scratch files, built as the plan's Task 3 Step 5 says): `UM` is the first reorganization's 174 rows plus the 74 branch rows, `UL` is both ledgers' `files` lists joined, `UR` is the first reorganization's listed edits plus the 12 listed in `BR` (81 files).

### 10.1 Decisions applied (owner chose the recipe defaults)

| Ruling | What was done |
|---|---|
| D1 path-only code edits | Applied (nine path literals in five files, section 10.4) and recorded in `docs/changelog.md`. The recorded r8 validation (`code_sha256` `f0b85690...`) is not reusable because the hashed code changed. |
| D2 three hash-quoted briefs pinned | `2026-10-25-intraday-feasibility-r8.md`, `2026-10-25-intraday-prospective-observation.md` and `2026-10-25-intraday-opus-verification-manifest.md` moved byte-for-byte and are in no rewrite scope. The tool already carried them as `BRANCH_PINNED_PATHS` (fifteen pinned paths with the first reorganization's twelve). |
| D3 stale hash annotated | The 2026-10-24 feasibility brief stays as it is on `main`. `_intraday_probe.json` (frozen, moved unchanged) records its hash as `51162768...`; the brief at `research/intraday/spy-momentum-feasibility/2026-10-24-intraday-feasibility.md` hashes to `33c00494...` because the first reorganization rewrote its links. Provenance only: `intraday_probe.py` writes the hash and nothing reads it back. The Stage 0A contract is unchanged (`cbb7354a...`). Recorded here and in the changelog; the brief was not edited. |
| D4 three new folders | `research/intraday/spy-momentum-prospective-observer` (17 files), `research/intraday/artifacts-2026-10-25-intraday-gap` (2) and `research/intraday/artifacts-2026-10-26-intraday-readiness` (25); the other 30 files went to the existing `research/intraday/spy-momentum-feasibility`. Paths follow the additions CSV's `proposed_path` column exactly. |

### 10.2 The three commits

| Commit | Content |
|---|---|
| `eeab3be3` docs: add the intraday translation manifest and ledger | 74-row manifest (kinds: 20 report, 5 brief, 22 result, 27 artifact); `manifest-check` `OK 74 rows`; ledger `OK ledger 74 files, 22 referencing files` (sha256 of all 74 before the move; 52 frozen, 3 of them pinned). |
| `c62e6c81` refactor: move intraday research into research/ (move only) | `move --dry-run`: `DRY-RUN 74 moves planned, 0 basename changes, 0 collisions`; `move`: `MOVED 74 files, 0 basename changes`. `git show -M --name-status --format= HEAD`: 74 lines, all `R100`, nothing else. |
| `fe3763b7` refactor: point intraday references at research/ | the reference rewrite and the hand edits (sections 10.3 and 10.4). |
| the records commit (this one) | `research/MOVED.csv`, `research/README.md`, `research/RESEARCH_INDEX.md`, `docs/changelog.md`, this section, the recipe corrections. |

### 10.3 Reference rewrite, by scope (each run with `--dry-run --report` first; every unresolved item read)

| Scope | Command | Files changed | Replacements | Unresolved |
|---|---|---|---|---|
| moved-markdown | `rewrite --manifest $BM --also-map docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv --scope moved-markdown` | 12 (of 22 mutable moved reports; 52 frozen files skipped) | 56 | 3, all `docs/research/_intraday_momentum_results.json` (a file never written); fixed by hand |
| fixed-markdown | `rewrite --manifest $UM --scope fixed-markdown --exclude deploy/intraday_observer` | 7: `README.md` 2, `docs/architecture.md` 3, `docs/changelog.md` 11, `docs/cli_reference.md` 8, `docs/next_steps.md` 2, two intraday plans 5 and 4 | 35 | 22 (below) |
| code | `rewrite --manifest $UM --scope code` | 5 | 8 | 13 (below) |

A first dry run of the last two scopes with the branch-only manifest gave 6 code replacements and extra "not in manifest" items; the 2026-10-24 brief and the Stage 0A contract live in the first reorganization's manifest, so those two scopes were run with the union manifest `UM`, as the recipe says. The moved-markdown run used the branch manifest plus `--also-map` of the first reorganization's manifest.

Fixed-markdown unresolved (22), all read: 3 are the `README.md` and changelog sentences that describe the first move on purpose; 3 are the "Create" placeholders in `2026-09-11-paper-trading-remediation.md` and 3 more are other older plans' globs (`2026-11-02-research-followups-master.md`, `2026-11-04-root-cause-investigation.md` twice); 2 are files an older plan only planned to create (`2026-11-02-research-reliability-next-steps.md`); 8 are the meta text of this translation's own plan and the readability plan; 1 is `docs/cli_reference.md:116`, 2 are the plans' bare prose paths (all three fixed by hand, section 10.4). None is an intraday path left behind.

Code unresolved (13): `scripts/intraday_momentum_research.py:20` (hand edit); `tests/lab/test_intraday_momentum_data.py:17` (`parents[2]` arithmetic, not a research path); `tests/scripts/test_driver_default_paths.py` four lines and `tests/scripts/test_research_retarget.py` seven lines, which are the first reorganization's tests and describe the old location on purpose.

### 10.4 Hand edits (path only)

| File | Before | After |
|---|---|---|
| `scripts/intraday_momentum_research.py` | `DEFAULT_VALIDATION = PROJECT_ROOT / "research/intraday/spy-momentum-feasibility/_intraday_momentum_validation.json"` (tool output, over 100 columns) and `DEFAULT_RESULT = PROJECT_ROOT / "docs/research/_intraday_momentum_results.json"` | `STUDY_DIR = PROJECT_ROOT / "research/intraday/spy-momentum-feasibility"`, `DEFAULT_VALIDATION = STUDY_DIR / "_intraday_momentum_validation.json"`, `DEFAULT_RESULT = STUDY_DIR / "_intraday_momentum_results.json"` |
| `src/ggTrader/lab/intraday_probe.py` | `DEFAULT_OUTPUT` and the two `DOCUMENTS` entries as long `PROJECT_ROOT / "research/..."` literals (over 100 columns) | `STUDY_DIR = PROJECT_ROOT / "research/intraday/spy-momentum-feasibility"`, `DEFAULT_OUTPUT = STUDY_DIR / "_intraday_probe.json"`, `STUDY_DIR / "2026-10-24-intraday-feasibility.md"`, `STUDY_DIR / "2026-10-25-intraday-stage0a-contract.md"` |
| `src/ggTrader/lab/intraday_momentum.py` | `PROTOCOL_PATH` and `VENDOR_REVIEW_PATH` as long literals | `STUDY_DIR = PROJECT_ROOT / "research/intraday/spy-momentum-feasibility"`, `PROTOCOL_PATH = STUDY_DIR / "2026-10-25-intraday-feasibility-r8.md"`, `VENDOR_REVIEW_PATH = STUDY_DIR / "_intraday_r8_vendor_adjustment_review_2026-10-25.json"` |
| `tests/lab/test_intraday_observer.py` | one 102-column line `/ "research/intraday/artifacts-2026-10-25-intraday-gap/spdr-etf-historical-distributions.xlsx"` | the same path split in two string parts (`/ "research/intraday/artifacts-2026-10-25-intraday-gap"` then `/ "spdr-etf-historical-distributions.xlsx"`) |
| `scripts/intraday_coverage_probe.py` | `docs/research/_intraday_probe.json` | `research/intraday/spy-momentum-feasibility/_intraday_probe.json` (97 columns, tool output only) |
| `docs/cli_reference.md:116` | `docs/research/_intraday_momentum_results.json` | `research/intraday/spy-momentum-feasibility/_intraday_momentum_results.json` |
| `docs/superpowers/plans/2026-10-25-intraday-r8-implementation.md:11` | `**Spec:** docs/research/briefs/2026-10-25-intraday-feasibility-r8.md` | `**Spec:** research/intraday/spy-momentum-feasibility/2026-10-25-intraday-feasibility-r8.md` |
| `docs/superpowers/plans/2026-10-25-intraday-stage0a.md:3` | `Spec: docs/research/briefs/2026-10-25-intraday-stage0a-contract.md` | `Spec: research/intraday/spy-momentum-feasibility/2026-10-25-intraday-stage0a-contract.md` |
| three moved reports (`...overall-plan-argon-review.md`, `...r8-implementation.md`, `...r8-validation-unresolved.md`) | `docs/research/_intraday_momentum_results.json` | `research/intraday/spy-momentum-feasibility/_intraday_momentum_results.json` |
| `docs/changelog.md` | (the recipe's "one directory citation") | none was left: the merged changelog's only remaining old-folder mentions are the first reorganization's own entry, which describes the move on purpose. A new entry for this translation was added at the top. |

`deploy/intraday_observer/OPERATOR_RUNBOOK.md` (old paths at lines 11 and 156) was excluded and not edited: the live observer runs the pinned release `280d4a0`; update the runbook when a new release is cut. `docs/next_steps.md`'s intraday item had two links rewritten (paths only); its dates and wording are stale and need an owner-facing wording pass, which this translation did not do.

`_intraday_momentum_results.json` has never been written (the r8 closure report says scoring did not run). The path now names the study folder; the reports that cite it as "never created" still say so, and `scripts/check_lens_report.py --paths-only` flags it in three moved reports, exactly as it flagged the old path on the original branch.

### 10.5 Gates

All run in the worktree after the records were written, with `PYTHONPATH=<worktree>/src` and `GGT_TEST_PYTHON=/home/flynn/ggTrader/.venv/bin/python` (`ggTrader.__file__` was checked to point inside the worktree, not at the main checkout the editable install names).

| Gate | Command | Result |
|---|---|---|
| nothing left in the old folder | `git ls-files docs/research` and `ls docs/research` | no tracked file; the folder does not exist |
| moved, branch mode | `$PY scripts/research_reorg.py verify --stage moved --manifest $BM --ledger $BL --branch-mode` (run after the move commit, before the rewrite) | exit 0: `OK stage=moved 74 files byte-identical to the ledger (branch mode)` |
| rewritten and final, branch mode | same with `--stage rewritten` and `--stage final`, plus `--rewrite-manifest $BR` (run after the rewrite commit, before the records) | exit 0: `OK stage=rewritten 74 files checked, 12 listed edits (branch mode)`; `OK stage=final 74 manifest files, 12 listed edits, 259 tracked files under research/ (branch mode)` |
| union gate, no branch mode | `$PY scripts/research_reorg.py verify --stage final --manifest $UM --ledger $UL --rewrite-manifest $UR` with the seven `--require-new` paths (`README.md`, `MOVED.csv`, the root-cause `README.md`, `reviews/2026-11-05-direction-reviews-consolidated.md`, `reference-sheet.md`, `findings/2026-11-05-findings-plain-english.md`, `2026-11-05-executive-summary.md`) | exit 0: `OK stage=final 248 manifest files, 81 listed edits, 259 tracked files under research/`. No `--allow-new` was needed: the records step adds no new file under `research/` (it edits `README.md`, `MOVED.csv` and `RESEARCH_INDEX.md`). The pinned count is derived as 15 (12 and 3). `--stage rewritten` on the union fails by design after the records step (the root-cause reading files are "unexpected" to it), as sections 7 and 9 already say; the branch-mode `rewritten` run above was taken before the records. |
| frozen and pinned bytes | independent Python: sha256 of each of the 74 files on disk against the ledger `BL` | 52 frozen and 3 pinned all match; 62 of 74 files match, the other 12 are exactly the 12 listed Markdown edits |
| r8 protocol hash | `PYTHONPATH=<worktree>/src $PY -c "from ggTrader.lab import intraday_momentum as m; print(m.protocol_sha256())"` | `be395d950c1da7abf010acf90ffbe461db597672a21dd72b73c0294ba7b45b85` (the code constant is unchanged and the moved r8 brief hashes to it) |
| focused intraday tests | `scripts/run_tests.sh tests/lab/test_intraday_probe.py tests/lab/test_intraday_momentum.py tests/lab/test_intraday_momentum_data.py tests/lab/test_intraday_observer.py -q` | 470 passed (131.9 s). Before the path edit the merge left 464 passed and 6 failed, all in `test_intraday_probe.py` (the old brief paths). |
| scripts tests | `scripts/run_tests.sh tests/scripts -q` | 299 passed (9.2 s), run again after the last documentation edit (section 10.6) |
| full suite, this branch | `scripts/run_tests.sh -q` | 2250 passed, 30 deselected, 1 warning, 449 s, exit 0 |
| full suite, `main` baseline | `scripts/run_tests.sh -q` in the main checkout (`6774b4d3`) | 1770 passed, 30 deselected, 1 warning, 310 s, exit 0 |
| tests failing only here | compare the two runs | none: zero failures in either run, so no environment-only failure (the four `.venv`/`ruff`/`run_capped` tests pass with the `.venv` link). The branch runs 480 more tests than `main` because it carries the intraday tests and the branch-mode tool tests. |
| ruff | `ruff check .` and `ruff format --check .` | `All checks passed!`; 301 files already formatted |
| checker | `scripts/check_lens_report.py --paths-only` on the 22 mutable moved reports | 19 exit 0; 3 exit 1, all with the same single complaint, `cited path does not exist: research/intraday/spy-momentum-feasibility/_intraday_momentum_results.json` (`...overall-plan-argon-review.md`, `...r8-implementation.md`, `...r8-validation-unresolved.md`). Pre-existing, not move-caused: on the original branch the same three files fail on `docs/research/_intraday_momentum_results.json`, a file that was never written. `research/README.md` and `research/RESEARCH_INDEX.md` exit 0; `docs/changelog.md` exits 1 with the same 48 complaint lines before and after the new entry (none mention intraday; they are older entries' unresolvable citations). |
| original branch untouched | `git rev-parse refs/heads/research/intraday-stage0a backup/intraday-stage0a-pre-translate`; `git -C <original worktree> status --short` | both `58919c098bb5680a8ec2df4b2490a76ebfe6a4b6`; the original worktree shows only its untracked `.venv` and `data/external/intraday_momentum_r8/` |
| live observer untouched | `ls -la --time-style=full-iso ~/.local/share/ggtrader` and the user unit files; `systemctl --user list-timers --all` | release tree and the eight unit files carry their 2026-10-26 timestamps; the four `ggtrader-intraday-observer-*` timers (recover, inputs, decide, archive) are listed and were not touched; `git diff 58919c09 HEAD -- deploy` is empty (nothing under `deploy/` differs from the original branch tip) |
| working tree | `git status --short` | only the untracked `.venv` link |

### 10.6 Scripts tests after the final edit

`scripts/run_tests.sh tests/scripts -q`: 299 passed in 9.2 s, run after the recipe corrections and this record were written.
