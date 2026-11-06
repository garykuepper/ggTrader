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
| verify rewritten | same with `--stage rewritten` | exit 0: `OK stage=rewritten 174 files checked, 68 listed edits` |
| verify moved | same with `--stage moved` (no rewrite manifest) | exit 1 by design after Task 9: exactly 68 `sha256 differs from ledger` lines, one per rewritten Markdown file, followed by one `NOTE` line (the exit code is unchanged). `moved` is only valid before the rewrite. |
| frozen digests | independent: `git show HEAD:<path>` piped to sha256 per ledger entry | 78/78 frozen match, 12/12 pinned match; 106/174 of all files match, the other 68 are exactly the 68 listed edits (174 - 106) |
| survivors | `git grep -I -n "docs/research"` (tracked files) | 2,670 hits in 58 files before this record was written (2,679 in 59 with its first version), 0 uncategorized; shifts when this record is re-committed (section 2) |
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
- `verify --stage moved` is only valid right after the move step; after the rewrite it correctly fails with the 68 listed Markdown differences.
- The allow-list matches by pattern, so an unexpected file that happens to match an allowed pattern would pass; the defaults are exact paths.
- The `ruff.toml` exclusion for frozen `.py` artifacts under `research/` was left unchanged (ruling R8d, out of spec scope); a repo-wide `ruff format .` could rewrite them. Optional owner follow-up. Today `ruff format --check .` passes (285 files already formatted).
- Survivor counts in section 2 shift whenever this document, the tool tests or any other file with old-path text is committed; the figures are for the stated commits.
- Results reflect commit `70d37120` (gate, tests and ruff also re-run after fix round 1); Task 14 repeats `verify --stage final` with the `--require-new` flags above once Tasks 11 and 12 have added their files, plus the survivors grep (`git grep`), the checker table and the guard diff.
