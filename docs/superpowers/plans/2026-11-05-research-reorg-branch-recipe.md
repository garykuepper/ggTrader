# Branch translation recipe: research reorganization

**Status: applied to the intraday branch on 2026-11-05 to 2026-11-06** (branch
`translate/intraday-stage0a`; record in section 10 of
`docs/superpowers/plans/2026-11-05-research-reorg-verification.md`). The momentum branch was
not translated, as section 5 says. The text below is the recipe as written, with two
corrections found while applying it (section 3 step 3 and the conflict table in section 4).
It was applied on the owner's go-ahead after the reorganization merged to `main` (locally).
Companion mapping: `docs/superpowers/plans/2026-11-05-research-reorg-branch-additions.csv`
(82 rows: 74 intraday, 8 momentum). Facts below come from read-only git queries on
2026-11-05; re-run the scan (section 1) before applying if either tip or `main` moved.

## 1. What each branch carries

| | branch research/intraday-stage0a | branch research/momentum-retest |
|---|---|---|
| Tip / merge-base with `main` | 58919c09 / 86225707 | d0e30056 / 2fb00094 |
| Ahead / behind `main` (42990c74) | 33 / 77 commits | 11 / 174 commits |
| Worktree | in use at `/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a` | none |
| docs/research paths vs merge-base | 74 added, 0 modified (no overlap with any manifest file) | 4 added, 4 modified, all 8 in the manifest |
| Unique content | all 74 files; 26 more added outside docs/research (code, tests, `deploy/`, plans) | none: 5 blobs identical to `main` (landed as bf0e8279), 3 are older text that `main` revised later |
| Code citing the old path | 5 files, 9 literals (section 4) | none beyond what `main` already has |

Rebase point: `main` is 42990c74 and does not contain the reorganization (branch
`reorg/research-top-level`, tip e83257ea at time of writing). The recipe applies to the tip of `main` **after**
the reorganization merges, never to the reorg branch itself. Read-only scan commands:
`git diff --name-status main...<branch> -- docs/research` and
`git diff main...<branch> -U0 | grep docs/research`.

## 2. Preconditions

1. The reorganization is merged to `main` and its verification passed.
2. `git worktree list` shows the intraday worktree; its owner agrees. Do not edit, check
   out in, or move that branch from this checkout. Work in a new worktree (step 2).
3. Both branch tips match the shas above, or this recipe's scan is re-run.
4. The decisions in section 6 are answered. D1 and D2 are blocking for intraday.

## 3. Steps (intraday; momentum is covered in section 5)

1. **Record and back up.** Note the tip (58919c09) and plan a backup ref
   (`git branch backup/intraday-stage0a-pre-translate 58919c09`; do not create it until applying).
2. **Isolate.** Create a new branch and worktree from that tip (`translate/intraday-stage0a`).
   The original branch and worktree stay untouched until the owner agrees to move them.
3. **Merge, do not rebase.** Merge rather than rebase because the branch cites
   its own commit shas as evidence (4ab01d1, 937e487, 8ea98fc, 280d4a0, the deployed release
   directory name), and a rebase would rewrite all 33. The branch modifies no existing
   docs/research file, so git's rename detection carries the moved files cleanly. Expected
   conflicts are in the table in section 4, not in research files. Run
   `git -c merge.directoryRenames=false merge main` in the new worktree. Do not use a plain
   `git merge main`: git's default directory-rename detection sees main's moves out of the old
   research folder, proposes moving all 74 branch files into `research/audits/event-timing/`,
   and silently puts them in the wrong place.
4. **Build the branch-only manifest** the tool reads (`current_path, proposed_path, folder,
   kind, notes`) from the 74 intraday CSV rows: `current_path` = `branch_path`, `folder` = parent
   of `proposed_path`, `kind` = report, result, brief or artifact. Dry-checked in memory: 74 rows
   load and pass the tool's layout rules. Then `scripts/research_reorg.py manifest-check
   --manifest <branch-manifest>` must pass: after the merge, docs/research holds exactly the
   74 added files.
5. **Extend the tool first (separate reviewed commit). Not yet implemented.** The current tool
   covers only the 174 reorg rows, so new branch files cannot be moved or mapped without this:
   - E1: add the three hash-pinned briefs to `PINNED_PATHS` (see D2); pinned count becomes 15.
   - E2: `rewrite --also-map <csv>` (does not exist yet) supplies extra mapping rows (the reorg manifest) without
     making them targets. Needed because re-running `moved-markdown` over the reorg rows would
     re-resolve already-rewritten links against their old locations.
   - E3: a branch mode for `verify` (does not exist yet) that skips the whole-tree count checks.
   `fixed-markdown` and `code` scopes need no extension: pass the union manifest
   (`main` manifest rows plus branch rows).
6. **Move.** `ledger --manifest <branch-manifest> --out <ledger>`, then `move ... --dry-run`,
   then `move`. Append the 74 pairs to `research/MOVED.csv` (the `moved-csv` subcommand writes
   a whole file, so merge by hand or use the union manifest).
7. **Rewrite**, each with `--dry-run --report` first, then for real:
   - `rewrite --manifest <branch-manifest> --also-map docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv --scope moved-markdown --report ...` (22 mutable reports and briefs; `--also-map` is E2, does not exist yet)
   - `rewrite --manifest <union> --scope fixed-markdown --exclude deploy/intraday_observer --report ...`
   - `rewrite --manifest <union> --scope code --report ...`
   Simulated in memory on the branch's own (unmerged) text: the code scope makes 8
   replacements and leaves one unresolved (`_intraday_momentum_results.json`, never created);
   the Markdown scopes leave the hand edits listed in section 4. The merged tree needs its own
   dry run, because the reorg already rewrote most mentions in the shared docs.
8. **Hand edits** (section 4 lists exact lines). Then add index entries: the three new folders
   in `research/README.md` and `research/RESEARCH_INDEX.md`.
9. **Gates** (section 5), then hand the translated branch to the owner. Moving the original
   branch ref is a separate owner step.

## 4. Edits and conflicts

### Code and test edits (tool output, then format)

F = `research/intraday/spy-momentum-feasibility`. Seven of the nine lines exceed 100 columns
after substitution (the two `_intraday_probe.json` lines, intraday_probe.py:35 and
intraday_coverage_probe.py:19, are 97 and need no wrap). Run `ruff format` and `ruff check`;
the formatter cannot split a string literal, so where E501 remains introduce a `STUDY_DIR`
constant instead.

```
src/ggTrader/lab/intraday_probe.py:35   docs/research/_intraday_probe.json -> F/_intraday_probe.json
src/ggTrader/lab/intraday_probe.py:37   docs/research/briefs/2026-10-24-intraday-feasibility.md -> F/2026-10-24-intraday-feasibility.md
src/ggTrader/lab/intraday_probe.py:38   docs/research/briefs/2026-10-25-intraday-stage0a-contract.md -> F/2026-10-25-intraday-stage0a-contract.md
src/ggTrader/lab/intraday_momentum.py:57  docs/research/briefs/2026-10-25-intraday-feasibility-r8.md -> F/2026-10-25-intraday-feasibility-r8.md
src/ggTrader/lab/intraday_momentum.py:63  docs/research/_intraday_r8_vendor_adjustment_review_2026-10-25.json -> F/_intraday_r8_vendor_adjustment_review_2026-10-25.json
scripts/intraday_momentum_research.py:19  docs/research/_intraday_momentum_validation.json -> F/_intraday_momentum_validation.json
scripts/intraday_momentum_research.py:20  docs/research/_intraday_momentum_results.json -> F/_intraday_momentum_results.json   (hand: file does not exist yet)
scripts/intraday_coverage_probe.py:19   docs/research/_intraday_probe.json -> F/_intraday_probe.json
tests/lab/test_intraday_observer.py:207 docs/research/artifacts-2026-10-25-intraday-gap/spdr-etf-historical-distributions.xlsx -> research/intraday/artifacts-2026-10-25-intraday-gap/spdr-etf-historical-distributions.xlsx
```

Hand edits in Markdown (the tool leaves these): docs/cli_reference.md:116 (the
`_intraday_momentum_results.json` default, same F path), docs/superpowers/plans/2026-10-25-intraday-r8-implementation.md:11
and 2026-10-25-intraday-stage0a.md:3 (bare prose paths to the r8 brief and the contract),
one directory citation in docs/changelog.md. `deploy/intraday_observer/OPERATOR_RUNBOOK.md`
(lines 11 and 156) is **excluded**: the active observer runs from the pinned release
280d4a0, whose layout keeps the old paths. Update the runbook only when a new release is cut.

### Merge-conflict risk (files the branch modifies that the reorg or `main` also edited)

| File | Branch change | Reorg edit | Expected |
|---|---|---|---|
| docs/next_steps.md | +21 lines | 7 lines | merges cleanly (checked when applied); the intraday item keeps its old dates and wording, which need an owner-facing wording pass; its two path mentions are rewritten in step 7 |
| docs/changelog.md | +87 lines | 40 lines | conflicts at the top of the file (both sides add entries there); keep main's newer entries first, then the branch's; the branch's path mentions are rewritten in step 7 |
| docs/architecture.md | +51 lines | 1 line | conflicted when applied (the one-line edit sits against a branch hunk); kept main's rewritten handoff link and the branch's intraday sections |
| README.md, docs/cli_reference.md | +11, +104 | 3, 3 | probable clean; branch's new mentions rewritten in step 7 |
| pyproject.toml, scripts/run_tests.sh | +2, +2 | none | no reorg conflict |
| scripts and src the branch only adds | new files | none | no conflict; code edits above |

## 5. Gates, momentum, rollback

**Gates on the translated branch:** `git ls-files docs/research` is empty; `verify` stages
(`moved`, `rewritten`, `final`) pass in branch mode (E3, does not exist yet), with the 49 frozen and 3 pinned files
matching the ledger byte-for-byte; `scripts/run_tests.sh tests/lab/test_intraday_probe.py
tests/lab/test_intraday_momentum.py tests/lab/test_intraday_momentum_data.py
tests/lab/test_intraday_observer.py tests/scripts/test_research_reorg.py`, then the full suite via
`scripts/run_tests.sh`; `ruff check .` and `ruff format --check .`; `intraday_momentum.protocol_sha256()`
returns be395d95... (proves the r8 brief moved unchanged); `scripts/check_lens_report.py --paths-only`
on the moved reports with failures classified as pre-existing or move-caused; the
pinned release directory and its timers are unchanged (do not restart them).

**Momentum:** do not translate. Every file it touches is identical to `main` or older text that
`main` revised, so any conflict is resolved by taking the `main` side. Whether to merge it at all
is the owner's call; the 8 CSV rows exist only to document the mapping.

**Rollback:** nothing touches the original branch or worktree until the owner agrees. To
abandon: remove the new worktree and delete `translate/intraday-stage0a`; the backup ref and
the original branch still point at 58919c09. After the original ref is moved, restore with
`git branch -f research/intraday-stage0a backup/intraday-stage0a-pre-translate`. That command
fails while a worktree has the branch checked out and would desync that worktree if forced:
remove or detach the worktree first, with the owner's agreement.

## 6. Decisions for the owner

- **D1 (blocking).** Path-only edits change the code identity the r8 pipeline hashes
  (`code_sha256`, recorded as f0b85690... in the validation results) and, for
  `intraday_probe.py`, the observer's file list. The source says any edit must be a deliberate,
  re-approved change. Options: (a) apply the path-only edits, record them in the changelog,
  never reuse the recorded validation (scoring is unapproved); (b) keep the branch on the old
  layout. The active observer is bound to release 280d4a0 and is unaffected either way.
- **D2 (blocking).** Three branch briefs are quoted by hash elsewhere (r8 protocol, prospective
  observation, Opus verification manifest). They must move byte-for-byte and stay out of every
  rewrite scope (E1). Their relative links will be dead, as for the existing twelve pinned files.
- **D3 (affects the reorg branch now).** `_intraday_probe.json` on the intraday branch records
  `document_hashes.strategy_brief` = 51162768... for the 2026-10-24 feasibility brief. The reorg
  rewrote that brief's links, so its bytes now hash to 33c00494.... The Stage 0A contract is
  unchanged (cbb7354a...). The hash is provenance only: `intraday_probe.py:499` writes it and
  nothing reads it back. Options: (a) freeze the brief byte-for-byte as a 13th pinned path
  (`PINNED_PATHS`, the ledger and the verify expected count change; its relative links go dead);
  (b) accept the new hash and annotate the probe's provenance in the index or MOVED notes; the
  file itself must not be edited. Owner's decision.
- **D4.** Three new folders: `spy-momentum-prospective-observer` (17 files), and two
  `artifacts-*` packages beside the study folders (2 and 25 files; the depth cap forbids nesting
  them). Alternative: put the 17 observer files in `spy-momentum-feasibility` (47 files).
- **D5 (other hash drift on `main`, outside both branches).** The reorg already changed the
  bytes of `docs/superpowers/plans/2026-10-22-event-evidence-resolution.md` (9cc5c1a4... to
  ef8ff7af...; recorded in `_event_evidence_resolution_workbook.json`, which is the only
  file that holds it) and of `scripts/production_replica_audit.py`
  (9d179ca4... to 7e0ac01f...; recorded as `producer.sha256` in
  `_event_audit_detector_provenance.json`). Both are provenance in frozen evidence; a full sweep
  is scheduled in the final review.

## 7. Branch-only paths cited (do not resolve on this checkout)

These exist only on the branches or after translation: the 74 intraday paths and 8 momentum paths
in the CSV; the intraday code, test and deploy files named in section 4; the three new folders
under research/intraday; and `_intraday_momentum_results.json`.
