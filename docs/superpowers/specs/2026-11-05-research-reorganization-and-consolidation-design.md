# Research Reorganization and Review Consolidation: Design

**Date:** 2026-11-05 · **Status:** draft for owner approval; builder check done, calls in section 14 · **Type:** repository reorganization plus two new navigation and review documents (no research is re-run, no trading code changes)

## 1. Intent

The owner reads ggTrader's research reports himself and is not a developer. They currently sit in one flat folder, `docs/research/`, with 144 top-level entries and about 70 raw result files mixed in with the reports. Reviewing the 2026-11 root-cause investigation meant opening about 45 loosely related files, including two separate model direction reviews and a verification of one of them.

The owner's decisions (his own words, over the conversation of 2026-11-05):
1. Research reports are different from ordinary documentation and should live at the **top level** of the repository in a folder called `research/`, not under `docs/`. This applies to **all** research reports, not just the latest investigation.
2. `research/` has **subfolders**, because there is a variety of research.
3. Keep the folders **relatively flat and not too deep**.
4. Categories must **not overlap**: a reader should know in one guess where a study is.
5. He suggested a general **audits** category, and confirmed the root-cause investigation is an audit.
6. The two model direction reviews (the Opus senior-quant review, and the Argon review with its verification) should be **combined into one report** so they are easy to review. Detailed reports stay separate, like the independent challenge.
7. He wants a start-here index for the investigation.

Why now: the investigation just finished and its reports, reviews and evidence are the first thing he has to read. Every further research arc will land in the same pile unless the structure changes first.

## 2. Goals, non-goals, anti-goals

**Goals**
- Move every file under `docs/research/` to a new top-level `research/` with the layout in section 4.
- Preserve git history (`git mv`), and keep each study's brief, amendments, report and raw results together.
- Add a root `README.md` (where do I start), a `MOVED.csv` (old path to new path for every file), and an investigation start-here README.
- Write one consolidated direction-review report from the Opus review and the verified parts of the Argon review.
- Keep the repository working: tests, lint, the path checker, the Docker image contents and every script that reads or writes research files must behave as before.

**Non-goals**
- No new research, no re-runs, no change to any scientific verdict, number, or conclusion in any existing report.
- No change to `src/` behavior, the live paper trader, cron, the container, `data/`, or the database.
- No change to `docs/` content other than reference updates (architecture, CLI reference, changelog, roadmap, next steps, working task list, `docs/superpowers/` plans and specs stay where they are).
- Moving or merging the unmerged branches `research/intraday-stage0a` or `research/momentum-retest` is not part of this design (ordering is an owner decision, section 9).

**Anti-goals (this would be a failure even if everything moved)**
- Silently rewriting the content of old reviews so they agree with each other, or hiding that an Argon claim was unverified.
- Breaking the checksum-bound provenance of the event-timing audit.
- Publishing research artifacts into the production Docker image.
- Making citations look verified when they are not (for example, a path checker that stops checking research paths).
- A compatibility copy of `docs/research/` that defeats the owner's requirement.
- Any recommendation to retire the sleeve, end paper trading, or shut down the project; taxable-account-only; no paid data (standing owner instructions in `CLAUDE.md`).

## 3. Constraints

- **Folder depth:** at most two folder levels below `research/` (category, then one folder per study). All files of a study sit directly in its folder; no `briefs/` or `results/` folders below a study. (Exception: the five existing `artifacts-*` packages keep their names and contents and count as the study folder.)
- **Names:** every existing basename is kept, with exactly three layout-forced exceptions (section 6). No other historical renaming.
- **Frozen evidence:** twelve research paths are explicitly bound by SHA-256 inside other result files, and the event-timing audit checks three briefs at run time. These files must move byte-for-byte unchanged, and so must their historical embedded path strings. The twelve (paths relative to the old `docs/research/`):
  `briefs/2026-10-21-event-timing-audit.md`, `briefs/2026-10-21-event-timing-audit-sources.md`, `briefs/2026-10-22-event-timing-source-conflict-proposal.md`, `event-timing-audit-implementation.md`, `2026-10-22-event-evidence-resolution-workbook.md`, `_event_evidence_resolution_workbook.json`, `_event_audit_metadata_inputs.json`, `_cost_charter_comparison_results.json`, `_event_evidence_export_file_search.json`, `_insider_identity_inventory_resolution_results.json`, `_insider_identity_inventory_acquisition.json`, `_production_replica_results.json`.
  All historical machine artifacts (the `_*.json` and CSV files) are treated as immutable by default; re-serializing JSON can change hashes.
- **Reference rewrites** are limited to mutable documents and code (section 7). Frozen files keep their old internal links; navigation for them is provided by `MOVED.csv`, the README and the index.
- **Safety rules for implementers:** never execute a script that has not been read; never run a script only to see its `--help` (an accidental run of a driver on 2026-11-04 rewrote price history for 19 symbols); no database writes; the live paper trader and the intraday observer timers are untouched; tests only via `scripts/run_tests.sh`.
- **Models:** per `AGENTS.md` (Claude only; Sonnet orchestrates, Opus for judgment, Haiku for bulk; a stronger model checks Haiku). The Gemini Argon and GPT Sol outputs used to inform this design were opinions, not sources of record.

## 4. The layout

Counts are existing files (174 recursively; four new files are added by this work).

```
research/
  README.md                   NEW  where do I start; link to the current review
  MOVED.csv                   NEW  old path -> new path for every moved file
  RESEARCH_INDEX.md           find a study by question, date, status, alias (A23, PEAD ...)
  RESEARCH_SNAPSHOT.md        generated strategy roster (regenerated by the research-snapshot skill)
  WEB_RESEARCH_CANDIDATES.md  idea backlog
  TEMPLATE-research-report.md report format
  prompts/                    reusable commissioning prompts (2)
  allocation/       (36)  does changing exposure, leverage or a portfolio mix help?
      century-leverage/ static-leverage/ cross-asset-sleeve/ core-spy-blend/ month-end-treasury/ fomc-drift/
      fx-hedge/ commodity-trend/ treasury-curve/ leveraged-index-rotation/ leveraged-trend/ artifacts-2026-09-23-web-batch/
      artifacts-2026-09-28-web-batch/ artifacts-2026-09-30-web-batch/
  audits/           (83)  can existing evidence, shared machinery or recorded conclusions be trusted?
      root-cause-2026-11-04/ (32 + 2 new)   event-timing/ (27)   production-replica/ (10)   rebaseline/ (8)
      research-process/ implementation/ comprehensive-review/ artifacts-2026-09-09-perf-review/
  stock-signals/    (23)  can a stock-selection mechanism work, and is the evidence available?
      factor-screen/ momentum/ net-issuance/ feasibility/ pead/ insider-cluster/ congress-trades/ short-interest/
      short-volume/ max-effect/ index-deletion/ pairs-stat-arb/ headline-sentiment/
  paper-system/     (7)   how the paper system measures, sizes, accounts for and operates its book
      cash-sweep/ indicator-warmup/ position-sizing/ tax-lots/ observation-readiness/
  programme/        (7)   shared research standards and methods
      evidence-standard/ wfo-methods/ initial-research-plan/
  intraday/ (9)  spy-momentum-feasibility/
  crypto/   (3)  execution-venues/ artifacts-2026-10-25-crypto-web-batch/
```

The full file-by-file mapping is the appendix `docs/superpowers/specs/2026-11-05-research-reorganization-mapping-draft.csv` (174 rows: current path, proposed path, folder, kind, notes). It is a **draft from a design review**; the plan's first task must reconcile it against the actual tree (several counts in earlier estimates differed) and produce the approved manifest. No file may be unmapped or have two destinations.

## 5. The ordered placement test (no overlap)

Apply top to bottom; the first match wins. A document that touches two categories gets cross-links and index tags, never two copies.

| Order | Test | Home |
|---|---|---|
| 1 | Reusable navigation, roster, idea backlog or report template? | `research/` root; reusable commissioning text goes to `prompts/` |
| 2 | Companion (brief, amendment, result, review) to an already identified study or investigation? | That study's folder, whatever its kind, asset or file type |
| 3 | Mainly checking the reliability of existing evidence or shared machinery across studies, reconstructing baselines, or auditing a deployed construction? | `audits/` |
| 4 | Establishing a shared experimental standard, or studying research-policy behavior? | `programme/` |
| 5 | An intraday-specific experiment or feasibility study? | `intraday/` |
| 6 | A crypto-specific mechanism or venue study? | `crypto/` |
| 7 | Changing exposure, leverage, index/ETF allocation or a blend of sleeves? | `allocation/` |
| 8 | Operating behavior or accounting of the paper system, without a primary return hypothesis? | `paper-system/` |
| 9 | Selecting individual stocks, screening stock factors, or assessing candidate data availability? | `stock-signals/` |

The root-cause investigation matches test 3 (it audits the whole research record), so its home is `audits/root-cause-2026-11-04/`. Because it is the item under review, the root `README.md` and `RESEARCH_INDEX.md` link to it prominently; there is no second physical copy and no symlink.

Borderline placements already decided (full list in the draft mapping notes): the WFO methodology studies go to `programme/wfo-methods/` (they define what the harness measures); anchor-leak and corrected re-baselines go to `audits/rebaseline/`; the production replica audit, its four amendments and the FOMC clustered result go to `audits/production-replica/`; position sizing and the live indicator warm-up go to `paper-system/`; the core-plus-SPY blend goes to `allocation/`; the 2026-06-28 execution plan goes to `programme/initial-research-plan/`.

## 6. Naming

- Keep every existing basename. Exactly three briefs collide with a report of the same name once they share a study folder, so they gain a `-brief` suffix: `2026-10-23-century-leverage.md`, `2026-09-28-cross-asset-sleeve.md`, `2026-09-30-static-fractional-leverage.md`.
- Going forward: reports `YYYY-MM-DD-topic.md`; registrations `YYYY-MM-DD-topic-brief.md`; amendments `YYYY-MM-DD-topic-amendment.md`; machine outputs `YYYY-MM-DD-topic-results.json` (or a specific suffix for a distinct artifact). Dates live in filenames, not in a year/month folder hierarchy. A changed protocol or a materially new study gets a new dated file; a finished result file is never reused.
- Folder names are short, lowercase, and based on the question the study answers.

## 7. What must be updated (mutable references only)

The path-bearing consumers, verified by read-only search in the design reviews (counts are planning figures to be recounted in plan task 1):
- **Scripts** (about 26 to 27) that write or read raw result files; each needs exact destination defaults for both outputs and input briefs (a plain prefix substitution is not enough, because outputs now belong to different study folders). This includes shell-scheduled `--out` arguments, Python constants and CLI help text. `scripts/lens2_build.py` and the WFO fold-attribution driver assume a common research root for several artifacts and need explicit per-artifact locations.
- **`src/`** (9 files mention the path; only `src/ggTrader/lab/event_timing_audit.py` (lines 47 to 51: three brief paths that are later hashed and checked) matters functionally; two paper-code files are comment-only). Updating the three brief constants is required; old checkpoints may not resume, and that is accepted (section 9).
- **Tests** (about 8 to 9; two actually check paths), **docs** (about 24 outside the research tree), **`AGENTS.md`** section 5 (it still says research reports live in `docs/research/` and are archived under `docs/`; the owner's decision overrides the old rule), and **six skill files** under `.claude/` (five `SKILL.md` plus one helper script). Update the owned sources, not generated copies; `CLAUDE.md` is a symlink to `AGENTS.md`.
- **`scripts/check_lens_report.py`** only verifies cited paths beginning `docs|src|scripts|tests|results`. It must also accept `research/` so citations are checked, with a negative test that a missing `research/...` path fails. Historical aliases for immutable evidence are handled separately and must not make invented paths pass.
- **`.dockerignore`** excludes `docs/` and Markdown but not `research/`; without a new entry about 64 JSON/CSV files, four Python artifacts, text and an archive enter the production image build context. Add the exclusion before or with the move.
- **Relative links** between documents (`../next_steps.md`, `../../superpowers/...`) change differently at each depth; resolve them from each file's new parent.
- Not in the checkout but affected: memory notes outside the repo and any bookmarked GitHub URLs. These cannot be fixed and are harmless.

## 8. The new documents

**8.1 `research/README.md`** (landing page): what `research/` is, the folder list with the one-line question each answers, a "current review" shortcut to the investigation README, and the rule for where new studies go (the ordered test). Short.

**8.2 `research/audits/root-cause-2026-11-04/README.md`** (investigation start-here): a three-item reading route (the synthesis, then the consolidated direction review, then the independent challenge), then a table of the lens reports and evidence files grouped under headings (summary, lenses, reviews, evidence), each with one line. It states plainly that the Phase 2 re-runs were not performed and carries the existing verification warnings about the Argon review. It is a route into the evidence, not another synthesis.

**8.3 `research/audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md`** (about 1,500 words): a new editorial comparison of the two direction reviews, not a concatenation and not a replacement for either. The originals, the verification and the challenge stay separate and linked. Content rules:
- Sources and status stated up front: the Opus review (reliable, single reviewer), the Argon review (a different model family, **unverified as written**; 43 of its 105 citations are broken or unsupported and several specifics are invented), and the Claude verification that says what survives.
- A side-by-side table by topic (paper-trader reliability, the 29 unrun checks, stock-picking power, leverage, tax, crypto, intraday): Opus position, Argon position, verification disposition, status (agree, differ, Argon-only, dropped).
- Consensus, disagreements with the evidence that settles each, and corrections that change interpretation. Each substantive statement cites a source line; only verified Argon claims are used. No new return computations.
- At most three ranked next-step options, each with the question it answers, the cheapest admissible step, prerequisites, a falsifier and a stop gate, framed as options for the owner (standing rules: no retirement or shutdown advice, taxable only, free data only, the paper trader stays frozen unless the owner decides otherwise, the owner's "no price-table writes" decision stays).
- One de-duplicated list of the owner's decisions (the Opus review lists eight; add the synthesis' open items: accept the synthesis, changelog and snapshot update), each with a default and what it unlocks.
- A short "what to ignore" list for the Argon specifics that did not survive verification.

A review step compares the consolidated report against its three sources: every claim traceable, nothing added, softened or dropped without saying so.

**8.4 Pointers:** one dated line at the top of the Opus review and the Argon review saying the consolidated report is the reading copy. Content otherwise unchanged. The Argon file keeps its existing unverified banner.

**8.5 `research/MOVED.csv`** and the index entries: `RESEARCH_INDEX.md` gains the question, main report, brief, evidence, status, date, limitations and aliases for each study dossier. Scientific verdicts are not changed while rewriting navigation.

## 9. Decisions for the owner

These change what gets built; defaults in brackets are the lead's recommendation.
1. **Frozen evidence:** accept that the twelve hash-bound files (and, by default, all historical machine artifacts) keep their old internal paths and links, with navigation and an old-to-new resolver provided outside them, rather than re-hashing or rewriting them. Fully clickable rewritten links in every old file and byte-identical pinned files cannot both be guaranteed. [Accept; immutable evidence plus external navigation.]
2. **Layout:** approve the nine-category, study-folder layout above, with the investigation under `audits/`. [Approve.]
3. **Editorial exception:** authorize the new README files, the index entries and the consolidated direction review (the move itself is otherwise reference-rewrite only). [Authorize.]
4. **Branch ordering:** the unmerged `research/intraday-stage0a` branch adds about 74 files under `docs/research/` and `research/momentum-retest` has its own changes. Either reconcile them first or translate their additions afterwards. [Translate afterwards, from the current branch tips; moving paths must not change either branch's research or operational status.]
5. **Three brief renames** with a `-brief` suffix. [Accept.]
6. **Stub at the old location:** no `docs/research/` stub at all, so nothing in the new layout depends on the old one. [No stub; GitHub links to the old paths will 404.]
7. **Event-timing audit checkpoints:** the audit's runtime resume checks path, code hashes and git revision, so old checkpoints may not resume after the move even though the briefs are byte-identical. [Accept; no resume is planned.]
8. **Push:** whether to push after the move. [Push only when told to; the Docker build publishes on every push to `main`.]

## 10. Verification (what "done" means)

- Same file count and bytes after the move, except explicitly listed reference-only edits and the three renames; every one of the twelve pinned files byte-identical to its pre-move digest; no deletion; no third directory level.
- No file unmapped, none with two destinations; `MOVED.csv` covers every moved file.
- `scripts/check_lens_report.py` (updated) exits 0 on every moved report and on the new documents, and a missing `research/...` path fails its negative test.
- `scripts/lens2_build.py` run into a scratch path reproduces the committed classifications; `scripts/guard_snapshot.sh` shows no difference from the baseline other than the expected `main:` hash line.
- A grep for `docs/research` finds only categorized survivors: frozen provenance, historical citations, or known prior defects. The Docker build context excludes `research/` (verified by inspection, not by building).
- Full test suite via `scripts/run_tests.sh` passes; `ruff check .` and `ruff format --check .` pass; no formatter change to frozen evidence.
- The consolidated report passes a source-traceability review by a fresh reviewer.
- Nothing under `src/` changes behavior; nothing is executed against the database, broker or network.

## 11. Sequence (for the plan to turn into tasks)

1. Reconcile the draft mapping against the actual tree; capture a migration ledger (source revision, hashes, inbound references, existing dead links). Approve the manifest.
2. Prepare consumers on the same branch: `.dockerignore`, the path checker, script defaults, tests, skills, `AGENTS.md`.
3. `git mv` per the manifest (move-only commit, separable from rewrites).
4. Rewrite mutable references; leave frozen sources intact; add the resolver data in `MOVED.csv`.
5. Verify offline (section 10).
6. Write the README files, the index entries and the consolidated report; review.
7. Reconcile the unmerged branches as decided.
8. Commit locally; push only on the owner's word.

## 12. Left to the builder

- Exact implementation of the `MOVED.csv` resolver in the checker (a lookup table is enough).
- The wording of the dossier descriptions in the index and README.
- Ordering of commits within the sequence above, provided the move-only commit stays separable.
- Reconciling the draft mapping's file counts and any file it missed or double-assigns (the manifest from plan task 1 is authoritative).
- The exact `.dockerignore` pattern.

## 13. Effort and risk (planning estimate)

About 13 to 21 hours of focused agent work across two to three working days, dominated by provenance preservation, consumer updates and review; the mechanical move is short. Main risks: breaking hash-bound evidence, admitting research artifacts into the production image, missing a path consumer, and colliding with the unmerged intraday branch. None requires running a research driver.

## 14. Calls made by the lead (settled without the owner; check these)

Added after a fresh builder read the spec and asked seven questions. None changes what is built; each is the lead's call and the owner can override it.

1. **Which files are byte-frozen and which are rewritten.**
   - *Byte-frozen set:* the twelve hash-pinned paths in section 3, every `_*.json`, `_*.csv`, `.txt`, `.py` and `.tgz` file, and the contents of the five `artifacts-*` packages.
   - *Rewritten set:* every other Markdown file, including all the investigation reports the owner will read, so their links work. Relative links are resolved from each file's new parent; backticked `docs/research/...` citations become `research/...` new paths.
   - Each rewrite is produced by one script that also emits a manifest of every file it changed; section 10's "explicitly listed reference-only edits" means exactly that manifest (plus the in-scope code and skill files and the two pointer lines). The byte-identity check compares everything not in the manifest.
2. **How the path checker is used.** `scripts/check_lens_report.py --paths-only` is run on every moved Markdown report and on the new documents; the full heading rules apply only to documents that are lens reports (the consolidated report and READMEs are not). Frozen files are not run through the checker. The old-to-new resolver (`MOVED.csv`) is used only for navigation and never by the checker's missing-path negative test, so an alias can never make an invented path pass.
3. **The unmerged branches.** This work does not touch `research/intraday-stage0a` or `research/momentum-retest`, their worktrees, or the observer. The plan produces a translation recipe only (a mapping for the branch's added files, including a placement for the roughly 74 new intraday files, and a rewrite script). The recipe is applied to a branch only on the owner's separate go-ahead, after that branch's owner has merged or agreed. The observer timers run from a pinned release directory and are unaffected by the move.
4. **Gating and branch.** The plan's first tasks (reconcile the manifest, capture the ledger, dry-run the rewrite) change nothing in the tree. The `git mv` task is held until the owner approves the section 9 decisions. All work happens on a new branch cut from `main` (suggested name `reorg/research-top-level`) and merges to `main` only when the owner says so.
5. **What "reproduces the committed classifications" means for `lens2_build.py`.** The script is run into a scratch path and compared on the classification fields (id, class, rules-out size, verdict type) of its JSON output, not byte-for-byte; the path strings it embeds in text will legitimately differ. The committed `_lens2_sensitivity.json` stays frozen (not regenerated). The script's single research-root constant is replaced by explicit per-artifact locations.
6. **The consolidated report adds no verification.** It restates only what the Opus review states and what the Argon verification marks as surviving. Any needed claim the verification does not cover is marked "not verified". It is one Opus drafting task, then a separate fresh-reviewer traceability check.
7. **The two pointer lines** at the top of the Opus and Argon reviews are allowed edits and appear in the rewrite manifest. The Argon banner's path to the verification file is updated to the new path (that file is not pinned).

## Addendum 2026-11-05b: sub-folders for a large study folder

Added after the first reading of the finished tree showed that one study folder, `audits/root-cause-2026-11-04/`, holds more than 30 files. This is the only change to the section 3 depth rule.

- A study folder with more than about 30 files may group its files one level deeper into sub-folders. Each sub-folder is named for what its files are (never for a number or a phase).
- Only `audits/root-cause-2026-11-04/` uses this today, with the sub-folders `findings/`, `lenses/`, `reviews/` and `records/`.
- The depth rule becomes five path components (`research/<category>/<study>/<sub-folder>/<file>`) for that prefix only. Every other study folder keeps four.
- The placement test in section 5 is unchanged: it picks the category and the study folder, and a sub-folder is chosen inside the study folder afterwards.
- `scripts/research_reorg.py` carries the exception as `DEFAULT_ALLOW_DEEPER`; `verify` and `manifest-check` accept `--allow-deeper PREFIX` (relative to `research/`, repeatable) to add another prefix.
