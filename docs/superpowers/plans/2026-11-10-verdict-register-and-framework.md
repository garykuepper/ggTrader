# Verdict Register and Evidence Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one complete, auditable verdict register with separate evidence and action labels, plus prospective return and defense registration templates.

**Architecture:** Maintain a structured inventory of discovered records and append-only verdict annotations; render a single browsable Markdown register from those records. Original study reports remain authoritative. Reconcile the root-cause audit, snapshot and later study reports, record unresolved and excluded entries, and annotate in verified batches; do not create statistical judgments automatically. Templates specify the fields future briefs must decide without changing the existing governing standard.

**Tech Stack:** Python 3, standard-library JSON and pathlib, Markdown, pytest via `scripts/run_tests.sh`, Ruff.

**Spec:** [2026-11-10-research-verdict-framework-design.md](../specs/2026-11-10-research-verdict-framework-design.md).

## Global Constraints

- Existing historical GO/NO-GO, stopped and unresolved results remain unchanged; annotations are additive and dated. No return tests, data changes, broker access, production changes, or automatic reopening of closed studies.
- The approved 2026-10-23 standard and all existing Part A/B0/B1 gates remain operative unless the owner separately approves a dated amendment. New numerical risk, wealth, evidence and defense thresholds are not supplied by this plan.
- Use free data and taxable-account assumptions; no recommendation to retire the stock sleeve, end paper trading, change account type, or shut down the project.
- Preserve untracked `data/` and existing owner work. Do not hand-edit `research/RESEARCH_SNAPSHOT.md`; use its regeneration workflow if available, otherwise link from the research README and index.
- Evidence labels reflect checked evidence, not model-generated classifications. Missing power alone is unknown power, not automatically invalid. Never assign a label from a historical verdict string alone. Original study records govern if index and source disagree.
- Delegation follows the owner's current ChatGPT quota headroom and model preferences; no private account records in delegate prompts. Fresh review and lead verification do not imply cross-family independence when only ChatGPT is used.

## Review Focus

- Multiple verdicts and annotation revisions can share a study and specification without collisions; duplicate record IDs fail the checker (Task 1).
- Unknown power and missing source are different: keep a valid observed result, and mark genuinely unverifiable claims unresolved (Tasks 1 and 2).
- Original replica model NO-GO, stopped holdout and unverified live parity must stay distinct (Task 2).
- Future evidence/action updates retain older annotations and original source references; a missing or unaccounted inventory record fails validation (Tasks 1–2).
- Historical already-seen data cannot become an untouched holdout or create a new defensive pass by changing objectives (Tasks 2–3).

## File map and data contract

- Create `research/programme/verdict-framework/inventory.json`: explicit inventory cutoff, sources, discovered records and exclusions; no silent deletion. The historical lens audit (`research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `classifications`) is one input, not the whole project.
- Create `research/programme/verdict-framework/annotations.json`: one immutable verdict record per `verdict_id`, and append-only annotation revisions keyed by `annotation_revision_id` with `study_id`, `specification_id`, `stage_id`, and `verdict_id`. Include original verdict, source paths, evidence label, protocol result, action label, dates, rationale, approval and uncertainties. Initial unknown fields remain explicit null or `UNRESOLVED`, never guessed.
- Create `scripts/verdict_register.py`: deterministic `validate(inventory: dict, annotations: dict, root: Path) -> list[str]`, `render(inventory: dict, annotations: dict) -> str`, and CLI `check` / `render`. `check` returns nonzero for duplicate record IDs, broken references, invalid labels or unaccounted inventory rows. `render` writes `research/VERDICT_REGISTER.md` only when validation passes; `check` compares the committed register with rendered output without writing.
- Create `tests/scripts/test_verdict_register.py`: pure fixtures in temporary directories, no real market or DB access; test via `scripts/run_tests.sh`.
- Create `research/VERDICT_REGISTER.md`: canonical browsable table with study/specification, original verdict, registered result, evidence, action and links, plus date, coverage and incomplete/review status.
- Create `research/programme/verdict-framework/2026-11-10-retrospective-review.md`: review protocol, dated coverage table, priority ordering and checked claims; add dated batch logs as work is reviewed. Records and raw source reports are never rewritten.
- Create `research/programme/verdict-framework/TEMPLATE-return-brief.md` and `TEMPLATE-defense-brief.md`: prospective registration templates, explicitly unapproved until each candidate's own brief and any required standard amendment are approved.
- Modify `research/README.md` and `research/RESEARCH_INDEX.md` for navigation; leave snapshot generation to its existing workflow.

### Task 1: Register contract and deterministic verification

**Files:** Create `scripts/verdict_register.py`, `tests/scripts/test_verdict_register.py`, inventory and annotation JSON files and `research/VERDICT_REGISTER.md`.

**Interfaces:** `validate(inventory: dict, annotations: dict, root: Path) -> list[str]`; `render(inventory: dict, annotations: dict) -> str`. Inventory has `cutoff`, `sources`, and `records`; each record has `inventory_id` and `disposition` (`pending`, `included`, or `excluded` with reason). An annotation has immutable `verdict_id`, existing `inventory_id`, `study_id`, `specification_id`, `stage_id`, `original_verdict`, source path, protocol result, and one or more dated revisions, each with unique `annotation_revision_id`, evidence/action labels and rationale. The script checks structure, links and coverage only; it does not assess whether a judgment is correct.

- [ ] Add tests asserting `validate` returns errors for duplicate verdict IDs or duplicate annotation-revision IDs, missing verdict/source links, included rows without a verdict, pending rows incorrectly counted as reviewed, and exclusions without a reason. Assert two annotations of the same verdict with distinct revision IDs pass, as do different stages of a single specification.
- [ ] Add tests asserting render includes separate original verdict, protocol, evidence, action and source columns; reports included, pending and excluded counts and `INCOMPLETE` when any row is pending; preserves both revisions in the detailed record and uses the latest explicitly dated revision in the summary without erasing history. Assert rendering twice is byte-identical and stale generated Markdown makes `check` fail without modifying it.
- [ ] Run `scripts/run_tests.sh tests/scripts/test_verdict_register.py`; expect failure because the module does not exist.
- [ ] Implement the interfaces, CLI and empty valid JSON documents; resolve project root dynamically, use atomic writes for render, and provide `--help`. Reject malformed paths outside the repository or nonexistent source files. Represent missing source evidence in an unresolved ledger record with a valid review note; do not fabricate a source.
- [ ] Run the same test command, `.venv/bin/ruff check scripts/verdict_register.py tests/scripts/test_verdict_register.py` and `.venv/bin/ruff format --check scripts/verdict_register.py tests/scripts/test_verdict_register.py`; expect success. Have a fresh Sol reviewer assess contract and preservation semantics; lead reads the diff and reruns checks.

### Task 2: Inventory, annotate and publish the retrospective register

**Files:** Modify the Task 1 JSON and register; create the dated review protocol and batch records; modify `research/README.md`, `research/RESEARCH_INDEX.md`.

**Interfaces:** Uses Task 1 validation/rendering; produces a coverage-accounted inventory and reviewed, source-linked annotations. Unreviewed rows remain `pending` and visibly incomplete until checked; the table never claims full coverage early.

- [ ] Freeze the inventory cutoff at execution time and record HEAD and dirty paths. Preserve `data/`. Enumerate the lens-2 audit classifications, distinguish its status-only items from tests using its own rules, reconcile with the snapshot roster and research index, and search dated study reports and amendments after the audit cutoff. Record the source and disposition of every discovered item, including duplicates and explicit exclusions. Independent review must sample and verify counts against raw source files; do not silently treat 114 historical rows as 114 independent tests.
- [ ] Write the batch protocol defining source precedence, scope, reviewer and lead checks, stable IDs, and priority: replica and current stock closures, defensive studies, then remaining inventory. Define completion as zero unaccounted discovered items, all included records reviewed, and all exclusions justified. Mark unavailable artifacts or uncertain inference unresolved; document unknown power separately. A closed verdict remains its original action unless a separately authorized decision changes it.
- [ ] For the replica, record separately original daily-bar model NO-GO, stopped/inconclusive holdout, and unverified exact live parity; record its limited power for a modest edge without changing the original primary-window rule. Independently cross-check the original brief, closure amendment, report and audit. Add a fixture/regression asserting these three scopes are never collapsed into a single result.
- [ ] Review and annotate each remaining batch against original briefs, reports, artifacts and the historical audit. For each, record original verdict, protocol result, evidence and action with rationale and sources; mark post-hoc analysis descriptive where appropriate. A newly invented threshold cannot be retroactively preregistered. Lead verifies every figure and samples every mechanical inventory batch; do not calculate new returns or rely on unavailable data.
- [ ] Generate the canonical Markdown with `python scripts/verdict_register.py render`; link it in the research README and index. Run `python scripts/verdict_register.py check` and `scripts/run_tests.sh tests/scripts/test_verdict_register.py`; both must pass. Fresh Sol reviews coverage, representative annotations, source fidelity and historical-verdict preservation. The lead checks findings and reruns checks. If a source is inaccessible, publish the scoped unresolved entry and coverage limitation rather than inventing a label.

### Task 3: Prospective registration templates and owner decisions

**Files:** Create the two templates; update `research/README.md` navigation and the retrospective review's completion status.

**Interfaces:** Return and defense templates consume the record fields and labels from the spec and Task 1; they are blank registration aids, not an approved amendment or authorization to run.

- [ ] Write the return template with candidate ID, research question, fixed dates, exposure, comparator, economic and statistical estimands (distinguish arithmetic excess from CAGR), thresholds/equality, uncertainty and multiplicity, validity and data-fault behavior, costs and tax scenarios, power, stage policy, stops, evidence/protocol/action fields and approval provenance.
- [ ] Write the defense template with all common fields plus one downside endpoint, the wealth-sacrifice/noninferiority margin, fixed passive and SPY comparisons, any lagged risk-matched diagnostic, crisis definition and coverage, treatment of previously seen regimes, forward stops and reporting when no stress event occurs. Require an explicitly approved dated amendment before a defensive path can replace any existing gate.
- [ ] Check both templates against the eight acceptance examples in the design. Confirm that missing decisions remain blank, existing B0/B1 policy screens do not gain a new statistical gate, and a proposed unapproved next question is HOLD rather than authorization to run. Link the templates from the research README.
- [ ] Run `python scripts/verdict_register.py check`, `scripts/run_tests.sh tests/scripts/test_verdict_register.py`, `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`; record results. Obtain fresh Sol review, then lead independently checks the diff and reruns the relevant checks.

### Task 4: Final gate and handoff

**Files:** all changed files and the dated review ledger.

- [ ] Run `scripts/run_tests.sh --full`; record passes, skips and any unavailable evidence. Run `python scripts/verdict_register.py check` and the repository Ruff checks. Do not claim any skipped test passed.
- [ ] Review coverage totals, missing/unresolved cases, original-verdict preservation, prospective-template status, and links. Ensure the register visibly identifies incomplete scope if any identified records remain pending and that no new return test was run.
- [ ] Report coverage, review and test outcomes, unavailable evidence, same-family review limits and decisions still needing the owner. Do not merge, deploy, change a registered gate, or start paper trading without separate authorization.
