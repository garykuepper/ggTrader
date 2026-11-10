# Repository Review and Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement approved slices task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and fix justified maintainability and reliability issues while preserving research and paper-trading behavior.

**Architecture:** Review the core in three independent areas, verify each finding, and produce a separate concrete implementation plan for each approved slice. The review determines the interfaces and tests; do not invent refactor targets before evidence exists. Each slice must stand alone and have its own verification and review.

**Tech Stack:** Python, PostgreSQL/TimescaleDB, vectorbt Rust engine, pytest via `scripts/run_tests.sh`, Ruff, graphify.

**Spec:** Owner's review-led, ChatGPT-only refactoring request and repository `AGENTS.md`; this is a review programme, not a spec for a predetermined code change.

## Global Constraints

- Use ChatGPT models only. Terra or Luna may build; fresh-context Sol reviews. The lead independently verifies findings, diffs, and checks. Record that same-family review is not cross-family independence; the owner has selected ChatGPT-only in place of the usual Opus review.
- Keep private account records out of delegate prompts. The owner reports plenty of ChatGPT quota; exact usage is not visible to the agent. Report usage where available and pause after each phase. If headroom becomes unknown for a later run, default to one bounded review, report usage, and pause.
- Preserve the stopped paper trader and existing timers; no live orders, cron changes, database writes, deployment, or deletion of `data/`.
- Preserve existing untracked work. Tests run only through `scripts/run_tests.sh`; full tier is required before merge. No merge or deployment is authorized by this plan.
- Maintain research reproducibility, pinned membership and evaluation windows, and honest NO-GO findings. Changes to research results require separate authorization.
- Do not change behavior merely to make a file smaller. Keep studies isolated from `src/` and follow existing import and documentation rules.

## Review Focus

These are review questions, not claims of defects. If a slice touches one, its slice plan must specify a behavioral test before implementation.

1. Naive UTC timestamps and adjustments at the backfill/live-loader seam must preserve stored dates and prices.
2. Missing, stale, or mismatched membership must continue to block live trading.
3. A failure between backup, commit, and insert-record persistence must retain its documented recovery behavior.
4. Fold selection, point-in-time membership, and simulation inputs must not introduce lookahead or change historical evidence.
5. Broker, database, or clock failures must not bypass risk controls or hang the test suite.

## File map

- Read: `docs/next_steps.md`, `docs/architecture.md`, `research/RESEARCH_SNAPSHOT.md`, and `AGENTS.md`.
- Review: `src/ggTrader/paper/`, `src/ggTrader/data/`, `src/ggTrader/engine/`, `src/ggTrader/lab/`, `scripts/backfill_missing_members.py`, and their tests. Sample study tooling and documentation where a finding crosses those seams.
- Create: `docs/superpowers/plans/2026-11-09-refactor-review-findings.md` as the evidence-backed review ledger, then a separate dated plan per approved slice.
- Modify: no production files in Tasks 1–2. Each subsequent slice plan must name exact files, interfaces, tests, commands, and expected results based on verified findings.

### Task 1: Baseline and scope

**Files:** Read the file map above; create the review ledger.

**Interfaces:** Consumes the current checkout; produces a baseline with commit, branch, dirty paths, command results, and any pre-existing failures.

- [ ] Record `git status --short`, `git branch --show-current`, and `git rev-parse HEAD`; do not alter untracked files.
- [ ] Run `scripts/run_tests.sh --fast`, `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .`; record exit status and failures in the ledger. Investigate failures before using the baseline as a comparison.
- [ ] Read the worklist and architecture and record production freezes and the review scope. Use the graph as a navigation aid, not proof; verify claims in source. Note that its last update was code-only and unsupported file types were excluded.

### Task 2: Evidence-backed independent reviews

**Files:** Read the areas below; write verified findings to the ledger.

**Interfaces:** Each reviewer returns file and line, evidence, impact, confidence, suggested remedy, and relevant existing tests. The lead produces a ranked, deduplicated ledger, including rejected findings and reasons.

- [ ] Dispatch fresh-context ChatGPT reviews for (a) paper and data, (b) engine, WFO and provenance, and (c) CLI, backfill, tests and documentation. Supply the global constraints and exclude private account records. Reviewers perform read-only analysis.
- [ ] Inspect `membership_weekly.py`, `membership_sources.py`, `scripts/backfill_missing_members.py`, and `data/live/cached_yfinance_loader.py` specifically; compare write conventions and their tests before proposing a shared helper. Check whether moving behavior would improve locality without changing semantics.
- [ ] Lead verifies every material claim in source and tests, ranks it by impact and confidence, and records whether a change is warranted. Large files alone are not findings.
- [ ] Report results and available usage, then pause for the owner to select slices. If no finding merits change, close the programme without refactoring.

### Task 3: Specify and execute each selected slice

**Files:** Create one dated slice plan under `docs/superpowers/plans/`; exact production and test paths follow from Task 2.

**Interfaces:** Consumes a verified ledger finding and owner-selected scope; produces one independently testable change with documented behavior and review.

- [ ] Write the slice plan with exact files, existing and proposed interfaces, behavioral tests and assertions, commands and expected outcomes. Cover applicable Review Focus conditions. If a test is already sufficient, identify it rather than duplicating it.
- [ ] Have a ChatGPT builder implement only the selected slice, preserving existing behavior and using the project's test runner. Run its targeted checks and Ruff.
- [ ] Give a fresh-context Sol reviewer the diff, requirement and relevant evidence, without the builder's reasoning. Review correctness, failure behavior, and whether tests protect the public behavior. Document the same-family limitation.
- [ ] Lead independently reads the diff and reruns checks; resolve substantiated findings and recheck. Report the result and available usage before selecting another slice. Repeat Task 3 only for selected findings.

### Task 4: Final verification and handoff

**Files:** All changed files and the review ledger; update `docs/changelog.md`, `docs/architecture.md`, or `docs/cli_reference.md` if the actual changes require them.

**Interfaces:** Consumes reviewed slices; produces a verification report and a reviewable branch, without merging or deploying.

- [ ] Run `scripts/run_tests.sh --full`, `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .`; record outcomes, skips, and investigate failures. Run any slice-specific tests and pin checks required by its plan. For changes affecting the engine, WFO, or paper signals, run the applicable real-tape oracle or live golden checks; if they cannot run, name the missing evidence and report verification as incomplete rather than passing the gate.
- [ ] Review the complete diff and documentation for behavioral changes, orphaned imports, stale instructions, and untouched owner data. Record residual risks and deferred findings.
- [ ] Report the branch, changes, checks, reviewer limitation, and available usage. Leave merge, deployment, and restarting cron to a separate owner decision.
