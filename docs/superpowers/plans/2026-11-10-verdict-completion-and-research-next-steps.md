# Verdict Completion and Research Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the two known register gaps as far as surviving evidence permits, prepare the branch for integration, and give the owner a concrete choice of shadow rules and next research objective.

**Architecture:** Original studies and prior decisions remain authoritative; add source-linked, append-only register annotations and retain uncertainty when evidence is absent. Prepare decision briefs and a candidate registration without treating a proposal, label, or elapsed time as approval to change gates or run a strategy.

**Tech Stack:** Markdown, existing verdict-register JSON and Python CLI; pytest through the capped repository runner; Git.

**Spec:** [Research verdict framework design](../specs/2026-11-10-research-verdict-framework-design.md); [original register plan](2026-11-10-verdict-register-and-framework.md).

## Global Constraints

- Preserve original GO/NO-GO, stopped and unresolved records, source artifacts, and the production freeze; no broker, account or production database access, trade, cron change, or return test is authorized by this plan.
- Preserve the approved 2026-10-23 Part A/B0/B1 rules for their existing scope. A defensive path or modified shadow rule requires a separate dated owner-approved amendment; do not apply B1 retrospectively to the cross-asset sleeve.
- Owner's instructions favor research and paper reliability; no recommendation to retire a sleeve, stop paper trading, shut down the project, or change taxable-account type.
- Treat previously inspected data as seen. Do not infer a complete history merely because the pending count reaches zero; keep `coverage_complete` false unless a documented reconciliation actually supports changing it.
- Reuse the register's append-only record IDs, sources and validation. Mark unverified results UNRESOLVED, with exact gaps; never invent a verdict or a numerical risk, wealth or evidence threshold.
- The owner has said ChatGPT quota is plentiful and Claude quota has recovered; exact remaining headroom is unknown. Use a small, phased review: Sol and Opus for final branch review, reporting scope and any provider failure. No private account records in delegate prompts. Before merge satisfy the repository's cross-family and dual-review rules.
- For external claim checking, follow the current research skill and DeepThought library workflow: retain eligible primary-source papers there where possible, record provenance and access limits, and do not redistribute PDFs without rights review.

## Review Focus

1. A found report mentioning a weight but not its result must not become a weight-specific outcome (Task 1).
2. An unlocated 10% trial must not be converted into evidence that no trial occurred (Task 1).
3. Rule conflict, missing contemporaneous observations and the separate stock B1 standard must not be conflated (Task 3).
4. A proposed benchmark or a brief selected after seeing data must not be called preregistered or treated as authorization (Tasks 3–4).
5. A passing test suite or reviewer opinion does not establish complete historical discovery or authorize production (Task 2).

## Files and deliverables

- Existing register inputs: `research/programme/verdict-framework/inventory.json`, `annotations.json`, `2026-11-10-retrospective-review.md`; generated `research/VERDICT_REGISTER.md`.
- New source-review note: `research/programme/verdict-framework/2026-11-10-midcap-and-sizing-source-review.md`.
- New shadow decision brief: `research/programme/verdict-framework/2026-11-10-cross-asset-shadow-decision-brief.md`.
- New objective decision brief: `research/programme/verdict-framework/2026-11-10-next-research-objective-brief.md`.
- Existing blank registration aids: `research/programme/verdict-framework/TEMPLATE-return-brief.md` and `TEMPLATE-defense-brief.md`.

### Task 1: Reconcile the two pending historical scopes

**Inputs:** `discovery-review-8bb3ce665d093d4e` (MidCap 50/50 and 70/30) and `discovery-review-8faf1f5e3f69a7de` (possible 10% sizing), the source links in `inventory.json`, and the existing MidCap and archived-trial annotations. **Output:** a dated note and source-backed dispositions; no new backtest.

- [x] Inventory relevant tracked reports, Git history and existing local artifacts read-only. For the MidCap variants check `docs/roadmap.md`, `docs/superpowers/specs/2026-06-24-midcap400-research-design.md`, and the historical blend implementation; identify exactly whether each weight has a recoverable outcome. For 10% sizing inspect `docs/archive/sp500_equity_strategy_research.md`, its invalidation and available run references. Do not search private account records or modify `data/`.
- [x] Write the source-review note with paths, versions, commands and findings for each scope, including missing artifacts and the distinction between a tested configuration and a recorded verdict. Cross-check every reported number against its source; label an unlocated result as unverified, not nonexistent.
- [x] If a distinct result is supported, add its own dated verdict and revision, preserving original wording and limitations. If the result cannot be established, leave the inventory item pending with a specific missing-evidence note; do not create a guessed verdict or silently exclude it. Retain the `INCOMPLETE` status where appropriate.
- [x] Run `.venv/bin/python scripts/verdict_register.py render`, then `.venv/bin/python scripts/verdict_register.py check` and `scripts/run_tests.sh tests/scripts/test_verdict_register.py`. Have a fresh reviewer check the source-to-record mapping; the lead verifies the cited facts and any correction.

### Task 2: Final branch integration gate

**Inputs:** Task 1 changes and the branch diff against local `main`; preserve the two local untracked FINRA/IRS PDFs. **Output:** reviewed, tested branch ready for an owner decision on integration.

- [x] Check `git status --short`, `git diff --check` for the current changes, register counts, original-verdict preservation, source links, and the retrospective note. The branch-wide diff also reports intentional Markdown hard breaks in the generated register and approved design; document those separately rather than treating whitespace as an unexamined failure. Verify that pending items and the overall coverage limitation appear accurately; do not hand-edit `research/RESEARCH_SNAPSHOT.md`.
- [x] Run `.venv/bin/python scripts/verdict_register.py check`, `scripts/run_tests.sh tests/scripts/test_verdict_register.py`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, and `scripts/run_tests.sh --full`. Record the exit codes and all material skips; investigate failures rather than increasing the memory cap.
- [x] Obtain fresh Sol and Opus reviews of the final branch and its evidence, with no private account records. Record model, scope and limitations; adjudicate findings against sources and rerun relevant checks. A reviewer does not certify an entire research history from a sample.
- [x] Prepare a concise integration summary and commit reviewed changes. Do not merge, push, deploy or enable paper trading unless separately authorized; any merge requires the repository's final full-suite and review gates.

### Task 3: Prepare the cross-asset shadow decision brief

**Inputs:** register record `discovery-review-fe12b892259fd8fc`; `research/allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve.md`; `docs/archive/next-steps-history-2026-11-03.md`; `research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md`; and the 2026-10-23 forward-paper standard. **Output:** an unapproved, source-linked decision brief; no shadow run.

- [x] Compare the original roughly three-month report with the parity-first/12-month rule, after-tax and rebalance assumptions, start/reset ambiguity, and the A23 interaction. Identify each source and any evidence that it was approved; distinguish a source-attested approval from independently verified timing.
- [x] State what is and is not known about observations: the recomputed overwritten CSV is not a contemporaneous decision archive; do not inspect private account values or infer a clean clock. Explain why the later B1 stock standard is not automatically applicable.
- [x] Give the owner concrete choices for governing rule, comparison, observation/ledger method, start/reset treatment and taxable-account accounting. Leave numerical tolerances and dates blank unless already authorized. Explain which choice would require a dated prospective amendment and explicitly label the brief **proposal; no authorization**.
- [x] Check the brief against the source records and register, preserving the earlier static GO, weak holdout and unresolved shadow as distinct. Obtain an independent review before presenting the decision.

### Task 4: Select an objective and prepare one feasible study

**Inputs:** [benchmark synthesis](../../../research/programme/benchmark-objectives/2026-11-10-objectives-benchmarks-consolidated.md), register, both templates and current `docs/next_steps.md`. **Output:** owner decision brief and, only after required choices, one proposed candidate registration.

- [x] Prepare a brief comparing return/stock-selection and downside-defense objectives; show SPY and the allocation displaced, sleeve versus whole-account effects, cash exposure, taxes and the limits of short forward samples. List no more than three feasible hypotheses and identify existing related verdicts and differences without reopening them.
- [ ] Ask the owner to choose objective, horizon, downside measure, passive alternative and acceptable wealth/return sacrifice as relevant. Do not fill those values by looking at study results. Continue independent source and data-access checks while awaiting the decisions.
- [ ] For the chosen question, check free point-in-time data, exact availability and provenance, benchmark, cost and taxable-account inputs. If infeasible, record the missing input and cheapest informative alternative; do not manufacture a feasible screen.
- [ ] Draft one dated brief from the appropriate template, fixing previously seen data, study family and multiplicity, endpoint and equality, comparator, costs/taxes, power or minimum detectable effect, validity faults, stage rules and stopping behavior. Retain existing gates unless a separate prospective amendment is approved. Clearly mark it **unapproved; no return test authorized**.
- [ ] Independently review the feasibility and brief; present the remaining owner decisions and only then seek any approval required by the applicable standard. No capital, broker paper, observer schedule or production authorization follows from a research brief.
