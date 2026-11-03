# Paper Evidence and Accruals Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the already reviewed research branch, reconstruct the available paper-trading evidence, resolve or document the sweep-design and accruals-feasibility gaps, and prepare a bounded factor-scope recommendation.

**Architecture:** Integrate the existing branch first, then reconcile evidence and investigate accruals in parallel; finish the sweep design after the evidence review. Each track produces a reviewable, source-linked document; the lead consolidates independent reviews, records unresolved items and updates the short worklist. This is a research and reliability plan, not a trading-system deployment plan.

**Tech Stack:** Git, Markdown, read-only broker/database records where available, Python/pytest through `scripts/run_tests.sh` for integration verification, primary-source web documentation.

**Spec:** `docs/working_task_list.md` (A1, A2, A4, B1–B4); `docs/research/briefs/2026-10-23-forward-paper-standard.md`; and, after Task 1, `docs/research/2026-11-02-paper-freeze-readiness.md`, `docs/research/2026-11-02-two-symbol-sweep-design.md`, and `docs/research/2026-11-02-accruals-feasibility.md`.

## Global Constraints

- Taxable account only; no paid datasets or subscriptions. Preserve historical NO-GOs, parked PEAD/insider research, and paper observations.
- Do not change paper production code, production configuration, containers, or place orders under this plan; do not run `docker compose pull` or `docker compose up` during the freeze. No calendar date alone lifts the freeze; a parity incident resets the window.
- Read broker/database evidence without mutation; never publish account numbers, credentials, broker or client order IDs, or real-account lot data, and never include them in delegate prompts. Assess paper-trading tickers and quantities for sensitivity before publication. A public report must cite safe, reproducible evidence references without publishing credentials or personal account records.
- Unknown or unavailable evidence remains unknown. Do not infer an owner decision, a clear parity window, or actual funding from silence.
- Do not inspect accrual portfolio returns until the candidate and test are registered under the existing standard. This plan does not authorize a return test, live trading, a new WFO policy, or a change to a closed verdict.
- Run tests only through `scripts/run_tests.sh`; never delete `data/` or its event-timing cache.
- The user previously authorized continuing without prompts; use judgment on routine design choices. Preserve owner-only decisions where explicitly required; record a recommendation and an open decision rather than inventing consent.

## Review Focus

1. Missing session inputs must remain unknown rather than becoming a clean parity day (Task 2 reconciliation checks).
2. A parity incident must reset the clock; a missing detection date or restart decision must not produce a fabricated eligibility date (Task 2 ledger checks).
3. A stock-sleeve cash shortage must not silently cause an extra SSO band rebalance or an unauthorized tax-lot sale (Task 3 scenario checks).
4. Published historical portfolios may contain revised data; a stated formation lag does not prove point-in-time accounting inputs (Task 4 source checks).
5. A broad or narrow interpretation of a factor closure must not change historical NO-GOs or silently authorize a new test (Task 5 scope checks).

## File map and dependencies

- `docs/research/2026-11-02-paper-freeze-readiness.md`: update the evidence inventory, incident ledger, unresolved gaps and readiness status; never commit raw personal records.
- `docs/research/2026-11-02-two-symbol-sweep-design.md`: update proposed cash, order and lot scenarios, identify decisions and support future tests; no implementation.
- `docs/research/2026-11-02-accruals-feasibility.md`: update primary-source findings, terms, revision limitations and the benchmark/cost protocol; no return values.
- `docs/research/2026-11-02-family-scope-review.md`: append a clearly labeled recommendation only; an owner ruling goes in the factor report and family review only when actually given.
- `docs/working_task_list.md` and `docs/next_steps.md`: consolidate verified status and the next one or two actions after the reviews.
- `docs/superpowers/plans/2026-11-02-paper-evidence-and-accruals-follow-up.md`: this execution record; do not overwrite the earlier plan.

Task 1 precedes Tasks 2–5. Task 2 first defines evidence, AVB and shadow scope, then builds the ledger. Tasks 2 and 4 can proceed in parallel; Task 3 incorporates Task 2 before final review. Parallel implementers use separate managed worktrees or return patches for the lead to apply and commit sequentially; never concurrently commit from a shared checkout. Task 5 is independent after Task 1. Task 6 follows all reviews.

## Delegation and review

- **Lead:** Codex executes integration and synthesis, retains responsibility for every conclusion, and verifies each delegate's cited evidence and diff.
- **Paper evidence:** native Codex lead with direct read-only access performs the reconciliation; Gemini Argon independently reviews the evidence classification and reset logic without modifying files.
- **Sweep design:** Codex Sol owns the design; Gemini Argon independently challenges the cash, fill and lot scenarios after the paper-evidence findings are available.
- **Accruals:** Codex Sol assesses sources and feasibility; Gemini Argon independently reviews the source interpretation and feasibility classification.
- **Factor scope:** lead Codex prepares the bounded recommendation. The owner-only ruling remains open unless explicitly given.
- Run only one Antigravity task at a time, with Argon reviews ordered paper evidence, sweep design, then accruals. Use DeepSeek Flash through **Pi**, the owner's preferred provider, only for a simple bounded extraction (such as a named field or exact passage); verify against the source. Gemini Flash is subject to the same bounded-task restriction. Neither forms audit conclusions.
- Before execution, discover the requested providers and exact model IDs through the active orchestration tools. If a requested provider or model is unavailable, report it and continue independent work without silently substituting a model. Do not delegate access to credentials or raw account data unnecessarily; share sanitized evidence summaries and safe paths for review.

---

## Task 1: Integrate and verify the existing branch

**Files:** Existing branch `research/reliability-next-steps`, based on `411cbbfb`, with recorded head `cc70734f`; repository `main`; the untracked earlier plan at `docs/superpowers/plans/2026-11-02-research-reliability-next-steps.md`.

**Interfaces:** Consumes the reviewed branch and the current remote state; produces an integrated baseline for Tasks 2–6. This plan's untracked file must remain intact throughout.

- [ ] Record `git status --short --branch`, `git worktree list`, the branch head and `git log --oneline main..research/reliability-next-steps`. Confirm the branch still contains `cc70734f` and inspect any later commits before integrating.
- [ ] Fetch the remote and compare local and remote `main`; inspect divergence before choosing a safe merge. If a conflict affects production or research conclusions, resolve it against the linked specs and verify the result. Do not force-push.
- [ ] Compare the untracked earlier plan with the tracked copy on the branch. If identical, move it to a temporary location outside the checkout before merging and restore it only if the merge fails; if different, preserve both and inspect the differences before proceeding. Never overwrite this new plan or touch `data/`.
- [ ] Integrate the branch into `main` with a fast-forward when possible; otherwise use a reviewed merge commit. Confirm `git log` and `git status` reflect the intended history and preserve untracked data.
- [ ] Run `scripts/run_tests.sh tests/lab/test_tax_lots.py tests/lab/test_wfo.py -q`; expect zero failures. Run `ruff check` and `ruff format --check` on `src/ggTrader/lab/tax_lots.py`, `tests/lab/test_tax_lots.py`, and `tests/lab/test_wfo.py`; expect success. If the environment cannot run a check, report the exact blocker rather than claiming success.
- [ ] Before pushing, inspect the running container read-only and record its image ID, registry digest and source commit if verifiable in the readiness document. Identify missing provenance explicitly; a mutable `:latest` tag is not a rollback anchor. The review observed a possible historical baseline at `291e6dd4`, but re-verify rather than assume it.
- [ ] Push the integrated `main` only after the above checks pass and confirm the pushed commit matches the intended head. A push triggers the image-publishing workflow and can update `:latest`; it does not itself pull or deploy the container. Do not pull or restart the production container during the freeze. Mark A3 integrated only after confirming the branch includes its fixes.

## Task 2: Reconstruct paper-trading evidence

**Files:** Modify `docs/research/2026-11-02-paper-freeze-readiness.md`.

**Interfaces:** Consumes Task 1 and the four parity criteria in the readiness document; produces a sanitized session/evidence inventory, incident ledger and `pass`, `incident`, or `unknown` classifications for Task 3 and the eventual freeze review.

- [ ] Identify which specific shadow book and rule the shadow-parity criterion applies to, using the approved source. If the source is ambiguous, document alternatives and treat the criterion as unresolved rather than selecting a book by assumption.
- [ ] Verify the documented pre-existing broker-frozen AVB condition, whether it counts against `MAX_POSITIONS`, and its effect under the literal parity rules. Record the condition and evidence; do not waive it or assert a clean window. Present a possible exception as an owner decision only where explicitly required.
- [ ] Inventory available dated evidence from the 2026-09-29 deploy through the latest completed NYSE session: effective configuration, paper snapshots/trades, broker positions and fills, expected signals, shadow decisions and incident records. Record source coverage and gaps without placing orders or changing records.
- [ ] For each session and criterion, link the safe evidence location and classify `pass`, `incident`, or `unknown`. Keep underlying sensitive account data local. Do not derive expected signals from executed orders alone. Verify whether historical expected signals, contemporaneous tape vintages and shadow outputs exist; do not treat a replay on revised data as original evidence. Where inputs or data-load results are insufficient to establish a criterion, classify that criterion as `unknown`.
- [ ] Inspect the deployed rule and logs for holiday and early-close handling, including the market-clock gate and errors. Classify each affected session using its observed behavior and evidence; do not assume that the cron schedule submitted an order or that a skipped run was a parity incident.
- [ ] Assess whether a read-only host-side daily evidence capture would fill future gaps; document its data sources, privacy boundary, operational effect, failure modes and authorization status. This plan does not install or schedule it.
- [ ] Record each substantiated parity incident with evidence, detection and effective restart dates, reviewer and reset rationale. If any required date or decision is unknown, leave the new eligibility date unknown; do not invent a clean window or a reviewer assignment.
- [ ] Check three cases in the resulting ledger: an unavailable input is `unknown`; a verified incident resets the window; no incidents recorded without complete evidence does not establish a clean period.
- [ ] Have Gemini Argon independently review the sanitized inventory and reset conclusions; lead Codex verifies each cited fact and resolves or records every material disagreement. Do not upload raw credentials or account records.
- [ ] Commit only the sanitized report and cite evidence provenance in the commit: `git add docs/research/2026-11-02-paper-freeze-readiness.md && git commit -m "docs: reconcile paper parity evidence"`.

## Task 3: Resolve sweep design scenarios

**Files:** Modify `docs/research/2026-11-02-two-symbol-sweep-design.md`.

**Interfaces:** Consumes the registered A23 rule and Task 2's evidence; produces proposed, explicitly labeled policy choices and a future implementation test matrix. It produces no runtime code.

- [ ] Identify the specific account, instruments, deployed paper sweep and owner's separate real-account A23 helper; cite the authority for each. Never provide real-account lots to delegates.
- [ ] Map cash flows for a daily stock purchase or exit, sweep contribution and month-end band check. Preserve the registered 70/30 target, 22–38% band, month-end observation and next-day trade convention, cash-first funding and highest-basis long-term lot rule.
- [ ] Compare selling SPY alone with pro-rata SPY/SSO sales for daily funding; identify the open policy decision and verify broker support for explicit tax-lot selection before assuming it exists. For mid-month cash shortfall, insufficient eligible lots, partial/rejected fills and stale prices, specify a recommended fail-safe behavior and clearly mark any change or rule not already authorized as a proposal requiring the applicable decision. No implied order submission.
- [ ] Write expected outcomes for: inside-band month end; outside-band month end; mid-month stock cash need; insufficient long-term lots; stale price; duplicate run after a partial fill. Ensure the third scenario does not assume an extra band rebalance and no scenario invents a tax-lot sale.
- [ ] Incorporate relevant Task 2 evidence and explicitly list any production behavior that could not be established from records.
- [ ] Obtain Gemini Argon's independent challenge; lead Codex checks each claim against the A23 report and source code, resolves or records material disagreements and leaves unapproved choices labeled proposed.
- [ ] Commit: `git add docs/research/2026-11-02-two-symbol-sweep-design.md && git commit -m "docs: resolve sweep design scenarios"`.

## Task 4: Resolve accruals feasibility gaps

**Files:** Modify `docs/research/2026-11-02-accruals-feasibility.md`.

**Interfaces:** Consumes the existing metadata review and forward-paper standard; produces `feasible`, `infeasible`, or `unresolved`, with a cited rationale and a proposed benchmark/cost approach. It consumes no return observations.

- [ ] Verify the source's stated usage/redistribution terms using primary sources; distinguish permission to perform local research from permission to publish raw files. If no authoritative terms are found, document the searches, the exact unresolved use, and an appropriate way to proceed without publishing raw files; do not infer a prohibition merely from copyright.
- [ ] Verify available accrual-specific information about accounting restatements, formation lags and historical vintages; record exactly what is known and unknown. The library's general return revisions do not alone prove or disprove point-in-time accounting inputs.
- [ ] Propose a specific long-only portfolio, benchmark, common date alignment and cost/turnover treatment before looking at returns. Compare the proposal to the Part A registered requirements, including Newey–West 12 lags, one-sided family-wise 0.00135 Holm procedure and the +1.0 pt/yr net CAGR criterion since 1994; distinguish methodology proposal from observed performance.
- [ ] Assess whether a historical portfolio-level Part A screen is possible separately from stock-level B0/B1 reconstruction. Identify any blocker, uncertainty, and the minimum evidence that would change the verdict.
- [ ] Have Gemini Argon independently check citations, point-in-time interpretation and verdict; lead Codex verifies the sources and records disagreements. Confirm no return observations were viewed.
- [ ] Commit: `git add docs/research/2026-11-02-accruals-feasibility.md && git commit -m "docs: refine accruals feasibility assessment"`. If feasible, identify a registered brief as the next task; do not run a screen under this plan.

## Task 5: Prepare factor-scope recommendation

**Files:** Modify `docs/research/2026-11-02-family-scope-review.md`; modify the factor report only if the owner has explicitly supplied a ruling.

**Interfaces:** Consumes the factor-screen closure and family-scope review; produces a proposed interpretation with reasons and an accurate status for S05, S07 and S08.

- [ ] Compare the narrow and broad readings of “alternate factor definitions” against the registered factor hypotheses, the closure text and the definitions in the family-scope review.
- [ ] Append a dated, clearly labeled wording-only addendum that recommends no candidate, and its practical consequences, including why S05, S07 and S08 remain undefined even under a narrow interpretation and why the ruling does not cover S04 accruals or the separate momentum closure.
- [ ] Confirm no closed verdict is revised, no candidate becomes test-ready solely from a wording interpretation, and no owner decision is recorded without an explicit owner ruling.
- [ ] Commit: `git add docs/research/2026-11-02-family-scope-review.md && git commit -m "docs: clarify factor scope recommendation"`.

## Task 6: Consolidate reviews and hand off

**Files:** Modify `docs/working_task_list.md` and `docs/next_steps.md`; commit the new plan with these updates if it is still untracked.

**Interfaces:** Consumes Tasks 1–5 and their review findings; produces the current ranked worklist and a concise evidence-based handoff.

- [ ] Verify each delegate's output and each report's citations, note unavailable providers and unresolved disagreements, and fix confirmed errors before recording status.
- [ ] Update the ranked worklist with completed, partial and blocked items; preserve the conditional tax-lot integration, WFO policy, funding-dependent A23 check, 2027-09-30 A23 annual review and future freeze-review gates. Update `next_steps.md` to no more than two actual next actions.
- [ ] Run `git diff --check`; verify all links to modified local files resolve and no account numbers, credentials, order IDs, real-account lot data or unregistered accrual return observations entered the reports. Run targeted tests again only if review fixes changed code.
- [ ] Commit the worklist and this plan: `git add docs/working_task_list.md docs/next_steps.md docs/superpowers/plans/2026-11-02-paper-evidence-and-accruals-follow-up.md && git commit -m "docs: consolidate research and parity follow-up"`.
- [ ] Report tested integration commit, evidence coverage, actual incidents or unknowns, accruals verdict, sweep decisions, factor-scope recommendation, independent review outcomes, and every unresolved blocker. Do not claim the freeze is lifted or that a strategy has passed.
