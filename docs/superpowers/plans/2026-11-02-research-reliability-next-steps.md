# Research and Paper Reliability Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare an evidence-based paper-trading readiness review, correct three known test/accounting defects, and determine whether a distinct accruals study can be registered using free data.

**Architecture:** Three independently reviewable tracks share the existing ranked worklist, not runtime interfaces. Paper work produces documents only; correctness work changes the standalone tax-lot library and a WFO test helper; research work establishes source feasibility before any return analysis. Decisions and later-stage work retain their existing gates.

**Tech Stack:** Markdown, Python, pandas, pytest via `scripts/run_tests.sh`, existing research artifacts.

**Spec:** [Ranked working task list](../../working_task_list.md), especially A1–A4 and B1–B4; [forward-paper standard](../../../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md); [family-scope review](../../../research/stock-signals/feasibility/2026-11-02-family-scope-review.md); [A23 report](../../../research/allocation/static-leverage/2026-09-30-static-fractional-leverage.md).

## Global Constraints

- Taxable accounts only; preserve all historical NO-GOs, parked event studies, and the existing paper-trading freeze.
- No paid data; do not alter paper production code, configuration, containers, or live orders during this plan. The sweep is design only.
- The parity window ends no earlier than approximately 2026-12-29 and any parity incident resets it; elapsed time alone is not clearance.
- Research results need a registered brief before returns are examined, pinned windows, an appropriate benchmark and raw JSON artifacts. A screen pass permits further research, not capital deployment.
- Run tests using `scripts/run_tests.sh`, never bare pytest; do not delete `data/`. Do not infer approval or a factor-scope ruling from silence.
- Each task can be reviewed and committed separately. Do not push or deploy as a consequence of completing this plan.

## Review Focus

1. Missing parity evidence must remain unknown, not be recorded as a clean day (Task 1 document checks).
2. Stock-sleeve cash needs must take priority without silently converting the monthly SSO band into daily trading (Task 2 scenario checks).
3. Fractional wash matching must preserve as-of history and avoid premature finality (Task 3 regression tests).
4. A tiny sale matched to a much larger replacement must retain the unmatched shares and basis (Task 4 regression test).
5. A skipped training fold must not shift subsequent gate outcomes to the wrong fold (Task 5 regression test).

## File map and execution strategy

- Create `research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md`: evidence criteria, incident ledger and pre-lift checks.
- Create `research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md`: proposed cash-flow rules, unresolved decisions and scenario-based test plan, no implementation.
- Modify `src/ggTrader/lab/tax_lots.py` and `tests/lab/test_tax_lots.py`: fix two independent wash-sale edge cases.
- Modify `tests/lab/test_wfo.py`: repair and test its scripted helper, without changing WFO production semantics.
- Create `research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md`: source metadata and cited feasibility verdict; a separate brief and run are conditional deliverables.
- Modify `docs/working_task_list.md` and `docs/next_steps.md` only after a task closes, recording evidence and the actual next actions. Preserve owner decisions as open unless explicitly supplied.

**Recommendation:** Execute natively by track and obtain an independent Codex review of tax accounting, WFO regression coverage, and the research/readiness conclusions before merging. Independent tracks could be delegated in parallel if speed matters, but per-task implementer/reviewer handoffs add overhead for these small changes; paper deployment consequences and accounting correctness justify focused review. No delegation occurs merely by approving this plan.

## Track A — Paper reliability (documents only)

### Task 1: Define evidence and the reset ledger

**Files:** Create `research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md`.

**Interfaces:** Consumes A2 and B2 in the working task list and the parity rules in `docs/next_steps.md`; produces a checklist and ledger consumed by Task 2 and the future lift review.

- [ ] Read the listed specs and parity artifact `research/audits/production-replica/_replica_parity_20261019.json`; distinguish historical replication uncertainty from prospective parity incidents.
- [ ] Document the four parity criteria: stock names ≤ `MAX_POSITIONS`, no missed exits, no orphan or untradable holdings, and shadow weights/rebalances match the rule. For each identify evidence, frequency, responsible role and pass/fail/unknown disposition; identify `paper_trades` and `paper_snapshots` as evidence locations without claiming they alone prove every criterion.
- [ ] Add a dated incident-ledger template with incident, evidence, detection date, reset decision, window start, earliest new review date and owner. Record only verified incidents; label absent evidence unknown. State any parity incident resets the clock and the starting date requires confirmation.
- [ ] Add pre-lift checks for the SSO fee and closure risk, actual distribution tax character versus modeled bounds, rollback, and the documented build/pull/deploy sequence. Record unverified facts as outstanding rather than assuming the old waiver still applies.
- [ ] Check document scenarios: an undocumented day stays unknown; an incident resets the start; a calendar date without evidence does not pass. Cross-check every assertion against a linked artifact and mark any unavailable source.
- [ ] Commit the document separately: `git add research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md && git commit -m "docs: define paper freeze readiness evidence"`.

### Task 2: Design the two-symbol sweep

**Files:** Create `research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md`.

**Interfaces:** Consumes Task 1 and the A23 report; produces a design and test plan only, with no runtime API.

- [ ] Document the registered 70/30 SPY/SSO target, 22–38% SSO band, month-end rebalance, cash-first funding, highest-basis long-term lots, and initial 10% intended-capital staging; cite the controlling source for each.
- [ ] Describe proposed handling of daily stock cash needs, insufficient cash, unavailable or fractional lots, missing prices, and interrupted/rejected orders. Explicitly distinguish required sale for liquidity from scheduled band rebalancing; identify any policy not fixed by the source as an open decision, not an approved rule.
- [ ] Write expected outcomes for five table-driven future tests: inside band, outside band at month end, mid-month stock cash shortfall, insufficient eligible long-term lots, and stale price or rejected order. For an unresolved policy, assert the implementation must stop or surface a decision rather than claim a predetermined trade.
- [ ] Cross-check design against Task 1 and the A23 report; document conflicts and confirm it authorizes no code, configuration, orders, or deployment.
- [ ] Commit the document separately: `git add research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md && git commit -m "docs: design two-symbol sweep readiness"`.

## Track B — Correctness (isolated from production paper trading)

### Task 3: Fractional wash-sale finality

**Files:** Modify `src/ggTrader/lab/tax_lots.py`; test `tests/lab/test_tax_lots.py`.

**Interfaces:** Preserve `TaxLotBook.loss_status(realization: Realization, asof: pd.Timestamp) -> str` and immutable realization/event behavior.

- [ ] Add `test_fractional_forward_matches_finalize_as_of_last_match`: buy 0.3 shares at $1 on 2025-01-01, sell 0.3 at $0.80 on 2025-03-01, repurchase 0.1 on each of March 2, 3 and 4. Assert provisional as of March 3, final as of March 4, disallowed approximately $0.06, original realization unchanged, and `verify_chain()` true. Retain the existing partial-forward-match test for historical as-of behavior.
- [ ] Run `scripts/run_tests.sh tests/lab/test_tax_lots.py::test_fractional_forward_matches_finalize_as_of_last_match -q`; confirm it fails on current code for finality.
- [ ] Update `loss_status` to compute as-of matched quantity from events belonging to the sale, using the existing quantity tolerance and split semantics; do not use later events to judge an earlier `asof`. Preserve the current elapsed-window behavior.
- [ ] Run `scripts/run_tests.sh tests/lab/test_tax_lots.py -q`; expect all tests to pass, including split and as-of cases. Add a split-before-match as-of regression if the existing suite does not cover this path.
- [ ] Commit: `git add src/ggTrader/lab/tax_lots.py tests/lab/test_tax_lots.py && git commit -m "fix: finalize fractional wash matches consistently"`.

### Task 4: Match a small sale to a large replacement

**Files:** Modify `src/ggTrader/lab/tax_lots.py`; test `tests/lab/test_tax_lots.py`.

**Interfaces:** Preserve `TaxLotBook.buy`, `sell`, event schema and `loss_status` signatures; consumes Task 3's corrected finality behavior.

- [ ] Add `test_tiny_loss_matches_large_replacement_lot`: buy 0.000001 XYZ at $100 on 2025-01-01, sell it at $80 on March 1, buy 2,000,000 shares at $1 on March 2 with sufficient cash. Assert one wash adjustment for 0.000001 shares and approximately $0.00002 disallowed, final loss status, total replacement quantity unchanged, and a valid event chain. Inspect the public sale output or ledger to verify the unmatched replacement basis remains $1/share.
- [ ] Run `scripts/run_tests.sh tests/lab/test_tax_lots.py::test_tiny_loss_matches_large_replacement_lot -q`; confirm current code fails because the match is skipped.
- [ ] Update `_match(self, sale: _Sale, lot: _Lot, date: pd.Timestamp) -> None` so a real positive matched quantity is not discarded relative to the larger lot; use the sold quantity as the appropriate scale and preserve lot splitting, basis and event accounting.
- [ ] Run `scripts/run_tests.sh tests/lab/test_tax_lots.py -q`; expect all tests to pass, including existing partial replacement and split tests.
- [ ] Commit: `git add src/ggTrader/lab/tax_lots.py tests/lab/test_tax_lots.py && git commit -m "fix: match small wash sales to large lots"`.

### Task 5: Keep scripted WFO gates aligned to folds

**Files:** Modify and test `tests/lab/test_wfo.py`.

**Interfaces:** Preserve `_scripted_wfo(monkeypatch, scripts, skip_train=(), breaker=None, policy=None)` and production `run_wfo` behavior.

- [ ] Add `test_scripted_wfo_gate_alignment_after_skipped_fold` using heterogeneous gate scripts: fold 0 skipped, fold 1 passes both gates, fold 2 fails NDH. Assert the returned per-fold outcomes and deployed winner/anchor correspond to folds 1 and 2, rather than scripts 0 and 1; use existing fold-result fields and fixtures.
- [ ] Run `scripts/run_tests.sh tests/lab/test_wfo.py::test_scripted_wfo_gate_alignment_after_skipped_fold -q`; confirm misalignment fails the test.
- [ ] In `_scripted_wfo`, associate each actual training dispatch with its fold index and have both gate stubs read the active fold's script. Handle the final live-parameter sweep without advancing a gate index.
- [ ] Run `scripts/run_tests.sh tests/lab/test_wfo.py -q`; expect all tests to pass. Run `ruff check src/ggTrader/lab/tax_lots.py tests/lab/test_tax_lots.py tests/lab/test_wfo.py` and `ruff format --check` on those paths.
- [ ] Commit: `git add tests/lab/test_wfo.py && git commit -m "test: align scripted WFO gates with folds"`.

## Track C — Research preparation and decisions

### Task 6: Accruals source feasibility, without returns

**Files:** Create `research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md`.

**Interfaces:** Consumes S04 in the family-scope review, the free-data inventory and forward-paper standard; produces a cited `feasible`, `infeasible`, or `unresolved` finding and evidence gaps for Task 7.

- [ ] Consult the original data provider's documentation for the named accruals portfolios. Record source URL, access date, first available observation, portfolio formation and accounting-data lag, revision behavior, available long-only portfolio definitions, and terms of use. Verify whether history reaches 1994 and whether benchmark and cost assumptions can be defined comparably. Do not download or inspect return values.
- [ ] Determine separately whether the published series supports the historical screen and whether stock-level point-in-time reconstruction is available; do not infer the latter from portfolio history. Cite evidence or label unknown for each field.
- [ ] Write a feasibility verdict and any blocking facts. If a needed fact is unknown, use `unresolved`; do not turn missing documentation into a pass. State that feasibility is neither a performance result nor permission to trade.
- [ ] Verify that every factual claim links directly to its primary source, and that no return results or candidate tuning entered this task.
- [ ] Commit: `git add research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md && git commit -m "docs: assess accruals data feasibility"`.

### Task 7: Conditional registered screen and verdict

**Files:** If Task 6 is feasible, create `docs/research/briefs/2026-11-02-accruals-screen.md`, a focused driver under `scripts/`, raw JSON `docs/research/_accruals_screen_results.json`, and a dated report under `research/`.

**Interfaces:** Consumes the Task 6 feasibility finding and the forward-paper standard; produces a reproducible Part A verdict. If Task 6 is infeasible or unresolved, do not execute this task; record its blocker in Task 8.

- [ ] Before examining returns, specify exact source and download version, fixed long-only portfolio, benchmark, sample windows, cost haircut, missing-data rules, hypothesis family and tests. Use the standard's Holm-adjusted Newey–West A1 threshold and A2 ≥ +1.0 percentage point/year since 1994; define all annualization, alignment, and sample calculations. State there is no cross-study error control.
- [ ] Review the complete brief against the source metadata and the standard; preserve any explicit owner approval requirement recorded in the worklist. Do not run until the brief is approved where required.
- [ ] Implement a reproducible, resumable-if-long-running driver following existing research drivers; record source identifiers, fixed parameters, raw inputs' provenance, results and failure reasons in JSON. Verify calculations on a small hand-calculated fixture before running actual data.
- [ ] Run the registered screen once, with fixed windows. Report A1 and A2 separately, sample limitations and no-go scope; compare with SPY only where the registered data and methodology support that comparison. A pass authorizes further research only.
- [ ] Run the driver's targeted tests through `scripts/run_tests.sh`, verify the report's numbers against JSON and refresh the research snapshot using its designated skill after the arc closes. Commit brief, driver, tests, JSON and report together or as reviewable sequential commits.

### Task 8: Decision record and handoff

**Files:** Modify `docs/working_task_list.md` and `docs/next_steps.md`; optionally update factor report and family-scope review only if the owner explicitly supplies a ruling.

**Interfaces:** Consumes completed Tasks 1–6 and, if executed, Task 7; produces the current 1–2 step worklist and a clear record of open decisions.

- [ ] Record completed work with links and verification, and leave incomplete or conditional items open. Do not mark the factor-scope decision or insider amendment resolved without an explicit owner decision; neither blocks the accruals feasibility check.
- [ ] Preserve future triggers: Part A pass before extensive tax-lot validation and integration; decide WFO options when a WFO study is proposed; stock-level accruals work requires a point-in-time fundamentals pipeline; month-end A23 check only once funded; freeze review only after qualifying evidence and its clock permit it.
- [ ] Confirm `next_steps.md` names no more than the next two actual actions and that both files agree about status. Do not change the original NO-GO findings or parked studies.
- [ ] Commit only the updated documentation: `git add docs/working_task_list.md docs/next_steps.md && git commit -m "docs: update research and reliability worklist"`.

## Final review and handoff

- [ ] Independently review the two accounting fixes for split/as-of semantics, rounding and unchanged ledger behavior; review the WFO helper test against actual fold outputs.
- [ ] Independently review readiness claims and the feasibility/brief against cited sources and the registered standard. Separate unknowns, assumptions and facts.
- [ ] Run the affected test files together using `scripts/run_tests.sh tests/lab/test_tax_lots.py tests/lab/test_wfo.py -q`; check `git diff --check` and `git status --short`. Report exact outcomes, remaining decisions, and untracked data without deleting it.
- [ ] Do not deploy or treat this review as a freeze lift. Present a concise completion report, with links to deliverables and any unresolved blockers.
