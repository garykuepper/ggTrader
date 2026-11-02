# WFO Attribution and Free-Data Stock Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 2026-11-01 research process audit's follow-ups into bounded work: (A) attribute what the walk-forward optimization (WFO) harness actually deployed in saved runs, (B) pin down and test its fallback/halt semantics, and (C, the main forward-looking track) inventory which stock-research questions can be tested on free, point-in-time data under the approved Part A → B0 → B1 standard.

**Architecture:** Track A reads existing saved fold records only (no reruns). Track B specifies intended semantics in a document first, then adds characterization tests that pin *current* behavior; `wfo.py` behavior changes only after an owner decision. Track C builds a no-return feasibility matrix with a selection rule fixed before any candidate is looked at, plus a requirements/gap spec (not an implementation) for the multi-stock after-tax lot engine that B0/B1 need.

**Tech Stack:** Python 3, pandas/numpy, pytest via `scripts/run_tests.sh`, Markdown + JSON artifacts, read-only Git inspection.

**Spec:** [Research process audit design](../specs/2026-11-01-research-process-audit-design.md) and its [report](../../research/2026-11-01-research-process-audit.md) §"Research gaps and follow-up boundaries" / "How follow-ups could be evaluated". Review inputs: Sol, Sonnet, Argon and Opus opinions in thread `bb103194-12e3-4475-b19d-beeb6db67aa3`; their consensus is summarized under "Decisions carried in" below.

## Decisions carried in

- Sequence: cheap attribution (Track A) → semantics spec + tests (Track B) run **alongside** the free-data inventory (Track C), which is the main forward-looking work. (Sol/Sonnet wanted B first; Opus wanted C first; this plan runs A/B short and C in parallel.)
- Not adopted as established: Opus's claim that a halt is "close to certain" for a positive-Sharpe strategy. It rests on assumed returns and dependence; Task 6 defines a model to test it, gated on owner decision.
- Nothing here reopens a NO-GO, changes the production freeze, or recommends retirement/account changes.

## Global Constraints

- Taxable account only; stocks first, crypto in scope; **free data only — no purchase or subscription** (owner, 2026-10-23).
- Do not recommend retiring the stock sleeve, ending paper trading or shutting the project down; do not suggest IRAs/retirement accounts.
- Preserve production freeze (~2026-12-29 lift), existing NO-GOs, stopped runs and event-study parking (PEAD/insider stay UNRESOLVED, no acquisition).
- No changes under `src/ggTrader/paper/`; no broker calls, orders, Telegram sends or Docker rebuilds.
- No market-return reruns and no new data acquisition in Tasks 1–5. Reviewed windows stay "seen"; never label them fresh OOS.
- Pre-DSR-fix gate results (fix `09f9f40`, committed 2026-10-19T16:04:44-07:00) and post-fix results are **never pooled**. Classify by the generating commit, not by filename date.
- Every cited number needs a raw artifact key (JSON path) or a file:line; label each statement *observation*, *calculation*, *hypothesis* or *unknown*. Define WFE/OOS/DSR/NDH/MDE on first use in each document.
- Tests run only via `scripts/run_tests.sh`; ruff (`line-length = 100`, rules E/F/W/I) must pass; core logic in `src/`, scripts have `main()` + `argparse` + `--help`.
- Delegation (owner rule): Codex/Argon for synthesis, audit conclusions and code review; Flash/Hermes only for bounded extraction, and their outputs get verified.
- Do not commit or push without an explicit owner request. Leave unrelated untracked items alone (`.claude/skills/*-workspace/`, `data/external/event_timing_audit/`).

## Review Focus

1. **Mislabeled halt state:** a fold row's `halted` is the state *after* the fold; reading it as the cause of that fold's anchor use is wrong (Tasks 1, 2).
2. **NaN WFE in the breaker:** `compute_wfe` can return NaN; `check_circuit_breaker` filters only `None`, so one NaN poisons the trailing average and silently suppresses a chronic-decay halt (Task 2).
3. **Winner-vs-anchor WFE mixture:** when the anchor is deployed, recorded WFE = anchor OOS Sharpe ÷ *winner* in-sample Sharpe; reading it as "the winner recovered" is wrong (Task 2).
4. **Skipped folds leave no row:** a fold with no training metrics is dropped, so fold counts can understate attempts (Tasks 1, 2).
5. **Feasibility mistaken for a verdict:** a "testable on free data" label read as a recommendation, or candidates ranked by realized returns (Task 3).

## Files

- Create `src/ggTrader/lab/wfo_attribution.py` — pure functions classifying saved fold rows.
- Create `scripts/wfo_fold_attribution.py` — driver: reads the seven fold-record JSONs, writes results JSON.
- Create `tests/lab/test_wfo_attribution.py`; append characterization tests to `tests/lab/test_wfo.py`.
- Create `docs/research/_wfo_fold_attribution_results.json`, `docs/research/2026-11-02-wfo-fold-attribution.md`, `docs/research/2026-11-02-wfo-fallback-semantics.md`.
- Create `docs/research/2026-11-02-free-data-stock-feasibility.md` + `docs/research/_free_data_feasibility_matrix.json`, and `docs/research/2026-11-02-after-tax-lot-engine-requirements.md`.
- Modify at the end only: `docs/research/RESEARCH_INDEX.md` (links), `docs/changelog.md` (one entry), `docs/next_steps.md` (worklist pointer). Do not hand-edit `RESEARCH_SNAPSHOT.md`.

## Gates (owner decisions)

- **G0 (before Task 1):** commit the 2026-11-01 audit deliverables (currently untracked/modified: report, ledger, plan, `RESEARCH_INDEX.md`)? Default: leave uncommitted until asked.
- **G1 (after Task 1):** does the attribution change how the audit report's WFO caveats read? If yes, amend the audit with a dated addendum — never rewrite verdicts.
- **G2 (after Task 2):** which semantics are *intended* for winner/anchor/deployed-policy WFE and for NaN? Only then may `wfo.py` change (separate approval; changes alter reproducibility of old runs).
- **G3 (after Task 3):** owner discussion of research direction (matches `next_steps.md` "Next"). This plan does not pick a candidate.

---

## Task 1: Attribute saved WFO fold records

**Files:**
- Create: `src/ggTrader/lab/wfo_attribution.py`, `scripts/wfo_fold_attribution.py`, `tests/lab/test_wfo_attribution.py`
- Output: `docs/research/_wfo_fold_attribution_results.json`, `docs/research/2026-11-02-wfo-fold-attribution.md` (use `docs/research/TEMPLATE-research-report.md`)

**Interfaces:**
- Produces: `classify_fold(row: dict) -> str` returning exactly `"gate_fail"` (not `gates_passed`; anchor used), `"gates_pass_anchor"` (`gates_passed` and `used_anchor`; by `wfo.py:731` this implies halted *entering* the fold), `"gates_pass_winner"`, or `"anomaly"` (`not gates_passed and not used_anchor`, which the code forbids).
- Produces: `summarize_run(rows: list[dict]) -> dict` with keys `n_folds`, `gate_fail`, `gates_pass_anchor`, `gates_pass_winner`, `anomaly`, `nan_wfe`, `none_wfe`, `fold_numbers_missing` (gaps in `fold_num`).
- Fold-row sources (observed schema `fold_rows` with `gates_passed`, `used_anchor`, `wfe`, `oos_sharpe`; `halted` only in two files): `_anchor_fix_reproduction_results.json` (`/core`), `_audit_reruns_20261019.json` (`/fomc_drift/wfo`, `/xs_momentum`, `/dual_momentum`), `_cross_asset_sleeve_results.json` (`/wfo`), `_ic_kelly_rebaseline_20260923.json` and `_pit_rebaseline_20260923.json` (`/ensemble`, `/ensemble_ic`, `/ensemble_kelly`), `_month_end_treasury_results.json` (`/wfo`), `_rebaseline_corrected_tape_20260822.json` (`/core`).

- [ ] **Step 1: Write failing tests** in `tests/lab/test_wfo_attribution.py`: `test_classify_fold_gate_fail`, `test_classify_fold_gates_pass_anchor`, `test_classify_fold_gates_pass_winner` (each with a minimal row dict), `test_classify_fold_flags_anomaly_when_neither_gates_nor_anchor`, `test_summarize_run_counts_sum_to_n_folds`, `test_summarize_run_counts_nan_and_none_wfe_separately` (`float("nan")` vs `None`), `test_summarize_run_reports_missing_fold_numbers` (rows 1,2,4 → `[3]`).
- [ ] **Step 2: Run to verify failure** — `scripts/run_tests.sh tests/lab/test_wfo_attribution.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `wfo_attribution.py` with the interfaces above (pure; no I/O; absent keys → treated as missing, not as False).
- [ ] **Step 4: Run to verify pass** — same command → PASS; `ruff check` and `ruff format --check` clean.
- [ ] **Step 5: Implement the driver** `scripts/wfo_fold_attribution.py` (`--out` path, `--help`): for each (file, run) pair record `classification counts`, the artifact JSON path, and `regime` = `pre_dsr_fix` or `post_dsr_fix`. Determine regime from `git log --format=%H -1 -- <file>` versus `git merge-base --is-ancestor 09f9f40 <that commit>`; if a file was regenerated later, record both and mark `regime: "unknown"` rather than guess. Write results to `docs/research/_wfo_fold_attribution_results.json` (include source file, JSON path, generating commit).
- [ ] **Step 6: Run the driver** — `python scripts/wfo_fold_attribution.py`. Expected spot checks (from the thread, to be reproduced, not assumed): PIT `ensemble_ic` = 6 gate_fail / 9 gates_pass_anchor / 2 gates_pass_winner of 17; corrected `xs_momentum` = 17 gate_fail of 17.
- [ ] **Step 7: Cross-check** three runs by hand against the raw JSON rows (one pre-fix, one post-fix, `fomc_drift` with 53 folds) and note any `halted` vs `gates_pass_anchor` inconsistency in the two files that carry `halted`.
- [ ] **Step 8: Write the report** from the template: per-run table, pre/post split, explicit "what this does not show" (winner's undeployed performance unknown; `halted` is post-fold; skipped folds invisible; run subset only). Include which historical NO-GOs relied on runs dominated by anchor folds, as an *observation* list with no verdict changes.
- [ ] **Step 9: Verify** — `python3 -m json.tool` on the results file; every number in the report traces to a JSON key; `git diff --check`. **Stop at G1.**

## Task 2: Specify and pin WFO fallback/halt semantics

**Files:**
- Create: `docs/research/2026-11-02-wfo-fallback-semantics.md`
- Modify (tests only): `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes: `check_circuit_breaker`, `check_shadow_reentry`, `compute_wfe`, `WfoState` (`src/ggTrader/lab/wfo.py:44-150`) and the fold loop (`wfo.py:728-800`).
- Produces: a semantics doc defining three separate quantities — *winner performance*, *anchor performance*, *deployed-policy performance* — and stating for each of WFE, `oos_sharpe`, `halted`, `used_anchor` which one it measures today; plus the characterization tests below.

- [ ] **Step 1: Write the semantics doc** (observations only, with `wfo.py` line refs): (a) anchor decision uses halt state *entering* the fold (`wfo.py:731`), recorded `halted` is *after* (`wfo.py:796`); (b) on anchor folds WFE = anchor OOS ÷ winner IS (`wfo.py:767`); (c) DSR trial count = `len(grid)` (`wfo.py:718`), so correlated variants are over-counted as independent; (d) fold with no train metrics is skipped without a row; (e) NaN WFE not filtered in `check_circuit_breaker` (`wfo.py:91`). For each, add an "intended behavior — OWNER DECISION NEEDED" line; do not pick one.
- [ ] **Step 2: Write failing/characterization tests** appended to `tests/lab/test_wfo.py`, each asserting **current** behavior and carrying a docstring naming the open decision:
  - `test_circuit_breaker_nan_wfe_suppresses_chronic_decay`: `wfe_history=[0.1, 0.1, 0.1, 0.1, float("nan")]`, `window_size=4` → `halted is False` today (excluding NaN would give avg 0.1 < 0.25 and halt).
  - `test_circuit_breaker_none_wfe_excluded_from_window`: `[0.1]*4 + [None]` → `halted is True` (control for the NaN case).
  - `test_compute_wfe_returns_nan_when_oos_nan`: `compute_wfe(1.0, float("nan"))` is NaN (not None).
  - `test_run_wfo_wfe_on_anchor_fold_uses_winner_is_sharpe`, `test_run_wfo_halted_field_is_post_fold_state`, `test_run_wfo_skips_fold_without_train_metrics`: build via the existing synthetic `_ohlcv` helper (see `test_run_wfo_reports_gate_status`, `tests/lab/test_wfo.py:587`) with `check_circuit_breaker`/gate outcomes monkeypatched so each case is hand-checkable; assert fold-row fields.
- [ ] **Step 3: Run** — `scripts/run_tests.sh tests/lab/test_wfo.py -q`. Expected: all pass (they pin current behavior). A test that *fails* means the semantics doc misread the code — fix the doc.
- [ ] **Step 4: Lint** — `ruff check tests/lab/test_wfo.py && ruff format --check tests/lab/test_wfo.py`.
- [ ] **Step 5: Do not edit `src/ggTrader/lab/wfo.py`.** **Stop at G2.** If the owner chooses a policy change, it becomes a separate plan: tests-first, flag-guarded, and every historical run re-labeled as produced by the old policy.

## Task 3: Free-data stock feasibility inventory (main track)

**Files:**
- Create: `docs/research/2026-11-02-free-data-stock-feasibility.md`, `docs/research/_free_data_feasibility_matrix.json`
- Consumes: `docs/research/WEB_RESEARCH_CANDIDATES.md`, `RESEARCH_INDEX.md`, `_research_process_audit_ledger.json`, `src/ggTrader/lab/french_data.py`, `fred_data.py`, the forward-paper standard brief.

**Interfaces:**
- Produces: JSON `{schema_version: 1, as_of, selection_rule, rows: [...]}`. Row keys: `id`, `family`, `economic_mechanism`, `side` (`long_only_feasible` | `needs_short` | `either`), `free_sources` (list, each with `url_or_name`, `point_in_time` (`yes`|`no`|`unknown`), `history_start`, `update_lag`), `part_a_testable` (needs ≥ 1994 free history), `b0_tape_ok` (clean PIT 2021-01-31→2026-04-30), `b1_live_source_continuity`, `status_vs_prior_work` (`untested` | `closed_nogo` | `parked` | `infeasible`) with link, `blocking_gaps`, `unknowns`. Allowed classification: `testable_part_a`, `testable_b0_b1_only`, `infeasible_free`, `closed_no_reopen`, `parked`.

- [ ] **Step 1: Fix the selection rule in the document before enumerating**: families are included by *coverage of unexamined mechanisms and data availability*, never by realized or published return; closed NO-GOs and parked event studies are listed only as `closed_no_reopen`/`parked` with their links.
- [ ] **Step 2: Enumerate stock mechanism families** from the candidate register, the audit ledger's `candidate_group` entries and the already-closed list (momentum, four classic factors, net issuance, value+momentum, PEAD/insider parked). Record for each whether prior work covered the same mechanism; do not count related variants as independent.
- [ ] **Step 3: Source checks (read-only, free):** for each remaining family confirm a free source exists, its point-in-time status, history start, update lag and license terms. Where a claim needs a web check, cite the URL and access date; mark `unknown` rather than infer. Delegation: Flash/Hermes may extract a named field from a named page; Codex/Argon draft the classification; verify extractions.
- [ ] **Step 4: Part A feasibility:** for each row state what registered design (windows, Holm family, costs) would be required and whether free history reaches 1994; compute **no returns**.
- [ ] **Step 5: B0/B1 feasibility:** state whether signals can be formed on the clean tape without lookahead (decision time vs data availability time) and whether the free source keeps updating for an 18-month forward shadow.
- [ ] **Step 6: Validate** — JSON loads; every row has ≥ 1 link; no row lacks a status; `grep` the doc for return/Sharpe figures from candidate testing (should be only citations of prior closed reports). Add a short "what would change this inventory" section.
- [ ] **Step 7: Add the matrix summary to the report** (counts per classification) and an explicit "no candidate is recommended here" line. **Stop at G3.**

## Task 4: After-tax multi-stock lot engine — requirements and gap spec (no build)

**Files:**
- Create: `docs/research/2026-11-02-after-tax-lot-engine-requirements.md`
- Consumes: forward-paper standard §§3, 4, 6 (lots, wash sales, T-bill cash, SPY on same basis), `src/ggTrader/lab/century_leverage.py` tax book, `a23_rebalance.py`, `replica.py`.

**Interfaces:**
- Produces: a requirements list (each testable), a table mapping each requirement to *exists / partial / missing* with a file:line, and a proposed test list (e.g., wash-sale replacement within ±30 days, partial lot sales, split/dividend adjustments, terminal-liquidation after-tax CAGR identical for strategy and SPY).

- [ ] **Step 1:** Extract every accounting requirement from the brief verbatim into a numbered list.
- [ ] **Step 2:** Inspect the existing tax book(s); record what they handle (single-asset leverage book) versus what multi-stock lots need.
- [ ] **Step 3:** Fill the exists/partial/missing table; flag the brief's note that the existing book's wash-sale handling is backward-only.
- [ ] **Step 4:** Write the proposed test list and a planning estimate (label "estimate, unmeasured"). Do not write engine code; a build needs its own plan and approval.
- [ ] **Step 5:** Verify every file:line reference resolves (`sed -n` each).

## Task 5: Statistical-power note for Part A (analytic, assumptions stated)

**Files:** Append a section to `docs/research/2026-11-02-free-data-stock-feasibility.md`.

- [ ] **Step 1:** State the Part A test (Holm-corrected Newey-West t ≥ 3, ≥ +1.0 pt/yr net since 1994, ~32 years).
- [ ] **Step 2:** Compute, with a short stdlib/numpy snippet recorded in the doc, the minimum detectable annual excess for t ≥ 3 as a function of assumed annual tracking error (a parameter grid, e.g., 2, 4, 6 pt/yr) and years of history, ignoring autocorrelation, then note how autocorrelation and multiple-family correction raise it. Label as *calculation under assumptions*, not a finding about any strategy.
- [ ] **Step 3:** State what the result means for the feasibility matrix (which mechanism types can't clear the bar even if real). No return data is read.
- [ ] **Step 4:** Spot-check the formula against one hand calculation.

## Task 6: Synthetic WFO policy characterization (conditional — only if G2 says the harness will be used again)

**Files:** Create `scripts/wfo_policy_synthetic.py`, `docs/research/2026-11-02-wfo-policy-synthetic.md`, `docs/research/_wfo_policy_synthetic_results.json` — **do not start without owner approval at G2.**

- [ ] **Step 1:** Pre-register scenarios before any run: null (zero-edge noise), useful signal at stated annual Sharpe values, regime-shift decay, correlated parameter grid. State seeds, number of replications and what each outcome would mean.
- [ ] **Step 2:** Measure for the *full* decision process: gate pass rate, anchor-use rate, halt entry/duration, re-entry, false promotion and false rejection. Report the halt rate for a positive-Sharpe strategy to test the Opus claim; do not import its conclusion.
- [ ] **Step 3:** Checkpoint per replication batch (runs may exceed 5 minutes); resumable; `--help` documented.
- [ ] **Step 4:** Report with results JSON; label as synthetic assumptions, not evidence about any real strategy.

## Task 7: Synthesize, link and verify

**Files:** Modify `docs/research/RESEARCH_INDEX.md`, `docs/changelog.md`, `docs/next_steps.md`.

- [ ] **Step 1:** Add links to the new reports in `RESEARCH_INDEX.md` under a "Process follow-ups" heading; do not reclassify historical verdicts.
- [ ] **Step 2:** Add one `docs/changelog.md` entry (what/why/no strategy or live-code change) and one `next_steps.md` pointer (1–2 steps deep per its rule).
- [ ] **Step 3:** Run the full gate: `scripts/run_tests.sh tests/lab -q`, `ruff check .`, `ruff format --check .`, `python3 -m json.tool` on every new JSON, local-link check on new Markdown, `git diff --check`, `git status` (only intended files changed; `src/ggTrader/paper/` untouched).
- [ ] **Step 4:** Scope audit of deliverables: no verdict override, no data purchase suggestion, no retirement/account recommendation, no fresh-OOS claim for reviewed windows.
- [ ] **Step 5:** Ask the owner whether to commit/push (default: no push without request).

---

## Master checklist

- [ ] G0 decided: audit deliverables committed or left local
- [ ] Task 1 — attribution script, tests, results JSON, report (G1)
- [ ] Task 2 — semantics doc, 6 characterization tests, `wfo.py` untouched (G2)
- [ ] Task 3 — free-data feasibility matrix + doc (G3)
- [ ] Task 4 — tax-lot engine requirements/gap spec
- [ ] Task 5 — Part A analytic power note
- [ ] Task 6 — synthetic policy study (only if approved at G2)
- [ ] Task 7 — index/changelog/next_steps, full verification, commit decision
- [ ] Final scope audit: production freeze, NO-GOs, paper code, free-data and taxable-account constraints all intact

## Timing (planning estimates, unmeasured)

Task 1: ~0.5 day. Task 2: ~1 day. Task 3: ~2–3 days (source checks dominate). Task 4: ~1 day. Task 5: ~0.5 day. Task 6: ~2–4 days if approved. Task 7: ~0.5 day.

## Self-review notes

- Coverage: Sol/Sonnet/Argon/Opus items map to Tasks 1 (attribution, three-way classes, post-fold `halted`), 2 (state machine, NaN, WFE mixture, skipped fold, DSR trial count), 3 and 4 (free-data inventory, tax-lot readiness), 6 (synthetic power, conditional).
- Open assumptions to confirm during execution: that `_audit_reruns_20261019.json` was generated after `09f9f40` (Task 1 Step 5 resolves it by Git, not by assumption); line numbers were verified 2026-11-02 (`wfo.py:91, 718, 731, 767, 796`); re-verify if the file changes before execution.
