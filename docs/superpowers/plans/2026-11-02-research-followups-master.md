# Research Follow-ups Master Plan (landing, WFO policy options, synthetic study, family-scope review, lot engine, snapshot, independent review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the parent (native) session; delegated work packages use `delegate_task` as specified in the Delegation Matrix. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the pushed research branch and complete every follow-up identified after the 2026-11-02 work: opt-in WFO policy options (G2), a synthetic WFO policy study, a documentary family-scope review, the after-tax multi-stock lot engine, a regenerated research snapshot, and an independent review of the earlier tasks, then merge to `main` without further prompts.

**Architecture:** Seven work packages (WP) with disjoint file ownership so delegates can run in parallel in the shared checkout. Code changes are additive and default to legacy behavior so every historical artifact stays reproducible. Delegates edit files but never commit; the parent verifies every delegate claim, commits per package, and is the only writer of the shared docs (index, changelog, next steps).

**Tech Stack:** Python 3.12 (`.venv`), pandas/numpy, pytest via `scripts/run_tests.sh`, ruff, `gh`, T3 `delegate_task`.

**Spec:** [Process audit design](../specs/2026-11-01-research-process-audit-design.md); inputs [fallback semantics](../../../research/programme/wfo-methods/2026-11-02-wfo-fallback-semantics.md), [lot-engine requirements](../../../research/paper-system/tax-lots/2026-11-02-after-tax-lot-engine-requirements.md), [feasibility inventory](../../../research/stock-signals/feasibility/2026-11-02-free-data-stock-feasibility.md), prior plan [2026-11-02-wfo-attribution-and-free-data-feasibility.md](2026-11-02-wfo-attribution-and-free-data-feasibility.md).

## Autonomy rules

The owner asked not to be prompted after execution starts. Decide and ledger every conflict as `Ruling: <what> — <why> — <cost if wrong>`. Stop only for: (1) a destructive or irreversible operation (including anything under `data/`), (2) a security-sensitive action, (3) CI failure or an unresolved Critical review finding after one fix pass: leave the PR open, report, and do not merge, (4) a plan so broken that every path is a guess. Everything else proceeds. The final message lists rulings and deferred minors; it asks no question.

## Global Constraints

- Taxable account only; free data only, no purchase or subscription; no retirement-account or sleeve-retirement recommendations (owner, 2026-10-20/23).
- Preserve production freeze (~2026-12-29 lift), all recorded NO-GOs, parked event studies and `src/ggTrader/paper/`. No broker calls, orders, Telegram sends, `docker compose pull/up`, or image deploys. Merging to `main` triggers the image-build workflow only; that is accepted, and nothing is pulled onto the live container.
- No market-return reruns and no new data acquisition. Reviewed windows are never called fresh OOS. No package selects a trading candidate; G3 stays "options listed, none recommended".
- Default behavior of every changed lab function is **unchanged** (legacy); new behavior is opt-in. The six characterization tests in `tests/lab/test_wfo.py` must pass unmodified.
- Core logic in `src/`; scripts have `main()`, `argparse`, `--help`; ruff `line-length = 100` rules E/F/W/I; tests only via `scripts/run_tests.sh`; type hints and one-line module docstrings.
- Delegation rule (owner): Codex Sol/Terra for code, audit conclusions, research synthesis and reviews; Pi/DeepSeek Flash only for simple bounded extraction or mechanical checks, verified before use; Sonnet for independent second reads and hand-calculation checks. If a delegate errors, returns unverifiable output, or its model is unavailable, redo the work natively and ledger it.
- Children never `git commit`/`push`, never touch files outside their ownership row, never delete anything. Parent commits after verifying.

## Delegation Matrix

| Package | Owner and model | Why |
|---|---|---|
| WP1 housekeeping, PR | Native | Tiny, shared-state operations |
| WP2 WFO policy options | **Terra** (`codex`, `gpt-6.1-terra`, reasoningEffort `high`); review by **Sol** | Medium-difficulty change to a core harness; golden-regression critical |
| WP3 synthetic study | **Terra** implements; interpretation drafted by **Sol** (`gpt-6.1-sol`, `high`) | Code is routine; conclusions are audit judgments |
| WP4 family-scope review | **Pi flash** (`pi`, `opencode/deepseek-v4.1-flash`, thinking `off`) extracts definitions; **Sol** synthesizes; **Sonnet** (`claudeAgent`, `claude-sonnet-5-5`, effort `high`) independent second read | Extraction is mechanical; classification is evidence synthesis |
| WP5 lot engine | **Sol** (`xhigh`) implements; **Sonnet** independently hand-computes expected values; a second **Sol** (`high`) reviews | Hardest logic (wash sales across year-end); needs cross-checks |
| WP6 snapshot | Native (`research-snapshot` skill) | Skill runs in the parent session |
| WP7 independent review of earlier tasks | **Terra** (`high`) reviews code/tests/docs; **Pi flash** mechanically re-checks numbers and links | Fresh eyes; cheap mechanical checks |
| WP8 integration, final review, merge | Native; final whole-branch review by **Sol** (`xhigh`) | Judgment task, most capable model |

Child task briefs must be self-contained (no parent history): include the goal, file ownership, constraints above, exact interfaces and test names from the package, the instruction "do not commit", and "report files changed, commands run and exact results".

## File ownership (parallel safety)

| Package | May create/modify |
|---|---|
| WP1 | `.gitignore`; PR operations |
| WP2 | `src/ggTrader/lab/wfo.py`, `tests/lab/test_wfo.py`, `research/programme/wfo-methods/2026-11-02-wfo-fallback-semantics.md` |
| WP3 | `src/ggTrader/lab/wfo_policy_synthetic.py`, `scripts/wfo_policy_synthetic.py`, `tests/lab/test_wfo_policy_synthetic.py`, `docs/research/2026-11-0*-wfo-policy-synthetic.md`, `research/programme/wfo-methods/_wfo_policy_synthetic_results.json` |
| WP4 | `docs/research/2026-11-0*-family-scope-review.md`, `research/stock-signals/feasibility/_family_scope_extraction.json`, `research/stock-signals/feasibility/_free_data_feasibility_matrix.json` (additive fields only) |
| WP5 | `src/ggTrader/lab/tax_lots.py`, `tests/lab/test_tax_lots.py`, `research/paper-system/tax-lots/2026-11-02-after-tax-lot-engine-requirements.md` (status column only) |
| WP6 | `research/RESEARCH_SNAPSHOT.md` and the two prompt templates (via the skill) |
| WP7 | review notes under `docs/research/2026-11-0*-independent-review-of-followups.md` only |
| WP8 (parent only) | `research/RESEARCH_INDEX.md`, `docs/changelog.md`, `docs/next_steps.md`, this plan's checkboxes |

## Review Focus

1. **Re-entry deadlock:** excluding anchor-fold WFE from the breaker history must not also feed `None` to `check_shadow_reentry`; while halted every fold deploys the anchor, so that would make re-entry impossible (WP2).
2. **NaN WFE:** under `nan_wfe="exclude"` a trailing NaN must not hide a chronic-decay halt; under `"legacy"` it still does (WP2).
3. **Wash sale across partial quantities and the sale itself:** replacement shares exclude shares being sold; a 100-share buy, 60-share loss sale and 40 prior shares must disallow only the matched part (WP5).
4. **Forward window across year-end:** a late-December loss sale with a January repurchase must amend the prior year's loss without rewriting the original ledger event (WP5).
5. **Oversell and corporate actions:** selling more than held raises `ValueError`; a split after a wash adjustment preserves basis and holding period (WP5).

---

## WP1: Housekeeping and PR (Native)

**Files:** Modify `.gitignore`.

- [ ] **Step 1:** Add to `.gitignore`: `.claude/skills/*-workspace/` (skill-eval scratch; fixes the 3 stray ruff errors in `ruff check .`). Do **not** delete these folders and do not touch `data/external/event_timing_audit/`.
- [ ] **Step 2:** Run `.venv/bin/ruff check .` → expected `All checks passed!`; `git status --short` shows no `-workspace` entries.
- [ ] **Step 3:** Commit `chore: ignore skill-eval workspaces`; push the branch.
- [ ] **Step 4:** Open a PR to `main` with `gh pr create` (title "Research follow-ups: process audit, WFO options, synthetic study, lot engine"; body lists packages and states: no live code change, defaults unchanged). Link it with `link_pull_request`. The body is completed in WP8.

## WP2: Opt-in WFO policy options (Terra, review Sol)

**Files:** Modify `src/ggTrader/lab/wfo.py`, `tests/lab/test_wfo.py`, `research/programme/wfo-methods/2026-11-02-wfo-fallback-semantics.md`.

**Interfaces:**
- Produces in `wfo.py`: `@dataclass(frozen=True) class WfoPolicy` with `nan_wfe: Literal["legacy", "exclude"] = "legacy"` and `anchor_fold_wfe: Literal["legacy", "exclude_from_breaker"] = "legacy"`.
- `check_circuit_breaker(state: WfoState, window_size: int = 4, wfe_floor: float = 0.25, nan_policy: Literal["legacy", "exclude"] = "legacy") -> WfoState` (existing signature plus one defaulted keyword).
- `run_wfo(..., universe_fn=None, policy: WfoPolicy = WfoPolicy())` (new trailing keyword).
- Each fold row gains `"halted_entering": bool` (state used for the anchor decision). `WfoResult` gains `skipped_folds: Tuple[Dict[str, Any], ...] = ()` with rows `{"fold_num", "train_start", "train_end", "test_start", "test_end", "reason": "no_train_metrics"}`; `fold_results` still omits skipped folds. Everything is additive; no existing key changes.
- `anchor_fold_wfe="exclude_from_breaker"` appends `None` instead of the mixed WFE to `wfe_history` on anchor folds, but `check_shadow_reentry` still receives the computed WFE.

- [ ] **Step 1: Write failing tests** appended to `tests/lab/test_wfo.py` (reuse `_scripted_wfo`, extend it to accept `policy`): `test_circuit_breaker_nan_policy_exclude_halts`, history `[0.1, 0.1, 0.1, 0.1, float("nan")]` with `nan_policy="exclude"` → `halted is True`; `test_circuit_breaker_nan_policy_legacy_unchanged` (same history, default → `halted is False`); `test_run_wfo_records_halted_entering` (fold 1 `halted_entering is False`, fold 2 `True` under the existing `halt_now` stub); `test_run_wfo_result_exposes_skipped_folds` (skip fold index 1 → `result.skipped_folds[0]["fold_num"] == 2`, reason `"no_train_metrics"`, and `fold_results` still lacks fold 2); `test_policy_exclude_from_breaker_keeps_shadow_reentry_wfe` (halted run: re-entry still evaluated with the numeric WFE; assert the run re-enters after two clean folds exactly as in legacy); `test_default_policy_leaves_fold_rows_identical_for_legacy_keys` (rows from default policy equal the six existing characterization scenarios' asserted values).
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_wfo.py -q` → Expected: new tests FAIL (missing `WfoPolicy`/keywords/fields), the six characterization tests PASS.
- [ ] **Step 3: Implement** the interfaces above in `wfo.py` with minimal edits (NaN filter `w is not None and w == w` only when `nan_policy == "exclude"`).
- [ ] **Step 4: Run** `scripts/run_tests.sh tests/lab/test_wfo.py tests/lab/test_wfo_attribution.py -q` → Expected: all PASS, characterization tests unmodified.
- [ ] **Step 5: Mutation check:** temporarily revert the re-entry argument (pass `None` on anchor folds) → `test_policy_exclude_from_breaker_keeps_shadow_reentry_wfe` must FAIL; restore. Verify `git diff src/ggTrader/lab/wfo.py` is only the intended change.
- [ ] **Step 6:** Update the semantics doc: for F1, F3, F5 and the F2(b) variant, state "implemented as opt-in `WfoPolicy` (default legacy); owner adoption pending"; document the re-entry deadlock hazard (Review Focus 1) as the reason F2(b) only changes breaker history. F4 and F6 remain documentation-only.
- [ ] **Step 7:** Full lab suite `scripts/run_tests.sh tests/lab -q`; `ruff check` and `ruff format --check` on touched files.
- [ ] **Step 8 (Sol review):** delegate a read-only review of the diff (checks: defaults unchanged, no behavior change for callers `lab/cli.py:189` and `lab/blend.py:112`, deadlock hazard covered). Apply Critical/Important findings with a failing test first.
- [ ] **Step 9:** Parent commits `feat(lab): opt-in WFO policy options (defaults unchanged)`.

## WP3: Synthetic WFO policy study (Terra, interpretation Sol; after WP2)

**Files:** Create the WP3 paths in the ownership table.

**Interfaces:**
- `wfo_policy_synthetic.py`: `simulate_run(rng: np.random.Generator, mu: float, p_gate: float, policy: WfoPolicy, n_folds: int = 17, p_nan: float = 0.05) -> Dict[str, float]` returning `anchor_share`, `halted_fold_share`, `ever_halted` (0/1), `n_halt_episodes`, `mean_halt_length`, `reentries`; `run_grid(seed: int, n_reps: int, mus, p_gates, policies) -> Dict[str, Any]`.
- Model (labelled assumptions, **not** the real NDH/DSR gates): winner IS Sharpe ~ N(mu + 0.5, 1.0); deployed OOS Sharpe ~ N(mu, 2.0) when the winner is deployed and N(0.5·mu, 2.0) when the anchor is deployed; OOS is NaN with probability `p_nan`; gates pass with probability `p_gate`; breaker and re-entry use the **real** `compute_wfe`, `check_circuit_breaker`, `check_shadow_reentry`, `WfoState`.
- Grid: `mu ∈ {0.0, 0.5, 1.0, 1.5}`, `p_gate ∈ {0.3, 0.7, 1.0}`, policies `{legacy, nan_wfe=exclude, anchor_fold_wfe=exclude_from_breaker}`, `n_reps = 5000`, `seed = 20261102`.

- [ ] **Step 1 (preregister, before any run):** write the report skeleton with scenarios, seed, and interpretation rules: (a) if P(ever halted) at `mu=1.0, p_gate=1.0` exceeds 0.5, a halt is likely for a positive-Sharpe strategy under this model, else not; (b) report whether `exclude` policies change halt rates by more than 5 percentage points; (c) every statement is "under these synthetic assumptions" and implies nothing about a real strategy or the real gates. Commit this file's preregistration text hash in the ledger.
- [ ] **Step 2: Write failing tests** `test_simulate_run_is_deterministic_for_seed`, `test_zero_p_gate_means_anchor_share_one`, `test_policy_legacy_matches_direct_breaker_calls` (hand-built 3-fold path compared to calling the real functions), `test_run_grid_resumable_output_schema`.
- [ ] **Step 3:** run → FAIL; implement the module and the script (`--out`, `--reps`, `--seed`, `--help`; runs well under 5 minutes so no checkpoint, noted in the docstring); run → PASS.
- [ ] **Step 4:** run the script; write `_wfo_policy_synthetic_results.json`; spot-check one cell by a 200-rep independent recompute.
- [ ] **Step 5 (Sol):** draft the interpretation against the preregistered rules only; parent verifies each cited number against the JSON, and records the answer to the "halt is near-certain" claim as supported or not supported under the model.
- [ ] **Step 6:** parent commits `research: synthetic WFO policy study`.

## WP4: Family-scope review (Pi extraction, Sol synthesis, Sonnet second read)

**Files:** Create the WP4 paths in the ownership table.

**Interfaces:** `_family_scope_extraction.json`: `{closed_reports: [{id, path, sort_variable, universe, weighting, rebalance, hypotheses: [...], source_lines: [...]}]}` for the momentum, four-factor and net-issuance screens (reports and briefs under `research/`). Matrix rows S02–S08 and S13 gain `family_scope_review: {outcome, rationale, evidence}` with `outcome ∈ {closure_applies_inferred, distinct_mechanism, undetermined}`. Recorded verdict fields must stay byte-identical.

- [ ] **Step 1 (Pi flash, extraction only):** brief: for each closed report fill the fields above with quoted source lines; do not judge any other family. Parent verifies every field against the source line (spot check all `sort_variable` and `hypotheses` entries).
- [ ] **Step 2 (Sol):** for each pending row compare sort variable, mechanism, universe, weighting, rebalance and registered hypotheses with the closed families using **definitions only**: no returns, no correlations. Output rationale per row, with the closed report's own scope statement quoted.
- [ ] **Step 3 (Sonnet):** independent second read of the same inputs without seeing Sol's outcomes; parent reconciles disagreements by choosing `undetermined` unless the evidence settles it.
- [ ] **Step 4:** write the review document; add the additive fields to the matrix; script check: `recorded_verdict`, `status_vs_prior_work` and `classification` of closed rows are unchanged versus `HEAD`, and every pending row now has `family_scope_review`.
- [ ] **Step 5:** `python3 -m json.tool` on both JSON files; links resolve; the document lists options for G3 but selects none and recommends none.
- [ ] **Step 6:** parent commits `research: family-scope review of pending feasibility rows`.

## WP5: After-tax multi-stock lot engine (Sol, Sonnet cross-check, Sol review)

**Files:** Create `src/ggTrader/lab/tax_lots.py`, `tests/lab/test_tax_lots.py`; edit only the status column of the requirements doc.

**Interfaces (all in `tax_lots.py`; the existing `Book` in `a23_rebalance.py` stays unchanged):**
- `@dataclass(frozen=True) class TaxParams: st_rate: float = 0.24; lt_rate: float = 0.15`. Costs are applied by the caller to prices (R13 frozen-cost convention), keeping the engine pure.
- `class TaxLotBook` with: `__init__(self, params: TaxParams, cash: float = 0.0)`; `buy(self, symbol: str, qty: float, price: float, date: pd.Timestamp) -> str` (lot id); `sell(self, symbol: str, qty: float, price: float, date: pd.Timestamp) -> List[Realization]` (raises `ValueError` if `qty` exceeds held); `split(self, symbol: str, ratio: float, date: pd.Timestamp) -> None`; `rename(self, old: str, new: str, date: pd.Timestamp) -> None`; `dividend(self, symbol: str, per_share: float, date: pd.Timestamp, qualified: bool) -> float` (credits cash, returns amount); `interest(self, amount: float, date: pd.Timestamp) -> None`; `year_end(self, year: int, asof: pd.Timestamp) -> float` (tax due; deterministic); `amended_tax(self, year: int) -> float` (delta tax for a closed year after later wash adjustments); `pay_tax(self, amount: float, date: pd.Timestamp) -> None` (raises `InsufficientCash` if cash < amount; funding sales are the caller's separate next-session decision); `liquidation_value(self, prices: Mapping[str, float], date: pd.Timestamp) -> float` (cash + market value − tax accrued − unrealized-lot tax; no state change); `terminal_liquidate(self, prices: Mapping[str, float], date: pd.Timestamp) -> float`; `events` (tuple of `LedgerEvent`), `verify_chain() -> bool`.
- `@dataclass(frozen=True) class Realization: lot_id, symbol, qty, proceeds, basis, gain, term ("ST"|"LT"), status ("final"|"provisional"), disallowed: float`. `LedgerEvent(seq, kind, date, payload, prev_hash, hash)` (sha256 chain; corrections are new events referencing `ref_seq`, never rewrites). `class InsufficientCash(Exception)`.
- Rules decided here: HIFO per the standard with tie-break (basis desc, acquisition date asc, lot id asc), no long-term-first rule. LT iff held more than one year. Wash sale: a loss sale is matched against shares of the same symbol bought within 30 days before or after the sale date, excluding the lots being sold; disallowed loss = loss × matched/sold quantity; it is added to the replacement lot's basis and that lot's acquisition date is moved earlier by the days the sold lot was held (tacked holding period). A loss is `provisional` until sale date + 30 days; a later purchase inside the window appends an adjustment event. Netting: ST and LT net separately, carryforwards keep character (`cf_st`, `cf_lt`), cross-netting only when one side is net negative, no $3,000 ordinary credit, net loss remains a carryforward valued at zero at the terminal date; unqualified dividends and interest taxed at `st_rate`, qualified dividends at `lt_rate`.

- [ ] **Step 1 (Sol):** write the failing tests listed in the requirements doc §4 with hand-computed literals: `test_wash_sale_forward_window_disallows_loss`, `test_wash_sale_backward_window_partial_quantity` (buy 40, buy 100, sell 60 at a loss → 40 matched share fraction), `test_wash_sale_excludes_lots_being_sold`, `test_wash_sale_adjusts_holding_period`, `test_wash_sale_loss_provisional_until_window_closes`, `test_late_december_loss_january_repurchase_amends_prior_year`, `test_hifo_tie_break_is_deterministic`, `test_carryforward_preserves_character`, `test_no_3000_ordinary_income_credit`, `test_split_preserves_basis_and_holding_period`, `test_rename_keeps_lot_dates`, `test_qualified_vs_unqualified_dividend_rates`, `test_interest_taxed_as_ordinary`, `test_pay_tax_raises_insufficient_cash_not_same_day_sale`, `test_oversell_raises_value_error`, `test_spy_benchmark_uses_identical_engine`, `test_liquidation_value_includes_unrealized_lot_tax_and_does_not_mutate`, `test_terminal_liquidation_after_tax_cagr_matches_hand_calculation`, `test_ledger_append_only_correction_preserves_original_event`, `test_ledger_chain_detects_tampering`.
- [ ] **Step 2 (Sonnet, independent):** from the rules text above and the test **descriptions only** (not the implementation), compute the expected numbers for the wash-sale, year-end amendment, carryforward and terminal-liquidation scenarios; parent diffs them against Sol's literals and resolves every mismatch by checking the rule text, correcting the wrong side.
- [ ] **Step 3:** run `scripts/run_tests.sh tests/lab/test_tax_lots.py -q` → Expected: FAIL (module missing).
- [ ] **Step 4 (Sol):** implement `tax_lots.py` to pass; run → PASS; `ruff` clean.
- [ ] **Step 5 (second Sol, read-only review):** checks rule-to-code fidelity against the requirements doc R1–R16, determinism, no mutation in valuation, and that `a23_rebalance.py` is untouched (`git diff` empty); apply Critical/Important findings with a failing test first.
- [ ] **Step 6:** update the requirements doc status column: R1–R9, R11, R12, R14, R16 → "implemented in `tax_lots.py` (library only, unvalidated against a real broker ledger)"; R10 → "supported (same engine, test)"; R13 stays "caller responsibility". No claim of tax validity.
- [ ] **Step 7:** parent commits `feat(lab): after-tax multi-stock lot engine (library only)`.

## WP6: Research snapshot (Native)

- [ ] **Step 1:** after WP2–WP5 are committed, invoke the `research-snapshot` skill to regenerate `RESEARCH_SNAPSHOT.md` and the two prompt-template context sections. Do not hand-edit them.
- [ ] **Step 2:** verify the snapshot still lists every recorded NO-GO unchanged and mentions the new follow-ups only as methods/feasibility entries; `git diff --stat` shows only the expected files.
- [ ] **Step 3:** parent commits `docs: regenerate research snapshot`.

## WP7: Independent review of the earlier work (Terra + Pi flash)

- [ ] **Step 1 (Pi flash, mechanical):** re-extract every number in `2026-11-02-wfo-fold-attribution.md` §2 from `_wfo_fold_attribution_results.json` and list mismatches; check that all local links in the four 2026-11-02 reports resolve. Parent re-verifies any reported mismatch before acting.
- [ ] **Step 2 (Terra, read-only):** review `wfo_attribution.py`, its driver and tests, the characterization tests, `2026-11-02-after-tax-lot-engine-requirements.md` against the standard §§3, 4, 6, and the index/changelog entries. Brief includes the Global Constraints and asks for severity-ranked findings with evidence.
- [ ] **Step 3:** fix Critical/Important findings with a failing test or a documented correction first; ledger Minor ones as deferred.
- [ ] **Step 4:** parent commits `fix: address independent review of research follow-ups` (only if there were fixes).

## WP8: Integration, final review, merge (Native, Sol review)

- [ ] **Step 1 (parent only):** update `RESEARCH_INDEX.md` (new rows: synthetic study, family-scope review, lot-engine library, WFO policy options), one `changelog.md` entry per package, and the `next_steps.md` pointer (open items now: owner adoption of WFO policy, direction G3, lot-engine validation).
- [ ] **Step 2:** full verification: `scripts/run_tests.sh tests -q` (expect no failure; a 137 exit means a hang, investigate and do not raise the cap), `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `python3 -m json.tool` on every new JSON, link check on new Markdown, `git diff --check`, and `git diff origin/main --stat -- src/ggTrader/paper` must be empty.
- [ ] **Step 3:** scope audit of all new documents: no verdict override, no paid-data suggestion, no retirement or account recommendation, no candidate recommendation, no fresh-OOS claim.
- [ ] **Step 4 (Sol, `xhigh`, read-only):** whole-branch review using the superpowers `review-package` script against the merge base, with this plan's Review Focus and ledger rulings. Re-grade by effect; fix Critical/Important in one pass (failing test first), ledger Minor.
- [ ] **Step 5:** push; update the PR body (packages, rulings, deferred minors, "defaults unchanged", "no live-code change"). Call `watch_pull_request` and end the turn; do not poll.
- [ ] **Step 6 (on wake):** if checks are green and no Critical remains, `gh pr merge --merge` (repository history uses merge commits); then `unwatch_pull_request`. If checks fail, diagnose once with systematic-debugging; if still red, leave the PR open and report. Do not deploy anything.
- [ ] **Step 7:** delete this plan's SDD workspace; write the final message: results per package, "Rulings I made", "Deferred minors", and what remains the owner's (adoption of any WFO policy option, direction G3). No questions.

---

## Master checklist

- [ ] WP1 housekeeping and PR
- [ ] WP2 WFO policy options + Sol review
- [ ] WP3 synthetic study (preregistered)
- [ ] WP4 family-scope review (Pi extract, Sol, Sonnet)
- [ ] WP5 lot engine (Sol, Sonnet hand-calc check, Sol review)
- [ ] WP6 snapshot regenerated
- [ ] WP7 independent review of earlier tasks
- [ ] WP8 full verification, Sol whole-branch review, PR merged
- [ ] Final scope audit: freeze, NO-GOs, paper code, free data, taxable account intact

## Parallelism and order

WP1 first. Then WP2, WP4 and WP5 can run concurrently (disjoint files). WP3 starts after WP2 is committed. WP6 after WP2–WP5. WP7 may run alongside WP6. WP8 last. Planning estimate (unmeasured): WP2 ~1 day, WP3 ~1 day, WP4 ~1 day, WP5 ~3–5 days, others ~0.5 day each; wall-clock shorter with the parallel delegates.

## Self-review notes

- Coverage: the six next steps offered earlier map to WP1 (land), WP2 and WP3 (G2 and Task 6), WP4 (G3 groundwork, no selection), WP5 (lot engine), WP6 (snapshot), WP7 (independent review).
- Open assumptions to confirm during execution: that `opencode/deepseek-v4.1-flash` on the `pi` provider is reachable (fall back to native if not); that merging the PR to `main` is covered by the owner's "do not prompt" instruction (it is a docs/tests/lab-only change with no deploy); the synthetic model parameters are assumptions, flagged as such in the report.
- Known limits: the synthetic study does not test the real NDH/DSR gates; the lot engine is unvalidated against a real broker ledger and corporate-action feed.
