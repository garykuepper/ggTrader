# Working Task List

**Last updated:** 2026-11-02 (revised after independent ranking by Codex Sol and Gemini Argon). **Status:** living document. `docs/next_steps.md` stays the 1–2-step worklist and mirrors only the next two steps; this file holds the full ranked queue, the calendar track, the reasoning and the glossary.

**How to use it.** Work **Track A** roughly in order, one item at a time. **Track B** items are dated or recurring obligations that run alongside, not inside, the queue. When an item finishes, mark it Done with the date and the commit or report that closed it, then promote the next item into `next_steps.md`. Follow the standing owner instructions in [`AGENTS.md`](../AGENTS.md#owner-instructions-2026-10-20): focus on research, strategy evidence and paper-trading reliability; taxable account only; no paid data; no retirement or sleeve-retirement recommendations.

**Ranking rule.** Importance to the stated priorities (trustworthy evidence, paper-trading reliability), then time-sensitivity, then how much an item unblocks or de-risks, then effort and reversibility, then information value (what we learn even from a NO-GO), then risk of wasted work. Effort figures are planning estimates, not measurements.

**Terms used below.** *WFO* = walk-forward optimization (train on past data, test on the next stretch). *WFE* = walk-forward efficiency, out-of-sample (OOS) Sharpe divided by in-sample Sharpe. *Anchor* = the defensive fallback parameter set the harness deploys when gates fail or the circuit breaker is halted. *Part A* = the long free-history test in the [forward-paper standard](research/briefs/2026-10-23-forward-paper-standard.md): **A1** is a Holm-corrected Newey–West t ≥ 3 on monthly excess returns; **A2** is at least +1.0 percentage point per year (pt/yr) net CAGR advantage since 1994. *B0* is a kill-only replay on the clean 2021–2026-04 tape; *B1* is an 18-month forward shadow book. *PIT* = point-in-time. *Owner* means Flynn decides.

## Execution update (2026-11-02)

The three A3 regression fixes were merged into `main` and pushed in integration
commit `9eaf0389`; 117 targeted tests and Ruff checks passed. The production
container was not pulled or restarted. The [readiness reconciliation](research/2026-11-02-paper-freeze-readiness.md)
finds a persistent untradable AVB condition and missing decision-time evidence;
a clean parity window is **not established**. The effective start, reset
and shadow scope remain unresolved. The [sweep design](research/2026-11-02-two-symbol-sweep-design.md)
now separates the paper sweep, proposed paper design and real-account helper;
its future funding policies remain proposals.

The [accruals feasibility review](research/2026-11-02-accruals-feasibility.md)
remains **unresolved** on usage, accounting revisions and cell metadata; it
proposes a benchmark/cost protocol without examining returns. A4b/c remain
conditional. The [factor wording recommendation](research/2026-11-02-family-scope-review.md#7-wording-only-recommendation-2026-11-02)
is not an owner ruling. The insider amendment remains unapproved.

## 1. Track A: ranked queue

| # | Item | Decides / does | Status | Hard prerequisites | Effort |
|---|---|---|---|---|---|
| A1 | Owner decision batch: factor-scope ruling; insider amendment | Owner | Open | None | Minutes |
| A2 | Freeze-readiness (part 1): qualifying evidence, incident ledger, pre-lift checklist | Codex drafts; Owner decides | Partially reconciled; AVB and evidence gaps unresolved | None | Small to medium |
| A3 | Fix three deferred code minors | Codex | Done 2026-11-02; integrated in 9eaf0389 | None | Under an hour |
| A4 | Accruals (S04): feasibility check → registered brief → one Part A run | Owner approves brief; Claude runs | Proposed | A4a metadata check passes; A1 (soft) | ~30 min check; about a day to run if feasible |
| A5 | Correct stale instructions in `next_steps.md` | Claude | Done 2026-11-02 (buy-or-park marked resolved) | None | Minutes |
| A6 | `TaxLotBook` validation and B0/B1 integration | Claude | Conditional | A candidate passes Part A | Several days |
| A7 | Decide whether to adopt any `WfoPolicy` option | Owner | Conditional | A WFO-based study is proposed | Minutes |
| A8 | Optional repo housekeeping (branches, stash) | Claude, with Owner OK | Optional | None | Minutes |

## 2. Track B: calendar and recurring

| # | Item | When | Who |
|---|---|---|---|
| B1 | Month-end A23 rebalance check | After each month's last NYSE close, once the real position is funded | Owner runs the advisory helper |
| B2 | Freeze-lift readiness review and deploy decision (part 2 of A2) | Date unknown until AVB, evidence and window start are adjudicated | Owner decides; Claude prepares |
| B3 | Research snapshot refresh | After the next closed research arc | Claude, via the `research-snapshot` skill |
| B4 | 2027-09-30 A23 annual review (including 15% vs 30%) | 2027-09-30 | Owner |

## 3. Item detail

### A1. Owner decision batch (minutes)
- **(a) Factor-scope ruling.** The factor-screen report says: "Do not re-propose or re-screen these factors on adjusted windows or alternate factor definitions" ([report §4](research/2026-10-21-factor-screen-nogo.md)). "These factors" are profitability (HiOP), value (HiBM), investment (LoINV) and the value-plus-momentum combo. "Alternate factor definitions" is not defined.
  - **Narrow reading:** it means the same four factors, retuned to rescue them. Then valuation variants (S05), quality (S07) and the q-factor rows (S08) are not closed by that sentence.
  - **Broad reading:** it also covers any redefinition of value, profitability or investment. Then S05, S07 and S08 are closed.
  - **Scope.** The sentence reaches S05, S07 and S08 only ([family-scope review](research/2026-11-02-family-scope-review.md)). It does **not** govern industry momentum (S06) or factor momentum (S13), which overlap the separate momentum closure, nor accruals (S04), reversal (S01, S02) or low risk (S03). A narrow ruling does not make S05, S07 or S08 test-ready: they remain `undetermined` until their constructions are defined, and S08 is `parked` pending a long-only implementability check.
  - Record the ruling in the factor report §4 and the family-scope review.
- **(b) Insider header-timestamp amendment.** [The DRAFT amendment](research/briefs/2026-10-23-insider-acceptance-time-amendment.md) proposes SEC header acceptance times; offline it reproduced 25/25 header-consistent and 9/25 API-inconsistent cached pairs, and its SEC documentation citation is unverified. Insider and PEAD studies are PARKED and nothing depends on this. **Leaving it unapproved is a valid answer**; reject it only if you want the loop closed. Low weight: do not let it delay A2 or A4.

### A2. Freeze-readiness, part 1 (do now)
- **Why this ranks high.** It is the paper-trading reliability work you prioritised, and the risk to avoid is treating the calendar as clearance. The 3-month parity window (since the 2026-09-29 deploy) runs to *about* 2026-12-29, **and any parity incident resets the clock** ([next_steps, "Months 0–3"](next_steps.md)). Exact historical live parity is also still unverified: 10 of the 32 audited orders are explained, and 19 of the 32 were manual orphan or legacy liquidations on 2026-09-29, so the figure describes missing decision inputs and cleanup trades, not a failed signal ([parity JSON](research/_replica_parity_20261019.json)).
- **Deliverables (documents only; no change to `src/ggTrader/paper/`, `.env` or the container before the window ends):**
  1. *Qualifying evidence.* Written criteria for what counts as a clean parity window: stock names ≤ `MAX_POSITIONS`, no missed exits, no orphan or untradable holdings, shadow weights and rebalances match the rule.
  2. *Incident ledger.* A dated record of any parity incident and its reset effect, with a named owner for the ongoing observation and the location of the evidence (`paper_trades`, `paper_snapshots`).
  3. *Pre-lift checklist.* SSO fund fee and closure check (the fee waiver expired 2026-09-30); reconcile actual SSO distribution tax character against the modeled bounds; rollback path; deploy order via the `ggtrader-deploy` skill (push, image build, `docker compose pull`, `up -d`).
  4. *Two-symbol sweep design and test plan.* Replace the single-symbol SPY sweep with the 70/30 SPY/SSO sweep and the 22–38% band, entering real money at 10% of intended capital first ([A23 report](research/2026-09-30-static-fractional-leverage.md)). The design must address how the **daily** stock-sleeve cash needs interact with a **month-end-only** band rebalance and highest-basis lot selling.
- **Open question for the Owner.** Whether the sweep code may be written and unit-tested on an unmerged branch before the window ends. Sol's review kept A2 to documents; Argon would code it now without deploying. The A23 brief says live implementation waits for the window; I default to documents only.
- **Done when.** Items 1–4 are written and approved. Part 2 (B2) is the actual lift review and deploy.

### A3. Deferred code minors — DONE 2026-11-02
The following were fixed with regression tests and integrated in `9eaf0389`. Do these now, not "at the next edit": `tax_lots.py` is parked behind A6 and may not be touched for months.
1. `tax_lots`: a fully matched fractional wash loss can stay `provisional`, because `loss_status` compares dollar totals exactly while `sell()` decides finality by quantity exhaustion (repro: sell 0.3 shares bought at $1 for $0.80, repurchase three 0.1-share tranches).
2. `tax_lots`: a tiny sold quantity matched against a much larger replacement lot is skipped by the exhaustion check, which is measured against the replacement lot.
3. `tests/lab/test_wfo.py::_scripted_wfo`: the gate-script index counts gate calls while sweeps use the fold index, so heterogeneous scripts misalign after a skipped fold (no current test exposes it).

### A4. Accruals (S04) Part A screen
- **Why this candidate.** The family-scope review rated it the only pending row `distinct_mechanism` (both reviewers). Source: Ken French "Portfolios Formed on Accruals".
- **A4a, metadata check (~30 minutes, no returns).** Confirm the file's start date reaches 1994 or earlier, its formation timing and lag, revision behavior and terms of use. An infeasibility finding is a valid result; stop there if so.
- **A4b, registered brief.** Before any computation (standard §2): the family, long-history source, fixed windows, cost haircut and exact tests, with the long-only portfolio chosen up front. A1 and A2 as defined above; Holm across the registered family; no cross-study error control. Owner approves the brief.
- **A4c, one run.** Raw results saved as JSON; dated report; NO-GO stated plainly if so.
- **How to read the result.** The A1 power note estimates the chance of passing t ≥ 3 and assumes illustrative tracking errors; it is not a candidate-specific probability, and it does not cover A2 or joint passage. The accruals effect is generally reported to have weakened after publication (background knowledge, not verified here). A NO-GO means this registered test failed its bar; it neither closes the whole free-data inventory (which is non-exhaustive) nor proves the true effect is below +1 pt/yr.
- **If it passes.** It authorizes further research, not capital. Stock-level B0/B1 for an accruals signal would also need a **point-in-time fundamentals pipeline (S11)**, which does not exist: SEC Financial Statement Data Sets start in 2009 Q1 and their acceptance-date field is unconfirmed. Scope S11 in A6, not before.

### A5. Correct stale instructions in `next_steps.md`
`next_steps.md` still lists, as an active step, "Owner: decide the memo's buy-or-park choice for each signal", while the same file records that both studies are PARKED and no paid data will be bought. Mark it resolved (park; no purchase). The insider amendment step is covered by A1(b). Documentation corrections like this go ahead of any branch or stash deletion.

### A6. `TaxLotBook` validation and integration (conditional on a Part A pass)
- **What exists.** `src/ggTrader/lab/tax_lots.py` (46 tests): HIFO lots, wash sales over ±30 days, character-preserving carryforwards, qualified vs unqualified dividends, splits and renames, a hash-chained append-only ledger. Library only; not wired into any strategy; not checked against a real broker ledger ([status](research/2026-11-02-after-tax-lot-engine-requirements.md)).
- **A6a, engine validation.** Hand-build one full year of real-style lots, including a wash sale that crosses year-end, and check every number.
- **A6b, study integration.** Not only receipts and outside holdings: merger handling beyond renames, distribution sourcing and qualification, the cost and tax of any tax-funding sale (R17), no spendable reserve (R18), provisional-fact handling (R19), outside-holdings reporting (R20), decision-before-fill commits with independent timestamp receipts, input and rules hashes and void-day handling (R21), benchmark consistency, tax-drag reporting, and the S11 point-in-time fundamentals pipeline.
- **Trigger.** Start only after a candidate passes Part A. Starting earlier risks all of the effort being wasted.

### A7. Adopt any `WfoPolicy` option?
- **Default.** Leave everything at legacy until a future brief actually uses walk-forward; then decide with the fold-split report in hand. The fixed-portfolio Part A screen in A4 does not use the WFO harness.
- **What is and is not a switch** (see [fallback semantics](research/2026-11-02-wfo-fallback-semantics.md)): F1 (`halted_entering`) and F5 (`skipped_folds`) are already reported unconditionally and need no decision. F2 (`anchor_fold_wfe`) and F3 (`nan_wfe`) are opt-in switches. F4 (whether non-finite folds break "consecutive negative") and F6 (the DSR trial count) have **no implementation yet**: adopting either would need code.
- **Evidence to weigh.** In the [synthetic study](research/2026-11-02-wfo-policy-synthetic.md), `nan_wfe="exclude"` moved halt probability by at most 0.021; `anchor_fold_wfe="exclude_from_breaker"` moved it by more than 0.05 in 5 of 12 cells, all with `p_gate < 1`. Synthetic assumptions only.

### A8. Optional repo housekeeping (destructive; needs Owner OK)
- **Merged old branches** (safe to delete): `cross-sectional-research`, `docs-roadmap-update-2026-06-26`, `fix/paper-trading-correctness-phase-a`, `ndh-variance-density-exempt`, `wfo-live-param-stability`. `worktree-paper-trading-remediation` is attached to a worktree and needs care.
- **Not merged, do not delete without a look:** `research/intraday-stage0a`, `research/momentum-retest`.
- **Stash:** `stash@{0}` ("WIP on main: 336c912 fix(lab): pass volume to _generate_signals in train_gate") is old; inspect before dropping.
- **Never delete:** `data/` (including the untracked 350 MB `data/external/event_timing_audit/` cache).

### B1–B4, calendar items
- **B1.** Run `scripts/a23_rebalance_check.py --lots <your lots CSV> [--new-cash N]`. It proposes trades only when SSO leaves 22–38%, restoring 30% with cash first and then the highest-basis long-term lots. It never places orders and has no drawdown logic (`docs/cli_reference.md`). Dormant until the real position is funded (not expected before the freeze lifts; I could not verify funding status from the repo).
- **B2.** The lift is an earliest eligibility date, not an appointment to deploy. Require the incident ledger (A2) to show a clean window and an explicit readiness judgment before any implementation.
- **B3.** `docs/research/RESEARCH_SNAPSHOT.md` was last updated 2026-11-02; refresh it after the next closed arc, not before.
- **B4.** Present the century-leverage trade-off: 30% has 10.71% after-tax CAGR vs 9.47% at 0% but trips the −90% ruin flag (−91.4% vs −84.6%); 15% avoids the flag at 10.18%, a 0.54 pt/yr cost vs 30% ([report](research/2026-10-23-century-leverage.md)). The registered rule keeps 30%. Also review the rate clause (about +0.8 pt/yr at ~4% bills) and SSO fee and fund-closure risk. No intra-year exits and no drawdown trigger.

## 4. Where the independent reviewers differed

Codex Sol and Gemini Argon each ranked the original ten items without seeing the other. Both put freeze-readiness near the top, both moved the deferred code minors up and the dormant month-end check out of the sequence, and both asked for an accruals metadata check before the brief. They differed on:
- **Factor ruling vs freeze-readiness first:** Argon put the five-minute ruling first; Sol put freeze-readiness first and the ruling third. This list puts the quick owner batch first because it costs minutes and blocks nothing else.
- **Insider amendment:** Sol ranked it ninth, Argon fifth. Here it is bundled into A1 at low weight.
- **Timing of code minors:** Sol attached them to the gates that use the code; Argon wanted them fixed immediately. This list does them now (A3) because they are under an hour.
- **Coding the sweep before the window ends:** see the open question in A2.

Each reviewer's biggest risk is kept in view: mistaking elapsed time for verified paper-trading reliability (Sol), and an unstated point-in-time fundamentals blocker between a Part A pass and B0/B1, plus design friction between the daily stock-sleeve sweep and a month-end band rebalance (Argon).

## 5. Glossary

- **`WfoPolicy`.** A settings object in `src/ggTrader/lab/wfo.py` with two opt-in switches for the walk-forward circuit breaker: `nan_wfe="exclude"` (a missing WFE no longer hides a halt) and `anchor_fold_wfe="exclude_from_breaker"` (folds run on the fallback stop counting toward the decay check; re-entry still sees them so a halted run can recover). Defaults reproduce all past results. Nothing uses it yet.
- **"Alternate factor definitions".** See A1(a).
- **`TaxLotBook`.** See A6: a standalone after-tax lot-accounting library for turning a shadow stock book into an after-tax number.
- **Fold attribution.** Splitting saved walk-forward folds into gate failure, gates passed but anchor deployed, and winner deployed. Read this split before interpreting any WFO out-of-sample Sharpe ([report](research/2026-11-02-wfo-fold-attribution.md)).

## 6. Standing decisions (do not re-open without the owner)

- Taxable account only; no IRA or retirement-account advice.
- No paid data or subscriptions, now or later; Part B on paid history is unreachable.
- No recommendation to retire the stock sleeve, end paper trading or shut the project down; a NO-GO is not such a request.
- Production freeze through about 2026-12-29 (resettable by a parity incident); PEAD and insider studies PARKED.
- Closed NO-GOs stay closed unless a specific reproducible defect is documented; see [`RESEARCH_INDEX.md`](research/RESEARCH_INDEX.md) and [`RESEARCH_SNAPSHOT.md`](research/RESEARCH_SNAPSHOT.md) for what has been tried.

## 7. Change log

| Date | Change |
|---|---|
| 2026-11-02 | Created after PR #13 (process audit follow-ups) merged; ten items ranked. |
| 2026-11-02 | Revised after independent ranking by Codex Sol and Gemini Argon: split into a ranked queue (A1–A8) and a calendar track (B1–B4); owner decisions batched; freeze readiness reframed as an evidence and incident gate; accruals gated on a metadata check; code minors promoted; S11 dependency and the corrected scope of the factor sentence added. |
