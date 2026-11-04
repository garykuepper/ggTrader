# Root-Cause Investigation: Why Haven't Stocks or Crypto Beaten SPY?

**Date:** 2026-11-04 · **Status:** draft for owner approval · **Type:** investigation design (diagnosis only)

## 1. Intent

The owner has put months into ggTrader and sees marginal results against SPY in both
backtests and Alpaca paper trading. The goal is **2x SPY buy-and-hold in raw dollar
return**, not 2x risk-adjusted return. The owner (37, small capital, taxable account only,
high risk tolerance) can accept losing what is put in.

This investigation finds out **why** the results are marginal. It does not propose
strategies, fixes or deployments. Its output is an evidence-backed list of root causes
and of what is sound, so the owner can decide what to do next with trustworthy
information. The owner cannot judge the technical quality of the work personally, so
every finding must be explained in plain language with its evidence.

## 2. Goals, non-goals, anti-goals

**Goals**
- Rank every material past result (stocks and crypto) by how far it can be trusted.
- Find what was done wrongly, and also what was done right and can be relied on.
- Verify that each strategy was implemented as specified, and that the numbers in the
  reports are true.
- Explain, with evidence, why paper trading and backtests show only marginal gains.
- Say plainly what a 2x-SPY raw-return target would require, based on measured evidence.

**Non-goals**
- Proposing new strategies, parameter changes, or deployments.
- Any change to the frozen production paper trader, the container, cron, or orders.
- Resolving the open owner decisions in `docs/working_task_list.md` (factor-scope wording,
  insider amendment, A23 sizing).

**Anti-goals (this would be a failure even if it ran cleanly)**
- Shading results positive to match the owner's hopes, or negative to look rigorous.
- Treating a NO-GO as a recommendation to retire the sleeve, end paper trading, or shut the
  project down. Standing owner instruction: no such recommendation, and no IRA or
  retirement-account advice. Taxable account only; no paid data.
- Re-running things that touch live state, or consuming memory until the host wedges.

## 3. Constraints

- **Memory:** every backtest or WFO re-run runs inside a transient cgroup, e.g.
  `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`, modelled on
  `scripts/run_tests.sh`. At most 2 runs in parallel. A kill at the cap (exit 137) is a
  finding to report, not something to retry with a higher cap. (Host: 31 GB RAM; the
  2026-09-16 incident hit 10 GB and wedged the machine.)
- **Reproducibility:** every re-run pins `--eval-start` and `--eval-end`, uses
  `--max-leverage` matching the config under test (production is 1.0, default is 2.0), and
  records raw results to JSON next to its report.
- **Read-only on live and data:** never delete or modify `data/` or database configuration;
  no database writes; no orders; do not touch the `ggtrader_live` container.
- **Tests:** run only via `scripts/run_tests.sh`.
- **Delegation** (per the `AGENTS.md` delegation preference dated 2026-10-22, broadened
  2026-11-03): Gemini Flash (agy) takes bulk and batch work, because its quota is generous,
  and may also implement well-specified research, documentation and code tasks, given the
  objective, context, constraints and expected verification. A stronger model
  independently verifies Flash's sources, conclusions, changes and checks before anyone
  relies on them. Codex owns final evidence synthesis, audit conclusions and protocol
  decisions. Gemini Argon, Codex or Claude Opus give independent second opinions.
  DeepSeek Flash (via Pi) is for simple, bounded tasks only, and its output is verified.
  The lead stays responsible for final decisions. Private account records stay out of
  delegate prompts.
- **Definitions:** expand abbreviations (WFO, OOS, IR, PIT, etc.) on first use in every
  report written for the owner.

## 4. Starting evidence (already gathered 2026-11-04)

A read-only audit by a senior-quant Opus agent covered stocks only. The full report,
including its trust table and the re-run costs, is saved at
[`docs/research/2026-11-04-phase0-trust-audit.md`](../../research/2026-11-04-phase0-trust-audit.md)
(unreviewed subagent output; some entries are marked "prose", meaning not recomputed).
Its main findings are the starting point, to be confirmed or overturned by the lenses below:

1. No real stock-picking edge over SPY was found. Sound tests mostly returned reliable NOs.
2. The live account is about 84% SPY, 5% cash, and 3 stocks at about 3.3% each. A stock
   sleeve of about 10% cannot move the account much, so "marginal versus SPY" is partly
   built in.
3. The early "beat SPY" figures (Sharpe 1.12, 0.99; CAGR about 16%) were produced by five
   bugs (bars stamped a day early, duplicate SPY rows, an anchor leak, a point-in-time
   universe leak, DSR units). The corrected core WFO figure is Sharpe 0.59 versus SPY 0.78.
   Gate counts from before 2026-10-19 are void.
4. WFO mostly tests the fallback parameters rather than parameter selection, so a WFO
   NO-GO is weak evidence. Pass bars (Holm-corrected t of at least 3) are unreachable on
   five years of data, and there is no clean holdout period.
5. The paper shortfall versus SPY is mostly configuration churn, not signal.
6. Live decisions cannot be reconciled with backtests (10 of 32 orders explained, no
   decision inputs archived). Paper snapshot value is $135-$320 above cash plus positions
   on every day since 2026-09-15, probably a dividend add-back in `paper/trader.py:1050-1055`
   (unproven).
7. The best free-data long-only large-cap effects measured after 1994 are +0.1 to +0.8
   percentage points a year. A 2x-SPY goal needs roughly +10 to +15, or about 2x leverage.

Prior art: `docs/superpowers/specs/2026-11-01-research-process-audit-design.md`,
`docs/research/2026-11-01-research-process-audit.md`,
`docs/research/2026-10-19-production-replica-audit-nogo.md`, and the 2026-10-20 strategy
triage (49 rows: 32 flawed-test, 8 untested, 4 clean NO). The investigation extends
these; it does not redo them without cause.

## 5. The investigation lenses

Each lens is one or more read-mostly subagents that report to the lead. Each lists the
**questions it must answer**.

### Lens 1 - Backtest validity (stocks)
- After the five known bugs, what contamination remains (data tape, benchmark, universe,
  fills, costs)? Which published numbers predate which fix?
- Does any current headline number still rest on an unrecoverable or drifting window?

### Lens 2 - Verdict audit
- For each NO-GO, was the test able to detect an edge (power), and was the pass bar the
  pre-registered one? Which NOs are reliable, unreliable, or inconclusive?
- Did strict pass bars (t of at least 3, plus 1 point a year net since 1994) reject small
  real effects? What would the verdicts look like under a looser, clearly-labelled bar?
- How many tests were run in total, and how does multiple testing change what "a win"
  would have meant?

### Lens 3 - Live versus backtest parity
- What does the paper trader do that the tests did not model (partial-bar signals, MAX
  positions, sweep sleeve, orphans, manual liquidations, dividend handling)?
- How much of the paper shortfall versus SPY is explained by each configuration era?

### Lens 4 - Target feasibility and how professionals do it
- Based on measured effects, what return-over-SPY is plausible for a one-person,
  long-only, free-data operation, and what would 2x raw return require (breadth, leverage,
  concentration, long-short)?
- How do large systematic firms generate returns (breadth, data, financing, costs,
  capacity), and which of those constraints apply to the owner? Sources must be cited and
  free.

### Lens 5 - Coverage gaps
- What was never tested, as opposed to tested badly? Candidates to check: small and micro
  caps, intraday horizons (including the unmerged `research/intraday-stage0a` branch, 33
  commits ahead of main), concentration, long-short, and cross-asset breadth.
- Where did hypotheses come from, and how correlated or crowded are they?

### Lens 6 - Implementation correctness
- Does each strategy's code match its written specification? The reviewer re-derives the
  spec from the research brief and compares entry/exit rules, lookback windows, rebalance
  timing, and point-in-time data use against `src/ggTrader/lab/strategies/`.
- Hand-replay a small window (a few symbols, a few weeks) and compare with the code's
  output.
- Does the live path in `src/ggTrader/paper/` run the same logic the backtests tested?
- Start with `ensemble` (the live strategy), then the strategies behind the headline
  numbers.

### Lens 7 - Research accuracy
- Recompute every headline number from its raw results file and flag mismatches.
- Do the academic claims relied on say what the reports say they do (factor papers, the
  leverage study)? Free sources only.
- Was each verdict reached under the pre-registered pass bar, or under one chosen after
  the result?

### Lens 8 - Crypto
- What was tried, with what universe, fees, venue, and benchmark? Did any crypto strategy
  beat buy-and-hold BTC or ETH (the correct benchmark, not SPY)?
- Fees: the 2026-06-08 edge search found fees dominated. Is that finding sound and complete?
- Scope of what was tested: long-only spot only? Were shorting, perpetual futures or
  funding-rate carry ever evaluated (carry was parked)?
- Data integrity: do the same bug classes found in equities (timestamp shift, duplicate
  rows, delisted coins missing from the universe, gaps) affect the crypto tape in
  TimescaleDB? Check read-only.
- Why did live crypto barely trade (venue migration, allowlist crash loop), and what live
  evidence therefore exists?

### Lens 9 - Statistical power and regime
- With the data actually available, what size of edge could each test have detected?
- How much do conclusions depend on 2021-2026 being a single (largely bull) regime?

## 6. Sequencing

1. **Phase 0 (done):** the senior-quant trust audit, stocks only.
2. **Phase 1 (read-only, parallel):** Lenses 1-9 start from the existing evidence.
   Lens 6 and Lens 7 begin with the results the Phase 0 audit ranked most important.
3. **Phase 2 (re-runs, owner-approved list):** the lead collects the unknowns each lens
   says only a re-run can settle, with cost estimates, and the owner approves the list
   before anything runs. Cheap candidates flagged already: the stock sleeve versus
   equal-weight S&P 500 (15-30 min), and French top-decile big-cap momentum after 1994
   (15 min). Hours-to-days items (after-tax replica, July event-study re-runs, a clean
   2010-2020 holdout) wait for a separate decision.
4. **Phase 3 (synthesis):** one report for the owner with the ranked root causes, a trust
   table across stocks and crypto, what is sound, and what remains unknown. Final synthesis
   is Codex's; Gemini Argon or Claude Opus independently challenges it before delivery.

## 7. Deliverables

- One report per lens under `docs/research/`, following the repository research-report
  template, with raw JSON artifacts for anything re-run.
- A synthesis report: `docs/research/2026-11-xx-root-cause-investigation.md`, written in
  plain language, with abbreviations expanded.
- Updates to `docs/changelog.md` and the research snapshot (via the `research-snapshot`
  skill) once the synthesis is accepted. No change to `docs/next_steps.md` priorities
  without the owner.

## 8. Success criteria

- Every material stocks and crypto result carries a trust classification with its
  verification method (recomputed from raw artifact, or taken from prose).
- The ranked root-cause list separates "method and process errors", "implementation and
  live-parity errors", and "the evidence is negative or the target is structurally hard",
  each with citations.
- The owner can read the synthesis without technical background and say what the
  evidence does and does not support.

## 9. Left to the builder

- How many subagents per lens, and which model runs each, within the delegation
  preference in section 3.
- Exact memory caps per run within the 6 GB / 2-parallel limits.
- Report file names and the exact table layouts.

## 10. Open for the owner

- Approve this design, then the Phase 2 re-run list when it is ready.

## 11. Calls made by the lead (settled without the owner; check these)

1. **Phase 0 audit** is saved in the repo (section 4 link). Lenses start from it as an
   unreviewed evidence base: they may confirm or overturn it, and entries marked "prose"
   are rechecked first.
2. **Scope of this plan:** Phases 1 and 2's proposal start once the owner approves this
   design. Phase 2 re-runs wait for the owner's approval of the list. Phase 3 may be issued
   with "unknown, re-run pending" entries and updated when re-runs finish, so the synthesis
   is one document with a revision history.
3. **Crypto in Lenses 6 and 7:** both lenses include crypto, working from git history
   (before the 2026-06-16 removal), old reports, and the TimescaleDB tape. Where the code,
   data or raw artifacts no longer exist, the result is classified "unverifiable, prose
   only" rather than trusted, and the success criterion is read as "classified or marked
   unverifiable".
4. **Defects found in the live path** (for example the dividend add-back discrepancy, missed
   exits, orphans): reported to the owner in chat as soon as a lens confirms them, not held
   for the synthesis. The investigation still changes nothing live; the fix is a separate
   decision. Lens 3 owns the snapshot-value discrepancy.
5. **The "looser bar" in Lens 2:** fixed before any result is seen. It is a sensitivity
   table, not a new pass bar: each result is shown at t of at least 2 and at least 3, with
   its effect size and confidence interval, labelled exploratory. No verdict changes on the
   looser bar alone.
6. **Budget:** no hard ceiling is set. The project's delegation preference settles the
   model mix (Flash for bulk because its quota is generous, a stronger model verifying,
   Codex for final synthesis; see section 3). Work is thorough by default within the memory
   limits, covering all 49 triage rows rather than a sample.
