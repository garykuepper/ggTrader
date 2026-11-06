# Root-Cause Investigation: Why Haven't Stocks or Crypto Beaten SPY?

**Date:** 2026-11-04 · **Status:** approved by the owner 2026-11-04 (Claude Max; quota not a constraint); revised after an independent Opus review · **Type:** investigation design (diagnosis only)

## 1. Intent

The owner has put months into ggTrader and sees marginal results against SPY in both
backtests and Alpaca paper trading.

**The target (stated by the owner, settled):** raw gains, 2x what holding SPY would have
made. Not risk-adjusted. In the investigation this is measured as raw total return
(price plus dividends) over the same period: SPY +14% means a target of +28%. Pre-tax is
the primary measure because the owner asked for raw gains; after-tax is reported beside it
because the account is taxable. Drawdown is reported but is not a pass or fail test, since
the owner can accept losing what is put in. The owner (37, small capital, taxable account
only, high risk tolerance) wants this measured, not softened.

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
- Proposing new strategies, parameter changes, or deployments. Lenses report what was
  ruled out and by what evidence, not options to pick from.
- Any change to the frozen production paper trader, the container, cron, or orders.
- Resolving the open owner decisions in `docs/working_task_list.md` (factor-scope wording,
  insider timestamp amendment). The A23 70/30 allocation is a scheduled review on
  2027-09-30, not an open decision, and is also out of scope.

**Anti-goals (this would be a failure even if it ran cleanly)**
- Shading results positive to match the owner's hopes, or negative to look rigorous.
  The investigation must be able to find that an edge exists; each lens answers a steelman
  question (section 5) so the work is not set up to confirm "no edge".
- Treating a NO-GO as a recommendation to retire the sleeve, end paper trading, or shut the
  project down. Standing owner instruction: no such recommendation, and no IRA or
  retirement-account advice. Taxable account only; no paid data.
- Re-running things that touch live state, or consuming memory until the host wedges.

## 3. Constraints

- **Memory:** every backtest or WFO re-run runs inside a transient cgroup, e.g.
  `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`, modelled on
  `scripts/run_tests.sh` (whose default is 8G; 6G is deliberately tighter so that two
  parallel runs, 12 GB, fit on the 31 GB host). At most 2 runs in parallel. A kill at the
  cap (exit 137) is a finding to report, not something to retry with a higher cap. (The
  2026-09-16 incident hit 10 GB and wedged the machine.)
- **Reproducibility:** every re-run pins `--eval-start` and `--eval-end`, uses
  `--max-leverage` matching the config under test (production is 1.0, default is 2.0), and
  records raw results to JSON next to its report.
- **Read-only on live and data:** never delete or modify `data/` or database configuration;
  no database writes; no orders; do not touch the `ggtrader_live` container.
- **Intraday observer is off-limits:** four user timers fire every weekday
  (`ggtrader-intraday-observer-recover`, `-inputs`, `-decide`, `-archive`, 12:15 to 13:20 PT)
  from the worktree `/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a`, activated
  under owner-approved commit `6f0362e0`. Do not stop, edit, restart, or write into those
  timers, that worktree, or the observer's archive. Reading them is allowed.
- **Broker reads:** read-only Alpaca account reads (positions, orders, account activities
  such as dividends) are allowed for Lens 3. No order, cancel, or configuration endpoint
  may be called.
- **Tests:** run only via `scripts/run_tests.sh`.
- **Models (owner instruction 2026-11-04; recorded in the `AGENTS.md` delegation
  preference dated 2026-11-04, which supersedes the 2026-10-22 Gemini/Codex preference):**
  Claude only. Before the delegation plan is written, the owner states current quota
  headroom and the plan is sized to it. Sonnet is the orchestrator. Opus takes critical
  tasks: the trust and verdict judgments, the implementation-correctness review, the
  root-cause ranking and the final synthesis. Haiku takes easy and bulk tasks: extracting
  numbers from reports and JSON, inventories, listing and cross-referencing, and
  file-by-file reads. A stronger model checks Haiku's output before anyone relies on it:
  at least 10% of each batch (minimum 5 items), plus every figure that reaches a report.
  The lead stays responsible for final decisions. Private account records stay out of
  delegate prompts.
- **Independence limit:** because the owner chose Claude only, no second model family
  reviews the work. The Phase 3 challenger is a fresh Opus agent that receives the raw
  results and the claims, not the synthesizer's reasoning, and is told to argue the opposite
  case. The synthesis states this limit plainly.
- **Definitions:** expand abbreviations (WFO, OOS, IR, PIT, etc.) on first use in every
  report written for the owner.

## 4. Hypotheses from Phase 0 (unreviewed; the lenses test them)

A read-only audit by a senior-quant Opus agent covered stocks only. The full report,
including its trust table and the re-run costs, is saved at
[`research/audits/root-cause-2026-11-04/2026-11-04-phase0-trust-audit.md`](../../../research/audits/root-cause-2026-11-04/2026-11-04-phase0-trust-audit.md)
(unreviewed subagent output; some entries are marked "prose", meaning not recomputed). An
independent Opus review of this spec found the audit leans one way in places. These are
starting hypotheses, not conclusions, and each lens states in advance what evidence would
overturn each hypothesis it touches.

1. No real stock-picking edge over SPY was found. Sound tests mostly returned NOs.
   *Qualifier:* the production-replica results file's top-level verdict is `INCONCLUSIVE`,
   stage `stopped`, reason `holdout:tainted_trade:primary`, while its `closure` block says
   NO-GO (stage `pretax_fail`, holdout INCONCLUSIVE, live replication UNVERIFIED)
   (`research/audits/production-replica/_production_replica_results.json`). The holdout PRIMARY arm did complete
   (non-citable, tainted); only its four sensitivity arms never ran. *Correction 2026-11-04:*
   an earlier version of this paragraph said no holdout arm completed; Lens 7 found that
   wording came from the results file's `audit_stop.holdout` string and the FOMC amendment,
   which describe an earlier stopped attempt. The audit graded the replica "Reliable NO,
   high trust"; Lens 2 decides which is right.
2. The live account is about 84% SPY, 5% cash, and 3 stocks at about 3.3% each. A stock
   sleeve of about 10% cannot move the account much, so "marginal versus SPY" may be partly
   built in. *Open:* whether the sleeve size or stock selection is the bottleneck (the
   replica records 998 of 1,317 days truncated and 83 halted).
3. The early "beat SPY" figures (Sharpe 1.12, 0.99; CAGR about 16%) were produced by five
   bugs (bars stamped a day early, duplicate SPY rows, an anchor leak, a point-in-time
   universe leak, DSR units). The corrected core WFO figure is Sharpe 0.59 versus SPY 0.78.
   Gate counts from before 2026-10-19 are void. (Taken from prose: that the five bugs
   alone caused the 1.12 headline.)
4. WFO mostly tests the fallback parameters rather than parameter selection, so a WFO
   NO-GO is weak evidence. Pass bars (Holm-corrected t of at least 3; the actual threshold
   differs by study, for example Holm alpha 0.00135 for the factor screen and about t 3.2
   for momentum) are unreachable on five years of data, and there is no clean holdout.
5. The paper shortfall versus SPY is mostly configuration churn, not signal (by era:
   about -1.9, -2.6 and +1.4 points; taken from prose).
6. Live decisions cannot be reconciled with backtests (10 of 32 orders explained, no
   decision inputs archived). Paper snapshot value exceeds cash plus positions by a
   non-zero amount from 2026-08-20: about $74 on 09-01, $156 for most of October, then
   exactly $320.24 each day from 10-30 to 11-04. The step pattern fits cumulative dividend
   accruals added back in `paper/trader.py:1051-1056`, unproven.
7. The best free-data long-only large-cap effects measured after 1994 are +0.1 to +0.8
   percentage points a year, so a 2x-SPY goal looks out of reach without leverage or a much
   larger effect. *Counter-evidence to weigh:* the momentum screen's size-matched
   comparison shows t 3.72 / 5.18 (+3.66 / +5.14 pp) and pre-1994 big-cap momentum shows
   t 3.56 (`_momentum_french_screen_results.json`). These may be the wrong comparators; Lens
   2 and Lens 7 decide. The "+10 to +15 pp needed" figure is from prose and is rechecked
   in Lens 4 against the section 1 definition.

Prior art: `docs/superpowers/specs/2026-11-01-research-process-audit-design.md`,
`research/audits/research-process/2026-11-01-research-process-audit.md`,
`research/audits/production-replica/2026-10-19-production-replica-audit-nogo.md`. The 2026-10-20 strategy
triage ("49 rows: 32 flawed-test, 8 untested, 4 clean NO") exists only in the lead's private
memory notes, not in the repo, so Phase 1 builds a committed inventory (section 6) and
measures coverage against it. The investigation extends prior work; it does not redo it
without cause.

## 5. The investigation lenses

Each lens is one or more read-mostly subagents that report to the lead. Each lists the
**questions it must answer**. Every lens also answers the same **steelman question**: what
is the strongest evidence that an edge exists, or that the result under review is sound,
and does it hold up? And it states, before looking, what evidence would overturn each
Phase 0 hypothesis it touches.

A lens is **done** when every question is answered, or marked "unknown" with the reason and
the cheapest test that would settle it.

### Lens 1 - Backtest validity (stocks)
- After the five known bugs, what contamination remains (data tape, benchmark, universe,
  fills, costs)? Which published numbers predate which fix?
- Does any current headline number still rest on an unrecoverable or drifting window?
- Are yfinance adjusted-close revisions and dividend/total-return handling consistent
  between the backtests and the paper ledger?

### Lens 2 - Verdict audit (runs after Lenses 1 and 6 for any result whose tape or code is in doubt)
- For each NO-GO, was the test able to detect an edge, and which NOs are reliable,
  unreliable, or inconclusive? (The power calculation itself belongs to Lens 9.)
- Did strict pass bars reject small real effects? Report the sensitivity table specified
  in section 11, item 5.
- How many tests were run in total, and how does multiple testing change what "a win"
  would have meant?
- Did the gate machinery (for example a circuit breaker that halts 87.5% of runs even with
  a true edge, per `2026-11-02-wfo-policy-synthetic.md`) reject candidates that a simpler
  test would have kept?
- How much did freedom in gate design, windows and benchmarks, revised over months, shape
  the verdicts, beyond simply counting tests?

### Lens 3 - Live versus backtest parity
- Relationship to the existing paper-parity plan (`docs/working_task_list.md`, Priority 1):
  this lens supplies evidence to that plan and does not write or replace it.
- What does the paper trader do that the tests did not model (partial-bar signals at
  12:45 PT, position caps, sweep sleeve, orphans, manual liquidations, dividend handling)?
  Do fills before the close behave the way the backtest's next-bar or close fills assume?
- How much of the paper shortfall versus SPY is explained by each configuration era?
- Own the snapshot-value discrepancy: compare against Alpaca account-activity dividend
  records (read-only).
- How long would frozen paper trading need to run to detect an effect of a given size in a
  sleeve of about 10%?

### Lens 4 - Target feasibility and how professionals do it
- **Measured ceiling (owner's question, 2026-11-04):** the owner knows 2x SPY is very
  ambitious and wants to find the maximum. What is the highest raw return over SPY that
  sound evidence in this project supports, for each approach tried, with a confidence
  interval and the conditions it needs (exposure, leverage, regime, costs, tax)? The answer
  is a measured number from existing and approved re-run evidence, not a forecast or a
  new strategy.
- Using the section 1 definition, what return-over-SPY would the target require, and how
  does that compare with each measured effect?
- Which of breadth, leverage, concentration and long-short were ruled out in this
  project's testing, and by what evidence? (Constraints and evidence, not options.)
- How do large systematic firms generate returns (breadth, data, financing, costs,
  capacity), and which of those constraints apply to the owner? Which structural
  advantages of a small account (no capacity limit) exist, and which were never used?
  Sources must be cited and free.
- After tax, do short holding periods turn a small pre-tax edge negative?

### Lens 5 - Coverage gaps
- What was never tested, as opposed to tested badly? Candidates to check: small and micro
  caps, concentration, long-short, and cross-asset breadth. Intraday is **prospective
  evidence in progress**, not a gap: the owner-approved observer timers (section 3) are
  running, so this lens reads their journal and readiness documents but never touches them.
- Where did hypotheses come from (for example LLM web research in
  `WEB_RESEARCH_CANDIDATES.md`), how correlated or crowded are they, and were most of them
  published anomalies known to decay after publication?

### Lens 6 - Implementation correctness (runs first, with Lens 1)
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
  the result? (This is the only lens that asks the pre-registration question.)

### Lens 8 - Crypto
- What was tried, with what universe, fees, venue, and benchmark? Did any crypto strategy
  beat buy-and-hold BTC or ETH (the correct benchmark, not SPY)? Does BTC beta explain any
  apparent alpha?
- Fees: the 2026-06-08 edge search found fees dominated. Is that finding sound and
  complete, and do venue fee tiers and quote-currency choice change it?
- Scope of what was tested: long-only spot only? Were shorting, perpetual futures or
  funding-rate carry ever evaluated (carry was parked)?
- Data integrity: do the same bug classes found in equities (timestamp shift, duplicate
  rows, gaps) affect the crypto tape? About 648 `-USD` symbols are still in TimescaleDB;
  check read-only. Were exchange-specific symbols (for example XBT, XETH) normalized before
  ranking, and is the coin universe survivorship-biased?
- Does the 252-day annualization in `lab/metrics.py` distort 7-day-a-week crypto
  statistics (`2026-11-01-research-process-audit.md`)?
- Why did live crypto barely trade (venue migration, allowlist crash loop), and what live
  evidence therefore exists?

### Lens 9 - Statistical power and regime
- With the data actually available, what size of edge could each test have detected?
  (Single home for all power questions.)
- How much do conclusions depend on 2021-2026 being a single (largely bull) regime?

## 6. Sequencing

1. **Phase 0 (done):** the senior-quant trust audit, stocks only.
2. **Phase 1a - inventory and foundations (read-only):** first task is a committed
   inventory file (`research/audits/root-cause-2026-11-04/2026-11-04-investigation-inventory.md`) listing every
   material result, strategy and study with its source. "Material" means it carries a
   headline number, a verdict, or a live deployment decision. Coverage in section 8 is
   measured against this file. In parallel, Lens 1 and Lens 6 start for the results whose
   tape or code is in doubt. The section 11 item 5 sensitivity-table specification is
   committed before Lens 2 opens any results.
3. **Phase 1b - remaining lenses (read-only, parallel):** Lenses 2 to 5 and 7 to 9, starting
   from the existing evidence. Lens 7 begins with the results the Phase 0 audit ranked most
   important, and its "prose" entries first.
4. **Phase 2 (re-runs, owner-approved list):** the lead collects the unknowns each lens
   says only a re-run can settle, with cost estimates, and the owner approves the list
   before anything runs. Cheap candidates already flagged: the stock sleeve versus
   equal-weight S&P 500 (15-30 min, needs a sleeve return series), and French top-decile
   big-cap momentum after 1994 (15 min). Hours-to-days items (after-tax replica, July
   event-study re-runs, a clean 2010-2020 holdout) wait for a separate decision.
5. **Phase 3 (synthesis):** one report for the owner with the ranked root causes, a trust
   table across stocks and crypto, what is sound, and what remains unknown. Written by
   Opus; challenged by a fresh Opus agent as described in section 3 before delivery.

**Checkpoints and early exit:** after each phase the lead reports quota used and what is
next, and the owner can pause. If Phase 1 confirms every Phase 0 hypothesis and no lens
raises an unknown that could change a verdict, Phase 2 is skipped and the lead says so.

## 7. Deliverables

- The inventory file (Phase 1a).
- One report per lens under `research/`, following the repository research-report
  template, with raw JSON artifacts for anything re-run.
- A synthesis report: `research/audits/root-cause-2026-11-04/2026-11-04-root-cause-investigation.md`, written in
  plain language, with abbreviations expanded, containing:
  - a one-page summary;
  - a traffic-light trust table with one plain sentence per row and a "what would change
    this conclusion" column;
  - answers to a fixed list of 5 to 7 questions the owner should be able to answer after
    reading (set in the plan, for example: is there any edge in the stock work, what is
    limiting the live account's return, what would 2x need, how reliable is each past NO).
- Updates to `docs/changelog.md` and the research snapshot (via the `research-snapshot`
  skill) once the synthesis is accepted. No change to `docs/next_steps.md` priorities
  without the owner.

## 8. Success criteria

- Every material stocks and crypto result in the inventory carries a trust classification
  with its verification method (recomputed from raw artifact, or taken from prose), or is
  marked "unverifiable, prose only".
- The ranked root-cause list separates "method and process errors", "implementation and
  live-parity errors", and "the evidence is negative or the target is structurally hard",
  each with citations.
- Each lens meets its "done" definition (section 5).
- The owner can answer the synthesis's fixed question list in plain language after reading
  the one-page summary.

## 9. Left to the builder

- How many subagents per lens, and which Claude model runs each, within the model rules in
  section 3.
- Exact memory caps per run within the 6 GB / 2-parallel limits.
- Report file names and the exact table layouts.

## 10. Open for the owner

- Approve the Phase 2 re-run list when it is ready. (Design approved and quota headroom
  stated as ample on 2026-11-04.)

## 11. Calls made by the lead (settled without the owner; check these)

1. **Phase 0 audit** is saved in the repo (section 4 link). Lenses start from it as an
   unreviewed evidence base: they may confirm or overturn it, and entries marked "prose"
   are rechecked first.
2. **Scope of this plan:** Phase 1 starts once the owner approves this design. Phase 2
   re-runs wait for the owner's approval of the list. Phase 3 may be issued with
   "unknown, re-run pending" entries and updated when re-runs finish, so the synthesis is
   one document with a revision history.
3. **Crypto in Lenses 6 and 7:** both lenses include crypto, working from git history
   (before the 2026-06-16 removal), old reports, and the TimescaleDB tape. Where the code,
   data or raw artifacts no longer exist, the result is classified "unverifiable, prose
   only" rather than trusted.
4. **Defects found in the live path** (for example the dividend add-back discrepancy, missed
   exits, orphans): reported to the owner in chat as soon as a lens confirms them, not held
   for the synthesis. "Confirmed" means reproduced from two independent sources (for example
   the database and a read-only broker record, or code plus a replay). The investigation
   changes nothing live; the fix is a separate decision.
5. **The "looser bar" in Lens 2:** a sensitivity table, not a new pass bar, with its
   specification committed before any result is opened. For each result it shows the
   effect in percentage points a year of raw CAGR versus SPY, with a confidence interval,
   next to the t-statistic judged at t of at least 2 and at least 3, and labels the table
   exploratory. No verdict changes on the looser bar alone. Because the goal is raw return,
   effect size in raw terms is reported alongside, not only Sharpe-type statistics.
6. **Budget and models:** the owner set the model mix (Claude only; section 3). No dollar
   ceiling has been stated, and the agent cannot see Claude usage, so the owner reports
   headroom before the plan is written. The plan batches Haiku work, keeps Opus to the
   judgment tasks, and reports usage after each phase so the owner can pause it.
7. **The 2x definition** in section 1 (raw total return over the same period, pre-tax
   primary, after-tax shown) is the lead's reading of the owner's "2x gains compared to
   just holding sp500" and "raw gains". The owner stated the raw-versus-risk-adjusted point
   directly; the pre-tax versus after-tax ordering is the lead's call.
