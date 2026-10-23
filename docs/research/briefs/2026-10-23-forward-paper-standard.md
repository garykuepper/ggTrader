# Brief: free forward evidence for future taxable stock candidates

**Status:** DRAFT 2026-10-23, awaiting owner approval

**Terms used below:**

- *Part A* asks whether the underlying idea survives a long-history test. *Part B*
  asks whether our exact version survives implementation, costs and taxes.
- *PIT* (point-in-time) means using membership and information available on each
  decision date, rather than today's list or later revisions.
- *CAGR* (compound annual growth rate) is the annualized geometric return.
  *pt/yr* means percentage points per year: subtract the two CAGRs, rather than
  divide one by the other.
- A *t-stat* measures signal relative to estimated noise under a statistical
  model. It is not the probability that an idea is true. *Newey-West (NW)* allows
  for correlated returns and changing variance. *Holm correction* controls the
  family-wide false-positive rate across the registered hypotheses.
- *Information ratio (IR)* is annualized excess return divided by annualized
  tracking error (the variability of returns relative to the benchmark).
- A *shadow book* records hypothetical holdings and fills without submitting
  broker orders. It is separate from ggTrader's production Alpaca paper trader.
- A *tax lot* records a purchase's quantity, basis and acquisition date. *ST/LT*
  means short-term/long-term tax character. A *wash sale* can defer a realized
  loss into a replacement lot's basis and holding period.
- *Append-only* means corrections add new records and preserve the originals.
  A *hash* identifies the exact frozen contents; it does not prove when those
  contents existed.

## 1. The question, and the decision it feeds

- **Question.** With paid data permanently ruled out, what free forward evidence
  could test a future stock candidate that has already passed Part A?
- **Reason for this dated brief.** The [owner decision of 2026-10-23](../../next_steps.md#current-research-decision-2026-10-23)
  says no paid dataset or subscription, now or later. The old Part B replay on
  paid 2000–2020 data is therefore unreachable as written.
- **Proposed replacement.** For all **future stock candidates**, replace that
  paid-history Part B with B0 (a kill-only replay) and B1 (a forward shadow).
  This changes the evidence available for our implementation; it does not turn
  a short forward record into a statistical proof.
- **Authority.** This document is a draft, not an approved amendment or permission
  to run a study, build a ledger, deploy anything or use real money. The conflicts
  in §8 remain explicit for owner approval.
- **Boundaries.** Part A stays unchanged. Closed NO-GOs stay closed, including
  momentum and the classic factor screen. PEAD and insider remain parked. This
  brief does not reopen them, change the production freeze or alter A23.

## 2. Part A: the idea, unchanged

The idea must pass on long free history (for example, Ken French): **t ≥ 3 with
Holm across the family, plus ≥ +1.0 pt/yr net since 1994**, exactly as in the
[momentum brief §3 and §5](2026-10-20-momentum-retest.md#3-the-standard-two-parts-owner-delegated-2026-10-20).

This shorthand does not replace the registered calculation. Momentum §5 uses
base-cost monthly excess returns, NW with 12 lags, and Holm at the family-wise
one-sided level 0.00135; individual Holm hurdles can exceed t = 3. Its A2 is the
CAGR difference after the cost haircut over 1994-01 → 2026-04, **not after tax**.
The [factor brief §6](2026-10-20-factor-screen.md#6-tests-significance-a1-and-since-1994-economic-persistence-a2)
likewise requires both significance and since-1994 economic persistence.

A future candidate must register its family, long-history source, fixed windows,
costs and exact tests before computation. This brief does not relax those tests,
pick a new family after seeing results or grant error control across successive
studies. A failure of Part A cannot be rescued by B0 or B1.

## 3. Part B0: a replay that can only kill

- **Rules.** Our exact rules, fixed before this replay; no parameter selection or
  choosing a different version after seeing the result.
- **Tape and window.** The clean PIT S&P 500 tape, **2021-01-31 → 2026-04-30**.
  Clean means **detector-screened, not certified**. The window has already been
  seen and is not an independent holdout. Warm-up is outside scored returns and
  must be pinned in the candidate brief.
- **Execution and cash.** Next-session fills, with the exact fill convention and
  costs frozen beforehand; T-bill cash; after-tax lot accounting. Use the same
  next-session-close convention as B1 unless a contradiction is brought back
  for approval before any run.
- **Benchmark and endpoint.** SPY buy-and-hold on the same starting capital,
  dates, cost and tax basis. Compare terminal-liquidation after-tax CAGRs under
  the primary tax assumptions in §6.
- **Kill.** If after-tax CAGR minus SPY's after-tax CAGR is **< −1.0 pt/yr**,
  stop this candidate and report **NO-GO**. Equality at −1.0 does not trigger
  this strict inequality.
- **Otherwise.** A B0 pass **proves nothing**; it only removes obvious losers.
  It can permit further research under an approved candidate brief, never a GO
  or capital authorization. Data or accounting failures mean no verdict, not a
  pass. The −1.0 hurdle conflicts with momentum §6; see §8.

## 4. Part B1: a forward shadow, minimum 18 months

**Rules frozen and hashed in git before day 1.** Freeze the candidate, universe
and eligibility rules, signals, timing, sizing, costs, free sources and fallback
rules, tax treatment, benchmark, start/end dates and quarterly review dates.
Register one primary book and an endpoint at least **18 calendar months** after
inception. Do not select a winning parallel book, stop early to bank a pass,
extend a failed endpoint until it passes or restart without preserving the
failed attempt. A rules change is a new prospective study needing approval;
the original record and verdict remain.

**Daily append-only shadow ledger.** Target weights are decided using only data
available before the next session; fills occur at that next session's close.
Record tax lots, T-bill cash and SPY on the same basis. Each day's decisions
must be **committed to git before the fill session**. A gap or late commit
**voids that day and is reported**. No orders go to the production paper trader,
and there are **no changes to `src/ggTrader/paper/` or the production freeze**.

The following proposed controls make those fixed terms auditable:

- **Before-session means before the session opens.** Record decision time,
  source-availability cutoff and intended fill date in UTC against a frozen
  exchange calendar. A no-trade day still needs a committed hold decision.
  After the close, append prices, fills, costs, distributions, cash, lots and
  benchmark wealth; never overwrite the decision.
- **A local git date is not a time witness.** Author/committer dates can be
  backdated and history rewritten. Before the fill-session open, obtain a free,
  independently timestamped receipt identifying the decision commit and its
  contents hash. Preserve that receipt and the preceding ledger hash. If that
  evidence is absent, treat the decision as late/unverified, regardless of its
  local timestamp. This draft does not set up or publish such a service.
- **Freeze inputs too.** Retain the input snapshot or immutable free-source
  artifact and its hash, availability time, target weights and intended lot
  sales. Each daily decision identifies the frozen rules hash used in B0 and
  B1. A later data correction is appended with its discovery time; it cannot
  rewrite the earlier decision or replace the originally observed input.
- **No selective missing days.** Publish every gap, late record, fault and
  correction. Do not erase the day, return or elapsed calendar time from either
  wealth path. Carry existing holdings through a void day and append actual
  valuation/cash events; no hypothetical trade can be inserted afterward.
  Halt new shadow trades while a decision or accounting fault is unresolved.
  Do not issue a B1 pass while a void day's effect is unresolved. Before day 1,
  owner approval must settle whether any void day disqualifies the study; this
  draft supplies no allowed-gap quota or automatic restart.
- **No favorable fill selection.** Freeze rounding, untradeable names, missing
  closes, splits, dividends, identity changes and delisting proceeds. Never
  silently drop a failed name or substitute a later convenient price. Report a
  data-quality stop when the frozen rules cannot value or fill it reliably.

## 5. B1 pass, stop and limits

- **Pass.** After the full registered period of at least 18 months, after-tax
  annualized excess over SPY is **≥ +1.0 pt/yr**, and no quarterly check has been
  **more than 15% cumulatively behind SPY**. Both conditions are required, with
  a complete, reconciled record. A valid full-period result below +1.0 is NO-GO;
  an unresolved evidence fault is UNRESOLVED, never a pass.
- **Annualized excess.** Subtract SPY's after-tax CAGR from the shadow's, using
  identical dates and elapsed time, net of costs and tax. Use unrounded values.
  Fix the annualization convention before day 1.
- **Quarterly check.** Check on the last trading day of each calendar quarter,
  from inception, using the same starting notional. Cumulative relative wealth
  is `shadow_after_tax_wealth / SPY_after_tax_wealth - 1`. A value **< −0.15**
  means more than 15% behind: **stop this candidate's shadow, NO-GO**. It is not
  a 15 percentage-point return difference or an absolute-drawdown stop. A
  quarterly breach cannot be cured by a later recovery or a new inception date.
- **Threshold conflict.** The supplied B1 wording is strict: exactly −15% does
  not breach it. Momentum §8's real-money stop is **15% or more** (≤ −0.15).
  These are transcribed separately; §8 flags the difference for approval.
- **Evidence limit.** Eighteen months cannot establish statistical significance.
  Under the illustrative approximation `t ≈ IR × sqrt(years)`, IR ≈ 0.36 over
  1.5 years gives expected **t ≈ 0.44**, not 3. This is an illustration, not a
  power calculation or detection guarantee. B1 is a reality check that the
  Part A idea survives **our implementation, costs and taxes going forward**.
  It does not replace long-history significance or recreate 2000–2020 regimes.

## 6. Taxable accounting: prerequisites, not a flat tax haircut

B0 and B1 need a tested multi-stock lot engine before a verdict is possible.
The [momentum brief §2 and §6](2026-10-20-momentum-retest.md#6-phase-2-what-do-costs-and-taxes-take-from-our-version-can-only-kill)
warn that the existing three-ETF Book's backward-only wash-sale approximation
is insufficient. This document builds no engine and asserts no tax validation.

Carry forward the registered primary model: $100,000 standalone notional;
24% ST/ordinary and 15% LT rates; HIFO (highest basis first) lot sales; annual
ST/LT netting and carryforwards with character preserved; no $3,000
ordinary-income credit; unused terminal carryforward valued at zero. Fix
deterministic lot tie-breaks. Apply identical assumptions to SPY buy-and-hold,
including reinvestment, costs and the source/timing of tax payments. Candidate
briefs must reconcile these details before day 1:

- **Wash sales.** Track replacements both 30 days before and 30 days after a
  loss sale, including partial quantities, reinvestment and overlapping lots.
  Carry disallowed losses into replacement basis and holding periods. A
  31-day re-entry block alone does not cover preceding purchases. Keep the
  forward window open: a loss is provisional until later replacements are
  known. Preserve decision-time records when appending tax adjustments.
- **ST/LT and distributions.** Preserve acquisition dates and adjusted holding
  periods through partial sales, washes and corporate actions. Separate
  qualified/unqualified dividends and T-bill interest. Momentum §6's dividend
  qualification rule is explicitly simplified; do not describe that model as
  an exact personal tax return. Register the free distribution source and the
  policy for qualification or classification not yet known at a review date.
- **Tax payments and cash.** Accrue liabilities, pay year-end tax from book cash,
  and account for any HIFO sales needed to fund it, their costs and resulting
  taxes. Those sales obey the same advance-decision/fill rule; no same-day
  retroactive tax trade. No unrecorded external top-ups or spendable tax reserve.
  Accrue T-bill income using information available then, taxed as ordinary
  income, rather than applying a subsequently revised rate backward.
- **Comparable wealth.** At quarterly checks, value both books on a consistent
  hypothetical-liquidation basis, including accrued tax and unrealized-lot tax;
  this is a valuation, not an extra trade. At the fixed endpoint use terminal
  liquidation with tax paid for both. Pending wash-sale or distribution facts
  make the check provisional; retain the original and any later adjustment,
  and never claim a pass before they are settled. A later confirmed quarterly
  breach still means NO-GO, even if the current book has recovered.
- **Standalone limit.** Do not assume losses can offset the owner's outside
  income or gains. Outside holdings and purchases can affect real-account wash
  sales; the shadow cannot certify their absence. Reconcile that exposure
  before any separately approved real-money pilot. Report tax-model limitations
  and ST, LT, dividend and cash tax drag explicitly.

## 7. After a B1 pass: cite the existing capital terms

Real money is only under the owner-approved
[momentum brief §8](2026-10-20-momentum-retest.md#8-phases-45-paper-trading-and-real-money-terms-fixed-now):
**5% pilot, 10% maximum, quarterly review, stop at 15% behind SPY**. Read its
full conditions: funding from the SPY/VOO core, never A23's SSO leg; 6 months of
clean broker paper before the pilot; 18 months clean paper plus 12 months of
pilot without a stop before growth; operational faults halt immediately;
whole-account reporting against the displaced SPY/A23 allocation.

This citation does not change those terms or claim that a shadow book proves
broker order parity or lot reconciliation. The old paid-Phase-3 prerequisite
and any substitution for broker paper remain conflicts in §8. **A B1 pass alone
does not authorize capital.**

## 8. Conflicts with the cited briefs: flagged, not silently resolved

1. **Paid Part B versus free forward evidence.** Momentum §3/§7 requires an
   exact-rules 2000–2020 paid replay; factor §2/§8 also routes a passer to paid
   Phase 2. The owner's no-paid-data decision closes those routes permanently.
   B0/B1 is the proposed replacement for future candidates only, effective only
   after approval. The old studies' failures and closures are not reconsidered.
2. **B0 kill bar.** Momentum §6 kills below **+1.0 pt/yr** after tax versus SPY;
   the supplied B0 kills below **−1.0 pt/yr**. This is a material loosening of
   the kill-only hurdle, transcribed here for explicit approval, not described
   as unchanged. The older MTUM/IJR comparisons are not restated as new B0/B1
   gates; whether they apply to a future candidate needs its approved brief.
3. **Shadow versus broker paper.** Momentum §8 specifies Alpaca Rule T paper at
   the freeze lift, 18 months checking mechanics, performance recorded but not
   judged. B1 instead has a performance bar and hypothetical fills. It changes
   neither production nor the freeze and cannot establish those broker checks.
   Whether a future shadow can satisfy any existing clean-paper time requirement
   is **unresolved**; this draft grants no equivalence.
4. **Real-money prerequisites.** Momentum §8 requires Phase 3 plus at least
   6 months of clean paper. Phase 3 is now unreachable as written. Replacing
   its evidence with B1 is the proposed amendment, not an already approved
   route to capital; the owner must approve how these prerequisites fit.
5. **15% boundary.** B1's supplied “more than 15%” is strict; momentum §8's
   real-money “15% or more” includes equality. Neither is rewritten here.
6. **Superseded retirement language.** Momentum §1/§8 says to retire ensemble
   picks at the freeze lift. The [standing owner instructions](../../../AGENTS.md#owner-instructions-2026-10-20)
   supersede that text. This brief recommends no retirement or shutdown and
   preserves paper observation and the production freeze.
7. **Tax-model limits.** Momentum §6's general netting paragraph mentions the
   $3,000 credit, while its primary-model bullets and §11 exclude it. This draft
   cites the explicitly registered primary treatment without rewriting that
   source. Its simplified dividend rule and unfinished multi-stock engine
   remain limits, not certified after-tax accounting.
8. **Part A's “independent” history.** Momentum §3 uses that word; factor §2
   acknowledges that published definitions were developed within the historical
   sample. This source qualification remains explicit: a Part A pass is not
   represented here as an independent temporal holdout. No Part A test or
   threshold is changed to resolve that tension.

## 9. Scope and approval

**Free data only. Taxable account only.** No dataset purchase, subscription,
account-type change, recommendation to retire anything, or production change
follows from this draft. No code, data, database work or return calculation is
part of this document task.

Owner approval is needed for this dated replacement and the conflicts above.
Before a future candidate starts, its own approved brief must pin the remaining
ledger timing/witness mechanism, invalid-day eligibility, tax conventions and
complete pass/stop rules. Approval of this standard would not itself authorize
that implementation or a real-money pilot.
