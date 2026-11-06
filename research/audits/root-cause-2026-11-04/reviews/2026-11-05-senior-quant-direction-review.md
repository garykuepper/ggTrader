> Reading copy (2026-11-05): see 2026-11-05-direction-reviews-consolidated.md

# Senior quant direction review of the root-cause investigation

**Date:** 2026-11-05 · **Type:** review plus recommendation (the owner asked for a direction; the
investigation itself was diagnosis only) · **Reviews:**
`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md` ("the synthesis") and
`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-challenge.md` ("the challenge").

**Abbreviations.** SPY = SPDR S&P 500 ETF (the benchmark). TR = total return (dividends
reinvested). pp/yr = percentage points a year. MDE = minimum detectable effect (smallest true
effect a test finds 80% of the time at its pass bar). TE = tracking error (yearly spread of a
book's return around its benchmark). t = t-statistic. Holm = a correction for testing several
ideas at once. WFO = walk-forward optimization. PIT = point in time. SSO = ProShares Ultra S&P
500 (2x daily SPY). A23 = the owner's adopted 70% SPY / 30% SSO allocation. P2-nn = a numbered
re-run proposal in the Phase 1 checkpoint. "Decision 0" = the owner's permission for re-runs to
write to the shared price cache (answered "no"). Every figure below is from the cited file;
figures marked **(my estimate)** show their arithmetic.

## Bottom line

The synthesis is sound on the question that matters: the built design (about 85% SPY) cannot
get to 2x SPY, and no stock-picking result after 1994 comes within reach of the gap. Its
"stock picking is undecided" is technically right but practically weak, because no affordable
free-data test can decide a 1 to 4 pp/yr edge. Its "only leverage reaches the gap" finding is
its most fragile positive statement: it rests on one window with 1.64% funding, and at
today-like funding my estimate puts even 2x exposure at about half the gap. My ranked
direction: (1) make the paper trader a trustworthy instrument (input archive, the halt-base
defect, the partial-bar measurement), (2) run the handful of cheap, no-DB-write Phase 2 items
that settle the leverage and power questions, (3) treat further stock-picking work as a small,
pre-registered, long-history screen budget whose realistic best case is about +1 pp/yr, not 2x.
Exposure is the only measured route to a much larger raw return; that is a risk decision for
the owner, not a research finding to act on.

## Assessment of the synthesis

**(1) Whole-account 2x for the built design is negative: agree, high confidence.** The
production replica was powered for a 2x-sized effect at whole-account level (MDE 4.24 pp/yr
against gaps of 9.57 and 6.14) and returned -0.64 and +1.00
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:52-58`;
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens9-power-and-regime.md:34-47`). The challenge rightly called the
first draft too symmetric on this point and the synthesis fixed it
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-challenge.md:38-59`). PARA cannot plausibly close a 7 to
9 pp/yr distance (challenger's bound of about 2.9 pp/yr, an inference;
`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:398-411`).

**(2) Stock picking is "undecided": agree with the label, push back on the reading.** Two
discounts:
- *"Power fails in 51 of 53" mostly means "power could not be computed".* Rule R5 counts a
  missing MDE as failing (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md:189`), and only 8 of
  144 rows had the inputs for an MDE (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens9-power-and-regime.md:23-33`).
  So most of the 51 are "unknown power, usually no raw file", not "demonstrably weak". The
  conclusion survives, but it is a statement about record-keeping as much as about power.
- *"Undecided" overstates how open the 2x question is for stock picking.* The required effect
  is far outside every measured one: after 1994 the registered screens run from -0.62 to +0.83
  pp/yr against a 2.34 gap; post-2013 momentum is negative
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:47-55`). Published anomalies lose
  about 58% after publication (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md:57-60`).
  Undecided at 1 to 4 pp/yr, yes; undecided at the 2x size, no.

**(3) Measured ceiling, only leverage reaches the gap: agree it is exposure, discount the
size.** The 2.0x arm was a post-hoc pick of five, has no t-statistic, fell 84.8%, and was
measured only at 1.64% average funding with no funding run
(`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:58-75, 283-292`). On the
synthetic 1993-2006 holdout the same arm made +0.96 against a 4.54 gap (lines 72-73). With
funding 2.5 points higher (about today's bills), the 1.3x arm fell from +1.32 to +0.52
(lines 245-247). **(my estimate)** Charging the 2.0x arm the same 2.5 points on its one unit of
borrowed exposure gives 4.15 - 1.0 x 2.5 = about +1.65 pp/yr, below the 3.55 gap. The same
linear rule on the 1.3x arm (1.32 - 0.3 x 2.5 = 0.57) lands close to its measured 0.52, so the
estimate is plausible but unmeasured; P2-04 would measure it in minutes.

**(4) Live is not what was backtested: agree, confirmed in code.** Six differences
(`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens6-implementation-correctness.md:238-251`). Note the replica
already models the cap, alphabetical fill and 3.3% sizing with live defaults (same table), so
the replica, not the WFO runs, is the relevant evidence for the live design. The unmodelled
items are the partial bar and the halt base.

**(5) Crypto did not beat BTC over three years: agree.** The out-of-sample wins were low-beta
books in a falling market (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md:31-49`).

**Single most likely way the synthesis is wrong:** the leverage ceiling is read as more
reachable than it is going forward, because the one window that "reaches the gap" had near-zero
funding costs. It matters a lot: leverage is the only route the synthesis says reaches the
target, and it is the route with ruin risk (century study 30% arm -91.4%, 1x base -84.6%;
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:294-305`). A second, smaller risk:
Lens 2's rules and classes came in one commit and the reproduction checked application, not
soundness (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:679-687`). I do not think
different rules would create a reliable GO; they could create a few reliable NOs, which would
not change the direction.

## Recommendation (ranked)

**1. Paper-trading reliability and parity (first).**
- *Evidence:* the paper shortfall is mostly idle cash and broker artifacts, not picks
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens3-live-parity.md:35-51`); the daily-loss halt fires at about
  2.7% because the saved value includes an uncredited dividend accrual
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens6-implementation-correctness.md:57-68`); partial bars leave
  the research tape only by accident of a stale symbol (lines 256-268).
- *Order:* (a) **measure** the partial-bar effect read-only (P2-14, about 1 hour, reads
  yfinance directly; `research/audits/root-cause-2026-11-04/findings/2026-11-04-phase1-checkpoint.md:527-540`); (b)
  **propose** an input archive per run (P2-29, lines 762-771), which is already the next step in
  `docs/next_steps.md:9-17`; (c) **propose** computing the daily-loss halt from the broker
  value; (d) document MNST and AVB as paper-broker artifacts in attribution (P2-05, 15 minutes
  read-only) rather than engineering around them; (e) do **not** change the 3-name cap, sizing
  or parameters to match the WFO: the replica already tests the live rules.
- *Payoff:* none in raw return; it makes every later live number interpretable. *Cost:* hours.
  *Risk:* any live change breaks the freeze; frame (b) and (c) as proposals through the normal
  deploy checks. *Needs:* owner decisions on (b) and (c). *Kill:* none; stop at "parity
  measured and archived". *Will not tell us:* whether there is alpha. A 10 pp/yr sleeve edge
  takes 23 to 40 years to detect on paper
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens3-live-parity.md:57-61`).

**2. The cheap Phase 2 items with no price-cache writes.**
- *Run:* P2-04 (2.0x at higher funding; minutes, frozen inputs, no DB), P2-02 (MDE of the French
  screens; minutes, offline), P2-01 (regime split from committed JSON), P2-05, P2-07, P2-08
  (parity reads) and P2-14 above (`research/audits/root-cause-2026-11-04/findings/2026-11-04-phase1-checkpoint.md:284-316`).
  Under 4 hours in total by the checkpoint's own costs.
- *Why these:* P2-04 decides whether leverage still "reaches the gap" at today-like funding;
  P2-02 decides whether the forward-paper standard's Part A can resolve a +1.0 bar at all.
- *Defer:* P2-03, P2-16, P2-17, P2-19, P2-21, P2-22, P2-25 to P2-28 write to the price cache
  and need Decision 0 (lines 261-263). P2-16 and P2-17 cannot change a class and the expected
  P2-17 result is non-significant (lines 567-570, 588-591), so they are poor value anyway.
- *Kill:* none needed; each is bounded. *Will not tell us:* whether any new edge exists.

**3. Re-powering stock picking: only as long-history, portfolio-level screens.**
- *Arithmetic:* MDE is about 3.84 x TE / sqrt(years) (the multiplier the challenge reproduced,
  `research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-challenge.md:76`). With Lens 2's indicative TE of 6 to 7%
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md:63-71`), **(my estimate)** detecting +1.0
  pp/yr at TE 6.5% needs (3.84 x 6.5 / 1.0)^2 = about 620 years; detecting the 2.34 post-1994
  gap needs about 114 years. A concentrated stock-level book is worse: the momentum replay's
  TE of 12.44% gave an MDE of 22.66 pp/yr on five years
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens9-power-and-regime.md:38`).
- *What this means:* no free-data design can "decide" a 1 to 4 pp/yr edge. Part A's t of 3
  (`research/programme/evidence-standard/2026-10-23-forward-paper-standard.md:48-50`) acts in practice as an
  effect-size bar of roughly 2.5 to 4.4 pp/yr; a pass would imply a large effect, a fail is
  uninformative below that.
- *Recommendation:* keep the accruals screen plan (`research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md:34-53`)
  as the one active candidate, single hypothesis, registered before returns, and accept that its
  best realistic outcome is about +1 pp/yr, not 2x. If P2-02 shows the French design cannot
  resolve its bar, say so in the brief before running. Annual-rebalanced portfolios also realize
  fewer short-term gains than the 25-day ensemble trades
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:84-89`); that tax point is my
  inference, not measured.
- *Cost:* hours each. *Kill:* fails Part A, or B0 below -1.0 pp/yr after tax
  (`research/programme/evidence-standard/2026-10-23-forward-paper-standard.md:80-84`). *Will not tell us:*
  anything about 2x.

**4. Exposure and leverage: present it as a risk decision, not a strategy.**
- *Evidence:* 1.3x +1.32 (window funding) and +0.52 (funding +2.5); 1.5x +2.37 and 2.0x
  +4.15 only at 1.64% funding; T-bill-financed exposure beat SSO (13.94% vs 12.71% at 1.3x)
  (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:245-249, 294-305`). Century study:
  30% SSO -91.4% in 1929-32 with the ruin flag; 15% avoids it at 0.54 pp/yr after-tax cost
  (`docs/next_steps.md:50-53`).
- *Payoff:* positive in expectation only if stocks beat the funding rate by enough to cover
  fees and volatility drag; likely about +0.5 pp/yr at 1.3x and (my estimate) about +1.65 at
  2.0x at today-like funding. *Risk:* drawdowns of 66% to 85% in-window and over 90% in a
  1929-style decline; a forced sale at the bottom is the ruin path. *Tax:* rebalancing
  realizes gains; the A23 terms already use cash first
  (`docs/next_steps.md:44-48`).
- *Recommendation:* run P2-04 and P2-18 so the scheduled 2027-09-30 A23 review sees
  funding-honest figures. I make no allocation recommendation. *Needs:* the owner's own
  drawdown tolerance in dollars. *Will not tell us:* the future equity premium.

**5. Other areas, ranked.** (a) **Intraday observer:** free, already running to 2027-11-03,
descriptive only (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md:315`); let it finish, do
not read early. Its rule holds SPY about 30 minutes a day, so even a positive result is small
in raw-return terms. (b) **Concentration:** P2-06 (about 15 minutes after a free download,
needs a storage decision; `research/audits/root-cause-2026-11-04/findings/2026-11-04-phase1-checkpoint.md:414-430`) is the one
cheap look. (c) **Small or micro caps:** a PIT, survivorship-free stock-level universe is not
available free (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md:30-47`); not worth effort.
(d) **Long-short:** lowers market exposure, so it works against a raw-return target unless
levered; low. (e) **Crypto beyond spot, options:** owner has crypto parked, the crypto tape is
contaminated, and no free options history was found; low.

**6. What I would put first that the synthesis does not.** (a) One benchmark everywhere: SPY
TR against Alpaca's end-of-day equity, plus an exposure-matched SPY/cash book, so the 5% cash
reserve (about 0.9 pp/yr) is not mistaken for stock-picking loss
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:105-117`). (b) After-tax as the primary
metric for any short-holding design. (c) A fixed yearly research budget (number of registered
hypotheses, Holm across them), since 72 tests at a t of 2 bar make a false win 81% likely
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:328-330`). (d) Give research a separate
copy of the price table so Decision 0 stops blocking re-runs and no script can write to the
table live signals read (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:604-644`).

## 30/90/180-day sequence with gates

**Days 0-30.**
1. Owner decides on accepting and merging the synthesis
   (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md:735-744`).
2. Register and run P2-04, P2-02, P2-01, P2-05, P2-07, P2-08, P2-14 (no DB writes).
   *Gate A:* if P2-04 puts 2.0x at today-like funding below the 2006-2026 gap, record that no
   tested exposure level reaches 2x at current rates. *Gate B:* if P2-14 finds partial-bar and
   final-bar decisions differ on more than a handful of the 30 sessions, the parity proposal
   must pick one bar for both live and replica.
3. Write the parity proposal (archive, halt base, bar choice) for the owner.

**Days 31-90.**
4. If approved, deploy the parity changes through the standard deploy checks and start a clean
   observation window under the readiness review (`docs/working_task_list.md:22, 26`).
5. Resolve the accruals feasibility items; register one brief; run Part A. *Gate C:* Part A
   fail closes it; a pass moves to B0 only.
6. If the owner approves a storage location, run P2-06. *Gate D:* below +2 pp/yr post-1994,
   concentration needs no further work.

**Days 91-180.**
7. Review the first 90 days of archived paper decisions for parity only, not performance.
8. Prepare the funding-honest leverage table for the 2027-09-30 review.
9. If Gate C passed, B0 replay; kill below -1.0 pp/yr after tax. Otherwise keep the budget
   unspent rather than open a new family by default.

**Not worth time:** more RSI/Bollinger ensemble levers (33 of 72 tests already;
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md:48-53`); P2-25 and P2-26 (5 to tens of hours,
cannot reach decisive power); the 2010-2020 holdout rebuild (days, survivorship risk;
`research/audits/root-cause-2026-11-04/findings/2026-11-04-phase1-checkpoint.md:748-760`); paper trading as alpha evidence;
crypto spot strategies.

## Decisions for the owner

1. Accept and merge the synthesis, or not.
2. Approve the seven no-DB-write Phase 2 items.
3. A separate research copy of the price table (an alternative to Decision 0).
4. The input-archive proposal for the frozen trader.
5. The daily-loss halt base proposal.
6. Where new free downloads (P2-06, accruals) are stored.
7. A yearly research budget (hypotheses and agent time).
8. Whether funding-honest leverage figures should reach the 2027-09-30 review early; the
   allocation itself stays the owner's call.

## What would change this

- P2-04 shows the 2.0x arm still clears the gap at +2.5 funding: leverage is more reachable than
  I estimate (it does not make it safer).
- P2-02 shows a French-screen MDE at or below +1.0 pp/yr: Part A is more decisive than I think,
  and direction 3 deserves more budget.
- P2-06 shows concentrated big-cap momentum above +2 pp/yr post-1994: concentration moves up.
- P2-14 shows large partial-bar divergence: parity moves ahead of everything.
- The intraday observer's full year shows a large, cost-surviving effect.
- A free, survivorship-free small-cap stock universe becomes available.

## Straight talk

Known: over 2021-2026, doubling SPY needed about 9.6 pp/yr more than SPY every year; over
2006-2026 about 3.5; the best stock-picking result after 1994 was +0.83, and the paper account
is about 85% SPY by design. Assumed, not known: future returns. **(my estimate)** If SPY earns
10% a year for 20 years, doubling its total return needs about 13.4% a year (1.10^20 = 6.73,
so SPY gains 573%; doubling to 1,145% means 12.45^(1/20) = 1.134), a gap of about 3.4 pp/yr.
Nothing in this project, and nothing the free data can test, shows stock picking delivering
that. The only measured way to that size is more stock-market exposure, which in the project's
own studies meant drops of 66% to 85%, and over 90% in a 1929-style crash, and which earns
less when borrowing costs are high, as they are now. Honest paths that remain: keep the paper
trader as a reliable, measured instrument; test one or two pre-registered long-history ideas a
year, expecting at best about +1 pp/yr; and make any leverage choice as a deliberate risk
decision with dollar drawdowns written down first. Beating SPY by a little is a realistic
research goal; doubling it through skill is not supported by any evidence here.

## Limits of this review

Single reviewer, Claude only (Opus), with no second model family, by the owner's standing
instruction (`CLAUDE.md:17-38`). Read-only: I ran no re-run, backtest or DB query, so every
figure is from committed reports and artifacts, and the estimates above are mine and
unmeasured. I read the synthesis, the challenge, the checkpoint's Phase 2 list and targeted
sections of Lenses 2 to 9, not every lens line. The tax remarks on rebalancing frequency are
inference. Owner decisions recorded only in the ledger copy
(`research/audits/root-cause-2026-11-04/records/2026-11-04-investigation-ledger.md`) were not re-verified.
