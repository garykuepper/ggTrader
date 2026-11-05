# Root-cause investigation: why the stock and crypto research has not beaten SPY by much

**Date:** 2026-11-05 · **Branch:** `investigation/root-cause-2026-11-04` (not merged, not pushed) ·
**Type:** synthesis of nine reviewed lenses; diagnosis only, no recommendations · **Design:**
`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` · **Status:** Task 14,
revised after the independent challenge (`docs/research/2026-11-04-root-cause-challenge.md`);
every objection is resolved in section 9. Phase 2 re-runs were not run, by the owner's decision of
2026-11-05.

**Abbreviations, expanded once.** SPY = SPDR S&P 500 ETF Trust, the benchmark. ETF = exchange-traded
fund. TR = total return (price plus dividends reinvested). CAGR = compound annual growth rate.
pp = percentage points; "pp/yr" = percentage points a year. bp = basis points (0.01%).
WFO = walk-forward optimization (pick settings on a past window, test them on the next unseen
window, repeat). OOS = out of sample (data the settings never saw). PIT = point in time (using
only what was known on each date, for example index membership). DSR = deflated Sharpe ratio (a
significance gate that discounts for how many settings were tried). NDH = the lab's
parameter-neighbourhood density gate. MDE = minimum detectable effect (the smallest true effect a
test finds 80% of the time at its pass bar). t-statistic (t) = an effect divided by its standard
error; t of 2 or 3 is the usual "probably not luck" bar. SE = standard error. CI = confidence
interval. NW = Newey-West (a standard error that allows for returns that are correlated day to
day). IR = information ratio (excess return divided by tracking error). TE = tracking error (how
far a book strays from its benchmark, as a yearly standard deviation). Holm = a correction for
testing several ideas at once. GO / NO-GO = the project's pass / fail labels. SSO = ProShares
Ultra S&P 500, a fund that targets 2x SPY's daily move. A23 = the owner-adopted 70% SPY / 30% SSO
allocation study. LETF = leveraged or inverse ETF. FOMC = Federal Open Market Committee (the US
Fed's rate meetings). IEF, TLT = 7-10 year and 20+ year Treasury bond ETFs. GLD = gold ETF.
PDBC = commodity ETF. MDY = SPDR S&P MidCap 400 ETF. RSP = equal-weight S&P 500 ETF. BTC, ETH =
bitcoin, ether. IBIT = iShares Bitcoin Trust ETF. DB = the project's TimescaleDB database. EOD =
end of day. PT / ET = US Pacific / Eastern time. P2-nn = a numbered Phase 2 re-run item from the
Phase 1 checkpoint.

**Plain terms used below.** *Lens* = one of the nine separate checks this investigation ran
(Lens 1 price data, 2 verdicts, 3 live account, 4 ceiling, 5 coverage, 6 code, 7 numbers, 8 crypto,
9 statistical power). *Sharpe ratio* = return per unit of ups and downs; higher is smoother, it is
not the same as higher total return. *Beta* = how much a book moves when the benchmark moves (beta
0.26 means about a quarter as much). *In-sample* = tested on the same data used to choose the
settings, so the result flatters itself. *12-1 momentum* = buy the stocks that rose most over the
past year, skipping the latest month. *French data library* = free, survivorship-free US stock
portfolio returns back to 1926 published by Professor Kenneth French. *Anchor / fallback* = the
default settings a walk-forward test uses when its gates reject the settings it just picked.
*Tier B* = Lens 4's middle soundness grade: the number recomputes on clean data, but nothing
proves it is statistically real (Tier A, reliable, is empty).

## 1. One-page summary

**The question.** Why has the research not come close to the owner's target, twice SPY's total
return over the same period (pre-tax first, after-tax beside it)
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md:10-16`), and what is the
most the evidence supports?

**The answer in plain words.**
- *2x with the design that was built* (about 85% SPY plus a small stock sleeve): negative. The
  production replica (the live rules re-run on history) could detect a whole-account excess of
  about 4.2 pp/yr, enough to see the 9.57 and 6.14 pp/yr its two windows needed, and returned
  -0.64 and +1.00 (`docs/research/2026-11-04-lens9-power-and-regime.md:396-399`). Most of the
  -0.64 is the drag of holding about 5% cash: against SPY plus the same cash it was +0.26 (t 0.20).
  Unmeasured caveat: it traded one stock with wrong prices (PARA).
- *Stock picking itself:* undecided. Of 72 measured tests, 0 are reliable, 11 unreliable, 58
  inconclusive and 3 stopped on their own terms; 42 more rows are status labels, not tests
  (`docs/research/2026-11-04-lens2-verdict-audit.md:29-50`). "Inconclusive" means the test could
  not decide, not that it found nothing. None shows an edge; none rules out 1 to 4 pp/yr.
- *Crypto:* nothing beat holding BTC over any three-year window; one-year wins were mostly
  in-sample, low-exposure years when BTC fell; the one-year funding-carry gain is not
  like-for-like (`docs/research/2026-11-04-lens8-crypto.md:31-49`).

**The measured maximum.** The target needs 9.57 pp/yr over 2021-2026, 3.54 over 2006-2026 and
0.77 over 1927-2026 (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:36-43`). Only the
largest leverage tested (2x SPY, +4.15 pp/yr over 2006-2026, -84.8% at worst) and pre-1993
momentum reached their gap. Leverage is exposure, not skill: at 1.3x and 1.5x, the same exposure
financed at T-bill rates earned more than the leveraged fund. After 1994, registered stock-picking
results ran from -0.62 to +0.83 pp/yr; the best 2021-2026 estimate, +4.37, has a CI of -7 to +16
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:77-83, 296-320`). **After tax:** the
74 paper stock round trips were all short-term; such a book needs about +2.42 pp/yr before tax
(+5.05 at 37%/20%) just to tie SPY after tax (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:84-89`).

**The biggest causes**, ranked by judgement (no cause's share is measured; see section 3):
1. The target is large next to every measured stock-picking effect.
2. The book is about 85% SPY and 9% stocks, so it cannot stray far from SPY.
3. Early "beat SPY" results came from leaks that favoured the strategy; they caused false
   optimism and chose the live design, not the shortfall.
4. The tests could not tell a small stock-picking edge from zero.
5. The paper shortfall (-3.76 pp at run-time broker prices, -4.57 on Alpaca's end-of-day equity)
   is mostly idle cash and broker quirks; live runs no tested configuration.
6. The search stayed narrow: large-cap, long-only, mostly one signal family.

**Solid:** known data bugs fixed; every headline found in its raw file matched; signal code does
what it says; paper accounting reconciles. **Unknown:** a 1 to 4 pp/yr edge; PARA's effect; how
live-versus-test differences change returns; 29 re-runs not run.

**Not done here:** no recommendation about trading, strategies, deployments, the paper account or
the account type, and no forecast. No second model family reviewed this work (the owner chose
Claude only); review used independent re-derivation by fresh agents, including a challenger that
checked 40 claims without seeing the author's reasoning.

*Length:* this summary is about 530 words including citation paths (not counting this note), above the 400-500 guide,
so that it can answer the seven questions on one page.

## 2. Answers to the owner's questions

**Q1. Is there any edge over SPY in the stock work?** For the design that was built (mostly SPY, a
3-name sleeve), the record shows no 2x-sized effect: the replica could have detected one at
whole-account level in both windows and found -0.64 pp/yr (2021-2026, t -0.69) and +1.00
(2010-2020 holdout, t 0.69, not citable because of bad prices)
(`docs/research/2026-11-04-lens9-power-and-regime.md:396-399`). Against a passive book that held
SPY in place of the stocks plus the same share of T-bill cash (about 5%), it was +0.26 pp/yr
(t 0.20), and +1.85 (t 1.57) in the holdout. So most of the -0.64 (about 0.9 pp/yr) is the drag of
holding about 5% in cash instead of SPY, not the stock picks, which were neither clearly helpful
nor harmful. Running it with same-day fills, 3 or 10 bp costs, or cash earning no interest gave
-0.42, -0.57, -0.83 and -0.83 (`docs/research/_production_replica_results.json`,
`windows.primary.arms.*.excess_vs_spy` and `excess_vs_exposure_matched`;
`scripts/production_replica_audit.py:57-63, 273`).
For stock picking itself, the evidence does not settle it: stocks were only 9.37% of that book,
so the same test could only see a sleeve excess above about 45 pp/yr
(`docs/research/2026-11-04-lens9-power-and-regime.md:49-54`). The long-history French-library
screens found momentum over 1927-2026 at t 3.05 (big stocks) and 3.11 (small), above a single t
of 3 but just short of the registered two-test correction; after 1994 the registered results run
from -0.62 to +0.83 pp/yr against the market, and momentum turned negative after 2013
(`docs/research/2026-11-04-lens2-verdict-audit.md:56-58, 938-943`;
`docs/research/2026-11-04-lens9-power-and-regime.md:520-528`). No edge has been shown; a small
one (1 to 4 pp/yr) has not been ruled out.

**Q2. What is limiting the live account's return versus SPY?** From 2026-06-24 the paper account
trailed SPY's total return. The size depends on which value is used
(`docs/research/2026-11-04-lens3-live-parity.md:25-33, 185-194`):
- **-3.76 pp** (+1.07% vs +4.84%, to 11-04) using the broker value the trader read at run time
  (cash plus shares times price, about 15:45 ET);
- **-4.57 pp** (-1.48% vs +3.09%, to the 11-03 close) using Alpaca's own end-of-day equity series;
  part of the 0.81 pp difference rests on an unproven hypothesis that Alpaca values the
  untradable AVB holding near zero at the close;
- **-3.45 pp** using the value the trader stores, which adds back dividends the broker never paid.

On the run-time basis, idle cash before the SPY cash sweep cost about 2.46 pp, the stock picks
0.88 pp, the SPY sleeve 0.15 pp and other items 0.29 pp (an estimate,
`docs/research/2026-11-04-lens3-live-parity.md:34-50`). About 0.85 pp of the stock loss is one
broker artifact: Alpaca's paper account never applied MNST's 2-for-1 split. Without it, the stock
picks netted about zero: up 1.8 pp early, down 3.3 pp in the July-September blend era, up 1.6 pp
since the configuration was frozen (`docs/research/2026-11-04-lens3-live-parity.md:255-271`).
Today the book is about 85% SPY, 5% cash and 3 stock slots at 3.3% each, so it tracks SPY by
construction (`docs/research/2026-11-04-lens3-live-parity.md:244-253`). Detecting a 10 pp/yr
sleeve edge from paper trading alone would take about 23 to 40 years
(`docs/research/2026-11-04-lens3-live-parity.md:56-60`).

**Q3. What is the measured maximum, and what would 2x SPY require?** The requirement depends on
the window: 9.57 pp/yr over 2021-2026, 9.54 over 2022-2026, 3.54 over 2006-2026 (3.55 with a
one-day-different SPY starting close), 2.34 over 1994-2026 and 0.77 over 1927-2026
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:157-183`). Leverage, on one basis
(each arm against SPY's true total return, 2006-2026, at that window's 1.64% average T-bill
rate): 1.3x (the registered A23 arm) +1.32 pp/yr, 1.5x +2.37, 2.0x +4.15; with funding 1.5 and 2.5
points higher, the 1.3x arm falls to +0.87 and +0.52. On the A23 study's own basis the same arms
read +1.61, +2.66, +4.44, +1.16 and +0.81
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:245-249, 266-281`). The 2.0x arm is the
largest exposure tested, not a ceiling of skill: it was picked after seeing five arms, has no
t-statistic, fell 84.8% at its worst and trailed SPY in 37.5% of rolling 10-year windows (that
rolling series reaches back into synthetic pre-2006 data). The same file shows that 1.3x or 1.5x
SPY exposure financed at T-bill rates would have earned 13.94% or 15.83% a year, more than the SSO
books' 12.71% or 13.76% (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:294-305`). For
stock picking, the highest registered figure after 1994 is +0.83 pp/yr (0.35 of the 2.34 gap); on
2021-2026 it is momentum's +4.37 pp/yr (0.46 of the 9.54 gap), whose 95% CI of -7.07 to +16.05 is
on the arithmetic mean excess, not the CAGR gap
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:315-320`). After tax, a book whose gains
are all short-term needs +2.42 pp/yr (24%/15% brackets) or +5.05 pp/yr (37%/20%) before tax just to
tie SPY held and taxed once (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:84-89`).

**Q4. How reliable is each past NO-GO?** None is reliable: of 53 NO results, 0 are reliable, 2 are
unreliable (judged against the 1.12 Sharpe baseline that later failed to reproduce) and 51 are
inconclusive (`docs/research/2026-11-04-lens2-verdict-audit.md:967-971`). A NO counts as reliable
only if its data are clean, its code was checked, its numbers recompute and it had the power to
see the effect it was meant to rule out; power fails in 51 of the 53
(`docs/research/2026-11-04-lens2-verdict-audit.md:181-191, 967-971`). The power test uses each
study's own pass bar where it had one (usually +1.0 pp/yr), which is stricter than the owner's
2x gap; measured against the 2x gap, the replica had enough power (Q1). What the NOs rule out: the
replica, a whole-account excess above about 4.2 pp/yr (if PARA proves immaterial); the momentum
replay, only an excess above about 22.7 pp/yr; the FOMC and month-end Treasury tests, calendar
effects above about 2.4 and 4.9 pp/yr in derived, non-SPY units; for every other NO the size is
unknown (`docs/research/2026-11-04-lens2-verdict-audit.md:972-977`). The six strongest NOs each
fail one condition only (`docs/research/2026-11-04-lens2-verdict-audit.md:52-61`). They all point
the same way and no test shows an edge.

**Q5. Did anything beat buy-and-hold in crypto?** Not over any three-year window. Every crypto
portfolio with a recoverable record lost to holding BTC over its three-year range, by 5 to 44 pp a
year; the same runs beat ETH in four of five cases, because ETH itself returned between -8% and
+22% over those three years (`docs/research/2026-11-04-lens8-crypto.md:206-219, 538-545`). The
headline 48.46% is real in its file, but it is an in-sample replay in one year when BTC fell
(-21.34% as that report stated; -17.46% on today's price data), on data later replaced, with
coins chosen by 2026 trading volume; BTC exposure was low (beta about 0.26) and the leftover gain
was not significant (t 1.51) (`docs/research/2026-11-04-lens8-crypto.md:43-49`). In the only
out-of-sample stretch (about seven months) the two books lost 9% and 19% while BTC fell 42%, so
they beat BTC by about 23 to 33 pp there; their low BTC exposure (beta 0.10 and 0.20) fits that,
and the leftover was not significant (t 0.43 and -0.44)
(`docs/research/2026-11-04-lens8-crypto.md:224-237`). Live crypto left no usable record: 0 trades
in the database (`docs/research/2026-11-04-lens8-crypto.md:442-461`).

**Q6. What was done right and can be relied on?**
- The five known data and code bugs are fixed: no weekend bars, no duplicate days, one SPY row
  per date (`docs/research/2026-11-04-lens1-backtest-validity.md:21-24, 97-120`).
- Of the 41 rows with a raw results file: 5 were recomputed from raw data and matched to the
  stored precision; 27 were read by hand (17 matched, 2 matched in part, 1 labelling mismatch
  for position sizing, and 7 whose headline number could not be found in the file); 9 were taken
  from an extraction pass without a re-check. No found number disagreed with its report
  (`docs/research/2026-11-04-lens7-research-accuracy.md:20-25, 112-138`).
- The signal code does what it says: four strategies hand-replayed on real prices with 0
  mismatched buy or sell dates (`docs/research/2026-11-04-lens6-implementation-correctness.md:18-27`).
- 11 of 14 academic citations checked say what the reports say; none misquotes a number
  (`docs/research/2026-11-04-lens7-research-accuracy.md:24, 204-205`).
- Most October studies set their pass bars before seeing results, with short, checkable gaps
  (the replica and momentum are the exceptions noted in A3)
  (`docs/research/2026-11-04-lens7-research-accuracy.md:97-101`).
- Corrections went the conservative way (FOMC t fell from 1.57 to 0.98; the DSR fix made gates
  stricter) (`docs/research/2026-11-04-lens2-verdict-audit.md:925-927`).
- The paper accounting reconciles to the cent apart from a known dividend add-back, and live
  fills before the close land on the close price on average (+0.3 bp)
  (`docs/research/2026-11-04-lens3-live-parity.md:98-124`).

**Q7. What is still unknown, and what would it cost to find out?** The Phase 1 checkpoint lists
29 re-runs or reads (P2-01 to P2-29), cheapest first; the order is by cost, not importance
(`docs/research/2026-11-04-phase1-checkpoint.md:280-312`). By the owner's decision none was run,
so each question below is "unknown, re-run not run". Items marked * write to the price cache and
need Decision 0, which the owner answered "no".

| Id | Unknown it would settle | Cost |
|---|---|---|
| P2-01 | How much of each 2021-2026 result sits in 2022 | minutes |
| P2-02 | Detectable effect for the three French screens | minutes |
| P2-03* | Month-end Treasury on current code | about 1 min compute |
| P2-04 | 2.0x SSO arm at higher funding cost | minutes |
| P2-05 | How Alpaca values AVB at the close | about 15 min |
| P2-06 | Concentrated big-cap momentum after 1994 | about 15 min plus a download |
| P2-07 | Rejected orders on days other than 09-25 | 20-30 min |
| P2-08 | Dividend accrual: missed or extra dividends | about 30 min |
| P2-09 | How many dividend-adjustment seams sit in the price data | about 30 min |
| P2-10 | Code check of the corrected FOMC study | 20-40 min |
| P2-11 | Direction of the one-day date shift in each July event study | about 70 min |
| P2-12 | Crypto code reads (carry Sharpe, benchmark, entry rules) | 1-1.5 h |
| P2-13 | Crypto recompute of located files; fee-tier gate replay | about 1 h |
| P2-14 | Effect of the unfinished 12:45 PT bar on live decisions | about 1 h |
| P2-15 | 9 unchecked rows and 2 citations | about 1 h |
| P2-16* | Replica with PARA excluded, plus trade and sleeve output | about 1 h |
| P2-17* | Stock sleeve against equal-weight S&P 500 (RSP) | 15-30 min after P2-16 |
| P2-18 | A23 funding and tax-character records (outside this scope) | about 1 h |
| P2-19* | PIT core and two variants on current code | about 1 h |
| P2-20 | Mapping tested anomalies to published-decay evidence | 1-2 h |
| P2-21* | Replica after tax with short-term lots | 2-4 h after P2-16 |
| P2-22* | Stored excess series for artifacts without a t-statistic | 2-4 h |
| P2-23 | Pass bars before 2026-09-10 in private transcripts | 1-2 h |
| P2-24 | Code check of 16 unexamined strategies | 5-11 h |
| P2-25* | July event studies on the corrected data | 5-20 h |
| P2-26* | Ungated and one-change re-runs for gate revisions | tens of hours |
| P2-27* | Crypto fee-tier re-run and full-range BTC beta | hours to a day+ |
| P2-28* | Clean 2010-2020 holdout | days |
| P2-29 | Longer paper trading with an input archive | months to years; needs a live change |

Four questions no re-run can settle: the window behind the 1.12 Sharpe, the raw output behind
1.14, the April-May 2026 crypto runs (their data were wiped) and whether SPY's duplicate rows
existed before 2026-06-20 (`docs/research/2026-11-04-phase1-checkpoint.md:272-276`).

## 3. Ranked root causes

Each item says what is **confirmed** (measured or read from code, data or a broker record) and
what is **inference** (a reading of the evidence). Confidence is high, medium or low, with the
reason.

**How the ranking was made.** It is the author's judgement, not a measurement: no item's share of
the shortfall versus SPY was measured, and several effects are unknown (A1, B1, B5, C6). Items are
ordered by how directly each bears on the owner's 2x question, using the measured facts each one
rests on. Three cautions: C1 (the target is large) is a property of the target, not a defect;
C1 and C3 overlap, because with a 9% sleeve the book's construction rather than the size of any
effect is the binding limit; and A1's leaks favoured the strategy, so they explain false
optimism and the choice of live design, not results trailing SPY. The confidence labels apply
to each item's facts, not to its rank.

### A. Method and process errors in the research

**A1. The early "beat SPY" results were built with leaks that favoured the strategy, and the live
design was chosen on them.** Confirmed. Every June-July positive headline (Sharpe 0.84, 1.12,
1.14; 16.2% CAGR; the 5-voter ensemble; Bollinger reversion; MACD) predates fixes for an anchor
leak (fallback settings fitted on the whole test span) and a PIT universe leak (14.1% of entries on
days the stock was not in the index), both of which favour the strategy; Lens 2 classes all nine
GO results "unreliable GO" (`docs/research/2026-11-04-lens1-backtest-validity.md:128-137`,
`docs/research/2026-11-04-lens2-verdict-audit.md:46, 196-207`). When the leaks were removed, results
fell: core Sharpe 0.99 to 0.59 after the PIT fix; 1.12 did not reproduce (0.97)
(`docs/research/2026-11-04-lens2-verdict-audit.md:920-924`). The live ensemble, 3% sizing and the
blend were adopted on these numbers (`docs/research/2026-11-04-lens6-implementation-correctness.md:199-208`).
Confidence: **high** that these results are unreliable; **unknown** how much each bug contributed.
This explains why the project believed it was ahead, not why results trail SPY.
*Phase 0 qualified:* the "corrected" 0.59 itself predates the DSR fix, which changes which
settings each fold uses and so its returns; and "the five bugs alone caused the drop from 1.12" is
not confirmed (`docs/research/2026-11-04-lens1-backtest-validity.md:357-370`).

**A2. The tests could not tell a small stock-picking edge from zero.** Confirmed. Power fails in 51
of 53 NO results, measured against each study's own bar
(`docs/research/2026-11-04-lens2-verdict-audit.md:967-971`). Only 8 of 144 rows have what is needed
to compute a detectable effect; the replica's is 4.24 pp/yr at whole-account level (enough for the
2x gap, but about 45 pp/yr for its stock sleeve), and momentum's 22.66 pp/yr
(`docs/research/2026-11-04-lens9-power-and-regime.md:23-57`). The French screens span up to a
century but, measured against the owner's post-1994 gap of 2.34 pp/yr, their indicative
detectable effect after 1994 is about 4.4 pp/yr, so for the 2x question they are less powerful than
the replica (`docs/research/2026-11-04-lens2-verdict-audit.md:63-71`). The walk-forward gates often
deployed a fallback: after the DSR fix the momentum replay used it in 17 of 17 folds; for that
replay the fallback was the plain 12-1 momentum rule holding 50 names, so its +4.37 pp/yr measures
plain momentum rather than a stand-in (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:238`).
In a synthetic model where the gates always pass and the true Sharpe is 1.0, 87.5% of runs are
halted at least once, about a third of folds use the fallback and about half of runs end halted
(`docs/research/2026-11-04-lens2-verdict-audit.md:757-766`). Where an artifact let the gate be
removed (two cases), the verdict did not change; for the other 48 gated results it is unknown
(`docs/research/2026-11-04-lens2-verdict-audit.md:831-845`). Confidence: **high** for the computed
rows; **unknown** for the 103 rows with no raw file. *Phase 0 qualified:* "t of 3 needs an IR near
1.3" holds only at 50% power; at 80% power it is 1.62 to 1.68
(`docs/research/2026-11-04-lens9-power-and-regime.md:58-67`).

**A3. Pass bars and records arrived late.** Confirmed. Nothing committed before 2026-09-10 sets a
per-study pass bar; 48 verdict rows have no brief, including every early "beat SPY" claim and every
July NO-GO; 103 of 144 rows (72%) have no raw results file
(`docs/research/2026-11-04-lens7-research-accuracy.md:23-25`). The replica's closure rule was
committed 9 seconds before its results and quotes them; it turned "no verdict" into NO-GO, not a
pass into a fail. The momentum brief was committed together with its results
(`docs/research/2026-11-04-lens7-research-accuracy.md:43-52`). Three gate loosenings in four days
(2026-06-24 to 06-27) preceded the 1.12 and 1.14 headlines and raised their gate counts
(`docs/research/2026-11-04-lens2-verdict-audit.md:876-881`). Confidence: **high** for the facts;
**low** for how much they moved any verdict (not measurable without one-change re-runs, P2-26).

**A4. Many tries and moving yardsticks.** Confirmed facts, inferred effect. 72 tests in 11
families; with zero true edge, at least one false t of 2 "win" is 81% likely across 72 independent
tests, while a t of 3 bar keeps it near 9% (`docs/research/2026-11-04-lens2-verdict-audit.md:692-706`).
Most bars had no significance test at all ("Sharpe above SPY's", "beat 1.12"), and the SPY
yardstick itself moved from Sharpe 0.58 to 0.78 to 0.91 as data and definitions changed
(`docs/research/2026-11-04-lens2-verdict-audit.md:707-716, 886-888`). One GO call, the cross-asset
80/20 book, rested on Sharpe while its raw CAGR trailed SPY by 1.09 pp/yr (2021-2026; 1.07 over
2016-2026); the core+SPY KEEP tied SPY's CAGR (12.963% vs 12.959%)
(`docs/research/2026-11-04-lens2-verdict-audit.md:217-224`;
`docs/research/2026-11-04-lens7-research-accuracy.md:161`). Confidence: **medium** that this
explains the early optimism; it does not explain the later NOs.

**A5. The search stayed narrow.** Confirmed as a description of coverage; its role as a cause is
an inference. 33 of 72 tests (46%) are one signal family (RSI/Bollinger mean reversion and its
levers); 44 of 51 single-stock tests are large-cap only; 65 of 72 are long-only. Micro caps,
options, futures contracts, international stock exposure and weekly rebalancing were never tested;
small caps, mid caps and concentrated books were tested only on flawed universes
(`docs/research/2026-11-04-lens5-coverage-gaps.md:30-47`). 62 of 82 web-research hypotheses were
never run (`docs/research/2026-11-04-lens5-coverage-gaps.md:54-57`). Confidence: **low** as a cause.
Whether any untested area holds an edge is unknown; Lens 5 does not say any would help.

### B. Implementation and live-parity errors

**B1. Live does not run the configuration any backtest measured.** Confirmed (code, run log, DB),
six facts (`docs/research/2026-11-04-phase1-checkpoint.md:188-200`;
`docs/research/2026-11-04-lens6-implementation-correctness.md:28-50, 236-252`):
1. Live uses setting values no walk-forward fold ever picked; two are outside the search grid.
2. Live computes signals at 12:45 PT from today's unfinished price bar.
3. Live fills about 15 minutes before the close on the same day; the walk-forward backtests fill
   at that day's final close and the replica a day later.
4. Live holds at most 3 stocks, picked alphabetically when more signals fire.
5. Live sizes each buy at 3.3% of account value; the walk-forward backtests use 3% of remaining cash.
6. The saved account value includes a dividend accrual, and the next day's 3% daily-loss halt
   starts from that inflated value, so the halt fires at about 2.7% real loss.

The replica models items 4 and 5 and the live risk rules, but not the same-day fill
(`docs/research/2026-11-04-lens6-implementation-correctness.md:144-148`). Its same-day-fill arm
(signals and fills on the signal day's close, the nearest available approximation) gave -0.42
pp/yr against -0.64 for next-day fills (`docs/research/_production_replica_results.json`,
`windows.primary.arms.lag0.excess_vs_spy.cagr_diff_pp`). Confidence: **high** that the differences
exist; the effect of the unfinished bar on returns is **unknown** (P2-14).

**B2. The dividend accrual and the broker's missing cash flows.** Confirmed. The daily gap between
the saved value and cash plus positions equals the code's cumulative dividend accrual, $320.24
from 10-30 onward (`docs/research/2026-11-04-lens3-live-parity.md:16-24`). The gap-equals-accrual
part comes from code and the DB, which are not independent of each other; the independent part is
the broker record: Alpaca's paper account has never credited a dividend, split, merger, interest
or any other cash flow since its $100,000 funding journal in 2024 (`docs/research/2026-11-04-lens3-live-parity.md:286-311`).
Two accrual rows are over-stated by $5.86 in total (AEP $5.76, SPY $0.10). Confidence: **high**.
*Phase 0 hypothesis 6 confirmed.*

**B3. Most of the paper shortfall is cash and broker artifacts, not stock picks.** Measured on the
run-time broker value; the split by component is an estimate. Idle cash cost about 2.46 of the
3.76 pp; broker artifacts about 1.0 pp (MNST split 0.85, uncredited SPY dividend 0.15); the
blend-era stock picks lost about 3.3 pp over July-September but the other eras gained, so the picks
net about zero without MNST (`docs/research/2026-11-04-lens3-live-parity.md:34-50, 255-271`). On
Alpaca's end-of-day equity the total shortfall is larger, -4.57 pp (Q2). Confidence:
**medium-high**. *Phase 0 hypothesis 5 qualified:* its era figures (-1.9 / -2.6 / +1.4 pp)
reproduce only with its own method (stored value against SPY price); on the run-time broker value
against SPY total return they are -3.02 / -2.28 / +1.60, so the first era is overturned under its
own pre-written rule (`docs/research/2026-11-04-lens3-live-parity.md:77-83, 223-228`).

**B4. Operational events.** Confirmed. The trader ran in dry-run by mistake from 07-15 to 07-30 and
placed no orders; 44 orders filled at the next day's open, costing about 47 bp (about $290,
estimate); three trading days had no run; 19 holdings were liquidated by hand on 09-29; AVB (merged
into EQR) is held at a frozen price and appears to be valued near zero at Alpaca's close
(hypothesis); rejected orders do not appear in Alpaca's history
(`docs/research/2026-11-04-lens3-live-parity.md:319-333`). Confidence: **high** for the events;
they explain part of the early-era shortfall.

**B5. Price-data faults the five fixes did not cover.** Confirmed facts; effects unmeasured.
PARA's prices belong to a different security over its whole index membership, and the replica
traded it 3 times without its check noticing; old and new dividend adjustments meet in "seams"
that show as false one-day moves (VZ -1.7% on 2026-07-20 against -0.2% in yfinance); AVB has
carried EQR's price since 2026-07-20; 14 index members have no prices in 2021-2026; the 2010-2020
data have 5,595 daily moves of 40% or more, so there is no clean holdout
(`docs/research/2026-11-04-lens1-backtest-validity.md:36-52`). The unfinished 12:45 PT bar is
overwritten later only by accident, because one stale symbol pulls every refresh back to 08-11
(`docs/research/2026-11-04-lens6-implementation-correctness.md:256-268`). Live signals read this
same table. On PARA's size: the independent challenger estimates, by inference and not by
measurement, that three long-only round trips capped at about 5% of the account could have cost at
most about 2.9 pp/yr over 5.25 years, far less than the 7-9 pp/yr distance to the target
(`docs/research/2026-11-04-root-cause-challenge.md:41-43`). Confidence: **high** that the faults
exist; **low** on their size.

**B6. Crypto-specific faults.** Confirmed. 43.7% of the Binance.US 4-hour bars are copies of Kraken
bars; delisted Binance.US coins were deleted; coins were chosen by trading volume at run time and
then back-tested (look-ahead); runs before 2026-05-05 used bars shifted 7 hours, later wiped
(`docs/research/2026-11-04-lens8-crypto.md:50-55, 130-141`). Live crypto stalled on sizing bugs, a
regime gate, a venue move and an allowlist crash loop and recorded 0 trades
(`docs/research/2026-11-04-lens8-crypto.md:454-461`). Confidence: **high**.

### C. The evidence is negative or the target is structurally hard

**C1. The target is large in the window most tests used.** Confirmed by two lenses' independent
arithmetic. 2021-2026 was a strong SPY window: CAGR 15.05% against 10.29% for the US market over
1927-2026, so doubling it needs 9.57 pp/yr (`docs/research/2026-11-04-lens9-power-and-regime.md:43-47, 68-78`).
Phase 0's "+10 to +15 pp/yr needed" is close only for five-year windows; over 10-20 years the gap
is 3.5 to 6.5 pp/yr (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:114-120`).
Confidence: **high**. *Phase 0 hypothesis 7 qualified.*

**C2. Every measured stock-picking effect is small next to the gap.** Measured. After 1994 the
registered French screens give -0.62 to +0.83 pp/yr against a 2.34 gap, and after 2008 or 2013
several sub-periods are negative (down to -0.97 for big-stock and -3.34 for small-stock
momentum); 2021-2026 momentum gives +4.37 against the 9.54 gap of its 2022-2026 test window, with a
CI that includes both zero and the full target
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:47-55`;
`docs/research/2026-11-04-lens9-power-and-regime.md:515-544`). Published stock anomalies lose about
58% of their return after publication on average (McLean and Pontiff, 2016), and the project's own
screens show that shape (`docs/research/2026-11-04-lens5-coverage-gaps.md:58-60`). Confidence:
**high** that measured effects are small; **not established** that effects are absent.

**C3. The book's construction keeps it close to SPY.** Confirmed arithmetic. With about 85% SPY and
9.37% stocks, a 9.57 pp/yr whole-account gap would need roughly +102 pp/yr from the stock sleeve
(illustration) (`docs/research/2026-11-04-lens9-power-and-regime.md:49-54`). Confidence: **high**.
*Phase 0 hypothesis 2 consistent* (`docs/research/2026-11-04-phase1-checkpoint.md:779`). Whether
sizing or selection is the bottleneck is unknown (P2-17 not run).

**C4. What reached the gap was exposure, not skill.** Measured. Only the largest leverage tested
(the post-hoc 2.0x SSO arm over 2006-2026, -84.8% drawdown, at that window's funding) and pre-1994
momentum reached their window's gap; no stock-picking result after 1994 did
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:77-83`). The leveraged-fund books
at 1.3x and 1.5x earned less than the same exposure financed at T-bill rates (Q3; the file has no
such figure for 2.0x), so what they show is exposure minus cost. In the century study the 30% SSO arm fell 91.4% in 1929-32 and the 1x base arm 84.6%
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:294-305`). The leverage rows reach
Lens 4's Tier B only through a judgement exception for unreviewed code
(`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:205-211`). Confidence: **medium**.

**C5. Tax on short holdings.** Derived from a stylised model. All 74 closed paper stock round
trips were short-term (median hold 25.5 days); such a book needs +2.42 pp/yr (24/15) or +5.05 pp/yr
(37/20) before tax just to tie SPY after tax over 2021-2026, and a +1.0 pp/yr pre-tax edge becomes
-1.08 or -2.55 (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:84-89, 434-441`). The
paper account itself is mostly SPY, and its total tax drag was not computed. Confidence: **medium**.

**C6. One regime.** Confirmed description; consequence unknown. 37 rows rest only on 2020-2026
data, a window with one bear market, a rate shock from 0% to 5% and an above-average SPY return;
the only cross-decade check (the replica's 2010-2020 holdout) is not citable
(`docs/research/2026-11-04-lens9-power-and-regime.md:567-579`). Confidence: **medium** that the
window matters; **unknown** in which direction.

**C7. Crypto: no evidence of beating BTC over three years.** Measured on recovered files and
today's data. No three-year window beat BTC; wins appear only in trailing 12-month windows when
BTC fell 17-40%, mostly in-sample, and in a seven-month out-of-sample stretch where low BTC
exposure fits the result. "Fees dominate" is supported as "costs matter at this turnover", but its
headline comparison changed four things at once (`docs/research/2026-11-04-lens8-crypto.md:31-42, 60-64`).
Confidence: **high** that there is no evidence of beating BTC over three years; **medium** on the
fee finding.

### Phase 0 hypotheses: what the lenses did to them

The Phase 0 audit is unreviewed and is not used as evidence here; this table only records how the
lenses tested it (`docs/research/2026-11-04-phase1-checkpoint.md:776-784`,
`docs/research/2026-11-04-lens2-verdict-audit.md:1062-1079`).

| Phase 0 claim | Outcome |
|---|---|
| Replica is a "Reliable NO, high trust" | Overturned: inconclusive (MDE 4.24 above its 1.0 bar; PARA traded unflagged); against the 2x gap it was adequately powered |
| Live book about 84% SPY; sleeve too small to move the account | Consistent (replica 85.33% SPY, 9.37% stocks; sleeve-scale MDE about 45 pp/yr) |
| Early beat-SPY figures came from five bugs; corrected core is 0.59 | Partly: 0.59 predates the DSR fix; "five bugs alone" not confirmed |
| WFO NO-GOs are weak; bars unreachable on five years | Confirmed; "IR near 1.3" partly overturned (1.62-1.68 at 80% power) |
| Paper shortfall mostly configuration churn (-1.9 / -2.6 / +1.4) | Qualified: -3.02 / -2.28 / +1.60 on the run-time broker value; idle cash is the largest part |
| Snapshot excess fits the dividend add-back | Confirmed |
| 2x SPY out of reach; "+10 to +15 pp needed" | Qualified: 9.57 pp/yr on 2021-2026, 0.77 to 3.54 on longer windows |
| French screens, FOMC, month-end, PIT core, ic/kelly, LETF timing "Reliable NO" | Each overturned to inconclusive (power, code check, DSR timing or survivorship) |
| July event studies: "two bugs favoured them, direction reliable" | Overturned: the direction of the date shift is unknown |
| Core+SPY KEEP "skeptical" | Overturned toward unreliable GO (it measured settings live does not run) |

## 4. Traffic-light trust table

**How the lights were set** (from the lens classifications, not re-judged here): **green** = Lens 2
"reliable" and recomputed by Lens 7; **red** = Lens 2 "unreliable", or Lens 6 found the code at run
time did not match the test's specification ("no"); **yellow** = a raw results file exists and was
recomputed or read (Lens 3, 7 or 8) but Lens 2 classes it inconclusive or gives no class; **grey** =
no raw results file (prose only) or no result yet. "Raw evidence" here also covers numbers
recomputed from the database and the broker (row 10) and from run files Lens 8 recovered (rows 32
and 33), although the inventory lists no raw file for those rows. The 36 lines cover the results
that carry a number or verdict, 95 of the 144 inventory rows
(`docs/research/_investigation_inventory.json`); 49 inventory rows (status labels, infrastructure
fixes, operations, proposals and two superseded documents) are not shown as separate lines. The
per-row states are in the lens JSON files listed in section 10, and the mapping of every inventory
row to its line, or its reason for not being shown, is in `docs/research/_synthesis_inventory_map.json`.

**There are no green rows**, because Lens 2 found 0 reliable results. Yellow is not "found nothing":
it means the number is real but the test could not decide. Where Lens 2 or Lens 9 computed it, the
"why" column says what size of effect the result rules out (Lens 2 `rules_out_larger_than_pp_per_yr`,
Lens 9 MDE at t of 3); everywhere else that size is unknown.

| # | Result | What was claimed | Light | Why, in one sentence | What would change this | Citation |
|---|---|---|---|---|---|---|
| 1 | Production replica, primary window 2021-2026 | The live rules lose to SPY: 14.19% vs 14.83% a year, t -0.69 | yellow | Numbers recompute exactly and it could see a 2x-sized whole-account effect, but it rules out only a whole-account excess above about 4.2 pp/yr (sleeve about 45), not its own +1.0 bar, and it traded the misidentified PARA; exposure-matched +0.26 (so about 0.9 pp/yr of the -0.64 is cash drag), same-day fills -0.42. | PARA shown immaterial (P2-16) clears the data condition; power against the +1.0 bar would still keep it inconclusive. | `docs/research/2026-11-04-lens2-verdict-audit.md:360-403` |
| 2 | Replica holdout 2010-2020 | +1.00 pp/yr vs SPY, t 0.69 (exposure-matched +1.85, t 1.57) | yellow | It ran but stopped on tainted trades, so its own verdict is inconclusive and it is not citable. | A clean, identity-checked 2010-2020 tape (P2-28). | `docs/research/2026-11-04-lens7-research-accuracy.md:32-42` |
| 3 | Replica live-parity check | 10 of 32 live orders reproduced | yellow | The figure is in its file, but the question was narrowed to a prior-day-bar model after the result, so exact replication is "unverified". | An archive of each live run's inputs (P2-29). | `docs/research/2026-11-04-lens7-research-accuracy.md:219-222` |
| 4 | S&P 500 core PIT walk-forward (with ensemble_ic and ensemble_kelly) | Core Sharpe 0.59 vs SPY 0.78; ic 0.41, kelly 0.50 | yellow | Read from file, but it predates the DSR fix, which changes which settings each fold used, and stores no t-statistic. | Re-run on current code with a stored excess series (P2-19). | `docs/research/2026-11-04-lens2-verdict-audit.md:1067` |
| 5 | June-July "beat SPY" headlines (1.12, 0.84, 1.14, 16.2% CAGR, 5-voter, Bollinger, MACD) | The ensemble beat SPY out of sample | red | Built with an anchor leak and a universe leak that favour the strategy, no raw file and no pre-set bar. | Nothing short of new tests; the 1.12 window cannot be recovered. | `docs/research/2026-11-04-lens2-verdict-audit.md:46` |
| 6 | August-September pre-PIT re-baselines (corrected-tape core 0.99 "Live Core", anchor-fix reproduction 0.97, blend reproduction 0.68, pre-PIT ic/kelly 0.61 / 34% CAGR) | The core still beat SPY after the tape fix; 1.12 did not reproduce | red | Each predates the PIT fix that cut the core to 0.59, and Lens 6 found the code at run time did not match an honest test. | Already superseded by row 4. | `docs/research/2026-11-04-lens2-verdict-audit.md:279, 314, 315` |
| 7 | Blend-to-core revert comparison | Blend Sharpe 0.69 vs SPY 0.78 | yellow | Read from file and code matches, but it predates the DSR fix and stores no t-statistic. | Re-run on current code (P2-26). | `docs/research/2026-11-04-lens2-verdict-audit.md:290` |
| 8 | Core+SPY blend "KEEP" | Sharpe 0.80 vs SPY 0.78 (CAGR 12.963% vs 12.959%) | red | It measured walk-forward-picked settings that live does not run, and 0.02 of Sharpe has no significance test. | Not changeable by re-reading; would need a test of the live settings. | `docs/research/2026-11-04-lens2-verdict-audit.md:256, 1077` |
| 9 | SPY cash sweep (live) | Putting idle cash into SPY | grey | No raw backtest file; on paper it tracked SPY's price with about zero error apart from one uncredited dividend. | A stored backtest of the sweep rule. | `docs/research/2026-11-04-lens3-live-parity.md:326` |
| 10 | Paper account vs SPY, 06-24 to 11-04 | Paper trails SPY | yellow | Recomputed by Lens 3 from the DB and Alpaca: -3.76 pp at run-time broker value, -4.57 on Alpaca's end-of-day equity, mostly idle cash and broker artifacts; the frozen-era +1.60 pp is noise. | Months to decades more paper data; a 10 pp/yr sleeve edge takes 23-40 years to detect. | `docs/research/2026-11-04-lens3-live-parity.md:185-194` |
| 11 | "Live runs what was tested" (live defaults, 3% sizing "matches" live) | The paper trader runs the backtested strategy | red | Lens 6 found different settings, bar, fill moment, position cap and sizing base. | Not a test result; the differences are in code. | `docs/research/2026-11-04-lens6-implementation-correctness.md:236-252` |
| 12 | Ensemble levers whose code did not match (ensemble_ic June, Nasdaq-100 core, Kelly sizing July) | Each failed to beat the baseline | red | Lens 6 "no"; two were judged against the 1.12 figure that later failed to reproduce (Lens 2 unreliable NO); Kelly July has no named comparator, so Lens 2 left it inconclusive. | New tests on current code against a valid comparator. | `docs/research/2026-11-04-lens2-verdict-audit.md:43, 225-228, 944-947` |
| 13 | Other ensemble levers (exit rules, trailing stops, VIX throttle, six-indicator, weekly RSI, conviction sizing, ML entry gate, position-sizing regimes) | Each failed to improve the ensemble | grey | No raw file (position sizing's reported number comes from an earlier run with no file), pre-fix data, no power figure. | Re-runs on current code (P2-26). | `docs/research/2026-11-04-lens7-research-accuracy.md:125` |
| 14 | Momentum S&P 500 replay (xs_momentum; dual_momentum is identical) | NO-GO: +4.37 pp/yr, t 0.76 | yellow | Clean data and checked code; every fold ran plain 12-1 top-50 momentum (the fallback), and it rules out only an excess above about 22.7 pp/yr. | Far more data or a lower-noise book; no listed re-run settles power. | `docs/research/2026-11-04-lens2-verdict-audit.md:59, 839` |
| 15 | Early momentum (June 2026) | +126.0% vs SPY +104.5% total return, lower Sharpe | grey | Prose only: no raw file, pre-fix machinery. | A pinned re-run with a stored result. | `docs/research/2026-11-04-lens7-research-accuracy.md:177-178` |
| 16 | French momentum screen (1927-2026) | NO-GO: full-window t 3.05 / 3.11 just short of the two-test correction; post-1994 +0.08 / +0.20 pp/yr | yellow | Clean, independently recomputed, but the file stores t and not the mean excess, so power is not established; bar committed with the results. | Storing the mean excess (P2-02); indicative MDE about 2.5 pp/yr over 99 years and 4.4 after 1994. | `docs/research/2026-11-04-lens2-verdict-audit.md:57, 63-71` |
| 17 | Factor screen (profitability, value, investment, value plus momentum) | NO-GO: best t 1.34; post-1994 -0.62 to +0.83 pp/yr | yellow | Clean and recomputed to 1e-12; only power is not established. | P2-02. | `docs/research/2026-11-04-lens2-verdict-audit.md:56` |
| 18 | Net issuance screen | NO-GO: t 0.26, +0.48 pp/yr since 1994 | yellow | Same as the factor screen: only power is not established. | P2-02. | `docs/research/2026-11-04-lens2-verdict-audit.md:58` |
| 19 | July event studies (earnings drift, insider buying, congress trades, short interest, short volume, index deletions, headline sentiment) | All NO-GO | grey | No raw files, no pre-set bars, a one-day date shift of unknown direction and the old DSR bug. | P2-11 (direction), P2-25 (re-runs on corrected data). | `docs/research/2026-11-04-lens2-verdict-audit.md:1072` |
| 20 | Other July stock studies and sleeves (MAX effect, idiosyncratic volatility, overnight gap, EMA cross, monthly tournament, pairs trading, diversification sleeves) | NO-GO or not deployable | grey | No raw files, pre-fix data, code unexamined or partly matched. | Pinned re-runs on current code (P2-25, P2-26). | `docs/research/2026-11-04-lens5-coverage-gaps.md:306` |
| 21 | MidCap 400 reversion | 15.0% CAGR / Sharpe 1.08 vs MDY 9.1% / 0.40 ("Researching") | red | Lens 6 "no"; current-list (survivorship) universe and pre-fix leaks; Lens 2 typed it a status label, so it is not counted as a GO, although it carried a measured claim. | New tests on a PIT mid-cap universe. | `docs/research/2026-11-04-lens2-verdict-audit.md:293` |
| 22 | Leveraged-ETF rotation and trend timing | NO-GO | grey | No raw files; a survivorship error that favoured the strategy plus unknown-direction items. | Re-runs on current code. | `docs/research/2026-11-04-lens2-verdict-audit.md:1078` |
| 23 | A23 static leverage, 70% SPY / 30% SSO (and the A24 trend rider) | After tax 12.00% vs SPY 10.40% a year, 2006-2026 | yellow | Clean data and read from file, but no t-statistic and code unexamined; +1.32 pp/yr against SPY's true total return at the window's funding, +0.52 at today-like funding, and below the same exposure financed at T-bill rates. A23 itself is outside this investigation's scope. | A code check; P2-18 for funding records. | `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:245-247, 294-305` |
| 24 | Century leverage study (1926-2026) | Keep 30% in a 2x fund; after tax +1.24 pp/yr | yellow | Read from file on clean data, no t-statistic; the 30% arm fell 91.4% in 1929-32. | A stored excess series (P2-22). | `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:250` |
| 25 | Sanity-check shortlist (turn of month, volatility-managed SPY, threshold and rebalancing tilts) | Small Sharpe improvements on quick checks | grey | One-off checks, no raw file, never registered tests. | Registered tests with stored results. | `docs/research/2026-11-04-lens2-verdict-audit.md:257, 285, 322, 325` |
| 26 | WFO policy synthetic study | The circuit breaker halts at least once in 87.5% of runs even with gates always passing and true Sharpe 1.0 | yellow | Pre-registered and read from file; the code was not examined and it models gates as coin flips. | A code check of the simulator. | `docs/research/2026-11-04-lens2-verdict-audit.md:757-766` |
| 27 | Pre-FOMC Treasury drift, corrected event study | NO-GO: day-clustered net t 0.98 (gross 2.26) | yellow | Clean data and power adequate in derived units (MDE about 2.37 pp/yr, not an SPY excess); the code check is missing, and the NO depends on its 10 bp round-trip cost, the stock default: at 2 bp the net t would be about 2.0, still below 3. | P2-10 would make it the first reliable NO if the code matches. | `docs/research/2026-11-04-lens2-verdict-audit.md:60`; `docs/research/2026-07-20-fomc-drift-nogo.md:179-181` |
| 28 | Month-end Treasury overlay | NO-GO: funded from SPY it trailed SPY by 0.08 pp/yr | yellow | The non-walk-forward test also failed, but the walk-forward predates the DSR fix (direction unknown); MDE about 4.94 pp/yr derived. | P2-03 on current code. | `docs/research/2026-11-04-lens2-verdict-audit.md:61, 837` |
| 29 | Cross-asset 80/20 (SPY plus TLT, GLD, PDBC), its 2007-2011 holdout and trend arm | Shadow GO on Sharpe 0.973 vs 0.896 | yellow | The GO rests on Sharpe with no significance test while raw CAGR trailed SPY by 1.09 pp/yr; holdout +3.08 pp/yr only in a crisis window. | A code check would make it "reliable GO" under Lens 2's rules, which ask a GO for no power test. | `docs/research/2026-11-04-lens2-verdict-audit.md:217-224, 1070` |
| 30 | July cross-asset studies (FOMC walk-forward, Treasury curve, currency hedge, commodity trend) | All NO-GO | grey | No raw files, pre-fix data, code unexamined. | Pinned re-runs on current code. | `docs/research/2026-11-04-lens2-verdict-audit.md:333-343` |
| 31 | Intraday SPY rule (prospective observer) | No claim yet | grey | Results do not exist; the observer runs from 2026-10-30 to 2027-11-03 and was not read. | Its own results when they exist. | `docs/research/2026-11-04-lens5-coverage-gaps.md:315` |
| 32 | Crypto Kraken walk-forward runs (48.46% vs BTC -21.34%; April runs) | Beat BTC | yellow | The figure re-derives from the stored equity curve, but it is an in-sample replay in one falling-BTC year with hindsight coin choice; each run lost to BTC over three years. | Not repairable: the data it used were wiped. | `docs/research/2026-11-04-lens8-crypto.md:97-156` |
| 33 | Crypto edge search 2026-06-08 | No deployable edge: +13.8% vs BTC +38.8% | yellow | Files located and benchmarks recomputed by Lens 8; Lens 7 has not recomputed them; universe survivorship and copied Binance.US bars. | P2-13. | `docs/research/2026-11-04-lens8-crypto.md:206-237` |
| 34 | Crypto textbook reset and fees | 36 gate passes frictionless, 0 at realistic fees | yellow | Gate counts recount from logs, but the inventory pointed at the wrong run and no portfolio formed even at zero cost. | P2-13, P2-27. | `docs/research/2026-11-04-lens8-crypto.md:356-392` |
| 35 | IBIT overnight versus daytime returns | Overnight +186% vs daytime -43% | yellow | Main figures match their file; the later split is prose only; MDE 59-73 pp/yr derived. | Not a strategy test; no listed re-run. | `docs/research/2026-11-04-lens7-research-accuracy.md:127`; `docs/research/2026-11-04-lens9-power-and-regime.md:403` |
| 36 | Crypto live trading, funding carry, March pipeline runs and literature-only ideas | "Modestly profitable" (+$9.88), carry +0.45%, others untested | grey | Prose only: trade records and scripts deleted, 0 trades in the DB, carry results not reproducible from files. | P2-12, P2-27 (carry inputs and code still exist). | `docs/research/2026-11-04-lens8-crypto.md:396-413, 442-461` |

**Count of lights:** green 0, yellow 20, red 6, grey 10 (36 lines).

## 5. What the live account is and does

**Composition.** In the frozen configuration since 2026-09-30 the paper book averaged about 85%
SPY (Phase 0 said 84%), 5% cash and about 10% in stocks: at most 3 stock positions at 3.3% of
account value each, plus the untradable AVB holding and some names carried over
(`docs/research/2026-11-04-lens3-live-parity.md:114-117, 244-253`). The account was funded with
$100,000 in 2024; its first snapshot in this period, on 2026-06-24, was about $102,459
(`docs/research/2026-11-04-lens3-live-parity.md:286-294`;
`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:392`).

**What that means for stock picking.** With about 9% in stocks, a stock edge has little room to
show. The replica of the same design could detect only a stock-sleeve excess of about 45 pp/yr
(an illustration scaled from its whole-account figure) (`docs/research/2026-11-04-lens9-power-and-regime.md:49-54`).
On paper, a 10 pp/yr sleeve edge would take 23 to 40 years to detect; over three months the
smallest detectable sleeve edge is about 96 to 126 pp/yr (`docs/research/2026-11-04-lens3-live-parity.md:351-361`).

**Paper result by era** (run-time broker value against SPY total return; the end-of-day column
uses Alpaca's own equity series) (`docs/research/2026-11-04-lens3-live-parity.md:185-194`):

| Era | Dates | What ran | Shortfall, run-time value | Shortfall, Alpaca EOD |
|---|---|---|---|---|
| E1 | 06-24 to 07-15 | Core ensemble, 60-68% idle cash | -1.21 pp | -1.29 pp |
| E2 | 07-15 to 07-30 | Accidental dry run, no orders | +1.90 pp | +1.55 pp |
| E3 | 07-30 to 08-21 | Three-sleeve blend, no sweep | -3.71 pp | -3.95 pp |
| E4 | 08-21 to 09-22 | Blend plus SPY cash sweep | -2.26 pp | -2.33 pp |
| E5 | 09-22 to 09-28 | Core revert plus sweep | +0.21 pp | +0.17 pp |
| E6 | 09-28 to 09-30 | Catch-up exits, manual liquidations | -0.27 pp | -0.19 pp |
| E7 | 09-30 to 11-04 | Frozen configuration | +1.60 pp | +1.40 pp (to 11-03) |
| All | 06-24 to 11-04 | | -3.76 pp | -4.57 pp (to 11-03) |

The frozen era's +1.60 pp came from five weeks of a 3-name sleeve; its t of 2.14 was found after
choosing among several era cuts, and over the whole period the sleeve's t is 0.73, so Lens 3
reads it as noise (`docs/research/2026-11-04-lens3-live-parity.md:371-375`).

**Confirmed live-versus-backtest differences** (section 3, B1): unfinished 12:45 PT bar, same-day
fill before the close, settings no backtest picked, 3-name alphabetical cap, 3.3% of account value
per buy, and a daily-loss halt that fires at about 2.7% because of the dividend accrual. How much
these change returns is unknown (`docs/research/2026-11-04-lens6-implementation-correctness.md:384-386`).

**Data integrity, in plain language** (`docs/research/2026-11-04-lens1-backtest-validity.md:36-66`):
the price table that both backtests and live signals read has one stock with the wrong prices
(PARA), places where old and new dividend adjustments meet and create false one-day moves, AVB
carrying EQR's price, and 14 index members missing. Backtests use dividend-adjusted prices; the
paper account trades at raw prices, its broker credits no dividends, and some past paper-versus-SPY
comparisons used SPY's price return without dividends.

## 6. Things that were found while investigating

**The accidental database refresh (2026-11-04, 23:48 PST).** An agent writing the Phase 1
checkpoint ran twelve scripts with `--help` to read their options. One of them,
`scripts/pead_correlation_check.py`, has no option parser and ran its body for about two minutes,
loading prices through the cached loader, which writes to the DB
(`docs/research/2026-11-04-phase1-checkpoint.md:31-39`). The investigation's read-only rule was
broken.
- **What changed (re-verified by SELECT on 2026-11-05):** 19 rows of the `symbol_inception` table
  carry the write time 23:48:18 PST, for APC, ARES, CAM, CSRA, CZR, EMC, GDDY, LITE, NE, PARA,
  PCL, POM, QRVO, SBNY, SEDG, SPLS, STI, VEEV and XYZ; no row in that table is later. The 19
  symbols hold 31,967 daily price rows (`docs/research/2026-11-04-phase1-checkpoint.md:40-55`), but
  not all of them were rewritten: CSRA, PARA, QRVO and SBNY still hold 4,388 rows dated before
  their newly recorded first date (598, 492, 2,922 and 376), so at most 27,579 rows were rewritten
  (`docs/research/2026-11-04-root-cause-challenge.md:91`). Those four series show a price-level jump
  at that date (PARA from 51.21 to 3,506.70 on 2021-02-12; SBNY, CSRA and QRVO also jump).
- **The PARA and SBNY jumps predate the incident.** The 2026-08-20 backup table shows the same kind
  of jump for PARA (51.21 to 101,500, about 29 times larger than today's) and the same post-jump
  SBNY prices. The incident did not create these breaks, but the data still carry them.
- **A lens figure changed:** PARA's maximum close in its index window is now 3,935.11 against
  113,900 measured by Lens 1; the high points shrank by the same factor (28.9) but the low end did
  not, so it is not a uniform rescaling. PARA is still the wrong security, but the PARA figures in
  Lens 1 and Lens 2 can no longer be reproduced from the DB; both reports carry an erratum
  (`docs/research/2026-11-04-phase1-checkpoint.md:65-77`; `docs/research/2026-11-04-lens1-backtest-validity.md:3-5`).
- **Verified unchanged:** the `ohlcv_no_data` table (last check 2026-10-19); no file under `data/`
  or `results/`; no order, container, observer or broker call; the guard snapshot; and SBNY's
  cited ratio, which still reproduces. No lens cites a price, volume or return for the other 17
  symbols (`docs/research/2026-11-04-phase1-checkpoint.md:61-82`).
- **The five current index members (ARES, GDDY, LITE, VEEV, XYZ), checked against the backup:** a
  SELECT-only comparison with the 2026-08-20 backup table shows their daily closes match through
  2026-08-20, with price ratios of 0.9895 to 1.0001 and daily-return differences of at most 0.35
  pp. **Still unknown:** their bars after 2026-08-20, which include the ARES paper trade bought
  10-15 and sold 10-22, and any other price rows refreshed in the same minutes, which cannot be
  identified because the price table keeps no write time. (Comparison by the challenger,
  `docs/research/2026-11-04-root-cause-challenge.md:93`, and re-run by the author with
  SELECT-only queries on 2026-11-05; the checkpoint had left it "not compared",
  `docs/research/2026-11-04-phase1-checkpoint.md:100-105`.)
- **The first check covered the wrong symbols.** The lead's first "bars unchanged" check used SPY,
  HAS and VZ, none of which is among the 19. The review found this and the checkpoint was
  corrected (`docs/research/2026-11-04-phase1-checkpoint.md:84-87`).
- **Owner decision (2026-11-05):** leave the 19 symbols as they are and document them; no restore.
  A new rule followed: no agent runs a script it has not read, and no script is run to see its
  help. (Source: the task ledger, which is gitignored; see section 8.)

**Phase 0 claims corrected.** Section 3's table lists them. The Phase 0 audit remains an unreviewed
document and is not evidence for anything in this report.

**Earlier incorrect statements corrected during review** (each found by a reviewer and fixed before
the lens was accepted; recorded in the gitignored task ledger):
- The design spec said no holdout arm of the replica completed. The primary holdout arm did complete
  (non-citable); only its four sensitivity arms never ran. The wording came from the results file's
  own `audit_stop.holdout` string and an amendment that describe an earlier stopped attempt; the spec
  was corrected (`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md:104-110`;
  `docs/research/2026-11-04-lens7-research-accuracy.md:32-42`).
- Lens 7 first said no per-study pass bar existed before 2026-09-24; a 2026-09-10 audit has success
  criteria.
- Lens 1 first said 16 index members had no prices; two map to existing tickers, so it is 14.
- Lens 6 first described the dividend add-back as affecting the reported value only; it also feeds
  the daily-loss halt.
- Lens 3 first misdated and misattributed an end-of-day offset (from 08-18, AVB alone fits).
- Lens 9 first called month-end Treasury "not computable"; it has a Welch t, and its detectable
  effect is now computed; Lens 9 also overstated what the replica could detect before scaling to
  the stock sleeve.
- Lens 2 first filled an "exposure" column for about 47 rows with text from other rows
  (`docs/research/2026-11-04-lens2-verdict-audit.md:1146-1156`).
- Lens 4 first mixed two bases in its headline (1.25x instead of 1.17x for the 2.0x arm) and left
  out the funding-rate condition.
- Lens 5 first over-scoped its one "evidence against" row; Lens 8 first overstated "all in-sample".
- This synthesis's first version had three wrong claims and several framing faults, found by the
  independent challenge and resolved in section 9.

**The usage-limit interruption.** One Opus dispatch for Lens 5 hit the five-hour usage limit at
about 21:50 PST on 2026-11-04 and was retried after the 22:20 reset; nothing was lost
(`docs/research/2026-11-04-phase1-checkpoint.md:812-815`).

## 7. Limits of this investigation

- **Claude only.** No second model family reviewed any part of this work, by the owner's
  instruction (`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md:84-87`). Each
  lens was reviewed once, fixed once and re-reviewed clean; reviewers re-derived key figures
  independently (for example Lens 2's classes were reproduced 113 of 114 by an independent
  classifier, and Lens 8's BTC beta was matched by an independent reviewer)
  (`docs/research/2026-11-04-lens2-verdict-audit.md:1162-1166`; `docs/research/2026-11-04-lens8-crypto.md:142-147`).
  The classifier reproduction checks that the rules were applied as written, not that the rules are
  sound; Lens 2's rules and classes arrived in one commit. The challenger is also a Claude agent, so
  it is not model-independent either.
- **Recomputation was partial.** Of the 41 rows with a raw file, 5 were recomputed from raw data, 27
  read and checked from the file, and 9 taken from an extraction pass without a re-check
  (`docs/research/2026-11-04-lens7-research-accuracy.md:112-117`).
- **103 of 144 rows are prose only** and cannot be verified at all
  (`docs/research/2026-11-04-lens7-research-accuracy.md:23`).
- **Lens 6 left 17 rows unexamined** for lack of time, and 16 more are unverifiable for other reasons
  (`docs/research/2026-11-04-lens6-implementation-correctness.md:321-335`).
- **Lens 9 could compute a detectable effect for 8 rows**; 33 rows have a file without the needed
  numbers and 103 have no file (`docs/research/2026-11-04-lens9-power-and-regime.md:23-33`).
- **No Phase 2 item was run** (P2-01 to P2-29; owner decision). Every question they would settle is
  "unknown, re-run not run".
- **The 19 rewritten symbols** make Phase 1 evidence on those symbols not exactly reproducible from
  the DB (section 6).
- **Lens rules contain judgements.** Lens 2's power rule tests a NO against its own pass bar where it
  has one and against the 2x gap only where it does not, so rows without a bar face an easier test.
  The rules ask a NO to prove power but do not ask a GO for a t-statistic; four GO results fail only
  the code check (`docs/research/2026-11-04-lens2-verdict-audit.md:181-191, 214-228`). Rule R1 typed
  some measured claims as status labels (for example MidCap 400 reversion), which keeps them out of
  the GO count. Lens 4 admitted the leverage and cross-asset rows to Tier B by a stated exception;
  rejecting it empties those ceilings (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:205-211`).
- **What a fresh reader can reproduce.** `scripts/lens2_build.py` rebuilds Lens 2's classifications
  exactly from committed files (it also reads one gitignored crypto results file)
  (`docs/research/2026-11-04-lens2-verdict-audit.md:1111-1117`). The other lenses' query and build
  scripts, and the challenger's, live only in the session scratchpad
  (/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/), which is
  not in the repository; a fresh reader has the committed JSON outputs and the methods described in
  each report's Verification section, not the scripts. The task ledger that records review rounds
  and owner decisions is gitignored, so those items cannot be checked from the repository.
- **Crypto rows** were classified from upstream inputs; Lens 8 found artifacts the other lenses
  treated as missing, but no crypto class changes under the written rules
  (`docs/research/2026-11-04-lens8-crypto.md:503-513`).

## 8. Owner decisions recorded and still open

The task ledger (`.superpowers/sdd/2026-11-04-root-cause-investigation/progress.md`) is gitignored,
so this section is the committed record of the decisions; the owner is the authority on them.

**Recorded (2026-11-05):**
- Phase 2: none of the 29 items is run.
- Decision 0 (allowing re-runs to write to the price cache): no.
- The 19 rewritten symbols: left as they are and documented; no restore.
- Task 13 (running approved re-runs): skipped as a result.
- The changelog entry and the research-snapshot update wait until the owner accepts this synthesis.

**Still open** (listed, not advised on):
- Whether to accept this synthesis, now that the challenge objections are resolved (section 9).
- Whether and when to merge the investigation branch; it is not merged and not pushed, and a merge
  changes one line of the guard baseline.
- The changelog and snapshot updates after acceptance; `docs/next_steps.md` is not changed without
  the owner.
- A23 funding and tax-character records (P2-18) sit outside this investigation's scope; A23 has a
  scheduled review on 2027-09-30 (`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md:38-40`).
- Whether any P2 item is wanted later.
- Deferred minor review notes in the ledger (wording and labelling items, none changing a class).

## 9. Challenge and resolutions

An independent fresh Opus agent, given the raw results, the inventory and this report's claims but
not the author's reasoning, argued the opposite case and checked 40 claims: 20 confirmed, 16
confirmed with a caveat, 3 wrong, 1 not checked. It raised 0 Critical, 7 Important and 12 Minor
objections, and concluded that the headline numbers hold but the first version was too lenient on
the owner's 2x question and mislabelled the paper shortfall
(`docs/research/2026-11-04-root-cause-challenge.md:55-59, 109, 238-249`). Each objection below is
numbered as in the challenge file. Before fixing, the author re-checked the challenger's evidence
against the raw artifacts and, where it involved the DB, with SELECT-only queries.

| # | Objection (short) | Disposition | What was done, and the evidence |
|---|---|---|---|
| 1 | "103 inconclusive" includes 42 status labels | **Fixed** | Summary now splits 72 tests (0 reliable, 9 unreliable GO, 2 unreliable NO, 58 inconclusive, 3 stopped on own terms) from 42 status labels; counts re-tallied from `docs/research/_lens2_sensitivity.json` (`verdict_type`: 53 NO, 16 GO, 3 test-inconclusive, 42 status). MidCap 400 reversion is now its own red row (21) and the R1 typing is disclosed in section 7. Lens 2's counts were not changed, because this report does not re-classify. |
| 2 | Headline too symmetric on the 2x question | **Fixed** (PARA bound left as inference) | Summary, Q1 and Q4 now state that the replica could detect a 2x-sized whole-account effect in both windows (4.24 vs 9.57; 4.22 vs 6.14) and found -0.64 and +1.00. The French screens are no longer called best-powered for the target (indicative 4.4 pp/yr after 1994 vs a 2.34 gap). The challenger's PARA bound (at most about 2.9 pp/yr) is quoted in B5 as its inference; it was not measured. |
| 3 | Paper shortfall mislabelled; -4.57 omitted | **Fixed** | Q2, B3, section 5 and row 10 now give each basis: -3.76 run-time broker value, -4.57 Alpaca end-of-day equity (to 11-03), -3.45 stored value (`docs/research/2026-11-04-lens3-live-parity.md:185-194`), with the AVB hypothesis caveat. |
| 4 | Leverage bases mixed; beta-matched figures omitted | **Fixed** | Q3 now gives all arms on one basis (true SPY TR: 1.3x +1.32, 1.5x +2.37, 2.0x +4.15; +0.87 / +0.52 at higher funding) and on the artifact basis beside it; adds beta-matched 13.94% / 15.83% vs 12.71% / 13.76% (`docs/research/_static_leverage_aftertax_results.json`, `beta_matched_pretax_cagr`, read by the author); notes the rolling 10-year series reaches into pre-2006 synthetic data (worst windows end 2009, so they start in 1999); calls 2.0x "the largest exposure tested". |
| 5 | Cross-group ranking asserted, not derived | **Fixed** | Section 3 now opens with how the ranking was made, says it is judgement, that no share is measured, that C1 and C3 overlap and that A1 explains false optimism rather than the shortfall; summary list marked "ranked by judgement". |
| 6 | Replica robustness and exposure-matched results omitted | **Fixed** (an earlier partial disagreement was withdrawn) | Summary, Q1, B1 and rows 1-2 now cite same-day fills -0.42, 3 bp -0.57, 10 bp -0.83, cash earning no interest -0.83, exposure-matched +0.26 (t 0.20) and holdout +1.85 (t 1.57), read from `docs/research/_production_replica_results.json`. The first revision disputed "mostly the cash reserve" by reading the -0.83 arm as "no cash reserve"; the task review showed that arm keeps the 5% reserve and only sets its interest to zero (`scripts/production_replica_audit.py:34, 273`; `config.reserve_pct` 0.05 in both arms, re-checked by the author). The comparison book holds SPY in place of the stocks plus the same T-bill cash (`scripts/production_replica_audit.py:57-63`), so about 0.9 pp/yr of the -0.64 is cash drag; the challenger's reading stands. |
| 7 | Incident imprecise for four symbols; part of the unknown checkable | **Fixed** | Re-verified by SELECT: 4,388 older rows for CSRA/PARA/QRVO/SBNY (598/492/2,922/376), PARA jump 51.21 to 3,506.70, the same jump (to 101,500) in the 2026-08-20 backup, and the five current members matching the backup through 2026-08-20 (ratios 0.9895-1.0001, daily-return differences up to 0.35 pp). Section 6 rewritten; bars after 08-20 remain unknown. Ledger citations removed from section 6 and the ledger's gitignored status stated in sections 7 and 8. |
| 8 | Sharpe-only GO calls trailing SPY: one, not two | **Fixed** | A4 now names only the cross-asset 80/20 (trails by 1.09 pp/yr); the KEEP blend tied SPY's CAGR (12.963% vs 12.959%). |
| 9 | Lights inconsistent with the stated rule | **Fixed** | Red now means Lens 2 unreliable or Lens 6 "no". The 0.97 anchor-fix reproduction, 0.68 blend reproduction and pre-PIT ic/kelly join red row 6; the blend revert (Lens 6 "yes") stays yellow as row 7; Kelly July (Lens 6 "no") moved to red row 12; MidCap 400 (Lens 6 "no") is red row 21. Count now 0 / 20 / 6 / 10. |
| 10 | Summary range omits negatives | **Fixed** | Summary, Q1, C2 and row 17 now give -0.62 to +0.83 after 1994, and the post-2008/2013 negatives in C2. |
| 11 | "Every headline with a raw file recomputes" contradicts section 7 | **Fixed** | Summary and Q6 now say every headline checked matched; 9 were not re-checked and 7 could not be located. |
| 12 | Momentum fallback framing | **Fixed** | A2 and row 14 now say the fallback in all 17 folds was plain 12-1 top-50 momentum, so +4.37 measures plain momentum (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:238`). |
| 13 | Full-century momentum test not reported | **Fixed** | Q1 and row 16 now give t 3.05 and 3.11 over 1927-2026 (p 0.0011 and 0.0009 in `docs/research/_momentum_french_screen_results.json`, read by the author), just short of the registered two-test correction. |
| 14 | FOMC NO depends on its cost assumption | **Fixed** | Row 27 now states the 10 bp round-trip cost (the stock default) and that at 2 bp the net t would be about 2.0. Checked from `docs/research/_fomc_clustered_event_study_20261020.json`: gross minus net mean difference is exactly 0.0010; SE = 0.0017678 / 2.2576; (0.0017678 - 0.0002) / SE = 2.00. |
| 15 | Crypto framing omits results that cut the other way | **Fixed** | Q5 now says the runs beat ETH in four of five full windows, the out-of-sample stretch beat BTC by about 23 to 33 pp with low beta, and that -21.34% is the report's own BTC figure (-17.46% on today's tape) (`docs/research/2026-11-04-lens8-crypto.md:206-237`). |
| 16 | 87.5% halt rate needs its definition | **Fixed** | A2 and row 26 now say "halted at least once, with gates always passing", plus about a third of folds on the fallback and about half of runs ending halted (`docs/research/2026-11-04-lens2-verdict-audit.md:760-761`). |
| 17 | Small internal inconsistencies | **Fixed** | Q3 explains 3.54 vs 3.55 (one-day-different SPY starting close, `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:182-183`); C2 labels 9.54 as the 2022-2026 test window; section 5 separates the $100,000 funding in 2024 from the about $102,459 first snapshot in this period. |
| 18 | Limits honest, with one qualifier | **Fixed** | Section 7 now says the 113-of-114 reproduction checks rule application, not rule soundness, and that the challenger is also a Claude agent. |
| 19 | Owner constraints respected | **No change needed** | The challenger found no recommendation, retirement or shutdown advice, account-type suggestion or banned phrase; the revised text was re-checked with the report checker. |

Totals: 18 fixed, 1 needing no change (objection 19), 0 disagreed, 0 left unresolved. The one
disagreement in the first revision (objection 6) rested on a misreading and was withdrawn after
the task review. What remains genuinely uncertain after the challenge is listed
in section 1 ("What is unknown") and section 7; the challenger's list agrees
(`docs/research/2026-11-04-root-cause-challenge.md:264-267`).

## 10. Index of evidence

Commits from `git log --oneline -- <file>` (newest first). Review rounds from the gitignored task
ledger `.superpowers/sdd/2026-11-04-root-cause-investigation/progress.md`.

| Evidence | File(s) | Commits | Review rounds |
|---|---|---|---|
| Design spec | `docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` | 5cf88940 (holdout-wording correction), 3e456008, 62e7ccf2, e79bf8ea, 9fd8d509, 886a319e, a385f49e | approved by owner; independent Opus review before approval |
| Phase 0 audit (unreviewed, not evidence) | `docs/research/2026-11-04-phase0-trust-audit.md` | a385f49e | none |
| Guard baseline | `docs/research/_investigation_guard_baseline_2026-11-04.txt` | 8eeda009 | Task 3: 1 review, clean |
| Inventory | `docs/research/_investigation_inventory.json`, `docs/research/2026-11-04-investigation-inventory.md` | 02a97b9e, 823071c0 | 1 review, 1 fix round, clean |
| Lens 2 table spec | `docs/research/2026-11-04-lens2-sensitivity-spec.md` | a5fe547b, 799b5a7d, c67f93e1 | 1 review, 2 fix rounds, clean |
| Lens 1 backtest validity | `docs/research/2026-11-04-lens1-backtest-validity.md`, `docs/research/_lens1_tape_state.json` | b195ccc0 (erratum), 6576811e, 77cecdab | 1 review, 1 fix round, clean |
| Lens 2 verdict audit | `docs/research/2026-11-04-lens2-verdict-audit.md`, `docs/research/_lens2_sensitivity.json`, `scripts/lens2_build.py` | b195ccc0 (erratum), 3dfacd83, 1b6c60d6 | 1 review, 1 fix round, clean |
| Lens 3 live parity | `docs/research/2026-11-04-lens3-live-parity.md`, `docs/research/_lens3_snapshot_gap.json` | a825594b, c145d1ed | 1 review, 1 fix round, clean |
| Lens 4 feasibility and ceiling | `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md`, `docs/research/_lens4_ceiling.json` | 18cac0da, bad6d855 | 1 review (Opus), 1 fix round, clean |
| Lens 5 coverage gaps | `docs/research/2026-11-04-lens5-coverage-gaps.md`, `docs/research/_lens5_coverage_matrix.json` | 9769a79d, e14494e2 | 1 review (Sonnet), 1 fix round, clean |
| Lens 6 implementation correctness | `docs/research/2026-11-04-lens6-implementation-correctness.md`, `docs/research/_lens6_verdicts.json`, `docs/research/_lens6_replays.json` | 659e3417, 36acba57 | 1 review, 1 fix round, clean |
| Lens 7 research accuracy | `docs/research/2026-11-04-lens7-research-accuracy.md`, `docs/research/_lens7_recompute.json` | cc6cb579, debcfaaa | 1 review, 1 fix round, clean |
| Lens 8 crypto | `docs/research/2026-11-04-lens8-crypto.md`, `docs/research/_lens8_tape_checks.json` | 392158e5, 6f5b0725 | 1 review (Sonnet), 1 fix round, clean |
| Lens 9 power and regime | `docs/research/2026-11-04-lens9-power-and-regime.md`, `docs/research/_lens9_power.json` | 7fe1ceb4, 21664be7 | 1 review, 1 fix round, clean |
| Phase 1 checkpoint (incident, P2 list) | `docs/research/2026-11-04-phase1-checkpoint.md` | b195ccc0 (incident correction and errata), 47fabb39 | 1 review (Sonnet), 1 fix round, clean |
| This synthesis | `docs/research/2026-11-04-root-cause-investigation.md` | e4362512 (first version); this revision | challenged once (next row) |
| Independent challenge | `docs/research/2026-11-04-root-cause-challenge.md` | 965d78c0 | fresh Opus agent, evidence and claims only |
| Inventory-to-table map | `docs/research/_synthesis_inventory_map.json` (built by a scratchpad script that asserts all 144 ids are mapped once) | this revision | task review fix round 2 |
