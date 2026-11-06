# Findings in plain English: why the research has not doubled SPY

**Date:** 2026-11-05. **Reading edition of the synthesis: the originals stay as the record.** Terms are defined where they first appear and in the reference sheet (`research/audits/root-cause-2026-11-04/reference-sheet.md`), which also maps each check and round name to its file.

## 1. The question and the short answer

The owner asked why the research has not come close to twice SPY's total return over the same period (pre-tax first, after-tax beside it) and what the evidence says is the most that can be reached. SPY is a fund that tracks the S&P 500. It is the benchmark for every result. Total return (TR) is price change plus reinvested dividends. The unit is pp/yr, percentage points a year.

The verdict:

- **The design that was built:** negative. The record shows no 2x-sized effect, and the test could have seen one. That design is about 85% SPY plus a small stock sleeve (the part of the account that runs one strategy). The production replica, which re-runs the live rules on history, could detect a whole-account excess of about 4.2 pp/yr. Its two windows needed 9.57 and 6.14 pp/yr. It returned -0.64 and +1.00.
- **Stock picking itself:** undecided. Of 72 measured tests, 0 are reliable. None shows an edge. None rules out an edge of 1 to 4 pp/yr.
- **Crypto:** nothing beat holding BTC over any three-year window.
- **The measured maximum:** only 2x leverage and momentum before 1994 reached the gap needed in their own windows. Leverage is exposure, not skill.
- **After tax:** a book that sells within a year needs about +2.42 pp/yr before tax just to tie SPY after tax.
- **Why:** six causes, ranked by judgement. No cause's share of the shortfall was measured.

In short: the target is large, the measured effects are small, and the tests were too blunt to tell small from zero.

Source: the synthesis, section 1 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`).

## 2. What the evidence says, by topic

**The built design against 2x.** The replica's primary window ran 2021-2026 and returned 14.19% a year against 14.83% for SPY, so -0.64 pp/yr, with a t of -0.69. The t-statistic is an effect divided by its standard error, and a t of 2 or 3 is the usual bar for "probably not luck". Its 2010-2020 holdout (data kept aside for a final check) returned +1.00 with a t of 0.69, and is not citable because of bad prices.

Against a passive book that held SPY in place of the stocks plus the same share of T-bill cash (short-term government debt), the replica returned +0.26 pp/yr with a t of 0.20, which cannot be told from zero. The cash reserve (5% of the account kept in cash) costs about 0.9 pp/yr relative to SPY, and the picks and other differences add back about +0.26, leaving the -0.64. Variants read -0.42 with same-day fills, -0.57 and -0.83 with costs of 3 and 10 bp (basis points, where one bp is 0.01%), and -0.83 with cash earning no interest.

Stocks were only 9.37% of that book. At sleeve level the same test could only see an excess above about 45 pp/yr. So the replica answers the 2x question for the whole account and leaves stock picking open. An unmeasured caveat: it traded one stock whose stored prices belong to another security (PARA).

**Stock picking itself.** The Verdict Trust check sorted past verdicts by whether the data were clean, the code checked, the number recomputed and the test powerful enough (power is the chance a test catches a real effect of a given size). Of 72 measured tests, 0 are reliable, 11 are unreliable (9 GO and 2 NO), 58 are inconclusive and 3 stopped on their own terms. Another 42 rows are status labels, not tests. "Inconclusive" means the test could not decide, not that it found nothing.

Of 53 NO results, none is a reliable NO, and power fails in 51. The momentum replay (buy the stocks that rose most over the past year, skipping the latest month), for example, rules out only an excess above about 22.7 pp/yr. The French data library (free, survivorship-free American stock portfolio returns back to 1926) gives the longest view. Momentum over 1927-2026 scored a t of 3.05 for big stocks and 3.11 for small, above a single t of 3 but just short of the correction for testing two ideas that was set in advance. After 1994 the registered results run from -0.62 to +0.83 pp/yr against the market, and momentum turned negative after 2013. No edge has been shown. A small one of 1 to 4 pp/yr has not been ruled out.

**Crypto.** Every crypto portfolio with a recoverable record lost to holding BTC over its three-year range, by 5 to 44 pp a year. The headline 48.46% is real in its file, but it is an in-sample replay (tested on the data used to choose the settings) in one year when BTC fell. It moved about a quarter as much as BTC, and the leftover gain was not significant (t 1.51). Live crypto left no usable record: 0 trades in the database.

**The measured maximum.** Doubling SPY needs 9.57 pp/yr over 2021-2026, 3.54 over 2006-2026 and 0.77 over 1927-2026. The gap shrinks as the window lengthens, and most tests used the short one. Only two things reached their window's gap. The first is the 2.0x SSO arm (a fund that targets twice SPY's daily move): +4.15 pp/yr over 2006-2026, with a worst fall of -84.8%. It was picked after seeing five arms, with no t-statistic. The second is momentum before 1994, which is the period before publication and which reverses after 2013. Leverage is exposure minus cost: at 1.3x and 1.5x, the same exposure financed at T-bill rates earned 13.94% and 15.83% a year, more than the leveraged-fund books' 12.71% and 13.76%. No stock-picking result after 1994 reached its gap. The best 2021-2026 estimate is momentum at +4.37 pp/yr, with a 95% confidence interval (the range of true values the data cannot rule out) of -7.07 to +16.05, so the data cannot tell it from zero.

**After tax.** All 74 closed paper stock round trips were short-term (held a year or less, taxed at ordinary income rates). Such a book needs about +2.42 pp/yr before tax at the 24%/15% brackets, or +5.05 at 37%/20%, just to tie SPY held and taxed once. A +1.0 pp/yr pre-tax edge becomes -1.08 or -2.55. This is a stylised model. The paper account's own tax drag was not computed.

Source: the synthesis, answers to questions 1 to 5 in section 2 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`), with the after-tax model in the Ceiling check report.

## 3. Why the target was missed

The ranking is the author's judgement. No cause's share of the shortfall was measured, and several effects are unknown. Causes one and two overlap, because with a sleeve this small the book's construction is the binding limit. Strength describes the facts behind each cause, not its rank.

Three terms before the table. A leak is information a test should not have had. Two matter here: the anchor leak (the fallback settings were fitted on the whole test span) and the universe leak (14.1% of entries fell on days the stock was not in the index). The Sharpe ratio is return per unit of swing, not total return. MDE is the minimum detectable effect, the smallest true effect a test finds 80% of the time.

| Rank | Cause in plain words | Strength of the evidence | The number that matters |
|---|---|---|---|
| 1 | The target is large next to every measured stock-picking effect. This is a property of the target, not a defect. | High: two checks reached it by independent arithmetic. | Gap of 9.57 pp/yr over 2021-2026. Best registered effect after 1994 is +0.83 against a gap of 2.34. |
| 2 | The book is about 85% SPY and 9% stocks, so it cannot stray far from SPY. | High: it is arithmetic. | A 9.57 pp/yr whole-account gap would need roughly +102 pp/yr from the sleeve (an illustration). |
| 3 | Early "beat SPY" results came from leaks that favoured the strategy, and the live design was chosen on them. They explain the false optimism, not the shortfall. | High that those results are unreliable. Unknown how much each bug contributed. | Core Sharpe fell from 0.99 to 0.59 after the universe fix. The 1.12 did not reproduce (0.97). |
| 4 | The tests could not tell a small stock-picking edge from zero. | High for the rows that could be computed, unknown for the 103 rows with no raw file. | Power fails in 51 of 53 NO results. Replica MDE 4.24 pp/yr for the account, about 45 for the sleeve. |
| 5 | The paper shortfall is mostly idle cash and broker quirks, and live runs no tested configuration. | Medium-high, because the split by component is an estimate. | -3.76 pp at run-time broker prices and -4.57 on Alpaca's end-of-day equity. Idle cash cost about 2.46 pp. |
| 6 | The search stayed narrow: large-cap, long-only, mostly one signal family. | Low as a cause, because coverage is confirmed but its role is inference. | 33 of 72 tests are one signal family. 62 of 82 web-research hypotheses never ran. |

Three further items sit below the top six. Pass bars (the rule, set in advance, that decides GO or NO-GO) arrived late: nothing committed before 2026-09-10 sets one per study. Many tries: 72 tests in 11 families, and with zero true edge, at least one false win at a t of 2 is 81% likely across 72 independent tests. One regime: 37 rows rest only on 2020-2026 data.

Source: the synthesis, section 3 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`).

## 4. What is solid, what is wrong and what is unknown

**Solid.** The five known data and code bugs are fixed (no weekend bars, no duplicate days, one SPY row per date). Of the 41 rows with a raw results file, 5 were recomputed and 27 read by hand, though 7 headline numbers could not be found in their files. No number found disagreed with its report. 9 were taken from an extraction pass without a re-check. The signal code does what it says: four strategies were replayed by hand on real prices with 0 mismatched buy or sell dates. 11 of 14 academic citations checked say what the reports say. The paper accounting reconciles to the cent apart from a known dividend add-back.

**Wrong: the leaks.** Every June-July positive headline predates the fixes for the two leaks above. The Verdict Trust check classes all nine GO results as unreliable, and removing the leaks lowered the results.

**Wrong or unfixed: price-data faults.** PARA's prices belong to a different security over its whole index membership, and the replica traded it 3 times. Old and new dividend adjustments meet in seams that show as false one-day moves. AVB, a stock that merged into EQR, has carried EQR's price since 2026-07-20. 14 index members have no prices in 2021-2026. The 2010-2020 data have 5,595 daily moves of 40% or more, so there is no clean holdout. On PARA's size, the challenger estimates by inference, not measurement, at most about 2.9 pp/yr, far less than the 7-9 pp/yr distance to the target.

**What live runs against what was tested.** Live does not run the configuration any backtest measured:

- Live uses setting values no fold ever picked, and two are outside the search grid. A fold is one pick-then-test slice of a walk-forward run (pick settings on a past window, test on the next unseen one, repeat).
- Live computes signals at 12:45 Pacific time from today's unfinished price bar.
- Live fills about 15 minutes before the close. The backtests fill at the day's final close.
- Live holds at most 3 stocks, picked alphabetically.
- Live sizes each buy at 3.3% of account value. The backtests use 3% of remaining cash.
- The saved account value includes a dividend accrual (a dividend booked before it is paid), so the daily-loss halt fires at about 2.7% real loss, not 3%.

These differences exist with high confidence. How much they change returns is unknown.

**The live account's shortfall.** From 2026-06-24 the paper account, held at the broker Alpaca, trailed SPY by 3.76 pp at the broker value the trader read at run time, or 4.57 pp on Alpaca's own end-of-day equity. Part of that 0.81 pp gap rests on an unproven hypothesis that Alpaca values the untradable AVB holding near zero. Idle cash before the SPY cash sweep (the live rule that puts idle cash into SPY while keeping a 5% reserve) cost about 2.46 pp. Alpaca never applied the MNST 2-for-1 split (0.85 pp) and never credited a dividend, leaving a $320.24 accrual that is not cash. Without the split the stock picks netted about zero. The frozen era (no configuration changes since 2026-09-30) returned +1.60 pp, which reads as noise: its t of 2.14 was found after choosing among several era cuts. Detecting a 10 pp/yr sleeve edge from paper trading alone would take about 23 to 40 years.

Live and the backtest agree on the name of the strategy and on very little else.

Source: the synthesis, sections 2, 3 and 5 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`).

## 5. What the checks cannot see

- **Claude only.** No second model family reviewed any of this work, by the owner's instruction. Each check was reviewed once, fixed once and re-reviewed clean. The Verdict Trust check's classes were reproduced 113 of 114 by an independent classifier, which shows the rules were applied as written, not that the rules are sound. The challenger is also a Claude agent, so it is not model-independent.
- **Recomputation was partial.** 103 of 144 rows are prose only and cannot be verified. The Code Correctness check left 17 rows unexamined for lack of time and 16 more are unverifiable. The Test Power check could compute a detectable effect for 8 rows.
- **Judgement sits inside the rules.** The Verdict Trust check tests a NO against its own pass bar where it had one, usually +1.0 pp/yr, stricter than the owner's 2x gap. It asks a NO to prove power but no GO for a t-statistic.
- **Reproducibility.** One script rebuilds the Verdict Trust classifications from committed files. The other checks' scripts lived only in a scratchpad outside the repository, so a fresh reader has the JSON outputs and method notes, not the scripts.
- **The price-history incident.** On 2026-11-04 an agent ran a script only to read its help text. The script had no option parser, ran its body and wrote to the database through the price loader. This broke the investigation's read-only rule. 19 symbols had their price history rewritten, at most 27,579 rows. Price jumps in PARA and one other symbol predate the incident. PARA's figures in two check reports can no longer be reproduced from the database, and both carry an erratum. Five current index members matched a 2026-08-20 backup through that date. Their later bars, and other rows refreshed in the same minutes, cannot be identified. The owner chose to leave the 19 symbols as they are.
- **What was not run.** The Optional Re-run List has 29 items. None was run, by the owner's decision. Every question they would settle stays "unknown, re-run not run". No re-run can settle four: the window behind the 1.12 Sharpe, the raw output behind 1.14, the April-May 2026 crypto runs (their data were wiped) and whether SPY's duplicate rows existed before 2026-06-20.
- **The independent challenge.** A fresh Opus agent, given the raw results and the claims but not the author's reasoning, argued the opposite case and checked 40 claims: 20 confirmed, 16 confirmed with a caveat, 3 wrong and 1 not checked. It raised 0 Critical, 7 Important and 12 Minor objections. Its verdict: the headline numbers hold, but the first version was too lenient on the owner's 2x question and mislabelled the paper shortfall. 18 objections were fixed and 1 needed no change.

The challenger found three wrong claims in the first draft. The synthesis fixed them, which is the intended order of events.

Source: the synthesis, sections 6, 7 and 9 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`).

## 6. Owner decisions

**Recorded (2026-11-05):**

- None of the 29 items on the Optional Re-run List is run.
- The no-writes rule: the owner answered no to re-runs writing to the shared price cache, which blocks every item tagged as doing so.
- The 19 rewritten symbols are left as they are and documented. No restore.
- The changelog entry and the research-snapshot update wait until the owner accepts the synthesis.
- A new rule: no agent runs a script it has not read, and no script is run to see its help.

**Still open** (listed in the synthesis, not advised on):

- Whether to accept the synthesis, now that the challenge objections are resolved.
- Whether and when to merge the investigation branch. The synthesis lists it as open, and a later note in the same file records a merge. I did not re-check.
- The changelog and snapshot updates after acceptance.
- A23 funding and tax-character records sit outside this investigation's scope. A23 has a scheduled review on 2027-09-30.
- Whether any Optional Re-run List item is wanted later.

The longer decision list is in the consolidated direction review (section 7).

Source: the synthesis, section 8 (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`).

## 7. Where to read more

| Read | For | File |
|---|---|---|
| Synthesis | The full findings, ranked causes, trust table and challenge resolutions | `findings/2026-11-04-root-cause-investigation.md` |
| Independent challenge | The opposite case and the 40-claim audit | `findings/2026-11-04-root-cause-challenge.md` |
| Main Evidence Round checkpoint | Limits, the database incident and the Optional Re-run List | `findings/2026-11-04-phase1-checkpoint.md` |
| Early Trust Audit | The unreviewed first audit, not used as evidence | `findings/2026-11-04-phase0-trust-audit.md` |
| Backtest Cleanliness check | What contamination is left in prices and code | `lenses/2026-11-04-lens1-backtest-validity.md` |
| Verdict Trust check | Which verdicts can be trusted, and for what size of effect | `lenses/2026-11-04-lens2-verdict-audit.md` |
| Paper-versus-Backtest check | Whether the paper trader matches the backtests | `lenses/2026-11-04-lens3-live-parity.md` |
| Ceiling check | The measured maximum and the target arithmetic | `lenses/2026-11-04-lens4-feasibility-and-ceiling.md` |
| Coverage check | What was never tested | `lenses/2026-11-04-lens5-coverage-gaps.md` |
| Code Correctness check | Whether the code does what the spec says | `lenses/2026-11-04-lens6-implementation-correctness.md` |
| Report Accuracy check | Whether reported numbers are true | `lenses/2026-11-04-lens7-research-accuracy.md` |
| Crypto check | What was tried in crypto and what it shows | `lenses/2026-11-04-lens8-crypto.md` |
| Test Power check | Whether the tests could see an effect | `lenses/2026-11-04-lens9-power-and-regime.md` |
| Opus and Argon reviews | The two frontier-model reviews and the check of the Argon review | `reviews/` |
| Consolidated direction review | The two reviews merged, with the owner decision list | `reviews/2026-11-05-direction-reviews-consolidated.md` |
| Reference sheet | Any term, check name or label | `reference-sheet.md` |

Source: the folder layout under `research/audits/root-cause-2026-11-04/`.
