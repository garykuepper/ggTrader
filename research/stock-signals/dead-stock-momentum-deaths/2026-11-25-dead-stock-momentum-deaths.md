# Dead-stock momentum deaths: why the rule held two to four times as many stocks that stopped trading

**Date:** 2026-11-25
**Study:** `research/stock-signals/dead-stock-momentum-deaths` (plan written down and frozen before any code: [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md); one dated amendment at its end)
**Question:** The random-baseline study found that the first test's momentum rule (each month buy the 50 stocks that rose the most over the past year, skipping the last month) held 31, 117 and 24 stocks on the day they stopped trading in the three windows, against about 13, 33 and 8 for 50 stocks drawn at random from the same list. Why?

## 1. In short

- **In these windows the strong risers stopped trading about 2.5 to 3.6 times as often as the average stock on the list.** Each month, among the stocks the rule could hold, the share that stopped trading within the next month was 4.4, 3.7 and 2.5 per 1,000 stock-months (2000 to 2004, 2005 to 2019, 2020 to 2024). Among the rule's 50 picks it was 11.2, 13.3 and 8.5 per 1,000. The ratio is 2.5, 3.6 and 3.4, with 95 percent intervals of 1.7 to 3.4, 3.0 to 4.2 and 2.1 to 4.7.
- **The rule's count matches what that rate implies.** This is a consistency check, not an independent prediction: the rates and the counts come from the same stock-months. A random 50 would have held 13.3, 33.1 and 7.3 stocks at death (the random-baseline study's draws: 13, 33, 8). The 50 top-ranked stocks actually died 33, 119 and 25 times, and the first test's ledger counted 31, 117 and 24. This study's counts are 2, 2 and 1 higher than the ledger's (6.5, 1.7 and 4.2 percent), below the brief's 10 percent warning line. The individual differences were not traced; the brief names buys the ledger could not fill as a possible reason, which this audit did not check.
- **The rate climbs with the momentum rank.** In 2005 to 2019 and 2020 to 2024 the death rate falls at every step from the top 50 down to the ranks below 500 (13.3, 8.2, 4.1 and 1.5 per 1,000, and 8.5, 5.1, 2.6 and 1.2). In 2000 to 2004 it does so except that ranks 51 to 150 sit below ranks 151 to 500.
- **The dying picks look like stocks that jumped in a day and then went flat.** 64 to 70 percent of the top-50 stocks that later died had a one-day gain of 20 percent or more inside the year their score reads, against 26 to 41 percent of the top-50 stocks that did not die and 6 to 12 percent of all stocks on the list. In the last month before they stopped, 73, 94 and 92 percent of them moved less than 10 percent either way. Their median rise in the 63 trading days before the decision was 14.7, 17.9 and 36.5 percent, against 7.1, 3.9 and 3.9 percent for dying stocks outside the top 50. None of the dying top-50 picks fell by more than half in its final month; 0 to 6 percent fell by 10 to 50 percent.
- **This is a description, not a cause.** A jump of 20 percent in a day is what a buy-out announcement looks like, and a flat final month is what a stock sitting at a deal price looks like. Earnings and news also make jumps, and these numbers cannot tell them apart. A separate filing search (the data-integrity study's phases 2 and 3) classified 103 of the first test's 141 held-at-death exits as acquisition-type: 88 with a checked identity and 15 name-only leads. Payment amounts and forms were not verified. That search did not cover 2000 to 2004.
- **The excess appears in most years.** Its concentration within years was not measured. The top-50 death count was above the random-50 expectation in all 5 years of 2000 to 2004, in 14 of 15 years in 2005 to 2019 and in 4 of 5 years in 2020 to 2024 (the exceptions: 2017 and 2021, each with one death).
- **No label.** The owner decides.

## 2. What was done

- **The measures were fixed before any code** (brief section 1): for every month-end decision day of the first test (the same universe A list, release 1, CRSP load 1, the same eligibility and scores), the usable stocks were ranked by momentum score; a stock "dies in the holding month" if its price history ends between the next trading day's execution and the day before the following month's execution, it has no successor series, and it is not a row cut off at the end of the CRSP file. Nothing in the account was simulated.
- **Checks.** The rebuilt top 50 equals the copied planner's targets on every decision day in all three windows (the run stops otherwise). The audit's top-50 death count was compared with the first test's saved count: 33 against 31, 119 against 117, 25 against 24. The "expected random 50" is the analytic count, `sum over months of 50 x deaths / usable stocks`, and agrees with the random-baseline draws' medians.
- **One review, one fix.** Codex Sol reviewed the code before any real run. It found that the one-day-jump measure counted calendar trading days where the score counts a stock's own valid closes, which differ when a stock has holes in its price history. I fixed it, wrote the two failing examples into tests, and recorded the change as a dated amendment in the brief before the real run (the span runs from the bar 252 valid closes before the stock's last valid close on or before the decision day to the bar 21 valid closes before it, both included, and a jump never bridges a missing close). The builder was Gemini 4 Neon.
- **Run times:** 144 seconds (2000 to 2004), 159 (2020 to 2024) and 217 (2005 to 2019), all on release 1 and load 1. The results were first produced on one working tree with the other windows' result files lying untracked next to them; because that marks the run as made on an unclean tree, all three were rerun one by one on a clean tree with each result committed before the next run (code commit `873c7967`, unchanged), and the numbers are identical. Raw numbers: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json).

## 3. Results

**A. Death rates and counts.** A stock-month is one usable stock on one decision day that has an execution day inside the window; usable means eligible, with a finite score and no series end before that day. Rates are deaths per 1,000 stock-months.

| Window | Stock-months | Deaths, all | Rate, all | Rate, top 50 | Ratio (95% interval) | Random 50 would hold | Top 50 died | Random-baseline draws, middle | First test's ledger |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2000 to 2004 | 56,485 | 251 | 4.44 | 11.19 | 2.52 (1.70 to 3.38) | 13.3 | 33 | 13 | 31 |
| 2005 to 2019 | 174,123 | 643 | 3.69 | 13.30 | 3.60 (3.02 to 4.21) | 33.1 | 119 | 33 | 117 |
| 2020 to 2024 | 57,312 | 142 | 2.48 | 8.47 | 3.42 (2.10 to 4.73) | 7.3 | 25 | 8 | 24 |

The interval comes from resampling whole months (2,000 resamples, all valid).

**A. Death rate by momentum rank band** (per 1,000 stock-months; deaths in brackets; stock-months in the band below the table):

| Window | Ranks 1 to 50 | 51 to 150 | 151 to 500 | 501 and below |
|---|---|---|---|---|
| 2000 to 2004 | 11.19 (33) | 5.42 (32) | 6.10 (126) | 2.22 (60) |
| 2005 to 2019 | 13.30 (119) | 8.21 (147) | 4.05 (254) | 1.45 (123) |
| 2020 to 2024 | 8.47 (25) | 5.08 (30) | 2.57 (53) | 1.22 (34) |

Stock-months per band (ranks 1 to 50, 51 to 150, 151 to 500, 501 and below): 2000 to 2004, 2,950, 5,900, 20,650 and 26,985; 2005 to 2019, 8,950, 17,900, 62,650 and 84,623; 2020 to 2024, 2,950, 5,900, 20,650 and 27,812.

**A. By year** (the year in which the holding month starts): usable stock-months, top-50 deaths, and the number a random 50 would have held.

| Year | Usable stock-months | Top 50 died | Random 50 expected |
|---|---:|---:|---:|
| 2000 | 9,763 | 8 | 4.9 |
| 2001 | 11,534 | 12 | 3.6 |
| 2002 | 11,692 | 3 | 1.5 |
| 2003 | 11,804 | 6 | 1.4 |
| 2004 | 11,692 | 4 | 1.9 |
| 2005 | 10,690 | 6 | 2.3 |
| 2006 | 11,655 | 7 | 2.9 |
| 2007 | 11,560 | 12 | 3.7 |
| 2008 | 11,618 | 15 | 2.7 |
| 2009 | 11,814 | 6 | 1.2 |
| 2010 | 11,714 | 7 | 2.0 |
| 2011 | 11,649 | 8 | 2.3 |
| 2012 | 11,682 | 12 | 1.8 |
| 2013 | 11,695 | 4 | 1.4 |
| 2014 | 11,627 | 9 | 1.4 |
| 2015 | 11,648 | 11 | 2.6 |
| 2016 | 11,711 | 12 | 2.8 |
| 2017 | 11,693 | 1 | 2.1 |
| 2018 | 11,742 | 5 | 2.1 |
| 2019 | 11,625 | 4 | 1.7 |
| 2020 | 10,657 | 7 | 1.4 |
| 2021 | 11,319 | 1 | 1.5 |
| 2022 | 11,642 | 6 | 1.9 |
| 2023 | 11,854 | 9 | 1.5 |
| 2024 | 11,840 | 2 | 1.0 |

**B. The stocks that died, by group** (the decision day that opens their last holding month):

| Window | Group | Stocks | Median rise in the 63 trading days before the decision | Final month: moved less than 10% either way | Final month: rose 10% or more | Final month: fell 10% or more |
|---|---|---:|---:|---:|---:|---:|
| 2000 to 2004 | In the top 50 | 33 | +14.7% | 73% | 21% | 6% |
| 2000 to 2004 | Not in the top 50 | 218 | +7.1% | 83% | 10% | 7% |
| 2005 to 2019 | In the top 50 | 119 | +17.9% | 94% | 6% | 0% |
| 2005 to 2019 | Not in the top 50 | 524 | +3.9% | 96% | 3% | 1% |
| 2020 to 2024 | In the top 50 | 25 | +36.5% | 92% | 8% | 0% |
| 2020 to 2024 | Not in the top 50 | 117 | +3.9% | 88% | 7% | 3% (plus 3% with no price on the decision day) |

**B. Quartiles of the dying stocks' profile** (lower quartile, median, upper quartile; valid counts differ from the group size only where a price was missing: 523 of 524 and 114 of 117 for the 63-day return of the not-in-top-50 group in 2005 to 2019 and 2020 to 2024; final-month moves below 0.5 and between 0.5 and 0.9, in the same order, are 0 and 2 (top 50) and 1 and 14 (others) in 2000 to 2004, 0 and 0 and 2 and 4 in 2005 to 2019, 0 and 0 and 2 and 1 in 2020 to 2024):

| Window | Group | Momentum score | Rank as a share of the usable list | Rise in the 63 trading days before the decision |
|---|---|---|---|---|
| 2000 to 2004 | In the top 50 | 0.92 / 1.18 / 3.10 | 0.017 / 0.029 / 0.040 | +0.3% / +14.7% / +36.1% |
| 2000 to 2004 | Not in the top 50 | 0.06 / 0.30 / 0.51 | 0.223 / 0.360 / 0.583 | +1.0% / +7.1% / +23.7% |
| 2005 to 2019 | In the top 50 | 0.60 / 0.92 / 1.28 | 0.010 / 0.026 / 0.041 | +3.3% / +17.9% / +49.0% |
| 2005 to 2019 | Not in the top 50 | 0.11 / 0.28 / 0.45 | 0.141 / 0.273 / 0.496 | +0.9% / +3.9% / +16.2% |
| 2020 to 2024 | In the top 50 | 0.70 / 1.12 / 2.90 | 0.003 / 0.020 / 0.042 | +1.5% / +36.5% / +88.2% |
| 2020 to 2024 | Not in the top 50 | 0.07 / 0.27 / 0.47 | 0.152 / 0.311 / 0.560 | +0.7% / +3.9% / +17.9% |

**C. One-day gains of 20 percent or more inside the span the score reads** (counted on each stock's own closes; share of stock-months flagged):

| Window | Top 50, later died | Top 50, did not die | All usable stocks |
|---|---|---|---|
| 2000 to 2004 | 67% (22 of 33) | 39% (1,125 of 2,917) | 12% (6,959 of 56,485) |
| 2005 to 2019 | 70% (83 of 119) | 26% (2,304 of 8,831) | 6% (9,932 of 174,123) |
| 2020 to 2024 | 64% (16 of 25) | 41% (1,211 of 2,925) | 11% (6,188 of 57,312) |

## 4. What this means for the earlier results, and what it does not say

- **What it adds to the dead-stock question.** In these three windows the top-50 band had the highest observed rate of ending price histories. That exposes the rule to more of the assumed zero-recovery events than random selection, which is why the worst reading hits this rule harder than chance. This audit does not measure the resulting account loss and does not establish why the rate differs. The price patterns (a rise, a one-day jump, a flat last month) are consistent with some acquisition-related endings, but they do not establish cash recovery or which payout scenario is closest to reality. The data-integrity study's filing search classified 103 of the 141 held-at-death exits as acquisition-type (88 checked, 15 name-only; amounts not verified); the payout rerun assumed last-price recovery for the 88 checked ones in its headline and for all 103 in a sensitivity case. For 2000 to 2004 no filings were searched.
- **A jump flag is not a buy-out.** Jumps were also common among top-50 stock-months that did not end in death (26 to 41 percent). The dying picks are higher still (64 to 70 percent), which is the only part that points at the buy-out story.
- **No account is simulated.** Nothing here says what the picks earned, what the deaths paid, or how the rule compares with SPY.
- **Same three windows as before**, so this adds no new market sample. The list is the 1,000 largest stocks priced at $5 or more; other stocks are not covered.
- **Stocks that end just after a window** are not counted as dying inside it, which can understate the rates slightly at the end.
- **Two minor test gaps are parked** (the code's handling of a bootstrap resample with no deaths, and a missing saved-file key, are correct by inspection but have no test). Neither arose in the real runs.

## 5. Files and how to reproduce

- Brief: [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md)
- Results: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json)
- Code: [code/](code/); driver [code/run.py](code/run.py); measures [code/momentum_deaths.py](code/momentum_deaths.py); input reader [code/momentum_inputs.py](code/momentum_inputs.py); engine copies `code/dm_*.py` with hashes in [code/frozen_copies.json](code/frozen_copies.json).
- To reproduce one window, from the repository root with a clean tracked tree: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-momentum-deaths/code/run.py --window dev --release-id 1 --crsp-load 1` (about 4 minutes; no random numbers except a fixed seed for the interval).
