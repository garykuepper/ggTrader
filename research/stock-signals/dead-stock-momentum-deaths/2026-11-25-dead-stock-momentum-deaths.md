# Dead-stock momentum deaths: why the rule held two to four times as many stocks that stopped trading

**Date:** 2026-11-25
**Study:** `research/stock-signals/dead-stock-momentum-deaths` (plan written down and frozen before any code: [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md); one dated amendment at its end)
**Question:** The random-baseline study found that the first test's momentum rule (each month buy the 50 stocks that rose the most over the past year, skipping the last month) held 31, 117 and 24 stocks on the day they stopped trading in the three windows, against about 13, 33 and 8 for 50 stocks drawn at random from the same list. Why?

## 1. In short

- **Strong risers stop trading two and a half to three and a half times as often as the average stock on the list.** Each month, among the stocks the rule could hold, the share that stopped trading within the next month was 4.4, 3.7 and 2.5 per 1,000 stock-months (2000 to 2004, 2005 to 2019, 2020 to 2024). Among the rule's 50 picks it was 11.2, 13.3 and 8.5 per 1,000. The ratio is 2.5, 3.6 and 3.4, with 95 percent intervals of 1.7 to 3.4, 3.0 to 4.2 and 2.1 to 4.7.
- **The rule's count is what that rate predicts.** A random 50 would have held 13.3, 33.1 and 7.3 stocks at death (the random-baseline study's draws: 13, 33, 8). The 50 top-ranked stocks actually died 33, 119 and 25 times, and the first test's ledger counted 31, 117 and 24. This study's counts agree with the ledger's to within 7, 2 and 4 percent, so the audit and the ledger describe the same rule.
- **The rate climbs with the momentum rank.** In 2005 to 2019 and 2020 to 2024 the death rate falls at every step from the top 50 down to the ranks below 500 (13.3, 8.2, 4.1 and 1.5 per 1,000, and 8.5, 5.1, 2.6 and 1.2). In 2000 to 2004 it does so except that ranks 51 to 150 sit below ranks 151 to 500.
- **The dying picks look like stocks that jumped in a day and then went flat.** 64 to 70 percent of the top-50 stocks that later died had a one-day gain of 20 percent or more inside the year their score reads, against 26 to 41 percent of the top-50 stocks that did not die and 6 to 12 percent of all stocks on the list. In the last month before they stopped, 73, 94 and 92 percent of them moved less than 10 percent either way. Their median rise in the three months before the decision was 15, 18 and 37 percent, against 7, 4 and 4 percent for dying stocks outside the top 50. None fell by more than half in the final month; 0 to 6 percent fell by 10 to 50 percent.
- **This is a description, not a cause.** A jump of 20 percent in a day is what a buy-out announcement looks like, and a flat final month is what a stock sitting at a deal price looks like. Earnings and news also make jumps, and these numbers cannot tell them apart. The filings that exist (for the 141 held dead stocks in 2005 to 2024) found an acquisition-type notice for 103 of them, which fits but is a separate piece of evidence. No filings exist for 2000 to 2004.
- **The excess is not a few bad months.** The top-50 death count was above the random-50 expectation in all 5 years of 2000 to 2004, in 14 of 15 years in 2005 to 2019 and in 4 of 5 years in 2020 to 2024 (the exceptions: 2017 and 2021, each with one death).
- **No label.** The owner decides.

## 2. What was done

- **The measures were fixed before any code** (brief section 1): for every month-end decision day of the first test (the same universe A list, release 1, CRSP load 1, the same eligibility and scores), the usable stocks were ranked by momentum score; a stock "dies in the holding month" if its price history ends between the next trading day's execution and the day before the following month's execution, it has no successor series, and it is not a row cut off at the end of the CRSP file. Nothing in the account was simulated.
- **Checks.** The rebuilt top 50 equals the copied planner's targets on every decision day in all three windows (the run stops otherwise). The audit's top-50 death count was compared with the first test's saved count: 33 against 31, 119 against 117, 25 against 24. The "expected random 50" is the analytic count, `sum over months of 50 x deaths / usable stocks`, and agrees with the random-baseline draws' medians.
- **One review, one fix.** Codex Sol reviewed the code before any real run. It found that the one-day-jump measure counted calendar trading days where the score counts a stock's own valid closes, which differ when a stock has holes in its price history. I fixed it, wrote the two failing examples into tests, and recorded the change as a dated amendment in the brief before the real run. The builder was Gemini 4 Neon.
- **Run times:** 145 seconds (2000 to 2004), 160 (2020 to 2024) and 218 (2005 to 2019), all on release 1 and load 1 at commit `873c7967`. Raw numbers: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json).

## 3. Results

**A. Death rates and counts.** A stock-month is one eligible stock on one decision day. Rates are deaths per 1,000 stock-months.

| Window | Stock-months | Deaths, all | Rate, all | Rate, top 50 | Ratio (95% interval) | Random 50 would hold | Top 50 died | Random-baseline draws, middle | First test's ledger |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2000 to 2004 | 56,485 | 251 | 4.44 | 11.19 | 2.52 (1.70 to 3.38) | 13.3 | 33 | 13 | 31 |
| 2005 to 2019 | 174,123 | 643 | 3.69 | 13.30 | 3.60 (3.02 to 4.21) | 33.1 | 119 | 33 | 117 |
| 2020 to 2024 | 57,312 | 142 | 2.48 | 8.47 | 3.42 (2.10 to 4.73) | 7.3 | 25 | 8 | 24 |

The interval comes from resampling whole months (2,000 resamples, all valid).

**A. Death rate by momentum rank band** (per 1,000 stock-months; deaths in brackets):

| Window | Ranks 1 to 50 | 51 to 150 | 151 to 500 | 501 and below |
|---|---|---|---|---|
| 2000 to 2004 | 11.19 (33) | 5.42 (32) | 6.10 (126) | 2.22 (60) |
| 2005 to 2019 | 13.30 (119) | 8.21 (147) | 4.05 (254) | 1.45 (123) |
| 2020 to 2024 | 8.47 (25) | 5.08 (30) | 2.57 (53) | 1.22 (34) |

**A. By year, top-50 deaths against the random-50 expectation:**
- 2000 to 2004: 8 against 4.9, 12 against 3.6, 3 against 1.5, 6 against 1.4, 4 against 1.9.
- 2005 to 2019: 6 against 2.3, 7 / 2.9, 12 / 3.7, 15 / 2.7, 6 / 1.2, 7 / 2.0, 8 / 2.3, 12 / 1.8, 4 / 1.4, 9 / 1.4, 11 / 2.6, 12 / 2.8, 1 / 2.1 (2017), 5 / 2.1, 4 / 1.7.
- 2020 to 2024: 7 against 1.4, 1 / 1.5 (2021), 6 / 1.9, 9 / 1.5, 2 / 1.0.

Years are those in which the holding month starts.

**B. The stocks that died, by group** (the decision day that opens their last holding month):

| Window | Group | Stocks | Median rise in the 63 trading days before the decision | Final month: moved less than 10% either way | Final month: rose 10% or more | Final month: fell 10% or more |
|---|---|---:|---:|---:|---:|---:|
| 2000 to 2004 | In the top 50 | 33 | +14.7% | 73% | 21% | 6% |
| 2000 to 2004 | Not in the top 50 | 218 | +7.1% | 83% | 10% | 7% |
| 2005 to 2019 | In the top 50 | 119 | +17.9% | 94% | 6% | 0% |
| 2005 to 2019 | Not in the top 50 | 524 | +3.9% | 96% | 3% | 1% |
| 2020 to 2024 | In the top 50 | 25 | +36.5% | 92% | 8% | 0% |
| 2020 to 2024 | Not in the top 50 | 117 | +3.9% | 88% | 7% | 3% (plus 3% with no price on the decision day) |

**C. One-day gains of 20 percent or more inside the span the score reads** (counted on each stock's own closes; share of stock-months flagged):

| Window | Top 50, later died | Top 50, did not die | All eligible stocks |
|---|---|---|---|
| 2000 to 2004 | 67% (22 of 33) | 39% (1,125 of 2,917) | 12% (6,959 of 56,485) |
| 2005 to 2019 | 70% (83 of 119) | 26% (2,304 of 8,831) | 6% (9,932 of 174,123) |
| 2020 to 2024 | 64% (16 of 25) | 41% (1,211 of 2,925) | 11% (6,188 of 57,312) |

## 4. What this means for the earlier results, and what it does not say

- **It explains the size of the dead-stock gap, not its sign.** Momentum picks recent winners, and recent winners are the stocks most likely to be taken over or otherwise to stop trading soon. The first test's worst reading (a dead stock pays nothing) therefore hits this rule harder than it hits chance, by construction of the picks. If most of these stocks were acquisitions paid in cash at around their last price, the best reading is the closer one. For 2005 to 2024 the filings say 103 of 141 were acquisition-type, and the payout rerun already used that. For 2000 to 2004 there is no such evidence, and the audit's flat-final-month profile is only a price pattern.
- **A jump flag is not a buy-out.** Top-50 stocks have many jumps in general (26 to 41 percent), because stocks that rose a lot usually did so in part through jumps. The dying picks are higher still (64 to 70 percent), which is the only part that points at the buy-out story.
- **No account is simulated.** Nothing here says what the picks earned, what the deaths paid, or how the rule compares with SPY.
- **Same three windows as before**, so this adds no new market sample. The list is the 1,000 largest stocks priced at $5 or more; other stocks are not covered.
- **Stocks that end just after a window** are not counted as dying inside it, which can understate the rates slightly at the end.
- **Two minor test gaps are parked** (the code's handling of a bootstrap resample with no deaths, and a missing saved-file key, are correct by inspection but have no test). Neither arose in the real runs.

## 5. Files and how to reproduce

- Brief: [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md)
- Results: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json)
- Code: [code/](code/); driver [code/run.py](code/run.py); measures [code/momentum_deaths.py](code/momentum_deaths.py); input reader [code/momentum_inputs.py](code/momentum_inputs.py); engine copies `code/dm_*.py` with hashes in [code/frozen_copies.json](code/frozen_copies.json).
- To reproduce one window, from the repository root with a clean tracked tree: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-momentum-deaths/code/run.py --window dev --release-id 1 --crsp-load 1` (about 4 minutes; no random numbers except a fixed seed for the interval).
