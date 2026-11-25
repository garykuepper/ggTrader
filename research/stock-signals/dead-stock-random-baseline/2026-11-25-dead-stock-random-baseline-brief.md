# Brief: Dead Stock Random Baseline

**Status:** FROZEN 2026-11-25, approved by the owner ("Ok execute") with the three calls below at their defaults: both arms, all three windows, 100 draws. Frozen before any code of this study exists; later changes are dated amendments at the bottom, never silent edits.
**Study:** stock-signals/dead-stock-random-baseline (kind: screen), written 2026-11-25.
**Decision fed:** which kind of rule to test next. The owner makes that decision. There is no pass bar and no GO/NO-GO label. The study says how the first test's momentum picks compare with picking the same number of stocks by chance, under the same costs and taxes.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these values or say which line it departs from.

- eval_start: 2000-01-03
- eval_end: 2024-12-31
- max_leverage: 1.0
- benchmark: SPY buy-and-hold through the same ledger, same dates, same costs and same tax rates
- costs: 18.3 bps per side (a basis point, bps, is 0.01 percent) of traded value on every buy and sell, dead-stock exits included. This is the headline cost of the first test. Its 6.4 and 40 bps figures are not rerun here.
- family_size: 2
- seed: 20261125 (every random draw takes a seed derived from it by a fixed rule written in section 3; nothing else in this study is random)

`family_size` is 2: the two random-pick arms of section 1. The three windows, three loss readings and draw counts are not tuned and are all reported.

**The three windows.** These are the first test's and the fresh test's windows, all already seen by the momentum rule:
- 2000-01-03 to 2004-12-31
- 2005-01-03 to 2019-12-31 (the first test's development window)
- 2020-01-02 to 2024-12-31 (the first test's holdout)

No window here is "unseen", and none needs to be: nothing in this study is tuned, and the random portfolios are not a rule under test. They are a yardstick.

## 1. Question and fixed arms

**Question, in plain words.** The first test's rule trailed SPY in all three windows. Is that because of how it picks stocks, or would any 50 stocks drawn by chance from the same list, bought and sold with the same costs and taxes, trail SPY by about as much? If chance does about as well as the rule, the rule's picks add nothing and the gap comes from the list, the equal weighting and the trading. If chance does clearly better, the picks hurt. If the rule sits near the top of the chance results, the picks help but not enough.

**The arms.** Both use universe A and the first test's ledger, with dead-stock loss 1.0, 0.5 and 0.0, cost 18.3 bps and the low tax set (24 percent short-term, 15 percent long-term). The high tax set is not run: in 2000 to 2004 both sets gave the same result because no tax was paid, and the first test's high set only changed the size of the tax bill.

- **Arm R (fresh dice).** Each month-end, hold 50 stocks chosen at random from the stocks that pass the same universe tests, equal weight, bought at the next trading day's close. This is the rule with only the ordering replaced by a random one.
- **Arm M (matched turnover).** Same, but each month the portfolio replaces the same number of names the first test's rule replaced that month, taken from its saved trades. Replaced names are drawn at random from the eligible stocks; the rest are kept. This answers the question "is it just that the rule trades a lot?".

**Draws.** 100 independent draws per arm per window. If a timing test (a throwaway draw with a seed not used afterwards, results discarded) shows the whole run will not fit in about 8 hours, the number is lowered for every arm and window equally, by a dated amendment before the real run.

**Not in this study:** universe B, other costs, any tuning, any new rule, SEC filing evidence about dying stocks, the 1995 to 1999 years.

## 2. Data and freeze before calculation

- **Prices, universe, dead-stock exit, SPY:** exactly the first test's: published combined release 1 and raw CRSP load 1, read only; every output records the release and the price hash.
- **The rule's numbers** come from the first test's saved files (`_dev-A.json`, `_holdout-A.json`) and the fresh test's `_fresh-A.json`, read only. The study never reruns the momentum rule.
- **Code.** The first test's engine files copied with a prefix and checked against the originals by hash, as the fresh test did. New code is a random-pick planner plus a summary step.
- **Check first.** Before any random draw, the copied engine must reproduce the rule's saved headline-cell numbers on the 2020 to 2024 window to the cent, with the rule's own momentum planner. Random draws are refused until that passes.

## 3. Registered model

The first test's model with one change: the order in which eligible stocks are taken.

- **Eligible stocks** on each decision day are those the first test's planner would consider: universe A tests passed, a finite score, not already dead on the decision day. The momentum score is still computed only so the eligible set is identical to the rule's; it does not rank anything here.
- **Arm R:** draw 50 at random without replacement from the eligible set (all of them if fewer than 50), equal weight.
- **Arm M:** let k be the number of names the rule's saved plan replaced at this decision day (new names bought). Keep 50 minus k of the current names, chosen at random, drop the others, add k random eligible names. If the portfolio holds fewer than 50 names (start, or after deaths) it is filled the same way. The exact rule for the first month and for forced exits is written in the code docstring before any draw and pinned by tests.
- **Seeds.** Draw d of arm a in window w uses the seed `20261125 + 1000000*window_index + 10000*arm_index + d`, with window_index 0, 1, 2 in the order above and arm_index 0 for R and 1 for M.
- **Decision days, trade timing, costs, taxes, dead-stock handling and SPY:** exactly the first test's.

## 4. Portfolio and accounting

The first test's section 4 and its dated amendments, unchanged: costs per side on every buy and sell; taxes assessed each year and paid in January from the book's own cash; a dead-stock exit realises its loss; both books fully sold on the last day; SPY through the same tax-lot ledger; no leverage.

## 5. Frozen output plan

Written to a results file with the data release and price hash:

- Per window, per arm, per loss reading: every draw's after-tax final value, yearly growth rate (CAGR), Sharpe ratio (return per unit of daily ups and downs; higher is better), largest peak-to-trough fall and yearly turnover; the spread of draws (lowest, 5th, 25th, middle, 75th, 95th, highest percentile).
- The gap to SPY for each draw, in points a year.
- **The rule's place among the draws:** for each window, arm and loss reading, the share of draws that ended behind the rule and the share ahead, with the rule's saved numbers.
- Cost and turnover of the draws next to the rule's.
- A check that the random portfolios behave as chance should: with dead stocks paid in full, before costs, their average should be close to the average of an equal-weighted hold of the same eligible list. This needs a short extra calculation, described in the report. If it fails, the report says the test is suspect.

The report states the result in words and gives no label.

## 6. Decision rule

None, as in the first test: the owner's four plain rules apply (test only on data the rule never saw; use only what was known on each day and include stocks that later died; subtract costs and taxes and compare to SPY over the same dates; paper trade first, the owner decides). If the answer changes a lot between loss levels or windows, that is reported as a finding.

## 7. What this cannot tell you, and the owner's checks

- **A yardstick, not a rule.** Chance results show what the list and the trading do on their own. They do not show what any other rule would do.
- **Not independent of the first test.** The windows and the list are the same, so this does not add a new market sample.
- **100 draws give the rule's position to about one percentage point of resolution.**
- **Same open questions as before:** what dying stocks paid is unknown (the three readings bracket it), costs before 2001 were probably higher than the grid, dividend tax is not modelled, a stock in trouble can trade at a worse price than the cost grid, and the matched-turnover arm copies the rule's turnover but not its picks' size or style.
- **Equal weighting against SPY's size weighting** is part of the gap and cannot be separated here from "the list".

**Execution boundary.** New study code goes in this study's `code/` folder; nothing in `src/` is touched, vectorbt is not imported, the paper trader stays stopped and the database is read only. The run is one heavy job at a time through the capped-run wrapper.

## Calls for the owner to check (answered 2026-11-25: both arms, all three windows, 100 draws)

1. Both arms (R and M).
2. All three windows.
3. 100 draws, with the cut-down rule if timing needs it.

## Amendments

**2026-11-25 (after the code was written and independently reviewed; before any real draw; no draw result seen).** Four points the review found the brief loose or silent on, all fixed before anything real runs. No registered parameter, window, arm, seed rule, cost, tax set or draw count changed.

1. **Arm M counts what the rule actually bought.** Section 3 said "taken from its saved trades". It is now: the number of names the rule bought for the first time that month, counted from the fills of the rule's reference cell (loss 1.0, 18.3 bps, low tax) that the same run has just reproduced from the first test's saved file to half a cent. A target the ledger could not fill is not a purchase and counts in the month it is finally bought; a buy that only resizes a held position does not count. The first test's trade CSV is not tracked by git, so it is not read; the reproduced fills are the same trades.
2. **The check comes first, on the window being run.** Section 2 said the 2020 to 2024 window. Each run now rebuilds the rule's three headline cells and SPY on its own window and compares them with that window's saved file (the first test's `_dev-A.json` and `_holdout-A.json`, the fresh test's `_fresh-A.json`) before any draw; no draw runs if one cell differs by more than half a cent. This covers the holdout reproduction and the other two windows. The holdout window is run first.
3. **The sanity check is fixed.** Section 5 said "close to". It is now: re-run the first 20 arm R draws of the window with dead stocks paid in full (loss 0.0), no cost and no tax; compare their mean yearly growth with an equal-weight hold of every eligible stock under the same settings. Fine when the difference is within two standard errors of the dice mean or within 0.5 percentage points a year, whichever is larger. If not, the report says the test is suspect. Draw counts for the sanity check are 20 per window.
4. **Timing.** A throwaway timing draw on the 2020 to 2024 window (seed index 9999, result discarded and kept apart from the real files) took 3 to 4 seconds a draw per arm, so 100 draws of both arms is about 15 minutes per window, plus loading. The 100 draws stand; the cut-down rule of section 1 is not used.

**2026-11-25, second note (after the second independent review; still before any real draw).** Three details of arm M and of resuming, fixed and pinned by tests; nothing registered changed.

5. **Arm M's fill rule when too few new names exist.** Each month it keeps `need - k` of the names it held that are still eligible, then buys `k` names not held last month. If fewer than that many such names exist, it buys all of them and takes the rest from names dropped this month.
6. **What counts as "held" when counting the rule's purchases.** A name counts as held while its position is above one billionth of its own largest position (so a tiny position being topped up is not a new name); the first version used a fixed share count, which the ledger does not.
7. **A resumed run must match the commit and the dependency lock too**, not only the study's own files, the window, the data and the saved file. A resume after any commit is refused; the window is simply rerun from the start.

