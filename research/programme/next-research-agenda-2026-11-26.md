# Next stock research: choices for discussion

**2026-11-26.** The available choices are an early-history test, slower trading, a rule that avoids suspicious jumps, and a screen of company measures. The first three compete for the same early data. the fourth overlaps years already used. Choose a question before inspecting results. No study is authorized by this agenda.

A later check must use data arriving after the rule is registered. The [window ledger](window-use-ledger.md#what-remains-available) explains the limits.

All studies use free data, a taxable account, information available on each decision date and stocks that later stopped trading. Register the exact rule, date bounds, cost assumptions, tax rates and comparison before scoring. Save raw results and report every tested variant. Compare with SPY, the S&P 500 fund, over the same dates after costs and taxes.

The held SPY history starts in 2000. a compliant same-date SPY comparison for earlier dates is not established. Resolve that conflict before scoring, rather than presenting a substitute as SPY. Apply costs to purchases and sales and taxes to realized gains, including short- and long-term rates. disclose dividend tax and other unmodeled items.

The [first test, sections 2 and 4](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested) describes the existing treatment.

### 1. Untouched decades

**Question:** Can one rule fixed in advance select stocks profitably on early CRSP, a university stock-price source, history including stocks that later stopped trading?

The [first test](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested) already tested monthly purchases of recent winners, so this is a new period, not a new discovery of that rule. The [ledger](window-use-ledger.md#what-remains-available) found no individual-stock rule score in 1968–1999, but earlier portfolio comparisons covered the era, one stock's rows were counted from 1992, and prices from August 1998 helped form early picks. Company filing lists also reached 1999.

The loaded CRSP copy has price, return and shares outstanding, but no trading volume or final removal returns, meaning actual recovery when trading ends is unknown. Use the owner's stated dead-stock reading assumption and show the other recovery scenarios. Register `--eval-start 1968-01-02` and `--eval-end 1999-12-31`, subject to available inputs and the SPY prerequisite above.

The exact rule and needed earlier input history are not established. Costs and taxes follow the common treatment above. If it works, consider a prospective check. if it does not, preserve the result and consider a different question. Effort: medium to large. Codex Terra or Sol builds. Gemini Neon reviews.

### 2. Slow it down

**Question:** Does the same recent-winner rule fare better with yearly rebalancing and one-year holds, so eligible gains receive the long-term tax rate?

The [first test, section 3](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#where-the-gap-comes-from) reports costs reduced yearly growth by about 1.7 to 1.9 percentage points and taxes by a further 1.5 to 3.7 points on its two previously used windows, 2005-01-03–2019-12-31 and 2020-01-02–2024-12-31. Monthly trading has already been tested. the yearly schedule is the proposed change.

CRSP is loaded. The only candidate historical stock-rule window identified by the [ledger](window-use-ledger.md#what-remains-available) is 1968-01-02–1999-12-31, subject to the same input-use exceptions and missing same-date SPY comparison as choice 1. Register `--eval-start 1968-01-02 --eval-end 1999-12-31` before scoring and account for prior input use.

Charge each trade and use the taxable account treatment above, applying the long-term rate only when the actual holding period qualifies. If it works, consider a prospective check. if it does not, retain the finding and consider another question. Effort: medium. Codex Terra or Sol builds. Gemini Neon reviews.

### 3. Death-aware momentum

**Question:** Does excluding stocks with a previously observed large one-day jump improve a recent-winner rule?

The [momentum deaths report, sections 2–3](../stock-signals/dead-stock-momentum-deaths/2026-11-25-dead-stock-momentum-deaths.md#2-what-was-done) describes dying picks with jumps of at least 20 percent in one day and mostly flat final months in the already used 2000-01-03–2004-12-31, 2005-01-03–2019-12-31 and 2020-01-02–2024-12-31 windows. A jump does not establish a takeover. The [ledger](window-use-ledger.md) also records earlier tests of avoiding large jumps. register a distinct rule and acknowledge that overlap.

CRSP is loaded. Test only on an untouched window: the candidate is 1968-01-02–1999-12-31, with the ledger's input-use exceptions and the missing SPY comparison resolved first. Pin `--eval-start 1968-01-02 --eval-end 1999-12-31`. register any threshold and inputs before scoring. Apply the common cost and tax treatment and recovery assumptions.

If it works, consider a prospective check. if it does not, keep the finding and avoid tuning to these results. Effort: medium. Codex Terra or Sol builds. Gemini Neon reviews.

### 4. McGill screen

**Question:** Which company measures, if any, rank next-month stock returns after costs and taxes?

The [ledger](window-use-ledger.md) records prior screens of profits, value, investment, momentum and other related measures. those are not new ideas merely because a different data set is used. The [inventory](../../docs/database-inventory-2026-11-26.md#in-the-database) records `kaggle_chars.mcgill` as loaded, covering large stocks from 2000-01-31 to 2023-12-29. This table has not been read for this screen. availability dates and usable measures are not established.

These dates overlap windows already spent by other rules. Before any score, make and record a window-use ledger decision on whether the precise measures, selection and dates can support an informative exploratory study. do not call the period untouched. Pin both `--eval-start` and `--eval-end` to the approved dates before running. their exact values are not established.

Register and report the number and names of measures tried: the count is not established, and more attempts raise the chance of a lucky result. Check when each measure was actually available, link to loaded prices and include dead stocks where supported. Use the common cost and tax treatment.

If something works, register it for prospective checking. otherwise retain the null findings. Effort: large. Codex Terra or Sol builds. Gemini Neon reviews.

### 5. Forward check

**Question:** Does a rule surviving choices 1–4 continue to work on data arriving after it is registered?

The [ledger](window-use-ledger.md#what-remains-available) records prior stock-rule and paper-sleeve use of the 2025–2026 Yahoo period, and the final 2026-11-05 bar covers only 18 stocks. No clean historical slice after 1999 remains, apart from the early CRSP period. Related historical winner tests are in the [first test](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested).

Register the surviving rule first. score only later-arriving data. Future prices and the final date are not yet loaded or established. Register `--eval-start` after registration and `--eval-end` at the predetermined future close. both exact dates are not established. Check coverage of stocks that stop trading, since survivor-only data cannot show their losses.

Apply the common cost and tax treatment and compare on identical dates with SPY. If it works, consider further evidence and paper trading under owner authorization. if it does not, preserve the result. Effort: medium plus elapsed collection time. Codex Terra or Sol builds. Gemini Neon reviews.

### 6. Costs that match the stock

**Question:** Would stock-specific trading costs change the result for a promising rule?

The [first test, section 3](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#where-the-gap-comes-from) found costs materially affected the monthly rule. it did not measure trading capacity because CRSP lacks volume. The [inventory](../../docs/database-inventory-2026-11-26.md#data-not-available-for-free) lists downloaded Quandl and Stooq files, with missing earlier dead stocks or survivor limitations and unverified Quandl licence and origin. They are not loaded for this purpose.

Load only if choices 1–4 show something and coverage, licence and backup conditions are resolved. the live database volume is treated as unbacked-up.

The dates depend on the chosen rule and a ledger decision. no clean additional historical scoring window is established. Register `--eval-start` and `--eval-end` before any approved run. exact dates are not established. Estimate costs using only then-available volume, retain comparable tax treatment and disclose gaps. If results survive, use the more realistic estimates in future evaluation. otherwise report the sensitivity.

Effort: large. Codex Terra or Sol builds. Gemini Neon reviews.

### 7. Merger documents

**Question:** Can merger documents establish what holders received when early-window stocks stopped trading?

The [early filings report, sections 1–2](../stock-signals/dead-stock-early-filings/2026-11-25-dead-stock-early-filings.md#2-what-was-done) found an acquisition-type filing for one of 33 held deaths in 2000-01-03–2004-12-31. a re-score using current labels would change almost nothing, and a label is not proof of payment. This search is available but not recommended ahead of the other choices.

The existing stock list and filing search are available. further free merger documents and actual payments are not established. The [ledger](window-use-ledger.md) marks the 2000-01-03–2004-12-31 rule window spent. Pin `--eval-start 2000-01-03 --eval-end 2004-12-31` if a retrospective re-score is separately approved. it would not be fresh evidence. Use the original cost and taxable-account treatment for any re-score.

If documents establish payments, improve the historical interpretation. if not, retain the uncertainty. Effort: large and uncertain. Codex Terra or Sol builds. Gemini Neon reviews.

## Parked, not recommended or ended

Earnings-surprise drift, the tendency for prices to keep moving after unexpected earnings, remains parked. Insider-trade studies, crypto, the intraday observer, and paper parity (matching paper decisions to research) also remain parked. AVB is a paper holding the broker marks untradable. The [ledger](window-use-ledger.md) and [verdict roster](../RESEARCH_SNAPSHOT.md) record related work and its limits. Parking makes no decision to end any of them.

## How the choice works

Choose the question whose uncertainty matters most. Choices 1–3 compete for the same early history, so registering and scoring one can make that window less informative for later choices. Choice 4 needs a ledger decision first. choice 5 needs time for new data. Nothing here changes the production freeze or authorizes a trade.
