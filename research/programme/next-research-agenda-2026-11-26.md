# Next stock research: choices for discussion

**2026-11-26.** The available choices are an early-history test, slower trading, a rule that avoids suspicious jumps, and a screen of company measures. The first three compete for the same early data. The fourth overlaps years already used. Choose a question before inspecting results. No study is authorized by this agenda.

A later check must use data arriving after the rule is registered. The [window ledger](window-use-ledger.md#what-remains-available) explains the limits.

All studies use free data, a taxable account, information available on each decision date and stocks that later stopped trading. Register the exact rule, date bounds, cost assumptions, tax rates and comparison before scoring. Save raw results and report every tested variant. Compare with SPY, the S&P 500 fund, over the same dates after costs and taxes.

SPY began in January 1993, and the held SPY history starts on 2000-01-03. CRSP, a university stock-price source, supplies a return for the whole US market, not SPY, for the earlier years. Its `vwretd` series weights companies by market value. A read-only database check on 2026-11-26 found no missing values on the loaded rows for 1968–1999. The query is recorded below.

For years through July 1998, may the CRSP whole-market return stand in for SPY? SPY began in 1993, so another option is to add free Yahoo SPY history for 1993–1999. The owner must decide before a run.

A market return series does not by itself establish the costs and taxes of an investable fund. Apply costs to purchases and sales and taxes to gains made when a holding is sold, including short- and long-term rates. Disclose dividend tax and other unmodeled items.

The [first test, sections 2 and 6](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested) describes the existing treatment.

## Candidates

### 1. Early decades

**Question:** Can one rule fixed in advance select stocks profitably on early history from CRSP, a university stock-price source that includes stocks that later stopped trading?

The [first test](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested) already tested monthly purchases of recent winners, so this is a new period, not a new discovery of that rule. The [ledger](window-use-ledger.md#what-remains-available) found no individual-stock rule score in 1968–1999, but earlier portfolio comparisons covered the era, one stock's rows were counted from 1992, and prices from August 1998 helped form early picks. Company filing lists also reached 1999.

A [published-portfolio screen](../stock-signals/momentum/2026-10-20-momentum-french-screen-nogo.md#subperiod-diagnostics-at-base-cost) already found large-stock recent winners gained 3.12 percentage points a year more than the market after assumed costs in 1927-01–1993-12. That is a portfolio result, not an individual-stock account test.

The loaded CRSP copy has price, return and shares outstanding, but no trading volume or final removal returns, meaning actual recovery when trading ends is unknown.

On 1973-01-02, 2,968 of 5,404 price rows are quote midpoints, the halfway point between quoted buy and sell prices, not recorded trades. On 1969-02-03, only 1,203 common stocks on the New York Stock Exchange (NYSE) meet the first test’s $5 price and positive-share requirements for a 1,000-largest list. The read-only queries are recorded below.

Use the owner's stated assumption about what holders recover when a stock stops trading, and show the other recovery scenarios.

Register `--eval-start 1969-02-03` and `--eval-end 1998-07-31`, subject to the chosen rule's input needs and the comparison decision above. The later end avoids prices already read from August 1998. The later start allows about a year of earlier prices.

The exact rule and needed earlier input history are not established. Costs and taxes follow the common treatment above. If it beats the chosen comparison after costs and taxes, prepare a forward check on later data. If it trails, keep the result and choose a different stock-selection question instead of tuning to this window. Effort: medium to large. Codex Terra or Sol builds. Gemini Neon reviews.

### 2. Slow it down

**Question:** Does the same recent-winner rule fare better if holdings are adjusted once a year and each stock is held for one year, so eligible gains receive the long-term tax rate?

The [first test, section 3](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#where-the-gap-comes-from) reports that costs and taxes together reduced annual growth by about 3.2 percentage points in 2005-01-03–2019-12-31 and 5.6 points in 2020-01-02–2024-12-31.

These figures apply to the 1,000 largest eligible stocks, assuming holders recover the last price when trading stops, with a 0.183 percent cost on each purchase and sale, based on one large trader’s 2009–2016 costs, and lower tax rates of 24 percent for short-term and 15 percent for long-term gains.

In those windows costs reduced growth by 1.71 and 1.89 points, and taxes by 1.46 and 3.72 points. In the earlier window, tax was zero under the separate assumption that holders recovered nothing. Monthly trading has already been tested. The yearly schedule is the proposed change.

CRSP is loaded. The only candidate historical stock-rule window identified by the [ledger](window-use-ledger.md#what-remains-available) is the earlier CRSP period, subject to the same input-use exceptions and comparison decision as choice 1. Register `--eval-start 1969-02-03 --eval-end 1998-07-31` before scoring. Prices before the start provide roughly one year of history and the end avoids the previously read prices from August 1998.

Charge each trade and use the taxable account treatment above, applying the long-term rate only when the actual holding period qualifies. If annual holding beats the chosen comparison after tax, prepare a later-data check of the annual rule. If it still trails, stop testing recent-winner variants on this spent window and choose another question. Effort: medium. Codex Terra or Sol builds. Gemini Neon reviews.

### 3. Death-aware recent-winner rule

**Question:** Does excluding stocks with a previously observed large one-day jump improve a recent-winner rule?

The [recent-winner deaths report, sections 2–3](../stock-signals/dead-stock-momentum-deaths/2026-11-25-dead-stock-momentum-deaths.md#2-what-was-done) describes dying picks with jumps of at least 20 percent in one day and mostly flat final months in the already used 2000-01-03–2004-12-31, 2005-01-03–2019-12-31 and 2020-01-02–2024-12-31 windows. A jump does not establish a takeover. The [ledger](window-use-ledger.md) also records earlier tests of avoiding large jumps. Register a distinct rule and acknowledge that overlap.

CRSP is loaded. Test only on the early window, subject to the ledger's disclosed exceptions and the comparison decision above. Pin `--eval-start 1969-02-03 --eval-end 1998-07-31`. Earlier prices provide roughly one year of history and the end avoids the previously read prices from August 1998. Register any threshold and inputs before scoring. Apply the common cost and tax treatment and recovery assumptions.

If the exclusion improves the registered comparison after costs and taxes, prepare a later-data check of the fixed rule. If it does not, leave the exclusion out of the next rule and do not tune it to this window. Effort: medium. Codex Terra or Sol builds. Gemini Neon reviews.

### 4. Company-measure screen

**Question:** Which company measures, if any, rank next-month stock returns after costs and taxes?

The [ledger](window-use-ledger.md) records prior screens of profits, value, investment, recent-winner rules and other related measures. Those are not new ideas merely because a different data set is used. The [inventory](../../docs/database-inventory-2026-11-26.md#in-the-database) records `kaggle_chars.mcgill` as loaded, covering large stocks from 2000-01-31 to 2023-12-29. This table has not been read for this screen. Availability dates and usable measures are not established.

The downloaded Gu–Kelly–Xiu file is another source: its exploration audit describes 94 lagged characteristics per stock-month, 4,117,300 rows from 1957–2021, and marks it usable. Unlike McGill it reaches the earlier period, but it is not loaded. A stage B import needs its backup gate and owner approval. See the [inventory](../../docs/database-inventory-2026-11-26.md#downloaded-but-not-loaded).

McGill’s 2000–2023 span and the 2000–2021 part of Gu–Kelly–Xiu overlap windows already spent by other rules. Before any score, make and record a window-use ledger decision on whether the precise measures, selection and dates can support an informative exploratory study. Do not call the period untouched. Pin both `--eval-start` and `--eval-end` to the approved dates before running. Their exact values are not established.

Register and report the number and names of measures tried: the count is not established, and more attempts raise the chance of a lucky result. Check when each measure was actually available, link to loaded prices and include dead stocks where supported. Use the common cost and tax treatment.

If a measure survives the registered comparison, fix that measure and prepare a later-data check. If none does, keep the findings and do not promote a measure for forward testing. Effort: large. Codex Terra or Sol builds. Gemini Neon reviews.

### 5. Forward check

**Question:** Does a rule surviving choices 1–4 continue to work on data arriving after it is registered?

The [ledger](window-use-ledger.md#what-remains-available) records prior stock-rule and paper-sleeve use of the 2025–2026 Yahoo period, and the final 2026-11-05 bar covers only 18 stocks. No clean historical slice remains after 1999. The only unused historical slice is the earlier CRSP period, 1968–1999, subject to the ledger’s noted exceptions. Related historical winner tests are in the [first test](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#2-what-was-tested).

Waiting alone will not collect the needed prices. The held Yahoo rows stop on 2026-11-05, no listed timer refreshes them, and normal coverage is about 623 symbols, fewer than the first rule’s 1,000 largest eligible NYSE and Nasdaq stocks. First start and freeze a new free daily collection covering the full eligible list before registering the rule, including stocks that later stop trading.

Register the surviving rule only after collection is in place. Score only later-arriving data. Future prices and the final date are not yet loaded or established. Register `--eval-start` after registration and `--eval-end` at the predetermined future close. Both exact dates are not established. Check coverage of stocks that stop trading, since survivor-only data cannot show their losses.

Apply the common cost and tax treatment and compare on identical dates with SPY. If it beats the registered comparison after costs and taxes, bring it to the owner for a paper-trading decision. If it does not, retain the result and do not advance that rule to paper trading. Effort: medium plus elapsed collection time. Codex Terra or Sol builds. Gemini Neon reviews.

### 6. Costs that match the stock

**Question:** Would stock-specific trading costs change the result for a promising rule?

The [first test, section 3](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#where-the-gap-comes-from) found costs materially affected the monthly rule. It did not measure how many shares could actually be traded because CRSP lacks volume. The [first test, sections 2 and 6](../stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md#6-what-this-cannot-tell-you) explains that limit. The [inventory](../../docs/database-inventory-2026-11-26.md#downloaded-but-not-loaded) lists downloaded Quandl and Stooq files, with missing earlier dead stocks or survivor limitations and unverified Quandl licence and origin.

They are not loaded for this purpose.

Load only if choices 1–4 show something and coverage, licence and backup conditions are resolved. The live database volume is treated as unbacked-up.

The dates depend on the chosen rule and a ledger decision. No clean additional historical scoring window is established. Register `--eval-start` and `--eval-end` before any approved run. Exact dates are not established. Estimate costs using only then-available volume, retain comparable tax treatment and disclose gaps.

If the result survives, use the stock-specific estimates in future evaluation. If it does not, do not advance the rule on the earlier cost estimate. Report the change.

Effort: large. Codex Terra or Sol builds. Gemini Neon reviews.

### 7. Merger documents

**Question:** Can merger documents establish what holders received when early-window stocks stopped trading?

The [early filings report, sections 1 and 3](../stock-signals/dead-stock-early-filings/2026-11-25-dead-stock-early-filings.md#3-results) found an acquisition-type filing for one of 33 held deaths in 2000-01-03–2004-12-31. Repeating the calculation using current labels would change almost nothing, and a label is not proof of payment. This search is available but not recommended ahead of the other choices.

The existing stock list and filing search are available. Further free merger documents and actual payments are not established. The [ledger](window-use-ledger.md) marks the 2000-01-03–2004-12-31 rule window spent. Pin `--eval-start 2000-01-03 --eval-end 2004-12-31` if a repeat calculation on the already used dates is separately approved. It would not be fresh evidence. Use the original cost and taxable-account treatment for any repeat calculation.

If documents establish payments, update the historical interpretation using those payments. If not, retain the range of recovery assumptions and avoid a payment-based rerun. Effort: large and uncertain. Codex Terra or Sol builds. Gemini Neon reviews.

## Parked, not recommended or ended

Earnings-surprise drift, the tendency for prices to keep moving after unexpected earnings, remains parked. Insider-trade studies, crypto, the proposed intraday observer, which would watch price movements within a trading day, also remain parked. Paper parity, matching paper decisions to research, and the related AVB paper holding that the broker marks untradable also remain parked.

The [ledger](window-use-ledger.md) and [verdict roster](../RESEARCH_SNAPSHOT.md) record related work and its limits. Parking makes no decision to end any of them.

## Market data check

The following read-only checks count CRSP prices recorded as negative, which mark quote midpoints, and the first test's eligible common stocks on the New York Stock Exchange (NYSE):

```sql
SELECT date, count(*) AS rows, count(*) FILTER (WHERE prc < 0) AS midpoint
FROM kaggle_crsp.daily
WHERE load_id = 1 AND date = '1973-01-02' GROUP BY date;
SELECT date, count(*) AS qualifying_nyse
FROM kaggle_crsp.daily
WHERE load_id = 1 AND date = '1969-02-03' AND exchcd = 1
  AND shrcd IN (10, 11) AND abs(prc) >= 5 AND shrout > 0
GROUP BY date;
```

The counts are 5,404 rows, 2,968 midpoints and 1,203 qualifying stocks, respectively.

The market-return check used this read-only database query:

```sql
SELECT min(date), max(date), count(*),
       count(*) FILTER (WHERE vwretd IS NULL),
       count(DISTINCT date), count(DISTINCT date) FILTER (WHERE vwretd IS NOT NULL)
FROM kaggle_crsp.daily
WHERE load_id = 1 AND date BETWEEN '1968-01-02' AND '1999-12-31';
```

The result was 41,725,030 rows, no missing values and 8,058 distinct dates with a value.

## How the choice works

Choose the question whose uncertainty matters most. Choices 1–3 compete for the same early history. One option is to register all three rules as separate arms of one run before scoring any of them. Report every arm and that all use the same window. Scoring one first can make the window less informative for later choices.

Choice 4 needs a window-use ledger decision, a recorded judgment about whether those dates can answer the specific question, first. Choice 5 needs time for new data.

Choice 6 depends on a promising earlier result and usable volume. Choice 7 is available for historical clarification but is not recommended ahead of the others.

For years through July 1998, may the CRSP whole-market return stand in for SPY? Alternatively, free Yahoo SPY history could be added for 1993–1999, the years after SPY began. Nothing here changes the production freeze or authorizes a trade.
