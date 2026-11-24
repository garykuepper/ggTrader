# All-stocks honest test bench: design (draft for owner approval)

Date: 2026-11-21. Status: draft. Nothing is built. Written in plain words for the owner; terms are defined where they first appear.

## Intent and why

The owner wants a daily stock-selection system over every US-listed common stock on the NYSE and Nasdaq. Success means making money, not losing it, and beating SPY (the S&P 500 fund) after trading costs and taxes in a taxable account. "No chill": the test starts with all NYSE and Nasdaq stocks, not a smaller set.

The library survey (`library-reading/CONSENSUS_REPORT.md`, revised 2026-11-21) found that most backtests mislead, mainly through two flaws. **Survivorship bias** means the test includes only stocks that still exist, so it leaves out the ones that failed. **Look-ahead bias** means the test uses facts that were not yet public on the test day. The survey also found no tested free source of dead-stock prices or delisting returns (what a holder received when a stock was delisted). This project builds the data foundation and the bias gauge so that later strategy tests can be believed. It scores no strategy.

## Goals

1. **Universe by date.** For every trading day from 2002 to today, the list of common stocks that were listed on the NYSE or Nasdaq, built from SEC filings (delisting notices, Form 25 and Form 15, and company filing lists) plus today's exchange symbol files. Each company is identified by its SEC company number (CIK), not by its ticker, because tickers get reused.
2. **Prices.** Daily prices for as many of those stocks as free sources give, stored as raw (not later-adjusted) prices together with split and dividend events, so the test can adjust them using only what was known on each date.
3. **Coverage report.** For each year, the share of the universe that has prices, split by living and dead stocks, by size, and by source. This is the bias gauge: it says how much of the market the test cannot see.
4. **First-reported company data.** Financial statements from the SEC's free data sets with the date each was filed, available from 2009. A test uses a figure only after its filing date, plus a delay.
5. **Costs and taxes as parameters.** Trading cost by stock size, from the measured studies now in the library, and the US tax rules by year, from the dated timeline (Appendix H of the report). A result is always shown at stated cost levels and under the tax rules of its own years.
6. **Forward tape.** From the day it starts, a daily snapshot of every listed stock's prices and listing status, kept on `/media/thesix`. After a year or more this is a fully honest record with no gaps, whatever the free history lacks.
7. **Benchmarks.** SPY and the Fama-French daily factor returns (built from a data set that includes dead firms).

## Non-goals

- No strategy is scored, tuned or recommended here.
- No paid data and no subscription.
- No change to the live or paper trader, which stays stopped.
- No machine learning work.

## Anti-goals (a result that would count as failure even if the code runs)

- A test that silently drops dead stocks, or reports a result on survivors only.
- A result presented without the coverage table for its years.
- A delisting outcome filled in with a guess that is not labeled as a guess.

## Constraints

- Free data only. Free sources impose terms: Yahoo allows personal use only and bars building a database from automated collection; Tiingo's free licence is internal use only and caps 500 symbols a month; the SEC asks for a contact in its user agent. The bench is for the owner's private research and is not redistributed. The owner decides whether the Yahoo and Tiingo terms are acceptable before they are used at scale (the project already reads Yahoo prices for live stocks).
- The four plain rules in `CLAUDE.md` hold: test on data the strategy never saw during tuning, use only what was known on each day, subtract costs and taxes and compare to SPY, paper trade first.
- Data lives on `/media/thesix` and in the database, with every refresh kept as a dated version; the repository keeps code and small dated fixtures only.

## What free data can and cannot give (checked 2026-11-21; `library-reading/gap-research/D_free_data_feasibility.md`)

- The SEC is the only free source that is complete and dated: filing lists from 1994, delisting notices (Form 25) from late 2002, and company financial data with filing dates from 2009. It holds no prices.
- No free price source tested keeps most dead companies from before about 2009. Yahoo returned "not found" for 17 of 17 dead names tried.
- The database already holds daily bars for 2,331 equity symbols from 2000 to 2026-11-05: 629 ending after October 2026 and 1,702 ending earlier. Some of those ended because the stock died; some may be data gaps. Which is which is not yet known.
- Listed US domestic companies, World Bank counts: 6,917 (2000), 4,279 (2010), 3,908 (2025). The count is not split by exchange or by common stock.
- Honest consequence: a free test is honest from about 2009 forward if the dead-name gaps are measured against the SEC list. For 2002 to 2008 the free path gives events (who left, and when) but not prices for most dead names.

## Approaches considered

| | What it is | Gain | Cost |
|---|---|---|---|
| A. 2009 and later only | Build the history from 2009 on; ignore earlier years | Quick; mostly honest | Drops the 2000 to 2002 crash and the 2008 crisis, the periods that punish a strategy most |
| B. Forward tape only | Start recording every stock daily from today | Fully honest | No history for a year or more; no test for a long time |
| C. Hybrid (recommended, owner's "no chill" read as widest scope) | Build the universe and coverage gauge from 2002; fill prices where free sources have them; measure the gap; report 2009 onward as the primary test window and 2002 to 2008 as a partial window with its coverage stated; start the forward tape at once | Widest honest scope, and the gap is measured, not hidden | More work than A; 2002 to 2008 results come with a stated gap |

The owner picked the widest scope (all NYSE and Nasdaq from the start). The author's call, to be confirmed by the owner: use C, with 2009 onward as the primary window.

## Stages (the build order for a plan)

1. **Universe and identity.** Load SEC company lists and filing indexes, Form 25 and Form 15 notices, and today's exchange files into a dated, append-only table keyed by CIK, with ticker history; flag reused tickers. Output: stocks listed per day, 2002 to today.
2. **Price coverage census.** Check which universe members the existing database already prices, then which free sources fill the gaps, in an order set by the terms and limits above. Output: the coverage table by year, living versus dead, and size.
3. **Dead-stock outcomes.** For each delisting, record the date and the kind of exit when the filing says it (merger, bankruptcy, moved to over-the-counter). Where no price exists, use a labeled assumption and always show a sensitivity range (for example 0%, -30% and -55% on the last price) beside the result.
4. **First-reported company data** from the SEC data sets (2009 on), joined by CIK and filing date.
5. **Forward tape** started on day one of the build, writing a dated snapshot each trading day.
6. **Bias gauge.** A fixed check that any later test must pass: coverage for its test years, with the share of delisted stocks missing and a plain-language bound on how much that can flatter the result.

## Success and failure

- Success: a coverage table that is measured against the SEC list for every year from 2002, a universe-by-date table, a first run of the bias gauge on the existing database, and a forward tape that has recorded without a missed day.
- Failure: the coverage for 2009 onward turns out too thin to say anything (the gauge shows the missing share of dead stocks is large), in which case the report says so and the forward tape becomes the main path. This outcome would still be a successful, honest finding.

## Calls made by the author after a builder read of this draft (the owner can change any of them)

- **Stage 0, validate the existing tape first.** Before the census, check that the 2,331 stored symbols carry raw or consistently adjusted prices, with no day shift, duplicate rows or ticker mix-ups (the project has hit all three before). If the tape is unreliable it is re-pulled, which counts against the source limits. The census and the first gauge run use the validated tape only.
- **Stage 7, costs, taxes and benchmarks.** Cost levels by stock size come from the measured library studies (large-company spreads of a few basis points, small-company spreads several times larger, all with their period caveats); tax rules by year come from the dated timeline; SPY and the published Fama-French daily files are downloaded, not rebuilt. These are parameters a later test reads, so this stage is small.
- **Starting list and independent check.** The 2002 starting list is built backward from today's exchange files plus Form 25 and Form 15 notices and company filing lists, and cannot see stocks that left before 2002. The project's own universe is checked three outside ways: the World Bank yearly counts, Form 25 counts per quarter, and a hand-checked sample of named dead companies (for example the 17 tried in the feasibility test). Disagreements are reported, not smoothed.
- **Unit of the universe.** Keyed by security within a company (CIK plus share class). In scope: domestic common stock, including REITs. Out of scope at the start, and labeled so in the tables: ETFs and funds, preferreds, warrants, units, SPACs before their merger, and foreign companies' ADRs.
- **Bias gauge output.** It sets no pass mark. It prints the missing share of delisted stocks by year and size bucket, and shows two bracket results for any test: missing stocks excluded, and missing stocks given the sensitivity exits above. The owner decides what is acceptable. Stocks with no price history have no size; they are counted in their own row.
- **Forward tape source.** The universe snapshot comes from the SEC and the exchange symbol files. Daily closes come from the first source whose terms the owner accepts (answer to open question 1); the fallback is the paper brokerage account's own market data, with its coverage limits written down. If no source covers all listed stocks daily, the tape records what it can and says so.

## Left for the builder

Table layout, source order, rate-limit handling, how to detect reused tickers, how the existing 1,702 early-ending symbols are classified, and the schedule for the forward tape. The plan should pin the evidence dates and keep every raw download with a hash.

## Open questions for the owner

1. Are the Yahoo and Tiingo free-use terms acceptable for private research at scale?
2. Is "2009 onward is the primary window, 2002 to 2008 is partial" acceptable?
