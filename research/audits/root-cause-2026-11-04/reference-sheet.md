# Root-cause reports: reference sheet

**Date:** 2026-11-05. **Purpose:** look up any term, check name or label used in the root-cause reports and read a plain-English meaning. No finance background assumed. Short source codes such as `synth:15-16` point to a file and line, and the Source key at the end maps each code to its file.

## 1. How to read these reports

- The reading reports name the checks and rounds. They never number them. This sheet maps each name to its file.
- The original reports keep their numbers because they are the record. Their file names carry the numbers, so the numbers for checks and rounds appear only in the File column of sections 4 and 5 and in code spans in section 6.
- A dash in a Source column means the term is common usage and no report defines it. Every other row was checked against the cited line.
- The forward-paper standard is approved by the owner: its status line says so (`standard:3`), and so does the note opening its conflicts section (`standard:224`). An older "draft" sentence survives in its section 1 (`standard:40`). Approval adopts the standard only: each idea still needs its own approved brief before any run (`standard:288`).

I have read all of it so you do not have to read all of it. Start with the Abbreviations table when a three-letter word stops you.

## 2. Abbreviations

Each plain meaning is at most 30 words and uses only terms that have their own row.

### Statistics and testing

Rows are alphabetical within each group. DSR builds on the Sharpe ratio, which is listed under S.

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| CI | Confidence interval: the range of true values the data cannot rule out. The reports use 95%. | Momentum's best 2021-2026 estimate, +4.37 pp/yr, has a CI of -7 to +16, so the data cannot tell it from zero. | `synth:70-72`, `synth:164` |
| DSR | Deflated Sharpe ratio: the chance a Sharpe ratio is real once you allow for how many settings were tried, since more tries breed lucky winners. The lab requires 0.80. | A lab gate. A fold that fails it runs on the anchor settings, so fixing a units bug in its calculation on 2026-10-19 changed returns, not only gate counts. It covers settings inside one test, not the 72 tests across the project (see Holm). | `synth:15-16`, `clean:28-34`, `src/ggTrader/lab/gates.py:190-215`, `src/ggTrader/lab/wfo.py:733-740` |
| Holm | A correction for testing several ideas at once. It raises the bar so that many tries do not produce lucky passes. | The French factor screens did not reach Holm significance (best t 1.34). | `synth:22-23`, `synth:533` |
| IR | Information ratio: excess return over the benchmark divided by tracking error. | It sets how big an edge a test needs to see (required IR 1.62 for the replica). | `synth:20-21`, `power:37` |
| IS | In-sample: tested on the same data used to choose the settings, so the result flatters itself. | One-year crypto wins over BTC were mostly in-sample. | `synth:36`, `crypto:31-37` |
| McLean-Pontiff | A 2016 study of 97 published stock predictors: on average they earn 26% less out of sample and 58% less after publication. | Published edges fade. The project's own momentum screen shows the same shape (Small t 3.38 before 1994, -0.77 after 2013). | `coverage:57-60` |
| MDE | Minimum detectable effect: the smallest true effect a test finds 80% of the time at its pass bar. | A NO only rules out effects above the MDE, not "no edge of any size". Analogy: a smoke alarm that trips only on a house fire. | `synth:17-18`, `verdicts:189-193` |
| NDH | The lab's parameter-neighbourhood density gate: a setting passes only if enough neighbouring settings also work, so an isolated lucky setting is rejected. | The second gate in a gated WFO, beside DSR. | `synth:16`, `clean:15` |
| NW t | Newey-West t-statistic: a t-statistic whose standard error allows for returns that are correlated from one day to the next. | The production replica's headline t (-0.69) is an NW t. | `accuracy:8-9`, `accuracy:31` |
| OOS | Out of sample: data the settings never saw. | Only OOS results count as evidence, and the early OOS figures carried leaks. | `synth:13-14` |
| PIT | Point in time (also written point-in-time): using only what was known on each date, for example index membership. | Tests that used today's index list instead of past membership favour the strategy. The 2026-09-23 membership fix is behind the core strategy's 0.59 Sharpe against SPY's 0.78. | `synth:14-15`, `synth:283`, `code:51-55`, `clean:29` |
| SE | Standard error: how much an estimate would wobble if the test were repeated on new data. Smaller means more precise. | It is the divisor in every t. | - |
| Sharpe ratio | Average return divided by how much returns swing day to day, scaled to a year. Higher means more return per unit of swing, not necessarily more total return. | The early "beat SPY" claims were Sharpe comparisons with no test of whether the gap was luck. | `synth:34`, `verdicts:103-105`, `src/ggTrader/lab/metrics.py:15-20` |
| t / t-statistic | An effect divided by its standard error (SE). A t of 2 or 3 is the usual bar for "probably not luck". | The Test Power check reports detectable effects at a t of at least 3 and at least 2. | `synth:18-19`, `power:35` |
| TE | Tracking error: how far a book strays from its benchmark, as a yearly standard deviation. | A book that is about 85% SPY has a TE of 2.61%, which is why its MDE is small. | `synth:21-22`, `power:36-38` |
| Welch t | A t-statistic that compares the averages of two groups without assuming they swing equally. | The month-end Treasury rule's t of 2.75 compares invested days with other days. | `coverage:326`, `ceiling:137` |
| WFE | Walk-forward efficiency: OOS Sharpe divided by IS Sharpe. | The circuit breaker stops a strategy when the trailing 4-window average WFE falls below 0.25. | `verdicts:18-19`, `verdicts:751` |
| WFO | Walk-forward optimization (also called walk-forward): pick settings on a past window, test them on the next unseen window, repeat. | The lab's backtest method. 50 of the 72 test verdicts rest on a gated WFO. | `synth:13`, `verdicts:106` |

### Returns and units

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| 2x | Two senses. As a target: twice SPY's total return over the same window. As leverage: twice the market exposure. | The target needs 9.57 pp/yr over 2021-2026. Only the 2x leverage arm and pre-1994 momentum each reached the gap needed in its own window. Other multiples such as 1.3x read the same way. | `synth:46-48`, `synth:68-70`, `ceiling:77-83` |
| B&H | Buy and hold: buy once, never trade. | Crypto results were judged against B&H BTC. | `crypto:22` |
| bp | Basis points: one bp is 0.01%. | Trading costs and fills are quoted in bp. The replica charged 5 bp. | `synth:12`, `audit0:25` |
| CAGR | Compound annual growth rate: the steady yearly rate that turns the start value into the end value. | The owner's target is a CAGR gap per window. | `synth:11` |
| MaxDD | Maximum drawdown: the largest fall from a peak to a later low. | The 2.0x SSO arm fell -84.8% at worst, against -55.3% for SPY. | `crypto:24`, `ceiling:70` |
| P&L | Profit and loss: money gained or lost. | Paper results are quoted in dollars of P&L as well as percentages. | - |
| pp, pp/yr | Percentage points, and percentage points a year: the plain difference between two percentages. 14.19% against 14.83% is -0.64 pp/yr. | Every edge in these reports is stated as pp/yr over SPY. | `synth:12`, `audit0:25` |
| PT / ET | American Pacific and Eastern time. | The live trader runs at 12:45 PT, before the close. | `synth:29`, `code:34-35` |
| RT | Round trip: a buy plus the matching sell. | The 74 closed paper stock round trips were all short-term. | `crypto:23`, `synth:73-74` |
| ST / LT | Short-term and long-term capital gains: a holding sold after one year or less, taxed at ordinary income rates, and one held more than one year, taxed at a lower rate. | All 74 paper round trips were short-term. Such a book needs about +2.42 pp/yr before tax just to tie SPY after tax. The one-year boundary is American tax law (outside the reports): the Ceiling check's "at least one year" (`ceiling:31`) is looser. | `ceiling:30-31`, `standard:22-23`, `synth:73-75` |
| T-bill | Treasury bill: short-term debt issued by the American government. | The comparison book holds SPY in place of the stocks plus the same share of T-bill cash. Leverage financed at T-bill rates earned more than the leveraged fund. | `power:18`, `synth:107-110`, `synth:69-70` |
| TR | Total return: price change plus dividends reinvested. | The target and the SPY comparison use TR. A price-only SPY series leaves out SPY's dividends. | `synth:11`, `parity:28-33` |
| UTC | Coordinated Universal Time, the world reference clock. It does not shift for daylight saving. | The price table stores naive UTC timestamps, and a write with a time zone shifted every bar. | `crypto:25`, `CLAUDE.md` |

### Funds and assets

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| A23 | The owner-adopted 70% SPY / 30% SSO allocation study. | Its registered 1.3x arm adds +0.81 pp/yr on the study's own basis when funding is priced 2.5 points higher, about today's bill rates, and +0.52 against SPY's true total return. | `synth:24-25`, `ceiling:58-62`, `ceiling:247` |
| A24 | A timing rule added to A23: a 10-month moving-average signal switches the SSO slice between SSO and SPY. NO-GO as a return improvement after tax. | After tax it returned 11.73% a year against 12.00% for plain 1.3x, with a smaller worst fall (-58.0% against -66.4%). | `inventory:65`, `backlog:2628` |
| BTC, ETH | Bitcoin and ether. | Crypto results are judged against holding BTC. | `synth:27-28` |
| ETF | Exchange-traded fund: a basket of assets that trades like a single stock. | SPY, SSO, TLT, IEF, GLD, PDBC and IBIT are ETFs. | `synth:10` |
| GLD | A gold fund. | Part of the cross-asset 80/20 sleeve: SPY plus TLT, GLD and PDBC. | `synth:26`, `coverage:327` |
| IBIT | iShares Bitcoin Trust ETF: a fund that holds bitcoin. | The only crypto result with a raw file is an IBIT study. | `synth:28`, `accuracy:23` |
| LETF | Leveraged or inverse ETF: a fund built to move a multiple of an index, or the opposite of it. | LETF timing rules were among the NOs downgraded to inconclusive. | `synth:25`, `synth:491` |
| MDY, MidCap 400 | The S&P MidCap 400 is an index of mid-sized American companies. MDY is the fund that tracks it. | The mid-cap reversion test made 15.0% a year against 9.1% for MDY, on a current-list universe with leaks, so it is not trusted. | `synth:537`, `challenge:15` |
| PDBC, DBC | Commodity funds. | PDBC is one third of the 20% cross-asset sleeve. | `ceiling:27-30`, `coverage:313` |
| S&P | Standard & Poor's, an index company. The S&P 500 is its list of 500 large American companies. | The stock universe and the benchmark both come from it. | - |
| SPY | A fund that tracks the S&P 500 and trades like a stock. It is the benchmark every result is compared with. | The live book is about 85% SPY, so it cannot stray far. | `synth:10-11` |
| SSO | A ProShares fund that targets twice SPY's daily move. | The leverage in A23. | `synth:23-24` |
| TLT, IEF | Treasury bond funds. IEF holds 7-10 year bonds and TLT holds 20+ year bonds. | TLT is part of the cross-asset 80/20 sleeve. IEF was the month-end Treasury rule's fund. | `synth:26`, `power:41` |

### Strategy and data terms

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| 12-1 momentum | Buy the stocks that rose most over the past year, skipping the latest month. | The momentum replay returned +4.37 pp/yr with a CI that includes zero. | `synth:37-38` |
| BB, RSI, EMA, MACD | Four price-pattern indicators: Bollinger Bands, relative strength index, exponential moving average, moving-average convergence/divergence. | Four of the five voters in the live ensemble. | `code:13-14`, `code:155` |
| EDGAR | The American Securities and Exchange Commission's public filings database. | The Argon review proposed it, but no EDGAR ingestion exists here. | `consol:185`, `verify:63` |
| Fama-French | Eugene Fama and Kenneth French, finance professors whose factor research defines the market, value, profitability and investment portfolios published in the French data library. | The long-history screens compare against the Fama-French market. | `accuracy:192`, `verdicts:486` |
| FOMC | Federal Open Market Committee: the American central bank's interest-rate meetings. | The pre-FOMC Treasury study is the one area tested with evidence against, with caveats. | `synth:25-26`, `coverage:41-46` |
| French data | Free, survivorship-free American stock portfolio returns back to 1926, published by Professor Kenneth French. | It allows long-history tests (1963 and 1927) without paid data. | `synth:38-39` |
| Kelly | Kelly sizing: set each bet's size from its estimated edge and risk, so a bigger edge means a bigger bet. | The ensemble_kelly strategy failed to beat the baseline. Its code did not match its spec and it has no named comparator, so it stays inconclusive. | `synth:528` |
| Mkt-RF, RF | Mkt-RF is the stock market's return above the risk-free rate, RF, the one-month T-bill return. Their sum is the market return. | The French market comparator is Mkt-RF plus RF. | `verdicts:486`, `power:164` |
| PEAD | Post-earnings-announcement drift: a stock tends to keep moving the way its earnings surprise pointed, for weeks after the report. | The one small-cap stock-picking arm tested was PEAD on a partial list of 200 names. | `coverage:22`, `coverage:35-36` |
| Shiller | Long-run American stock and dividend data compiled by Professor Shiller, used in the century study, a long-history test of static leverage. | The frozen copy's dividends end 2026-06. | `audit0:42`, `accuracy:200`, `checkpoint:663` |
| vbb | The volume-confirmed Bollinger voter, the fifth voter in the live ensemble. | The 5-voter set is the live default. | `code:14`, `consol:154` |

### Names and codes

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| agy | The Antigravity agent harness: a program that runs automated agents. | It ran the Argon review. | `consol:25` |
| Alpaca activity codes | About 25 activity codes of Alpaca, the paper account's broker: `DIV` and variants for dividends, `SPLIT`, `MA`, `REORG`, `REO`, `NC`, `SPIN`, `INT`, `CSD`, `CSW`, `JNLC`, `JNLS`, `FEE`, `REG`, `TAF`, `CAT`. | A search of the dividend, split, merger, interest and journal codes since 2024-06-01 returned only the $100,000 funding journal. The broker never credited a dividend. | `parity:21-26`, `parity:291-296` |
| Argon | The Gemini 4 Argon model, run through agy. | It wrote one of the two frontier-model reviews. The review carries a banner marking it as unverified model output. | `verify:3` |
| EQR | Equity Residential's stock ticker (public fact, not named in the reports). AvalonBay (AVB) merged into it, so the AVB holding cannot be traded. | AVB's stored prices have copied EQR's since 2026-07-20. The held position was $746, 0.7% of the account. | `parity:330`, `clean:47` |
| French portfolio names | Labels for stock portfolios sorted by a trait: profitability `HiOP`, cheapness `HiBM`, low investment `LoINV`, prior return `HiPRIOR`, share issuance `HiNI` and `NegNI` (net buybacks), size `BIG` and `SMALL`. | Used in the factor, momentum and net issuance screens. `BIG` `NegNI` returned t 0.26. | `coverage:305`, `ceiling:242`, `synth:534` |
| MNST, AVB | Stock ticker symbols for Monster Beverage and AvalonBay. | The broker never applied MNST's 2-for-1 split (-0.85 pp of the paper shortfall). AVB's price series has been EQR's since 2026-07-20. | `consol:58`, `consol:71`, `parity:45-46`, `clean:47` |
| Opus | The Claude Opus model. | It wrote the other frontier-model review, the senior-quant direction review. | `consol:13` |
| PARA | A ticker whose stored prices belong to a different security for its whole S&P 500 membership. | The production replica traded it 3 times, and its effect is unmeasured. | `clean:37-44` |
| Ticker symbols | Short exchange codes for a listed stock or fund, such as `AAPL`, `VZ` and `HAS` (Hasbro). Kraken, a crypto exchange, writes BTC as `XBT` in pair codes such as `XBTUSD`. | About 120 stock tickers and 40 crypto pair codes appear across the reports. Binance.US is another crypto exchange. | `CLAUDE.md` |

### Verdict labels

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| GO, NO-GO | The project's pass and fail labels for a test. | A NO-GO is a result, not a reason to stop the work. | `synth:23`, `consol:64` |
| inconclusive | The test could not decide because at least one condition fails or is unknown. It does not mean the test found nothing. | 51 NO rows, 7 GO rows, 3 tests with their own inconclusive verdict and 42 status labels. | `verdicts:44-49`, `synth:60-62` |
| NO | A measured test reported fail, no edge, underperformance or did not reproduce. | The Verdict Trust check classed 53 NO test verdicts. | `verdicts:181` |
| reliable GO | A GO with a clean tape, checked code, a recomputed number and a pass bar set in advance. | None of the 16 GO test verdicts qualifies. | `verdicts:191` |
| reliable NO | A NO whose data, code, recomputed number and statistical power all check out. | None of the 53 NO test verdicts qualifies. | `verdicts:191` |
| unreliable GO | A GO built on contamination that favours the strategy, or on code or numbers that failed their checks. | 9 rows, including every early "beat SPY" figure. | `verdicts:46`, `verdicts:191` |
| unreliable NO | A NO from a flawed test: the code check failed and it was judged against the later-voided 1.12 baseline, or its numbers do not recompute. | 2 rows. | `verdicts:43`, `verdicts:191` |

## 3. Plain-English glossary

Words that are not abbreviations but sound like they should be. Analogies are labelled and never change the meaning. A dash in Source means the word is common usage.

Some of these words mean two things. Section 6 lists the repeat offenders.

| Word | Plain meaning | Source |
|---|---|---|
| accrual (anomaly) | A stock-return pattern linked to how much of a firm's earnings are accounting accruals rather than cash. Not tested, only feasibility-checked. | `verdicts:247` |
| accrual (dividend) | A dividend the trader books as owed before the broker pays it. Here the broker never paid, so the paper account carries $320.24 that is not cash. | `parity:15-22` |
| anchor | The default settings a walk-forward test falls back to when its gates reject the settings it just picked. The anchor leak fitted them on the whole test span. | `synth:39-40`, `synth:283` |
| arm | One variant inside a test, such as the 1.3x or 2.0x leverage arm. | `ceiling:210-214` |
| artifact | A saved raw output file, usually a data file, behind a reported number. A "broker artifact" is a side effect of how the broker works, not a real result. | `accuracy:5`, `parity:44-45` |
| bar (pass) | The rule, set in advance, that decides GO or NO-GO. | `accuracy:11` |
| bar (price) | One period of price data: open, high, low, close and volume. Here usually one trading day. | - |
| carry | Return from holding a position rather than from price moves, such as funding payments on a crypto futures position. | `crypto:31-33` |
| cash reserve | The 5% of the account the live trader keeps in cash. It costs about 0.9 pp/yr relative to SPY. | `synth:56` |
| catastrophe stop | A forced sale when a holding is down 25% or more from what it cost. Live fired it once, on 2026-09-25. | `code:247`, `parity:333`, `src/ggTrader/paper/catastrophe_stop.py:33` |
| contamination | Anything in the data or code that tilts a result, such as leaks, wrong prices or survivorship. | `clean:1` |
| deflated | Marked down for how many settings were tried. See DSR. | `synth:15-16`, `src/ggTrader/lab/gates.py:190-215` |
| drawdown | The fall from a peak to a later low, shown as a percentage. | - |
| ensemble | The live strategy: five voters. It enters when at least 2 of the 5 vote buy, and exits when RSI crosses up through 50 or 2 vote exit. | `code:155-156` |
| event study | Measuring returns around a dated event, such as an FOMC meeting. | `coverage:41-46` |
| exposure | How much of the account sits in a given thing. The replica held 9.37% stocks and about 85% SPY. | `power:50` |
| factor | A stock trait, such as value or profitability, that is tested for earning a return. | - |
| falsifier | A result, written down before the evidence is read, that would show the working view wrong. | `crypto:72-73` |
| fold | One pick-then-test slice of a walk-forward run. The lab uses overlapping monthly folds. | `CLAUDE.md` |
| frozen | No configuration changes. The paper trader has been frozen since 2026-09-30. | `parity:177` |
| gates | Pass or fail checks inside a walk-forward run, mainly DSR and NDH. A fold that fails them runs on the anchor settings. | `clean:28-34` |
| holdout | Data kept aside and not used to choose settings, then used for a final check. The replica's 2010-2020 holdout is not citable because of bad prices. | `accuracy:32-37` |
| hurdle | A minimum result an idea must clear, such as the kill-only replay's -1.0 pp/yr after tax. | `standard:80-86` |
| idle cash | Cash not invested. Before the SPY sweep it cost 2.46 pp of the 3.76 pp paper shortfall. | `parity:38-41` |
| leak | Information a test used that it should not have had. Two favoured the strategy here: the anchor leak and the universe leak (14.1% of entries on days the stock was not in the index). | `synth:280-285`, `clean:132-133` |
| leverage | Holding more market exposure than your cash, by borrowing or through a fund like SSO. 1.3x means 1.3 dollars of market per dollar. | - |
| lookahead | A test that uses information not yet available at the decision time. | `clean:93` |
| multiple testing | Testing many ideas raises the chance that one passes by luck. Under zero edge, 72 tests at t of at least 2 give about 1.6 false wins. | `verdicts:96-101` |
| net issuance | How many shares a company issued or bought back. The tested long side was companies that bought back more shares than they issued. | `ceiling:242` |
| parity | Live matches the test: same decisions, same day, same prices. | `parity:1` |
| partial bar | Today's price bar before the market closes. Live signals at 12:45 PT use it. Backtests use the final close. | `clean:193-195`, `code:34-39` |
| PIT core | The S&P 500 stock-selection strategy run walk-forward with point-in-time membership. Sharpe 0.59 against SPY's 0.78. | `synth:520` |
| power | The chance that a test catches a real effect of a given size. These reports use 80%. | `synth:17-18` |
| pre-registered | The pass bar was written down and committed before the result was seen. | `accuracy:11` |
| regime halt | The walk-forward circuit breaker standing a strategy down. It triggers when the trailing 4-window average WFE is below 0.25 or after two straight negative OOS Sharpes. | `verdicts:751-752` |
| replica | The live trader's rules re-run on history, to ask what live would have earned. | `synth:53` |
| reversion | Betting that a stock that fell will move back toward its recent average. The live ensemble is a reversion strategy. | `coverage:53` |
| seam | A jump in stored prices where two downloads with different dividend adjustments meet, making a false one-day move. | `clean:42-45` |
| sensitivity | How a result changes when an input changes. The sensitivity table shows each result at two significance levels. | `sensitivity:13-17` |
| shadow | A book that records hypothetical holdings and fills without sending broker orders. | `standard:20-21` |
| sleeve | The part of the account that runs one strategy, such as the 3-stock sleeve beside the SPY holding. | - |
| status label | A verdict row that is not an edge test, such as feasibility, design, literature or operations. The Verdict Trust check counts 42 of them. | `verdicts:181` |
| survivorship | Testing only companies that still exist today, so the failures are missing and results look better. Analogy: reviewing only the services that are still up. | `verdicts:1078` |
| sweep (cash) | The live rule that puts idle cash into SPY while keeping a 5% reserve. | `synth:56`, `synth:138`, `synth:525` |
| sweep (settings) | Trying a grid of settings to see which works best. | `CLAUDE.md` |
| tape | The `ohlcv` price table in the database that every stock backtest reads. Analogy: the shared fixtures every test loads. | `clean:16-17` |
| tax lot | A record of one purchase: quantity, cost and acquisition date. TaxLotBook is the project's library that holds them. | `standard:22-23`, `verify:30` |
| test family | A group of related tests counted together when correcting for many tries. The 72 measured tests fall in 11 families. | `synth:328` |
| total return | Price change plus reinvested dividends. See TR. | `synth:11` |
| tracking error | See TE. How far a book strays from its benchmark. | `synth:21-22` |
| universe | The list of stocks a test may pick from. | - |
| voter | One indicator rule inside the ensemble that votes to buy or not. | `code:155` |
| wash sale | A tax rule: a loss on a sale is disallowed when substantially identical stock is bought within 30 days before or after. It applies only to loss sales, which is why the SPY sweep's trade count alone does not show wash-sale harm. | `verify:50` |

## 4. The nine checks

The investigation ran nine separate checks. The reports are in the `lenses/` folder. The file name is the only place the numbers appear. Each verdict is that report's own conclusion.

| Name | The question in one line | File | Verdict from the report |
|---|---|---|---|
| Backtest Cleanliness check | What contamination is left in the prices and code after the five known fixes? | `2026-11-04-lens1-backtest-validity.md` | The five known bugs are fixed, but PARA is the wrong security, stored prices mix adjustment bases and AVB tracks EQR. Of 144 rows, 55 are contaminated, 18 clean and 71 unknown. `clean:21-47` |
| Verdict Trust check | Which NO-GO and GO verdicts can be trusted, and for what size of effect? | `2026-11-04-lens2-verdict-audit.md` and `2026-11-04-lens2-sensitivity-spec.md` | None of the 53 NO test verdicts is a reliable NO and none of the 16 GO test verdicts is a reliable GO. That does not mean an edge was missed. It means no NO-GO can be read as ruling out an edge of the size it was meant to rule out. `verdicts:34-35` The second file fixes the sensitivity table before any result was opened. `verdicts:29-35`, `sensitivity:1-20` |
| Paper-versus-Backtest check | Does the live paper trader match what the backtests measured? | `2026-11-04-lens3-live-parity.md` | Paper trailed SPY by 3.76 pp since 2026-06-24. Idle cash before the SPY sweep is -2.46 pp. The snapshot gap is dividend accrual the broker never paid. `parity:15-45` |
| Ceiling check | Can the target be reached at all, and what is the measured maximum? | `2026-11-04-lens4-feasibility-and-ceiling.md` | The target needs 9.57 pp/yr over 2021-2026 and 3.54 over 2006-2026. Only the 2x leverage arm (+4.15 pp/yr, -84.8% worst fall, picked after the results) and pre-1994 French momentum each reached the gap needed in its own window. Leverage is exposure, not skill, and pre-1994 momentum is the pre-publication period and it reverses after 2013. No stock-picking result after 1994 reached its gap, and no result is graded fully reliable. `ceiling:35-46`, `ceiling:57-83` |
| Coverage check | What was never tested? | `2026-11-04-lens5-coverage-gaps.md` | The record is narrow: 44 of 51 single-stock tests are large-cap only and 65 of 72 are long-only. Of 28 areas, 6 were never tested, 4 tested badly, 17 inconclusive and 1 shows evidence against. 62 of 82 backlog ideas never ran. `coverage:30-58` |
| Code Correctness check | Does the code do what the spec says, and does live run what was tested? | `2026-11-04-lens6-implementation-correctness.md` | The arithmetic is right for the four rules replayed (0 mismatches), but the live trader does not run the configuration the backtests measured. `code:15-50` |
| Report Accuracy check | Are the reported numbers true, do the sources say what the reports say, and were pass bars set first? | `2026-11-04-lens7-research-accuracy.md` | Every headline with a raw file is true. 103 of 144 rows have no raw file. 11 of 14 citations checked say what the reports say. The synthesis narrows this: 7 headlines could not be found in their files and 9 were not re-checked. `accuracy:20-29`, `synth:203-207` |
| Crypto check | What was tried in crypto, what does the evidence show and what is unknown? | `2026-11-04-lens8-crypto.md` | No crypto result beat buy-and-hold BTC over any three-year window. The tape has crypto-specific defects: 43.7% of Binance.US 4-hour bars copy Kraken (both are crypto exchanges). `crypto:31-60` |
| Test Power check | Could the tests even see an effect? | `2026-11-04-lens9-power-and-regime.md` | Only 8 of 144 rows have the inputs for an MDE. The replica could detect 4.24 pp/yr at book level, about 45 pp/yr at stock-sleeve level (a rough scaling, not a recomputation). `power:23-54`, `power:51-52` |

## 5. The three rounds

Three rounds, then the write-up that closed the investigation. Nothing on the Optional Re-run List was run, by the owner's decision, which keeps its running cost at zero.

| Name | What it was | File | Result |
|---|---|---|---|
| Early Trust Audit | A single unreviewed senior-quant audit of the stock results, stocks only. Its claims were numbered `H1` to `H7` later and tested by the checks. | `findings/2026-11-04-phase0-trust-audit.md` | No real stock-picking edge over SPY was found. Its view that the sound tests were mostly reliable NOs was overturned: the Verdict Trust check found none. `audit0:9-17`, `verdicts:29-31` |
| Main Evidence Round | An inventory of 144 results, then the nine checks above, reported in one checkpoint. | `findings/2026-11-04-phase1-checkpoint.md` | The re-run list could not be skipped under the design's rule: the checks did not confirm every Early Trust Audit claim, and they raised unknowns that could change a verdict. `checkpoint:773-777` |
| Optional Re-run List | 29 numbered items, `P2-01` to `P2-29`, ordered cheapest first. An item tagged "writes to the price cache" (the synthesis and the Argon review mark these with *, `synth:224`) would write to the shared price cache, which the no-writes rule forbids, so it cannot run as written. `checkpoint:261` | `findings/2026-11-04-phase1-checkpoint.md` (items in `checkpoint:286-314`) | None run, by the owner's decision of 2026-11-05. `synth:7-8` |
| Synthesis and Independent Challenge | The synthesis of the nine checks, then a challenger that argued the opposite case and checked 40 claims. | `findings/2026-11-04-root-cause-investigation.md` and `findings/2026-11-04-root-cause-challenge.md` | Of 40 claims: 20 confirmed, 16 confirmed with a caveat, 3 wrong, 1 not checked. Every objection is resolved in the synthesis. `synth:746-751` |

Three review files came afterwards and sit in `reviews/`: the Opus review, the Argon review and a verification of the Argon review. A fourth file consolidates the two reviews. All four are in the Source key.

## 6. Other labels

The reading reports use the names in the second column. The original files use the codes in the first. The codes sit in code spans because the numbers and letters belong to the originals.

### Named checkpoints, stages and decisions

| Label | Name used in the reading reports | Meaning | Source |
|---|---|---|---|
| `Gate A` | the leverage-reach checkpoint | If `P2-04` puts 2.0x leverage at today-like funding below the 2006-2026 gap, record that no tested exposure level reaches 2x at current rates. | `opus:191-192` |
| `Gate B` | the partial-bar checkpoint | If `P2-14` finds partial-bar and final-bar decisions differ on more than a handful of the 30 sessions, the parity proposal must pick one bar for live and replica. | `opus:192-194` |
| `Gate C` | the long-history-test checkpoint | A fail on the long-history test closes the idea. A pass moves it on to the kill-only replay only. | `opus:200-201` |
| `Gate D` | the concentration checkpoint | After `P2-06` (concentrated big-cap momentum after 1994): below +2 pp/yr, concentration needs no further work. | `opus:202-203` |
| `Part A` | the long-history test | Does the idea survive long free history, such as the French data? The standard's pass bar: a t of at least 3 after a Holm correction across the registered family (the corrected hurdle can be higher than 3), and at least +1.0 pp/yr since 1994 after trading costs. Not after tax: tax first applies in the kill-only replay. The Argon review misstates it as after tax. | `standard:49-50`, `standard:55-56`, `standard:80`, `argon:98` |
| `B0` | the kill-only replay | Runs once on 2021-2026. If the after-tax CAGR minus SPY's falls below -1.0 pp/yr, the idea is killed. A pass proves nothing. | `standard:65-83` |
| `B1` | the forward shadow test | A forward shadow book run for at least 18 months. | `standard:88` |
| `Decision 0` | the no-writes rule | The owner's permission for re-runs to write to the shared price cache. Answered no. It blocks every re-run item tagged as writing to the price cache. | `opus:16-17`, `checkpoint:827-834` |

`Part B` in older briefs is a paid-data replay of the exact rules. The standard replaced it for future stock ideas with the kill-only replay and the forward shadow test (`standard:3`).

### Owner questions, account eras and grades

| Label | Name used in the reading reports | Meaning | Source |
|---|---|---|---|
| `30-day`, `90-day`, `180-day gate` | Argon's day checkpoints | Argon's own checkpoints, unrelated to the four Opus checkpoints above. | `argon:128-140` |
| `A1-A5`, `B1-B6`, `C1-C7` | ranked root-cause items | A: method and process errors in the research. B: implementation and live-parity errors. C: the evidence is negative or the target is structurally hard. | `synth:278`, `synth:349`, `synth:420` |
| `A1-A27`, `B1-B15` | research-backlog idea ids | Idea ids in the web-research backlog: `A1` to `A27` and `B1` to `B15`. A10 is the month-end Treasury sleeve, A12 a month-end SPY-to-IEF tilt, A14 a turn-of-month rotation. A23 and A24 are the leverage studies. | `backlog:586`, `backlog:1041`, `backlog:3038`, `backlog:1605`, `backlog:1665`, `backlog:1731`, `backlog:2567`, `backlog:2628` |
| `C`, `C*`, `W`, `NC` | challenge result codes | Confirmed, confirmed with caveat, wrong, not checked. | `challenge:63-64` |
| `claim N` | a claim in the Argon verification | Row N of the claim check table. | `verify:43-47` |
| `Decisions 1-13` | the owner decision list | One list in the consolidated review: accept the synthesis, changelog entry, any re-run item, research copy of the price table, input archive, daily-loss halt base, storage for downloads, yearly research budget, leverage figures, one benchmark, sweep tax churn, reopening closed NO-GOs, deferred minor review notes. | `consol:100-118` |
| `E1-E7` | account eras | Paper account stretches split at each configuration change: core ensemble (06-24 to 07-15), accidental dry run (to 07-30), 3-sleeve blend (to 08-21), blend plus SPY sweep (to 09-22), S&P 500 core plus sweep (to 09-28), missed-exit catch-up (to 09-30), frozen (to 11-04). | `parity:171-180` |
| `F1`, `F2` and `N1`, `N2` | falsifiers | Each check wrote its own set before reading evidence. The same code means different things in different reports. | `coverage:556` |
| `H1-H7` | the Early Trust Audit's seven hypotheses | Stock-trust hypotheses tested by the later checks. | `synth:476` |
| `H4a`, `H4b`, `H4c` | the ceiling hypotheses | H4a: the best long-only large-cap effects after 1994 are +0.1 to +0.8 pp/yr. H4b: +10 to +15 pp/yr is needed. H4c: 2x SPY is out of reach without leverage or a much larger effect. The Ceiling check confirmed H4a on its own terms, overturned H4b as a general statement and confirmed H4c for leverage and pre-publication momentum. | `ceiling:96-122` |
| `L4-X1`, `L4-X2` | the two Ceiling check exceptions | X1: fixed-weight books whose code was never examined. X2: non-registered arms and sub-windows. | `ceiling:205`, `ceiling:212` |
| `Options 1-3` | the consolidated review's three options | 1: paper-trader measurement and parity. 2: funding-honest leverage figures for the 2027-09-30 A23 review. 3: one pre-registered long-history stock screen (accruals). | `consol:62-98` |
| `Q1-Q7` | the owner's seven questions | Edge over SPY, what limits live return, the measured maximum, reliability of each NO-GO, crypto, what holds up, what is still unknown. | `synth:105-221` |
| `R1-R7` | the Verdict Trust check's seven classification rules | Fixed before any row was classified: verdict type, data condition, code condition, recompute, power, classes, rules-out size. | `verdicts:179-193` |
| `Rank 1-5`, `rec N` | recommendation ranks | The Argon review ranks five directions, and the Opus review numbers its recommendations. | `argon:64-113`, `opus:93-148` |
| `STATUS`, `TEST-INCONCLUSIVE` | verdict types | STATUS is a feasibility, design or operations label. TEST-INCONCLUSIVE is a test whose own verdict is inconclusive or invalid. | `verdicts:181` |
| `Syn` | the synthesis | The challenge file's short name for the synthesis file. | `challenge:63` |
| `Tier A`, `Tier B`, `Tier C` | soundness grades | A: reliable. B: the number recomputes on clean data, but nothing proves it is statistically real. C: everything else, listed but not used for the ceiling. | `ceiling:194-201`, `synth:40-41` |

### Labels that mean two things

| Label | First meaning | Second meaning |
|---|---|---|
| `B1` | Root-cause item: live does not run the configuration any backtest measured | The forward shadow test, and backlog idea `B1` (a stablecoin stress signal) |
| `A1`, `A2` | Root-cause items, and idea ids in the research backlog | The two halves of a long-history test: A1 significance, A2 size since 1994 |
| `H1-H4` | Hypotheses from the Early Trust Audit | Hypotheses in the factor screen |
| `F1` | A falsifier in one check | A different falsifier in each other check |
| `Q1` | An owner question | A calendar quarter, as in "2024 Q1" |
| `NC` | Not checked (challenge result code) | An Alpaca activity code |
| gate | The four Opus checkpoints above | The Argon day checkpoints, the lab's DSR and NDH gates, and the "stop gate" conditions in the consolidated review |
| bar | A price bar | A pass bar |
| accrual | A dividend booked before it is paid | The accruals anomaly |

The name `FundingCarryBTC Phase 4` in `crypto:403` is an old crypto study, not a round of this investigation.

## Source key

Source codes in the tables cite a short code and a line number, such as `synth:15-16`, in the file below. Paths are from the repository root.

| Code | File |
|---|---|
| `synth` | `research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md` |
| `challenge` | `research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-challenge.md` |
| `checkpoint` | `research/audits/root-cause-2026-11-04/findings/2026-11-04-phase1-checkpoint.md` |
| `audit0` | `research/audits/root-cause-2026-11-04/findings/2026-11-04-phase0-trust-audit.md` |
| `clean` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens1-backtest-validity.md` |
| `verdicts` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md` |
| `sensitivity` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-sensitivity-spec.md` |
| `parity` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens3-live-parity.md` |
| `ceiling` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md` |
| `coverage` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md` |
| `code` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens6-implementation-correctness.md` |
| `accuracy` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens7-research-accuracy.md` |
| `crypto` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md` |
| `power` | `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens9-power-and-regime.md` |
| `opus` | `research/audits/root-cause-2026-11-04/reviews/2026-11-05-senior-quant-direction-review.md` |
| `argon` | `research/audits/root-cause-2026-11-04/reviews/2026-11-05-argon-synthesis-review.md` |
| `verify` | `research/audits/root-cause-2026-11-04/reviews/2026-11-05-argon-review-verification.md` |
| `consol` | `research/audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md` |
| `inventory` | `research/audits/root-cause-2026-11-04/records/2026-11-04-investigation-inventory.md` |
| `backlog` | `research/WEB_RESEARCH_CANDIDATES.md` |
| `standard` | `research/programme/evidence-standard/2026-10-23-forward-paper-standard.md` |
