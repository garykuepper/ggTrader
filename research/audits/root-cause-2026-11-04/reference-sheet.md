# Root-cause reports: reference sheet

**Date:** 2026-11-05. **Purpose:** look up any term, check name or label used in the root-cause reports and read a plain-English meaning. No finance background assumed. Short source codes such as `synth:15-16` point to a file and line, and the Source key at the end maps each code to its file.

## 1. How to read these reports

- The reading reports name the checks and rounds. They never number them. This sheet maps each name to its file.
- The original reports keep their numbers because they are the record. Their file names carry the numbers, so the File column in sections 4 and 5 is the one place the numbers appear.
- A dash in a Source column means the term is common usage and no report defines it. Every other row was checked against the cited line.
- The forward-paper standard is approved by the owner (its own line 3, `standard:3`). Its section 1 still calls itself a draft, so the standard contradicts itself and this sheet follows line 3.

I have read all of it so you do not have to read all of it. Start with the Abbreviations table when a three-letter word stops you.

## 2. Abbreviations

Each plain meaning is at most 30 words and uses only terms that have their own row.

### Statistics and testing

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| Sharpe ratio | Return per unit of ups and downs. Higher means a smoother ride. It is not the same as higher total return. | The early "beat SPY" claims were Sharpe comparisons with no test of whether the gap was luck. | `synth:34`, `verdicts:103-105` |
| DSR | Deflated Sharpe ratio: a Sharpe ratio marked down for how many ideas were tried before this one looked good. More tries, more chance one is lucky. | A lab gate. A fold that fails it runs on the anchor settings, so fixing its units on 2026-10-19 changed returns, not only gate counts. | `synth:15-16`, `clean:28-34` |
| t / t-statistic | An effect divided by its standard error (SE). A t of 2 or 3 is the usual bar for "probably not luck". | The Test Power check reports detectable effects at a t of at least 3 and at least 2. | `synth:18-19`, `power:35` |
| SE | Standard error: how much an estimate would wobble if the test were repeated on new data. Smaller means more precise. | It is the divisor in every t. | - |
| NW t | Newey-West t-statistic: a t-statistic whose standard error allows for returns that are correlated from one day to the next. | The production replica's headline t (-0.69) is an NW t. | `accuracy:8-9`, `accuracy:31` |
| CI | Confidence interval: the range of true values the data cannot rule out. The reports use 95%. | Momentum's best 2021-2026 estimate, +4.37 pp/yr, has a CI of -7 to +16, so the data cannot tell it from zero. | `synth:70-72` |
| Holm | A correction for testing several ideas at once. It raises the bar so that many tries do not produce lucky passes. | The French factor screens did not reach Holm significance (best t 1.34). | `synth:22-23`, `verdicts:56` |
| MDE | Minimum detectable effect: the smallest true effect a test finds 80% of the time at its pass bar. | A NO only rules out effects above the MDE. It does not mean "no edge of any size". | `synth:17-18`, `verdicts:189-193` |
| TE | Tracking error: how far a book strays from its benchmark, as a yearly standard deviation. | A book that is about 85% SPY has a TE of 2.61%, which is why its MDE is small. | `synth:21-22`, `power:36-38` |
| IR | Information ratio: excess return over the benchmark divided by tracking error. | It sets how big an edge a test needs to see (required IR 1.62 for the replica). | `synth:20-21`, `power:37` |
| IS | In-sample: tested on the same data used to choose the settings, so the result flatters itself. | One-year crypto wins over BTC were mostly in-sample. | `synth:36`, `crypto:31-37` |
| OOS | Out of sample: data the settings never saw. | Only OOS results count as evidence, and the early OOS figures carried leaks. | `synth:13-14` |
| WFO | Walk-forward optimization: pick settings on a past window, test them on the next unseen window, repeat. | The lab's backtest method. 50 of the 72 test verdicts rest on a gated WFO. | `synth:13`, `verdicts:106` |
| WFE | Walk-forward efficiency: OOS Sharpe divided by IS Sharpe. | The circuit breaker stops a strategy when the trailing 4-window average WFE falls below 0.25. | `verdicts:18-19`, `verdicts:751` |
| PIT | Point in time: using only what was known on each date, for example index membership. | Tests that used today's index list instead of past membership favour the strategy. The 2026-09-23 membership fix is behind the core strategy's 0.59 Sharpe against SPY's 0.78. | `synth:14-15`, `synth:283`, `code:51-55` |
| NDH | The lab's parameter-neighbourhood density gate: a setting passes only if enough neighbouring settings also work, so an isolated lucky setting is rejected. | The second gate in a gated WFO, beside DSR. | `synth:16`, `clean:15` |

### Returns and units

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| pp, pp/yr | Percentage points, and percentage points a year: the plain difference between two percentages. 14.19% against 14.83% is -0.64 pp/yr. | Every edge in these reports is stated as pp/yr over SPY. | `synth:12`, `synth:29-30` |
| bp | Basis points: one bp is 0.01%. | Trading costs and fills are quoted in bp. The replica charged 5 bp. | `synth:12`, `audit0:25` |
| CAGR | Compound annual growth rate: the steady yearly rate that turns the start value into the end value. | The owner's target is a CAGR gap per window. | `synth:11` |
| TR | Total return: price change plus dividends reinvested. | The target and the SPY comparison use TR. A price-only SPY series leaves out SPY's dividends. | `synth:11`, `parity:28-33` |
| MaxDD | Maximum drawdown: the largest fall from a peak to a later low. | The 2.0x SSO arm fell -84.8% at worst, against -55.3% for SPY. | `crypto:24`, `ceiling:70` |
| 2x | Two senses. As a target: twice SPY's total return over the same window. As leverage: twice the market exposure. | The target needs 9.57 pp/yr over 2021-2026. Only 2x leverage reached its gap. Other multiples such as 1.3x read the same way. | `synth:46-48`, `synth:68-70` |
| P&L | Profit and loss: money gained or lost. | Paper results are quoted in dollars of P&L as well as percentages. | - |
| RT | Round trip: a buy plus the matching sell. | The 74 closed paper stock round trips were all short-term. | `crypto:23`, `synth:73-74` |
| B&H | Buy and hold: buy once, never trade. | Crypto results were judged against B&H BTC. | `crypto:22` |
| T-bill | Treasury bill: short-term debt issued by the American government. | The comparison book holds SPY in place of the stocks plus the same share of T-bill cash. Leverage financed at T-bill rates earned more than the leveraged fund. | `power:18`, `synth:107-110`, `synth:69-70` |
| ST / LT | Short-term and long-term capital gains: a holding sold within one year, and a holding kept at least one year. | All 74 paper round trips were short-term. Such a book needs about +2.42 pp/yr before tax just to tie SPY after tax. | `ceiling:30`, `synth:73-75` |
| PT / ET | American Pacific and Eastern time. | The live trader runs at 12:45 PT, before the close. | `synth:29`, `code:34-35` |
| UTC | Coordinated Universal Time, the world reference clock. It does not shift for daylight saving. | The price table stores naive UTC timestamps, and a write with a time zone shifted every bar. | `crypto:25`, `CLAUDE.md` |

### Funds and assets

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| SPY | A fund that tracks the S&P 500 and trades like a stock. It is the benchmark every result is compared with. | The live book is about 85% SPY, so it cannot stray far. | `synth:10-11` |
| S&P | Standard & Poor's, an index company. The S&P 500 is its list of 500 large American companies. | The stock universe and the benchmark both come from it. | - |
| ETF | Exchange-traded fund: a basket of assets that trades like a single stock. | SPY, SSO, TLT, IEF, GLD, PDBC and IBIT are ETFs. | `synth:10` |
| LETF | Leveraged or inverse ETF: a fund built to move a multiple of an index, or the opposite of it. | LETF timing rules were among the NOs downgraded to inconclusive. | `synth:25`, `synth:491` |
| SSO | A ProShares fund that targets twice SPY's daily move. | The leverage in A23. | `synth:23-24` |
| A23 | The owner-adopted 70% SPY / 30% SSO allocation study. | Its registered 1.3x arm adds +0.81 pp/yr when funding is priced 2.5 points higher, about today's bill rates. | `synth:24-25`, `ceiling:58-62` |
| A24 | The sleeve trend-rider study, run beside A23 with the same leverage arms. | A23 and A24 share one driver script. | `verdicts:426`, `verdicts:452` |
| TLT, IEF | Treasury bond funds. IEF holds 7-10 year bonds and TLT holds 20+ year bonds. | TLT is part of the cross-asset 80/20 sleeve. IEF was the month-end Treasury rule's fund. | `synth:26`, `power:41` |
| GLD | A gold fund. | Part of the cross-asset 80/20 sleeve: SPY plus TLT, GLD and PDBC. | `synth:26`, `coverage:327` |
| PDBC, DBC | Commodity funds. | PDBC is one third of the 20% cross-asset sleeve. | `ceiling:27-30`, `coverage:313` |
| BTC, ETH | Bitcoin and ether. | Crypto results are judged against holding BTC. | `synth:27-28` |
| IBIT | iShares Bitcoin Trust ETF: a fund that holds bitcoin. | The only crypto result with a raw file is an IBIT study. | `synth:28`, `accuracy:24` |

### Strategy and data terms

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| 12-1 momentum | Buy the stocks that rose most over the past year, skipping the latest month. | The momentum replay returned +4.37 pp/yr with a CI that includes zero. | `synth:37-38` |
| PEAD | Post-earnings-announcement drift: a stock tends to keep moving the way its earnings surprise pointed, for weeks after the report. | The one small-cap stock-picking arm tested was PEAD on a partial list of 200 names. | `coverage:22`, `coverage:35` |
| FOMC | Federal Open Market Committee: the American central bank's interest-rate meetings. | The pre-FOMC Treasury study is the one area tested with evidence against, with caveats. | `synth:25-26`, `coverage:41-46` |
| French data | Free, survivorship-free American stock portfolio returns back to 1926, published by Professor Kenneth French. | It allows long-history tests (1963 and 1927) without paid data. | `synth:38-39` |
| EDGAR | The American Securities and Exchange Commission's public filings database. | The Argon review proposed it, but no EDGAR ingestion exists here. | `consol:35`, `verify:63` |
| BB, RSI, EMA, MACD | Four price-pattern indicators: Bollinger Bands, relative strength index, exponential moving average, moving-average convergence/divergence. | Four of the five voters in the live ensemble. | `code:13-14`, `code:155` |
| vbb | The volume-confirmed Bollinger voter, the fifth voter in the live ensemble. | The 5-voter set is the live default. | `code:14`, `consol:31` |

### Names and codes

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| MNST, AVB | Stock ticker symbols for Monster Beverage and AvalonBay. | The broker never applied MNST's 2-for-1 split (-0.85 pp of the paper shortfall). AVB's price series has been EQR's since 2026-07-20. | `consol:33`, `parity:46-50`, `clean:47` |
| PARA | A ticker whose stored prices belong to a different security for its whole S&P 500 membership. | The production replica traded it 3 times, and its effect is unmeasured. | `clean:37-44` |
| agy | The Antigravity agent harness: a program that runs automated agents. | It ran the Argon review. | `consol:32` |
| Argon | The Gemini 4 Argon model, run through agy. | It wrote one of the two frontier-model reviews. The review carries an UNVERIFIED banner. | `verify:3` |
| Opus | The Claude Opus model. | It wrote the other frontier-model review, the senior-quant direction review. | `consol:40` |
| French portfolio names | Labels for stock portfolios sorted by a trait: profitability `HiOP`, cheapness `HiBM`, low investment `LoINV`, prior return `HiPRIOR`, share issuance `HiNI` and `NegNI` (net buybacks), size `BIG` and `SMALL`. | Used in the factor, momentum and net issuance screens. `BIG` `NegNI` returned t 0.26. | `coverage:305`, `ceiling:242` |
| Alpaca activity codes | About 25 broker account-activity type codes: `DIV` and variants for dividends, `SPLIT`, `MA`, `REORG`, `REO`, `NC`, `SPIN`, `INT`, `CSD`, `CSW`, `JNLC`, `JNLS`, `FEE`, `REG`, `TAF`, `CAT`. | A search of all of them since 2024 found a funding journal and small fees: the broker never credited a dividend. | `parity:21-26`, `parity:291-296` |
| Ticker symbols | Short exchange codes for a listed stock or fund, such as `AAPL`, `VZ` and `HAS` (Hasbro, not a word). Kraken crypto pair codes such as `XBTUSD` use `XBT` for BTC. | About 120 stock tickers and 40 crypto pair codes appear. Each is explained only where a report leans on it. | `CLAUDE.md` |

### Verdict labels

| Term | Plain meaning | Why it matters here | Source |
|---|---|---|---|
| GO, NO-GO | The project's pass and fail labels for a test. | A NO-GO is a result, not a reason to stop the work. | `synth:23`, `consol:155` |
| NO | A measured test reported fail, no edge, underperformance or did not reproduce. | The Verdict Trust check classed 53 NO test verdicts. | `verdicts:181` |
| reliable NO | A NO whose data, code, recomputed number and statistical power all check out. | None of the 53 NO test verdicts qualifies. | `verdicts:191` |
| unreliable NO | A NO from a flawed test: the code check failed and it was judged against the later-voided 1.12 baseline, or its numbers do not recompute. | 2 rows. | `verdicts:43`, `verdicts:191` |
| reliable GO | A GO with a clean tape, checked code, a recomputed number and a pass bar set in advance. | None of the 16 GO test verdicts qualifies. | `verdicts:191` |
| unreliable GO | A GO built on contamination that favours the strategy, or on code or numbers that failed their checks. | 9 rows, including every early "beat SPY" figure. | `verdicts:46`, `verdicts:191` |
| inconclusive | The test could not decide because at least one condition fails or is unknown. It does not mean the test found nothing. | 51 NO rows, 7 GO rows, 3 tests with their own inconclusive verdict and 42 status labels. | `verdicts:44-49`, `synth:60-62` |

## 3. Plain-English glossary

Words that are not abbreviations but sound like they should be. Analogies are labelled and never change the meaning. A dash in Source means the word is common usage.

I have tried to keep each entry shorter than the explanation of why it matters. I failed on a few.

| Word | Plain meaning | Source |
|---|---|---|
| minimum detectable effect | The smallest effect a test would usually catch. If it is bigger than the edge you care about, a NO means "could not see". Analogy: a smoke alarm that trips only on a house fire. | `synth:17-18` |
| walk-forward | Pick settings on a past window, then test them on the next unseen window, and repeat across history. | `synth:13` |
| fold | One pick-then-test slice of a walk-forward run. The lab uses overlapping monthly folds. | `CLAUDE.md` |
| gates | Pass or fail checks inside a walk-forward run, mainly DSR and NDH. A fold that fails them runs on the anchor settings. | `clean:28-34` |
| anchor | The default settings a walk-forward test falls back to when its gates reject the settings it just picked. The anchor leak fitted them on the whole test span. | `synth:39-40`, `synth:283` |
| tape | The `ohlcv` price table in the database that every stock backtest reads. Analogy: the shared fixtures every test loads. | `clean:16-17` |
| replica | The live trader's rules re-run on history, to ask what live would have earned. | `synth:53` |
| shadow | A book that records hypothetical holdings and fills without sending broker orders. | `standard:20-21` |
| sweep (cash) | The live rule that puts idle cash into SPY while keeping a 5% reserve. | `synth:138`, `synth:525` |
| sweep (settings) | Trying a grid of settings to see which works best. | `CLAUDE.md` |
| drawdown | The fall from a peak to a later low, shown as a percentage. | - |
| survivorship | Testing only companies that still exist today, so the failures are missing and results look better. Analogy: reviewing only the services that are still up. | `verdicts:1078` |
| lookahead | A test that uses information not yet available at the decision time. | `clean:93` |
| point-in-time | Using only what was known on each date. See PIT. | `synth:14-15` |
| regime halt | The walk-forward circuit breaker standing a strategy down. It triggers when the trailing 4-window average WFE is below 0.25 or after two straight negative OOS Sharpes. | `verdicts:751-752` |
| power | The chance that a test catches a real effect of a given size. These reports use 80%. | `synth:17-18` |
| deflated | Marked down for how many ideas were tried. See DSR. | `synth:15-16` |
| hurdle | A minimum result an idea must clear, such as the kill-only replay's -1.0 pp/yr after tax. | `standard:80-86` |
| tracking error | See TE. How far a book strays from its benchmark. | `synth:21-22` |
| holdout | Data kept aside and not used to choose settings, then used for a final check. The replica's 2010-2020 holdout is not citable because of bad prices. | `accuracy:32-37` |
| pre-registered | The pass bar was written down and committed before the result was seen. | `accuracy:11` |
| falsifier | A result, written down before the evidence is read, that would show the working view wrong. | `crypto:72-73` |
| artifact | A saved raw output file, usually a data file, behind a reported number. A "broker artifact" is a side effect of how the broker works, not a real result. | `accuracy:5`, `parity:44-45` |
| bar (price) | One period of price data: open, high, low, close and volume. Here usually one trading day. | - |
| bar (pass) | The rule, set in advance, that decides GO or NO-GO. | `accuracy:11` |
| partial bar | Today's price bar before the market closes. Live signals at 12:45 PT use it. Backtests use the final close. | `clean:193-195`, `code:34-39` |
| sleeve | The part of the account that runs one strategy, such as the 3-stock sleeve beside the SPY holding. | - |
| arm | One variant inside a test, such as the 1.3x or 2.0x leverage arm. | `ceiling:210-214` |
| ensemble | The live strategy: five voters, entering when enough of them agree. | `code:155` |
| voter | One indicator rule inside the ensemble that votes to buy or not. | `code:155` |
| universe | The list of stocks a test may pick from. | - |
| exposure | How much of the account sits in a given thing. The replica held 9.37% stocks and about 85% SPY. | `power:50` |
| leverage | Holding more market exposure than your cash, by borrowing or through a fund like SSO. 1.3x means 1.3 dollars of market per dollar. | - |
| contamination | Anything in the data or code that tilts a result, such as leaks, wrong prices or survivorship. | `clean:1` |
| accrual (dividend) | A dividend the trader books as owed before the broker pays it. Here the broker never paid, so the paper account carries $320.24 that is not cash. | `parity:15-22` |
| accrual (anomaly) | A stock-return pattern linked to how much of a firm's earnings are accounting accruals rather than cash. Not tested, only feasibility-checked. | `verdicts:247` |
| parity | Live matches the test: same decisions, same day, same prices. | `parity:1` |
| frozen | No configuration changes. The paper trader has been frozen since 2026-09-30. | `parity:177` |
| carry | Return from holding a position rather than from price moves, such as funding payments on a crypto futures position. | `crypto:31-33` |
| total return | Price change plus reinvested dividends. See TR. | `synth:11` |
| test family | A group of related tests counted together when correcting for many tries. The 72 measured tests fall in 11 families. | `synth:328` |
| sensitivity | How a result changes when an input changes. The sensitivity table shows each result at two significance levels. | `sensitivity:13-17` |
| reversion | Betting that a stock that fell will move back toward its recent average. The live ensemble is a reversion strategy. | `coverage:53` |
| seam | A jump in stored prices where two downloads with different dividend adjustments meet, making a false one-day move. | `clean:42-45` |
| PIT core | The S&P 500 stock-selection strategy run walk-forward with point-in-time membership. Sharpe 0.59 against SPY's 0.78. | `synth:520` |
| net issuance | How many new shares a company issued or bought back. The tested long side was net buyers. | `ceiling:242` |
| cash reserve | The 5% of the account the live trader keeps in cash. It costs about 0.9 pp/yr relative to SPY. | `synth:56` |
| catastrophe stop | A forced sale when a holding falls 25%. Live fired it once, on 2026-09-25. | `code:247`, `parity:333` |
| idle cash | Cash not invested. Before the SPY sweep it cost 2.46 pp of the 3.76 pp paper shortfall. | `parity:38-41` |
| multiple testing | Testing many ideas raises the chance that one passes by luck. Under zero edge, 72 tests at t of at least 2 give about 1.6 false wins. | `verdicts:96-101` |
| factor | A stock trait, such as value or profitability, that is tested for earning a return. | - |
| event study | Measuring returns around a dated event, such as an FOMC meeting. | `coverage:41-46` |
| tax lot | A record of one purchase: quantity, cost and acquisition date. TaxLotBook is the project's library that holds them. | `standard:23-24`, `verify:30` |
| status label | A verdict row that is not an edge test, such as feasibility, design, literature or operations. The Verdict Trust check counts 42 of them. | `verdicts:181` |

## 4. The nine checks

The investigation ran nine separate checks. The reports are in the `lenses/` folder. The file name is the only place the numbers appear. Each verdict is that report's own conclusion.

| Name | The question in one line | File | Verdict from the report |
|---|---|---|---|
| Backtest Cleanliness check | What contamination is left in the prices and code after the five known fixes? | `2026-11-04-lens1-backtest-validity.md` | The five known bugs are fixed, but PARA is the wrong security, stored prices mix adjustment bases and AVB tracks EQR. Of 144 rows, 55 are contaminated, 18 clean and 71 unknown. `clean:21-47` |
| Verdict Trust check | Which NO-GO and GO verdicts can be trusted, and for what size of effect? | `2026-11-04-lens2-verdict-audit.md` and `2026-11-04-lens2-sensitivity-spec.md` | None of the 53 NO test verdicts is a reliable NO and none of the 16 GO test verdicts is a reliable GO. That does not mean an edge was missed. The second file fixes the sensitivity table before any result was opened. `verdicts:29-35`, `sensitivity:1-20` |
| Paper-versus-Backtest check | Does the live paper trader match what the backtests measured? | `2026-11-04-lens3-live-parity.md` | Paper trailed SPY by 3.76 pp since 2026-06-24. Idle cash before the SPY sweep is -2.46 pp. The snapshot gap is dividend accrual the broker never paid. `parity:15-45` |
| Ceiling check | Can the target be reached at all, and what is the measured maximum? | `2026-11-04-lens4-feasibility-and-ceiling.md` | The target needs 9.57 pp/yr over 2021-2026 and 3.54 over 2006-2026. Only 2x leverage (+4.15 pp/yr, -84.8% worst fall) reached its gap, and that is exposure, not skill. No result is graded fully reliable. `ceiling:35-46`, `ceiling:57-72` |
| Coverage check | What was never tested? | `2026-11-04-lens5-coverage-gaps.md` | The record is narrow: 44 of 51 single-stock tests are large-cap only and 65 of 72 are long-only. Of 28 areas, 6 were never tested, 4 tested badly, 17 inconclusive and 1 shows evidence against. 62 of 82 backlog ideas never ran. `coverage:30-58` |
| Code Correctness check | Does the code do what the spec says, and does live run what was tested? | `2026-11-04-lens6-implementation-correctness.md` | The arithmetic is right for the four rules replayed (0 mismatches), but the live trader does not run the configuration the backtests measured. `code:15-50` |
| Report Accuracy check | Are the reported numbers true, do the sources say what the reports say, and were pass bars set first? | `2026-11-04-lens7-research-accuracy.md` | Every headline with a raw file is true. 103 of 144 rows have no raw file. 11 of 14 citations checked say what the reports say. `accuracy:20-29` |
| Crypto check | What was tried in crypto, what does the evidence show and what is unknown? | `2026-11-04-lens8-crypto.md` | No crypto result beat buy-and-hold BTC over any three-year window. The tape has crypto-specific defects: 43.7% of Binance.US 4-hour bars copy Kraken. `crypto:31-60` |
| Test Power check | Could the tests even see an effect? | `2026-11-04-lens9-power-and-regime.md` | Only 8 of 144 rows have the inputs for an MDE. The replica could detect 4.24 pp/yr at book level, about 45 pp/yr at stock-sleeve level. `power:23-54` |

## 5. The three rounds

Three rounds, then the write-up that closed the investigation. Nothing on the Optional Re-run List was run, by the owner's decision, which keeps its running cost at zero.

| Name | What it was | File | Result |
|---|---|---|---|
| Early Trust Audit | A single unreviewed senior-quant audit of the stock results, stocks only. It wrote seven hypotheses, H1-H7, that later checks tested. | `findings/2026-11-04-phase0-trust-audit.md` | No real stock-picking edge over SPY was found. `audit0:9-17` |
| Main Evidence Round | An inventory of 144 results, then the nine checks above, reported in one checkpoint. | `findings/2026-11-04-phase1-checkpoint.md` | Coverage, status of each check and findings carried forward. `checkpoint:1-30` |
| Optional Re-run List | 29 numbered items, `P2-01` to `P2-29`, ordered cheapest first. An item marked * writes to the price cache and needs the no-writes rule. | `findings/2026-11-04-phase1-checkpoint.md` (items in `checkpoint:280-312`) | None run, by the owner's decision of 2026-11-05. `synth:7-8` |
| Synthesis and Independent Challenge | The synthesis of the nine checks, then a challenger that argued the opposite case and checked 40 claims. | `findings/2026-11-04-root-cause-investigation.md` and `findings/2026-11-04-root-cause-challenge.md` | Of 40 claims: 20 confirmed, 16 confirmed with a caveat, 3 wrong, 1 not checked. Every objection is resolved in the synthesis. `synth:746-751` |

Three review files came afterwards and sit in `reviews/`: the Opus review, the Argon review and a verification of the Argon review. A fourth file consolidates the two reviews. All four are in the Source key.

## 6. Other labels

The reading reports use the names on the right. The original files use the codes on the left.

### Named checkpoints, stages and decisions

| Label | Name used in the reading reports | Meaning | Source |
|---|---|---|---|
| Gate A | the leverage-reach checkpoint | If `P2-04` puts 2.0x leverage at today-like funding below the 2006-2026 gap, record that no tested exposure level reaches 2x at current rates. | `opus:191-192` |
| Gate B | the partial-bar checkpoint | If `P2-14` finds partial-bar and final-bar decisions differ on more than a handful of the 30 sessions, the parity proposal must pick one bar for live and replica. | `opus:192-194` |
| Gate C | the long-history-test checkpoint | A Part A fail closes the idea. A pass moves it to B0 only. | `opus:200-201` |
| Gate D | the concentration checkpoint | After `P2-06` (concentrated big-cap momentum after 1994): below +2 pp/yr, concentration needs no further work. | `opus:202-203` |
| Part A | the long-history test | Does the idea survive long free history? The standard's pass bar: Holm t of at least 3 and at least +1.0 pp/yr net of costs since 1994 (not after tax). The Argon review restates it as after tax. | `consol:28`, `standard:47-55`, `argon:98` |
| B0 | the kill-only replay | Runs once on 2021-2026. If the after-tax CAGR minus SPY's falls below -1.0 pp/yr, the idea is killed. A pass proves nothing. | `standard:65-83` |
| B1 | the forward shadow test | A forward shadow book run for at least 18 months. | `standard:88` |
| Decision 0 | the no-writes rule | The owner's permission for re-runs to write to the shared price cache. Answered no. It blocks every `P2` item marked *. | `opus:16-17`, `checkpoint:827-834` |

There is no Part B in the sources. The standard replaced the old paid Part B with B0 and B1.

### Owner questions, account eras and grades

| Label | Name used in the reading reports | Meaning | Source |
|---|---|---|---|
| Q1-Q7 | the owner's seven questions | Edge over SPY, what limits live return, the measured maximum, reliability of each NO-GO, crypto, what holds up, what is still unknown. | `synth:105-221` |
| E1-E7 | account eras | Paper account stretches split at each configuration change: core ensemble (06-24 to 07-15), accidental dry run (to 07-30), 3-sleeve blend (to 08-21), blend plus SPY sweep (to 09-22), S&P 500 core plus sweep (to 09-28), missed-exit catch-up (to 09-30), frozen (to 11-04). | `parity:171-180` |
| Tier A, B, C | soundness grades | A: reliable. B: the number recomputes on clean data, but nothing proves it is statistically real. C: everything else, listed but not used for the ceiling. | `ceiling:194-201`, `synth:40-41` |
| L4-X1, L4-X2 | the two Ceiling check exceptions | X1: fixed-weight books whose code was never examined. X2: non-registered arms and sub-windows. | `ceiling:205`, `ceiling:212` |
| R1-R7 | the Verdict Trust check's seven classification rules | Fixed before any row was classified: verdict type, data condition, code condition, recompute, power, classes, rules-out size. | `verdicts:179-193` |
| A1-A5, B1-B6, C1-C7 | ranked root-cause items | A: method and process errors in the research. B: implementation and live-parity errors. C: the evidence is negative or the target is structurally hard. | `synth:278`, `synth:349`, `synth:420` |
| F1, F2 and N1, N2 | falsifiers | Each check wrote its own set before reading evidence. The same code means different things in different reports. | `coverage:556` |
| H1-H7 | the Early Trust Audit's seven hypotheses | Stock-trust hypotheses tested by the later checks. | `synth:476` |
| Options 1-3 | the consolidated review's three options | 1: paper-trader measurement and parity. 2: funding-honest leverage figures for the 2027-09-30 A23 review. 3: one pre-registered long-history stock screen (accruals). | `consol:152-196` |
| Decisions 1-12 | the owner decision list | One list in the consolidated review: accept the synthesis, changelog entry, any re-run item, research copy of the price table, input archive, daily-loss halt base, storage for downloads, yearly research budget, leverage figures, one benchmark, sweep tax churn, reopening closed NO-GOs. | `consol:217-232` |
| STATUS, TEST-INCONCLUSIVE | verdict types | STATUS is a feasibility, design or operations label. TEST-INCONCLUSIVE is a test whose own verdict is inconclusive or invalid. | `verdicts:181` |
| C, C*, W, NC | challenge result codes | Confirmed, confirmed with caveat, wrong, not checked. | `challenge:63-64` |
| Rank 1-5, rec N | recommendation ranks | The Argon review ranks five directions, and the Opus review numbers its recommendations. | `argon:64-113`, `consol:156` |
| 30-day, 90-day, 180-day gate | Argon's day checkpoints | Argon's own checkpoints, unrelated to the Opus Gates A-D. | `argon:128-140` |

### Labels that mean two things

| Label | First meaning | Second meaning |
|---|---|---|
| B1 | Root-cause item: live does not run the configuration any backtest measured | The forward shadow test |
| A1, A2 | Root-cause items | Sub-rows in the Verdict Trust check, and research-candidate headings in the Coverage check |
| H1-H4 | Hypotheses from the Early Trust Audit | Hypotheses in the factor screen |
| F1 | A falsifier in one check | A different falsifier in each other check |
| Q1 | An owner question | A calendar quarter, as in "2024 Q1" |
| NC | Not checked (challenge result code) | An Alpaca activity code |
| gate | The Opus checkpoints, Gates A to D | The Argon day checkpoints, the lab's DSR and NDH gates, and the "stop gate" conditions in the consolidated review |
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
| `standard` | `research/programme/evidence-standard/2026-10-23-forward-paper-standard.md` |
