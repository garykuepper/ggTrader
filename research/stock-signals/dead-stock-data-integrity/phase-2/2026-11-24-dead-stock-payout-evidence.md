# Dead-stock payout evidence: what the SEC's filings say about 141 stocks that stopped trading

**Date:** 2026-11-24 (version 2: rewritten after a second matching pass and four independent reviews)
**Study:** `research/stock-signals/dead-stock-data-integrity` (an audit; the brief was frozen on 2026-11-24 before any real run, and one dated amendment, the second matching pass, was added after the first run; no strategy was rerun)
**Question:** For the 141 dead stocks the first test held in its main stock list, what do the SEC's public filings say about why each one stopped trading, and which loss figures could a later rerun of the first test use for each stock?
**Earlier note:** the first half of this study, a scan of our own price rows, is [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md) (section 6 gives its headline).

## 1. In short

- **The first test could not tell why its dead stocks stopped trading.** A dead stock is one that stopped trading before the end of the data. The first test's answer swung from far behind SPY (the S&P 500 fund) to ahead of it depending on a guess: did holders get nothing back, half, or the full last price? This study looked for the SEC's own record of each stock's end.
- **For 102 of the 141 stocks the filings report an acquisition-type event.** Within 90 days of the stock's last trading day the company filed a report with the SEC carrying at least one of three item numbers the SEC uses for an acquisition, a change of control, or a change in shareholder rights. That is 85 of the 117 stocks in the development window (2005 to 2019) and 17 of the 24 in the holdout window (2020 to 2024). Of the 102, 88 rest on a notice we could read and check and 14 on a company-name match alone (section 3).
- **The strength of that evidence varies.** 74 of the 102 carry the SEC's acquisition-completed item (2.01); 28 do not, and 12 of those rest on a single weak signal. Section 4 lists them apart.
- **For the other 39 the filings tell us nothing:** 26 had no matching notice by company name, 8 have no company name in our data, and 5 matched a company but carried no acquisition-type item.
- **No bankruptcy turned up among the 141.** None of the 107 stocks whose filings we read carried the SEC's bankruptcy item. That agrees with the earlier finding that none of the 141 fell more than about 7 percent in its last month. It is not proof that no holder lost money (section 8 says why the search is narrow).
- **The suggested loss is 0.0 for the 102 and no figure for the other 39.** 0.0 means holders got the full last price. It is the brief's assumed scenario for acquisitions. It is plausible for cash and stock deals alike, because the last price before a deal closes already tracks the deal, but the filings give no amount and no form of payment. The two published substitute losses (30 and 55 percent, from the research literature) were available for bankruptcies but were not needed, because no bankruptcy was found.
- **This does not make the first test's result better than it already was.** Its best case for the rule was already the 0.0 reading: 2.0 points a year behind SPY in 2005 to 2019 and 1.6 points ahead in 2020 to 2024. Zero loss was the best previously tested uniform scenario; the study does not measure per-stock payouts. This study computed no return and reran nothing.
- **A second matching pass is a lead, not a confirmation.** After the first run 37 named stocks had no match. A revised name-cleaning rule (called v1.1, written after seeing that run) recovered 11 of them. They carry filings that look right, but their identity was not confirmed (section 3.3 prints all 11).
- **Nothing here is a GO or NO-GO label.** Paper trading comes before any real money, and the owner decides.

## 2. What was done, and what was not

**What was done.**

- **The list.** The 141 stocks are the dead stocks that the first test's main stock list held when they died: 117 in its development window (2005 to 2019, the period used while the rule was built) and 24 in its holdout window (2020 to 2024, the later period run once). The first test's own result files name them. Company names come from McGill, a table in our own database that gives each stock a name (cut off at 16 characters), and the price rows come from CRSP, a university database of US stock prices that keeps stocks that later stopped trading.
- **The match to a company.** For each stock with a name (133 of 141), we looked for a Form 25, the notice that a stock is being taken off an exchange, filed under a company name matching the stock's name, in the window from 30 days before its last trading day to 180 days after.
- **The filings.** The SEC (the US securities regulator) keeps every company filing on its free public site, EDGAR. We read 80 quarterly index files, from 2005 quarter 2 to 2025 quarter 1, with none skipped (the 180-day window after a July 2024 exit reaches January 2025; the brief's "nothing later than 2024-12-31" applies to price data, not to filings). Then, for each matched stock, we read its list of 8-K reports, the report a company files when something material happens, within 90 days either side of the last trading day.
- **How we asked.** A contact line supplied by the owner was sent to the SEC with every request, as the SEC requires. It is not written in any file of this study.
- **What it cost the SEC site.** The first run made 357 downloads. The later runs used saved copies: the clean rerun made 2 new downloads and the run with the second matching pass made 35 (24 plus 11). In no run did an 8-K list download fail, and no 8-K had a missing item list.

**What was not done.**

- No strategy was rerun and no return or comparison with SPY was computed. The rerun is a separate plan, for the owner to ask for.
- No payout amount was read from a deal document. We read the labels the SEC puts on each filing, not the filings' text.
- The Yahoo side of our price data was not scanned, and most of the first test's wider stress list was not covered (section 9).

## 3. Results

### 3.1 How many stocks have evidence, by group and window

Each stock falls in one group. The table counts all 141 (from `_payouts.csv`, recounted in section 12 of [`_hand_checks.txt`](../_hand_checks.txt)). The short names in the first column are the labels used in the data files; the second column says what each means.

| Label in the files | What it means | Development (117) | Holdout (24) | All (141) | Notice read and checked | Name match only |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `acquired_or_merged` | the filings report an acquisition, merger or change of control | 85 | 17 | 102 | 88 | 14 |
| `no_8k_found` | matched a company, but no acquisition, bankruptcy or delisting item found | 5 | 0 | 5 | 2 | 3 |
| `no_filing_match` | no Form 25 notice found under an equal name | 21 | 5 | 26 | 0 | 0 |
| `name_unknown` | no company name in our data to match | 6 | 2 | 8 | 0 | 0 |
| `bankruptcy_filed` | the filings report a bankruptcy | 0 | 0 | 0 | 0 | 0 |
| `delisted_other` | only a delisting notice | 0 | 0 | 0 | 0 | 0 |
| `ambiguous_match` | the name matched more than one company | 0 | 0 | 0 | 0 | 0 |

- **"Notice read and checked"** means we opened the Form 25 notice and confirmed that the company named in it as the issuer is the one we matched and that the security named is common or ordinary shares. **"Name match only"** means the company was found by name but its notice was not readable (6 stocks) or the match came from the second pass and its identity was not otherwise verified (11 stocks); section 3.2 and 3.3 explain each.
- **Share with evidence.** 102 of 141 is 72 percent; 85 of 117 is 73 percent in the development window; 17 of 24 is 71 percent in the holdout.
- **By exchange.** Of the 102, 48 traded on the New York Stock Exchange and 54 on Nasdaq. Of the 141, 65 were on the first, 68 on the second, and 8 (the unnamed ones) have no exchange recorded.
- **Against the price label.** The earlier note labelled each stock's last month of prices. Of the 102, 78 were labelled "calm like a deal" and 24 "unclear". Of the 34 "unclear" stocks, 24 have acquisition evidence and 10 do not. No stock was labelled as a collapse.

### 3.2 How many matched a Form 25 filing, and how sure we are

| Result | Stocks |
| :--- | ---: |
| Stocks with a company name | 133 |
| Matched by the first rule (v1): notice read and checked | 90 |
| Matched by the first rule (v1): notice could not be read | 6 |
| Matched only by the second pass (v1.1): notice read and checked, identity a lead | 11 |
| Found only a notice for a non-common security (Alpharma, cut off as `ALPHARMA INC-A`) | 1 |
| Found no matching filing | 25 |
| More than one company matched, or a different company's notice | 0 |
| Stocks with no company name | 8 |

- **The match rate is 107 of 133 named stocks, 80 percent** (96 by the first rule, 72 percent, and 11 more from the second pass).
- **How the 90 checked first-rule matches were made.** 76 had names that were equal after cleaning (capitals, punctuation and endings such as INC and CORP removed, and a fixed list of short forms such as CONTL for CONTINENTAL spelled out). 14 matched by the 16-character rule: McGill cuts company names off at 16 characters, so a name of exactly 16 characters was allowed to match a company whose cleaned name starts with it (for example `AMYLIN PHARMACEU`). 29 roster names are exactly 16 characters long, and 26 of them matched in the first pass.
- **What the check found.** All 90 passed: the issuer in each notice is the matched company and the security is common. No notice was rejected for naming a different issuer.
- **The 6 unreadable matches.** All six (Titan, Premcor, Spinnaker, PacifiCare, Vintage Petroleum, Burlington Resources) are exits from 2005 and 2006, and the notice's file was not found at the expected address (the accession numbers, the SEC's number for one filing, all begin 9999999997). Three carry acquisition evidence (Titan, PacifiCare, Burlington) and three carry none (Premcor, Spinnaker, Vintage).
- **A reproducibility note.** The first run had 89 checked matches and 7 unreadable ones. The seventh was Momenta (2020): the SEC server failed to deliver its filing page after three tries. On the clean rerun the page was delivered, the notice was read, and Momenta became a checked match. So "could not be read" can be a passing server failure; the six 2005 and 2006 cases gave the same "not found" answer in every run, which points to a missing file.

### 3.3 The second pass, rule v1.1: 11 more stocks, all leads

**What happened.** The first run left 37 named stocks without a Form 25 match. After seeing that, the study's lead wrote a revised name-cleaning rule, v1.1, and added it to the brief as a dated amendment. So v1.1 was written **after** seeing the first run and is **not** a pre-registered rule. It is applied only to stocks the first rule left unmatched, uses the same filing window, the same check that the notice's issuer is the matched company and the same common-share check, and never changes a first-rule result. Every stock matched only by v1.1 is labelled so in every output file. It fixes four generic defects in how names are cleaned, the same way for our names and the SEC's:

1. **State tags.** The SEC often adds a tag such as `/DE/` or `/NY` (a state) to a company name; v1.1 strips it. Example: our `KERR-MCGEE` against the SEC's `KERR MCGEE CORP /DE`.
2. **Share-class marks.** A trailing `-A`, `-B`, `CL A` or `CLASS A` is dropped. `ALPHARMA INC-A` then found a company, but only a notice for a non-common security, so it did not count.
3. **Plurals.** A fixed list of twelve plural words is folded to the singular, for example `TECHNOLOGIES` to `TECHNOLOGY`, so our `ESTERLINE TECH` matches the SEC's `ESTERLINE TECHNOLOGIES CORP`.
4. **Cut-off endings.** A 16-character name is allowed to match the SEC's full name before its ending is dropped, so `ATHLON ENERGY IN` matches `Athlon Energy Inc.`. (HCC Insurance, `HCC INS HLDGS IN`, was already matched in the first pass.)

**Result.** It recovered 11 stocks. They are **leads, not confirmations**: the notice itself was read and its issuer is the company found, but whether that company is the stock in our data rests on the name alone. All 11 are development-window stocks. Here is every one:

| Our name (CRSP/McGill) | Last trading day | Company the notice was filed under | Form 25 signed | Nearest acquisition-type 8-K (days from last trading day) | Acquisition items found |
| :--- | :--- | :--- | :--- | ---: | :--- |
| KERR-MCGEE | 2006-08-10 | KERR MCGEE CORP /DE | 2006-08-16 | -45 | 3.03 |
| RSA SECURITY | 2006-09-15 | RSA SECURITY INC/DE/ | 2006-09-15 | -77 | 3.03, 5.01 |
| MEDIMMUNE INC | 2007-06-18 | MEDIMMUNE INC /DE | 2007-06-18 | 0 | 3.03, 5.01 |
| ANHEUSER BUSCH | 2008-11-17 | ANHEUSER-BUSCH COMPANIES, INC. | 2008-11-26 | +7 | 3.03, 5.01 |
| CONTL AIRLINES | 2010-09-30 | CONTINENTAL AIRLINES INC /DE/ | 2010-10-04 | +1 | 3.03, 5.01 |
| EL PASO CO | 2012-05-24 | EL PASO CORP/DE | 2012-05-25 | 0 | 2.01, 3.03, 5.01 |
| MCMORAN EXPLOR | 2013-06-03 | MCMORAN EXPLORATION CO /DE/ | 2013-06-04 | +1 | 2.01, 3.03, 5.01 |
| ATHLON ENERGY IN | 2014-11-12 | Athlon Energy Inc. | 2014-11-14 | +1 | 2.01, 3.03, 5.01 |
| INTL RECTIFIER | 2015-01-13 | INTERNATIONAL RECTIFIER CORP /DE/ | 2015-01-15 | 0 | 3.03, 5.01 |
| CABLEVISION SYS | 2016-06-21 | CABLEVISION SYSTEMS CORP /NY | 2016-06-21 | 0 | 2.01, 3.03, 5.01 |
| ESTERLINE TECH | 2019-03-13 | ESTERLINE TECHNOLOGIES CORP | 2019-03-14 | +1 | 2.01, 3.03, 5.01 |

"Form 25 signed" is the date on the notice itself (the files do not keep the date the SEC received it). A minus sign means the 8-K was filed before the last trading day, a plus sign after.

To the eye these pairs look like well-known takeovers. That is the author's general knowledge, not evidence in the files, and it is why they are labelled leads.

### 3.4 The 39 stocks with no evidence

- **26 stocks with no matching filing** (21 development, 5 holdout). The recorded reason for all of them is "no equal name found". 9 of the 26 names are 10 characters or shorter and 8 are a single word (`CEPHEI`, `DADE`, `FOUNDATION`, `GBT`, `JUNO`, `LYONDELL`, `PRINCIPIA`, `SPARK`); 2 are exactly 16 characters. Many are short McGill names that the SEC would register under a longer legal name. Matching by ticker or CUSIP (the 9-character security code) was not tried: EDGAR has no free lookup by CUSIP, and a ticker lookup would need another source. One of the 26, Alpharma, did match a company, but only a notice for a non-common security.
- **8 stocks with no company name** (6 development, 2 holdout). They could be named from another source; that was not done.
- **5 stocks that matched a company but carried no acquisition-type item** (all development). Three (Premcor, Spinnaker, Vintage Petroleum) are the 2005 to 2006 cases with an unreadable notice from the SEC's early electronic period (accession numbers beginning 9999999997). Two (Maverick Tube and Pharmion) have checked notices filed under the rule cited as `17 CFR 240.12d2-2(a)(3)`. That citation is the only thing the file prints about the rule; the study's lead understands it to cover a stock that ceased to exist through a merger, consolidation or cash payment, but its wording was not checked here. It is a second, weaker kind of evidence and was not used in the class label. It also does not set these two apart: every one of the 101 readable notices cites the same rule. Their 8-Ks in the window carried other items (such as 1.01, 2.02, 5.02, 8.01).

## 4. How strong the evidence is, and what it does not show

**What the label means.** A stock is called `acquired_or_merged` when the company filed an 8-K within 90 days either side of the stock's last trading day that carried at least one of three item numbers. The SEC's own instructions for Form 8-K (checked on 2026-11-24, section 9 of `_hand_checks.txt`) name them:

- 2.01: Completion of Acquisition or Disposition of Assets;
- 3.03: Material Modification to Rights of Security Holders;
- 5.01: Changes in Control of Registrant.

Among the 102: item 5.01 appears for 95 stocks, 3.03 for 89 and 2.01 for 74. 88 of the 102 also carry 3.01, the notice of delisting.

**Strongest and weaker cases, apart.** Counted from the 8-K items in `_payouts.json`:

- **Strongest, 74 stocks:** the item 2.01 appears in at least one 8-K near the exit, and the nearest acquisition-type 8-K is within 30 days of the last trading day. Of the 74, 68 are first-rule matches with a checked notice, 1 (Titan) is a first-rule match with an unreadable notice, and 5 are second-pass leads.
- **Weaker, 28 stocks:** no item 2.01, only 3.03 and/or 5.01. Of the 28, 20 have a checked notice, 2 an unreadable one and 6 are second-pass leads. Examples: Ventana Medical Systems has one 8-K carrying only item 3.03, 28 days before its last trading day; CV Therapeutics has two 8-Ks carrying 3.03 without 5.01 or 2.01, 30 and 56 days before.
- **Weakest, 12 of those 28:** the nearest acquisition-type 8-K is more than 30 days away, or a single 8-K carries a single item 3.03 or 5.01. They are listed here (days from the last trading day, minus = before):

| Company (as in our data) | Last trading day | Items | Days | How matched |
| :--- | :--- | :--- | ---: | :--- |
| PACIFICARE HEALT | 2005-12-20 | 5.01 | +59 | first rule, notice unreadable |
| BURLINGTON RES | 2006-03-31 | 3.03 | -1 | first rule, notice unreadable |
| KERR-MCGEE | 2006-08-10 | 3.03 | -45 | second pass, lead |
| RSA SECURITY | 2006-09-15 | 3.03, 5.01 | -77 | second pass, lead |
| PHELPS DODGE | 2007-03-19 | 5.01 | +4 | checked |
| NEW RIVER PHARMA | 2007-04-19 | 5.01 | +5 | checked |
| WEBEX COMM | 2007-05-25 | 5.01 | +7 | checked |
| CHAPARRAL STEEL | 2007-09-14 | 3.03 | -64 | checked |
| VENTANA MED SYS | 2008-02-19 | 3.03 | -28 | checked |
| GOLDEN TELECOM | 2008-02-28 | 5.01 | -9 | checked |
| APP PHARMA | 2008-09-10 | 5.01 | +6 | checked |
| LONGS DRG STRS | 2008-10-30 | 5.01 | -9 | checked |

PacifiCare is the weakest of all: one 8-K, carrying only item 5.01, filed 59 days after its last trading day, and a name-only match. The other 16 stocks without item 2.01 carry both 3.03 and 5.01 or two 8-Ks; they are listed in section 12 of the hand checks. These groups only grade the 102; they do not change any label or suggested loss.

**What the label does not tell us.**

- **Cash or shares.** A takeover can pay in cash or in shares of the buyer; the filing label does not say which, or how much. For a deal paid in shares, the target's last price tracks the offer, so a loss of 0.0 is a reasonable reading. What the buyer's shares do afterwards is a separate holding decision, not a loss on the target.
- **Not only takeovers.** The SEC's instructions describe item 2.01 as the company or a subsidiary completing the acquisition or the disposition of a significant amount of assets, so it also covers a company selling assets, and a company buying something. Item 3.03 covers any material change to the rights of a class of shares, and 5.01 any change in who controls the company, so either can appear in events that are not a merger. That is why the weaker group matters.
- **Not the same event.** A filing 90 days before the last day could belong to an unrelated event.

**Bankruptcy (0 stocks).** Item 1.03 is the SEC's label for Bankruptcy or Receivership. It appears in none of the 656 8-K reports read for the 107 matched stocks. The rule says bankruptcy wins whenever any filing in the window carries item 1.03, even if another filing carries acquisition items, regardless of order or dates. That is a limit, not a tested finding on real data: if a company filed for bankruptcy and then sold its assets within the window, the label would still read bankruptcy. A test with two separate filings, in both orders, protects the rule in code; no such case exists in this data.

**Agreement with price labels.** The table's cross-check column reads `conflict` only when a bankruptcy stock has a calm label or an acquisition stock has a collapse label. It shows 0 conflicts, 78 agreements (acquisition evidence with a calm last month) and 63 "not comparable". With no collapse label and no bankruptcy anywhere, a conflict could not occur, so the zero says little.

**The note column.** Every row of `_payouts.csv` with a suggested loss now carries "filings report an acquisition or merger event; payment amount and form not verified". The rows with no figure say why there is none.

## 5. The suggested losses

| Group | Exchange | Suggested loss | Basis | Stocks |
| :--- | :--- | ---: | :--- | ---: |
| acquisition, merger or change of control | any | 0.0 | the filings (an assumed scenario, see below) | 102 (85 development, 17 holdout; 88 checked, 14 name match only) |
| bankruptcy | NYSE or AMEX | 0.30 | literature fill | 0 |
| bankruptcy | Nasdaq | 0.55 | literature fill | 0 |
| all other groups | any | none | none | 39 |

- **What a loss is.** It is the share of the last price a holder is assumed to lose when the stock stops trading. 0.0 means none.
- **Why 0.0 is a reasonable reading for acquisitions, and what it is not.** It is the brief's assumed scenario, fixed before any run. In a takeover the last traded price already tracks what the buyer pays, in cash or in shares, so selling at the last price is a close stand-in for being paid out. It is not a measured payout: the filings do not give amounts or form, and a premium paid after the last trade, or a deal that closes below the last price, would move the true figure.
- **The literature fills were not used.** A literature fill is a published average loss that researchers have used when the data does not show what a delisted stock was worth; it is not a measurement of these stocks. 0.30 comes from Shumway (1997), *The Delisting Bias in CRSP Data*, Journal of Finance 52(1), 327-340: page 340 (library PDF page 15) says omitted delisting returns average about -30 percent, and Table I on page 329 (library PDF page 4) gives -41.56 percent for NYSE and AMEX performance delistings where a return exists. 0.55 comes from Shumway and Warther (1999), *The Delisting Bias in CRSP's Nasdaq Data and Its Implications for the Size Effect*, Journal of Finance 54(6), 2361-2379: the abstract on page 2361 (library PDF page 2), equation 3 on page 2370 (page 11) and the conclusion on page 2378 (page 19) give -55 percent for Nasdaq. Both copies in our library are marked "unverified", and the page numbers come from the library's text, not from the printed journals. Their ids and fingerprints are in `_payouts.json` under `literature_fills`.
- **For the other 39 stocks the table gives no figure.** A later rerun would keep all three readings (1.0, 0.5 and 0.0) for them, as the brief says.
- **A tax note.** The first test treated every exit as a taxable sale at the last price. Stock-for-stock deals and some foreign reincorporations are often tax-free exchanges for the holder, so that treatment slightly overcharges tax on those stocks. This gives a direction only; it was not measured.

## 6. Phase 1 headline: our own price rows

The full note is [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md). Its headline, in the note's own numbers: six fixed checks over 20,379,285 daily rows of 9,422 CRSP stocks (2004-01-01 to 2024-12-31) found no duplicate rows, no missing or zero prices and no missing share counts, and flagged 345 one-day spikes that reversed at once, 251 runs of five or more missing trading days and 249 runs of an unchanged price on ten rows in a row, 845 flags in all. Only two of the 845 fall on a day the first test actually held the stock (both spikes, both in the wider list's holdout run), and none fall in the main-list runs; the Yahoo side of our data was not scanned, and a flag is a place to look, not proof of an error.

## 7. What this means for the first test (a question for a later plan, not a number)

- **For 102 of the 141 held dead stocks the filings report an acquisition-type event, and 0.0 (the full last price) is a plausible reading of the payout.** For the other 39 the question stays open.
- **For the 102 the 1.0 and 0.5 readings look too harsh,** though the filings do not state an amount. The first test's brief had registered the 1.0 reading as its headline.
- **The best case for the rule is still the 0.0 reading already in the first report.** At the middle cost (18.3 bps, 0.183 percent of each trade's value, per side) and low tax, development: 6.1 percent a year against SPY's 8.1, so 2.0 points behind. Holdout: 14.1 against 12.5, so 1.6 points ahead, a lead that was gone at the highest cost. Zero loss was the best previously tested uniform scenario; this study does not measure per-stock payouts. A rerun that uses a different figure per stock could differ somewhat from the uniform 0.0 result, for example because of a premium paid after the last trade, tax-free exchanges, or deals paid in shares. This report does not say in which direction.
- **The question for a rerun plan.** Should a rerun apply a loss of 0.0 to the 102 stocks with an acquisition-type event (perhaps the strongest 74 first) and keep all three readings (1.0, 0.5, 0.0) for the other 39? The rerun was not done here, and no number in this section comes from one.

## 8. What these results do not tell you

- **The 141 are a special group.** They are the dead stocks a momentum rule (buy the past year's winners among the 1,000 largest stocks) happened to hold. Such a rule tends to buy takeover targets after a deal is announced, because their prices have risen, and tends not to hold collapsing stocks. So "no bankruptcy found" and "mostly takeovers" are partly built in by how the stocks were chosen. They do not carry over to dead stocks in general, or to rules that buy falling or small stocks.
- **The bankruptcy search is narrow.** It covers only the matched company's own filings and only the 90 days around the last trade. A filing by a parent or successor company, or a bankruptcy filed earlier than 90 days before the last trade, is missed.
- **A name match is not an identity check** for the 17 name-only stocks (11 second-pass leads and 6 unreadable notices).
- **The first test's own limits carry over,** including that CRSP has no trading volume.

## 9. Gaps that remain

- **39 stocks have no evidence:** 26 with no matching filing, 8 with no company name, 5 with a match but no acquisition item.
- **17 stocks rest on a name match alone:** 14 of the 102 and 3 of the 5. Eleven are second-pass leads; six are 2005 and 2006 exits with an unreadable notice.
- **28 of the 102 lack the acquisition-completed item (2.01),** and 12 of those rest on a single weak signal (section 4).
- **Notices that could not be read.** The notices for six of the 2005 to 2006 matches were not found at the expected address, and why was not checked.
- **Items only.** The label reads the SEC's item numbers. No filing text, deal price or payment form was read.
- **No volume.** The CRSP file has no volume, so how much of a position could be sold is unknown, here and in the first test.
- **Yahoo side not scanned.** Our combined price data also holds Yahoo history for linked stocks; none was checked.
- **The stress list is mostly not covered.** The first test's wider stress list held 107 dead stocks in the development window and 35 in the holdout in total. Of those, 24 and 9 are also among the 141 and were covered here; the other 83 and 26 were not looked up.
- **Name rule.** Matching is strict. 26 named stocks found no filing, and some may have one under a name our rules do not connect. Matching by ticker or CUSIP was not tried.

## 10. Decisions made along the way, and what each costs if wrong

- **A prefix rule for 16-character names.** McGill cuts names at 16 characters, so an exact match would miss those. A 16-character name matches a filer whose cleaned name starts with it; two different filers starting with it give "ambiguous", and nothing is picked. Cost if wrong: a wrong company could be matched (none was ambiguous or rejected; 14 of the 90 checked first-rule matches used the rule) or a right one missed.
- **Two confidence levels.** "Notice read and checked" and "name match only". Cost if wrong: a name-only match could attach another company's 8-K to a stock. Seventeen of the 141 depend on it (14 of the 102 and 3 of the 5), so they are counted separately in every table.
- **Seven groups.** The brief had six; "more than one company matched" was added, and non-common-only or rejected notices were folded into "no filing match". A failed download is not an eighth group; it is counted apart (none occurred). Cost if wrong: none seen.
- **The second pass.** Written after the first run to recover obvious naming mismatches, applied only to stocks the first rule left unmatched, and its 11 matches are leads and never counted as confirmed. Cost if wrong: if a lead is the wrong company, 11 of the 102 would carry another company's filings; the table in section 3.3 lets a person check.
- **Who built and checked it.** Antigravity Gemini 4 Neon built several tasks (price labels, integrity checks, the download client, the 8-K reader), Claude Sonnet built the rest, and ChatGPT Codex Sol reviewed every task. A first draft of this report was then read independently by Claude Opus, Codex Sol, Gemini Neon and the study's lead; this version applies their findings. Cost if wrong: a number nobody independent read.
- **The contact line.** A privacy guard blocked the first attempt to use the owner's own contact line in a command. It was not worked around: the owner typed it into a private card and it was set as an environment variable. Nothing in the files holds it. Every commit in this repository carries the owner's normal git identity in its author header; no address is written in this report. Cost: a delay.

## 11. How the run went, how to reproduce it, and the pin

- **The run.** The price-row scan, the Form 25 matching and the payouts ran first on reviewed code. Reviews of the code that writes the result files then found problems, which were fixed with tests (the details are in the technical log at the end). The three phases were then rerun from a clean saved state of the code, once more after the second matching pass was added, and every number in this report comes from that last rerun.
- **What changed between runs.** Run 1: 96 matched (89 checked, 7 unreadable), 91 acquisitions. Run 2, same rules on clean code: 96 matched (90 checked, 6 unreadable), 91 acquisitions, because one SEC download that failed the first time succeeded. Run 3, with the second pass: 107 matched (101 checked, 6 unreadable), 102 acquisitions (88 checked, 14 name-only). The price-row scan was identical in all runs.
- **To reproduce all three phases,** run these from the repository root. You need release 1 and CRSP load 1 (and McGill load 2) in the `ggtrader` database, the contact line in the environment variable `SEC_USER_AGENT`, and the saved SEC files in `/media/thesix/ggTrader-data/sec-delistings` (a rerun without them downloads everything again, about 360 requests). Each command needs `--force` because the result files already exist.

  ```
  .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --phase integrity --release-id 1 --crsp-load 1 --mcgill-load 2 --force
  .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase form25 --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
  .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase payouts --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
  ```

- **The pin.** `PIN.json` fingerprints the code and the results so a later change is caught; it pins the three result files and records the roster and `_payouts.csv` as inputs. It is marked "partly" reproducible because the price scan needs our database and the SEC phases need network access unless the saved copies are present.
- **Report place.** This report keeps the file name that was asked for but sits in the folder `phase-2/`: the study tool accepts only a plainly named report at the top of the study folder, and the earlier note already holds that name.

## 12. Files

- Brief: [2026-11-24-dead-stock-data-integrity-brief.md](../2026-11-24-dead-stock-data-integrity-brief.md) (with its dated amendments, the last of which adds the second pass); earlier note: [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md).
- Results: `_integrity.json`, `_roster.csv`, `_form25_matches.json`, `_payouts.json`, `_payouts.csv` (every one of the 141 stocks, one row each); hand checks: [`_hand_checks.txt`](../_hand_checks.txt) (sections 9 to 13 belong to this report; section 12 is the recount behind this version).
- Code: [`code/`](../code/) (the download client, the Form 25 and 8-K readers, the payout table and the driver for the SEC phases, `code/sec_pilot.py`).

## 13. Words used

- **SEC (Securities and Exchange Commission):** the US regulator. **EDGAR:** its free public site of company filings. **Master index:** the SEC's quarterly list of every filing, with company, form type and date.
- **Form 25:** the notice that a stock is being taken off an exchange. **Issuer:** the company whose shares are named in it.
- **8-K and its items:** the report a company files when something material happens. Each report lists item numbers saying what kind: 1.03 bankruptcy or receivership; 2.01 completion of an acquisition or disposition of assets; 3.01 notice of delisting; 3.03 material change to shareholder rights; 5.01 change in control.
- **CIK:** the SEC's number for a filing company. **Accession number:** the SEC's number for one filing. **CUSIP:** a 9-character code that identifies a security.
- **CRSP:** a university database of US stock prices that keeps stocks that later stopped trading. **McGill:** a table in our database that gives each CRSP stock a company name. **Permno:** CRSP's permanent number for one stock, which survives ticker changes.
- **Development window, holdout:** the first test's two periods, 2005 to 2019 (used while the rule was built) and 2020 to 2024 (the later period, run once).
- **Dead stock:** a stock that stopped trading before the end of the data.
- **Evidence class:** which of the seven groups the filings put a stock in (table in section 3.1). **Checked or name-only:** whether the notice was opened and checked, or the stock was matched by name alone. **Lead:** a match from the second pass, whose identity rests on the name.
- **Literature fill:** a published average loss used when the true figure is missing; not a measurement of these stocks.
- **bps (basis points):** hundredths of a percent; 18.3 bps is 0.183 percent of the value of each trade.
- **Provenance:** the record inside a result file of which version of the code, roster and rules produced it.

## 14. Technical log

<details>
<summary>Open for the engineering history of the run (not needed to read the results)</summary>

- **What review found, in order.** (1) The Form 25 phase wrote no provenance block, which the study tool requires; both SEC phases now write one, and the payouts phase refuses a Form 25 file without a complete one. (2) A result file that was a symbolic link could have hidden a changed code file from the "is the working tree clean" test (the working tree is the folder of files under version control); only the three result files, by exact name, are now left out of that test, and both phases refuse to run if an output path is a link. (3) The checks on that block were tightened: a real 40-character commit, a true or false clean flag, exact hashes (a stray new line no longer passes), a full-run marker, and the same roster and short-form list as the code now. (4) The list of fingerprinted code files was extended with the shared parsers of the downloaded indexes and notices and two shared helper files. (5) The Form 25 file and the roster are each read once, so the fingerprint recorded (a SHA-256, a code computed from a file's bytes) is of the bytes actually used. (6) A test helper was moved so the tests also pass in the study tier's second test mode, and the test for this study's `run.py` loads it by path so another study's `run.py` cannot be picked up. The loss-note wording was changed from "holders were paid" to "filings report an acquisition or merger event; payment amount and form not verified". The coverage counts were fixed so the name-only subcounts always add up.
- **Commits.** The price-row scan's recorded head is `0bb88e00`; the Form 25 and payouts phases of the final run recorded head `ba313c7c`, both from a clean tree.
- **Downloads per run.** Run 1: Form 25 261 downloads, payouts 96. Run 2: 2 and 0. Run 3: 24 and 11 (saved copies used: 267 and 96).
- **Numbers that changed between run 2 and run 3.** Matched 96 to 107; checked 90 to 101; name matches counted by rule: first rule 96, second pass 11; acquisitions 91 to 102; unmatched 37 to 25 (plus 1 non-common-only); `no_filing_match` 37 to 26; checked and name-only inside the acquisitions 88 and 3 to 88 and 14; agreements with price labels 70 to 78; windowed 8-K reports 559 to 656 (stocks read 96 to 107).
- **The first run's own history.** `_hand_checks.txt` sections 10 and 11 keep the earlier counts; section 12 holds this version's recount and section 13 the commits and the change list.

</details>
