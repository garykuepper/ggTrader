# Dead-stock payout evidence: what the SEC's own filings say about 141 stocks that stopped trading

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-data-integrity` (an audit; the brief was frozen on 2026-11-24 before any real run; no strategy was rerun)
**Question:** For the 141 dead stocks the first test held in its main stock list, what do the SEC's public filings say about why each one stopped trading, and which loss figures could a later rerun of the first test use for each stock?
**Earlier note:** the first half of this study, a scan of our own price rows, is in [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md) (section 6 gives its headline).

## 1. In short

- **The first test could not tell why its dead stocks stopped trading.** Its answer swung from far behind SPY (the S&P 500 fund) to ahead of it depending on a guess: did holders get nothing back, half, or the full last price? This study looked for the SEC's own record of each stock's end.
- **For 91 of the 141 stocks the filings point to a takeover or merger.** For each of the 91, the company filed a report with the SEC within 90 days of its last trading day that carried at least one of three item numbers the SEC uses for an acquisition, a change in who controls the company, or a change in shareholder rights. That is 74 of the 117 stocks in the development window (2005 to 2019) and 17 of the 24 in the holdout window (2020 to 2024). Of the 91, 88 rest on a notice we could read and check, and 3 on a company-name match alone (section 3).
- **For the other 50 the filings tell us nothing.** 37 found no matching notice by company name, 8 have no company name in our data, and 5 found a notice but no acquisition-type item. The question stays open for these 50.
- **No bankruptcy turned up among the 141.** None of the 96 stocks whose filings we read carried the SEC's bankruptcy item. That agrees with the earlier finding that none of the 141 fell more than about 7 percent in its last month. It is not proof that no holder lost money: 45 stocks have no filings to read at all.
- **The suggested loss is 0.0 for the 91, and no figure for the other 50.** 0.0 means holders got the full last price, the "full last price" reading of the first test. The two published substitute losses (30 percent and 55 percent, from the research literature) were available for bankruptcies but were not needed, because no bankruptcy was found.
- **This cannot make the first test's result better than it already was.** Its best case for the rule was already the 0.0 reading: 2.0 points a year behind SPY in 2005 to 2019 and 1.6 points ahead in 2020 to 2024. This study can only show which stocks support that reading. It computed no return and reran nothing.
- **The filings do not say what holders were paid.** They show that an acquisition-type event happened. They do not say whether holders got cash or shares in the buyer, or how much (section 4).
- **Nothing here is a GO or NO-GO label.** Paper trading comes before any real money, and the owner decides.

## 2. What was done, and what was not

**What was done.**

- **The list.** The 141 stocks are the dead stocks (stocks that stopped trading before the end of the data) that the first test's main stock list held when they died: 117 in the development window and 24 in the holdout window. The first test's own result files name them.
- **The match to a company.** For each stock with a name (133 of 141), we looked for a Form 25, the notice that a stock is being taken off an exchange, filed under a company name matching the stock's name, in the window from 30 days before its last trading day to 180 days after.
- **The filings.** We read 80 quarterly index files of the SEC's filing database (EDGAR), from 2005 quarter 2 to 2024, with none skipped. Then, for each matched stock, we read its list of 8-K reports within 90 days either side of the last trading day.
- **How we asked.** A contact line supplied by the owner was sent to the SEC with every request, as the SEC requires. It is not written anywhere in this study.
- **What it cost the SEC site.** The first run made 261 downloads for the Form 25 phase and 96 for the 8-K phase (one filing list per matched stock), 357 in all. The clean rerun on the final code (section 10) used the saved copies of nearly all of them and made 2 new downloads. In neither run did an 8-K list download fail, and no 8-K had a missing item list.

**What was not done.**

- No strategy was rerun and no return or comparison with SPY was computed. The rerun is a separate plan, for the owner to ask for.
- No payout amount was read from a deal document. We read the labels the SEC puts on each filing, not the filings' text.
- The Yahoo side of our price data was not scanned, and the first test's wider stress list was not covered (section 7).

## 3. Results

### 3.1 How many stocks have evidence, by class and window

The evidence class says what the filings showed for each stock. The table is every one of the 141 stocks, from `_payouts.csv` (counted again in section 11 of [`_hand_checks.txt`](../_hand_checks.txt)).

| Evidence class | Development (117) | Holdout (24) | All (141) | Read from a notice we could check | Matched by name only |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `acquired_or_merged` | 74 | 17 | 91 | 88 | 3 |
| `no_8k_found` | 5 | 0 | 5 | 2 | 3 |
| `no_filing_match` | 32 | 5 | 37 | 0 | 0 |
| `name_unknown` | 6 | 2 | 8 | 0 | 0 |
| `bankruptcy_filed` | 0 | 0 | 0 | 0 | 0 |
| `delisted_other` | 0 | 0 | 0 | 0 | 0 |
| `ambiguous_match` | 0 | 0 | 0 | 0 | 0 |

- **Share with evidence.** 91 of 141 is 65 percent of all the stocks; 74 of 117 is 63 percent in the development window; 17 of 24 is 71 percent in the holdout.
- **By exchange.** Of the 91, 39 traded on the New York Stock Exchange and 52 on Nasdaq. Of the 141, 65 were on the first, 68 on the second, and 8 (the unnamed ones) have no exchange recorded.
- **Against the price label.** The earlier note labelled each stock's last month of prices. Of the 91, 70 were labelled "calm like a deal" and 21 "unclear". Of the 34 "unclear" stocks, 21 have acquisition evidence and 13 have none. No stock was labelled as a collapse.

### 3.2 How many matched a Form 25 filing, and how sure we are

| Result | Stocks |
| :--- | ---: |
| Stocks with a company name | 133 |
| Matched one filing company, notice read and checked (`confirmed_common`) | 90 |
| Matched one filing company by name, notice could not be read (`unconfirmed_unreadable`) | 6 |
| Found no matching filing (`no_name_match`) | 37 |
| More than one company matched (ambiguous), or a different company's notice (issuer mismatch) | 0 |
| Stocks with no company name | 8 |

- **The match rate is 96 of 133 named stocks, 72 percent.**
- **How the 90 checked matches were made.** 76 had names that were equal after cleaning (capitals, punctuation and endings such as INC and CORP removed, and a fixed list of short forms such as CONTL for CONTINENTAL spelled out). 14 matched by the 16-character rule: our name source cuts company names off at 16 characters, so a name of exactly 16 characters was allowed to match a company whose cleaned name starts with it (for example `AMYLIN PHARMACEU`). 29 roster names are exactly 16 characters long, and 26 of them matched.
- **What "checked" means.** We opened the notice itself and confirmed that the company named in it as the issuer is the one we matched and that the security named is common or ordinary shares, not preferred shares, warrants, notes or rights. All 90 passed, and no notice was rejected for naming a different issuer.
- **The 6 name-only matches.** All six (Titan, Premcor, Spinnaker, PacifiCare, Vintage Petroleum, Burlington Resources) are exits from 2005 and 2006, and the notice's file was not found at the expected address (the accession numbers all begin 9999999997). Three carry acquisition evidence (Titan, PacifiCare, Burlington) and three carry none (Premcor, Spinnaker, Vintage).
- **A reproducibility note.** The first run had 89 checked and 7 name-only matches. The seventh was Momenta (2020, a holdout stock): the SEC server failed to deliver its filing page after three tries. In the clean rerun the page was delivered (2 new downloads), the notice was read and checked, and Momenta became a checked match. So "could not be read" can be a passing server failure; the six 2005 and 2006 cases gave the same "not found" answer in both runs, which points to a missing file rather than a failed download. Nothing else changed between the runs: the same 96 of 133 stocks matched, the same 37 did not, and every evidence class kept its count.

### 3.3 The 37 stocks with no matching filing

All 37 have the same recorded reason, `no_name_match`: no filing under an equal name in the window. The data does not say why. What the names look like:

- **3 names are exactly 16 characters long** (`PHILA CONSOL HLD`, `PHARMACEU RESOUR`, `ATHLON ENERGY IN`). The 16-character rule was applied to them, but no company's name began with theirs. Their short forms (PHILA, HLD, RESOUR) are not in the fixed short-form list.
- **10 contain a short form that is in the list** (for example `VALEANT PHARMA`, `CONTL AIRLINES`, `INTL RECTIFIER`, `CABLEVISION SYS`). Spelling the short form out did not produce a name equal to any filer's.
- **8 are a single word** (`DADE`, `LYONDELL`, `CEPHEI`, `JUNO`, `FOUNDATION`, `SPARK`, `PRINCIPIA`, `GBT`), which is hard to match exactly because a filer's registered name usually adds words.
- **16 are other names** (for example `ANHEUSER BUSCH`, `SPRINT NEXTEL`, `KERR-MCGEE`, `UNITED DEFENSE`, `MEDIMMUNE INC`), several of which contain short forms that are not in the list (`VARIAN SEMI`, `FOREST LABS`, `PROMETHEUS BIO`, `SANTANDER CONS`).
- **The exits are spread over 2005 to 2023.**

So a stock in this group is "unmatched" because our name rule found no equal name, not because we showed it had no filing. The brief fixed the rule (equal names after cleaning, plus the 16-character prefix) before any SEC result was seen, and no looser matching was added afterwards.

## 4. What the evidence classes mean, and their limits

**`acquired_or_merged` (91 stocks).** The company filed at least one 8-K, the report a company files when something material happens, within 90 days either side of the stock's last trading day, carrying item 2.01, 3.03 or 5.01. The SEC's own instructions for Form 8-K (checked on 2026-11-24, section 9 of `_hand_checks.txt`) name them:

- 2.01: Completion of Acquisition or Disposition of Assets;
- 3.03: Material Modification to Rights of Security Holders;
- 5.01: Changes in Control of Registrant.

Among the 91: item 5.01 appears for 85 stocks, 3.03 for 78 and 2.01 for 69. 80 of the 91 also carry 3.01, the notice of delisting. 22 of the 91 have no item 2.01, and 11 have just one of the three acquisition items, which is a weaker signal.

What this does **not** tell us:

- **Cash or shares.** A takeover can pay in cash or in shares of the buyer. The filing label does not say which, or how much. The first test's 0.0 reading treats the last traded price as what the holder got. For a deal paid in shares, the holder gets the buyer's shares, which then move on their own.
- **Not only takeovers.** The SEC's instructions describe item 2.01 as the company or a subsidiary completing the acquisition or the disposition of a significant amount of assets, so it also covers a company selling assets, and a company buying something. Item 3.03 covers any material change to the rights of a class of shares, and 5.01 any change in who controls the company, so either can appear in events that are not a merger. The label says "something of this kind happened within 90 days", not "this company was bought".
- **Not the same event.** A filing 90 days before the last day could belong to an unrelated event.

**`no_8k_found` (5 stocks).** A filing company was matched, but none of its 8-Ks in the window carried one of the items above, 1.03 or 3.01. They carried other items (such as 1.01, 2.02, 5.02, 7.01, 8.01 and 9.01). This means "no evidence found", not "no reason exists". Two are checked matches (Maverick Tube and Pharmion) and three are name-only.

**`bankruptcy_filed` (0 stocks).** Item 1.03 is the SEC's label for Bankruptcy or Receivership. It appears in none of the 559 8-K reports read for the 96 matched stocks. The rule says bankruptcy wins whenever any filing in the window carries item 1.03, even if another filing carries acquisition items, regardless of order or dates. That is a limit, not a tested finding on real data: if a company filed for bankruptcy and then sold its assets within the window, the label would still read bankruptcy. A test with two separate filings, in both orders, protects the rule in code; no such case exists in this data.

**What "none found" does and does not mean.** No bankruptcy was found, which fits the earlier price-path finding, but 45 stocks (37 unmatched, 8 unnamed) have no filings read at all. A holder of one of those could have lost money. The finding is limited to the 96 stocks we could read.

**Agreement with price labels.** The table's cross-check column reads `conflict` only when a bankruptcy stock has a calm label or an acquisition stock has a collapse label. It shows 0 conflicts, 70 agreements (acquisition evidence with a calm last month) and 71 "not comparable". With no collapse label and no bankruptcy anywhere, a conflict could not occur, so the zero says little.

**A wording caution.** The note column in `_payouts.csv` says "acquired or merged: holders were paid" for these 91. Read that as "the filings show an acquisition-type item". The files are kept as the run wrote them.

## 5. The suggested losses

| Evidence class | Exchange | Suggested loss | Basis | Stocks |
| :--- | :--- | ---: | :--- | ---: |
| `acquired_or_merged` | any | 0.0 | filing evidence | 91 (74 development, 17 holdout; 88 checked, 3 name-only) |
| `bankruptcy_filed` | NYSE or AMEX | 0.30 | literature fill | 0 |
| `bankruptcy_filed` | Nasdaq | 0.55 | literature fill | 0 |
| all other classes | any | none | none | 50 |

- **What a loss is.** It is the share of the last price a holder is assumed to lose when the stock stops trading. 0.0 means none.
- **The literature fills were not used.** They are average losses that researchers have used when the data does not show what a delisted stock was worth. They are not measurements of these stocks. 0.30 comes from Shumway (1997), *The Delisting Bias in CRSP Data*, Journal of Finance 52(1), 327-340: page 340 (library PDF page 15) says omitted delisting returns average about -30 percent, and Table I on page 329 (library PDF page 4) gives -41.56 percent for NYSE and AMEX performance delistings where a return exists. 0.55 comes from Shumway and Warther (1999), *The Delisting Bias in CRSP's Nasdaq Data and Its Implications for the Size Effect*, Journal of Finance 54(6), 2361-2379: the abstract on page 2361 (library PDF page 2), equation 3 on page 2370 (page 11) and the conclusion on page 2378 (page 19) give -55 percent for Nasdaq. Both copies in our library are marked "unverified", and the page numbers come from the library's text, not from the printed journals. Their ids and fingerprints are in `_payouts.json` under `literature_fills`.
- **For the other 50 stocks the table gives no figure.** A later rerun would keep all three readings (1.0, 0.5 and 0.0) for them, as the brief says.

## 6. Phase 1 headline: our own price rows

The full note is [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md). Its headline, in the note's own numbers: six fixed checks over 20,379,285 daily rows of 9,422 CRSP stocks (2004-01-01 to 2024-12-31) found no duplicate rows, no missing or zero prices and no missing share counts, and flagged 345 one-day spikes that reversed at once, 251 runs of five or more missing trading days and 249 runs of an unchanged price on ten rows in a row, 845 flags in all. Only two of the 845 fall on a day the first test actually held the stock (both spikes, both in the wider list's holdout run), and none fall in the main-list runs; the Yahoo side of our data was not scanned, and a flag is a place to look, not proof of an error.

## 7. What this means for the first test (a question for a later plan, not a number)

- **For 91 of the 141 held dead stocks the filings support reading the payout as the last price,** the 0.0 reading. For the other 50 the question stays open.
- **For the 91 the filings also make the 1.0 and 0.5 readings look too harsh,** though they do not state an amount. The brief of the first test had registered the 1.0 reading as its headline.
- **The best case for the rule is still the 0.0 reading already in the first report.** At the middle cost (18.3 bps, 0.183 percent of each trade's value, per side) and low tax, development: 6.1 percent a year against SPY's 8.1, so 2.0 points behind. Holdout: 14.1 against 12.5, so 1.6 points ahead, a lead that was gone at the highest cost. This study cannot improve on that.
- **The question for a rerun plan.** Should a rerun apply a loss of 0.0 to the 91 supported stocks and keep all three readings (1.0, 0.5, 0.0) for the other 50? The rerun was not done here, and no number in this section comes from one.

## 8. Gaps that remain

- **50 stocks have no evidence:** 37 with no matching filing, 8 with no company name, 5 with a match but no acquisition item.
- **3 of the 91 and 3 of the 5 rest on name-only matches** (all six are 2005 to 2006 exits). The matching company was never confirmed through its notice.
- **Notices that could not be read.** The notices for six of the 2005 to 2006 matches were not found at the expected address, and why was not checked.
- **Items only.** The label reads the SEC's item numbers. No filing text, deal price or payment form was read.
- **No volume.** The CRSP file has no volume, so how much of a position could be sold is unknown, here and in the first test.
- **Yahoo side not scanned.** Our combined price data also holds Yahoo history for linked stocks; none was checked.
- **The stress list is not covered.** The first test's wider list held 107 more dead stocks in the development window and 35 in the holdout (counted from its result files); none was named or looked up here.
- **Name rule.** Exact matching after cleaning is strict. 37 named stocks found no filing, and some of those may have one under a name our fixed rules do not connect.

## 9. Decisions made along the way, and what each costs if wrong

- **Prefix rule for 16-character names (ruling R5).** McGill cuts names at 16 characters, so an exact match would miss those. A 16-character name matches a filer whose cleaned name starts with it; two different filers starting with it give `ambiguous`, and nothing is picked. Cost if wrong: a wrong company could be matched (none was ambiguous or rejected; 14 of 90 checked matches used the rule) or a right one missed (3 of the 37 misses are 16-character names).
- **Two confidence levels (ruling R11).** `confirmed` (notice read, issuer and common shares checked) and `name_only` (unique name match, notice unreadable). Cost if wrong: a name-only match could attach another company's 8-K to a stock. Three of the 91 depend on it, so those three are counted separately in every table.
- **Seven evidence classes (ruling R14).** The brief had six; `ambiguous_match` was added for names matching two or more companies, and non-common-only or rejected notices were folded into `no_filing_match`. Cost if wrong: none seen, since 0 stocks fell into it. A failed download is not an eighth class (ruling R15); it is counted apart (0 occurred).
- **Who built and checked it.** Antigravity Gemini 4 Neon built the price-label, integrity-check, download-client and 8-K tasks, Claude Sonnet the rest, and ChatGPT Codex Sol reviewed every task. The Phase 2 whole-study reviews (Sol and Opus) are still to come. Cost if wrong: a build or number nobody independent read; the reviews are where that is caught.
- **The contact line.** A privacy guard blocked the first attempt to use the owner's own contact line in a command. It was not worked around: the owner typed it into a private card and it was set as an environment variable. Cost: a delay.

## 10. How the run went, and how the study was closed

- **Report name and place.** This report keeps the file name that was asked for, `2026-11-24-dead-stock-payout-evidence.md`, but sits in the folder `phase-2/`. The study tool accepts only names of the form date, study name and an optional `-brief`, `-nogo` or `-amendment` at the top of the study folder, and the earlier note already holds the plain name for 2026-11-24.
- **The first run and what review found.** The three result phases first ran on code that had been reviewed and committed, and the report was first written from that run. Reviewing the code that writes the result files then found problems, fixed one after another with tests:
  1. The Form 25 phase wrote no provenance block (the record of which commit, code files, roster and rules made a file), which the study tool requires. The code now writes the block in both SEC phases, and the second phase refuses a Form 25 file that does not carry a complete one.
  2. A result file that was a symbolic link (a shortcut to another file) could have hidden a changed code file from the "is the working tree clean" test. Only the three result files, by their exact names, are now left out of that test, and both phases refuse to run if an output path is a link.
  3. The checks on that block were too loose. They now need a real 40-character commit, a true or false clean flag, exact hashes (a stray new line no longer passes), a marker that the run was a full one and not a trial, and the same roster and short-form name list as in the code now.
  4. The list of fingerprinted code files left out the shared parsers of the downloaded indexes and notices and two shared helper files; all four were added.
  5. The Form 25 file and the roster file are each read once, so the fingerprint recorded is that of the bytes actually used.
  6. A test helper was moved so the tests also pass in the study tier's second test mode.
  The wording of the loss note in `_payouts.csv` was also changed from "holders were paid" to "filings report an acquisition or merger event; payment amount and form not verified", because the filings show an event, not a payment.
- **The clean rerun.** After the last fix the controller reran all three result phases (the price-row scan, the Form 25 matching and the payouts) from a clean committed tree. Every number in this report comes from those reruns. Compared with the first run: 90 checked and 6 name-only matches (was 89 and 7); of the 91 with a suggested loss, 88 checked and 3 name-only (was 87 and 4); 76 checked matches with equal cleaned names (was 75); the SEC downloads were 2 (was 357). The price-row scan and every evidence class count came out the same.
- **Pin.** `PIN.json` pins `_integrity.json`, `_form25_matches.json` and `_payouts.json` as results, and the roster and `_payouts.csv` as inputs, by SHA-256. Reproducibility is "partly": the price-row scan needs our database, and the SEC phases need network access and a contact line (the saved copies make a rerun cheap and the SEC site may change).

## 11. Files

- Brief: [2026-11-24-dead-stock-data-integrity-brief.md](../2026-11-24-dead-stock-data-integrity-brief.md); earlier note: [2026-11-24-dead-stock-data-integrity.md](../2026-11-24-dead-stock-data-integrity.md).
- Results: `_integrity.json`, `_roster.csv`, `_form25_matches.json`, `_payouts.json`, `_payouts.csv` (every one of the 141 stocks, one row each); hand checks: [`_hand_checks.txt`](../_hand_checks.txt) (sections 9, 10 and 11 belong to this report; section 10 is the first run's count and section 11 the recount on the clean rerun).
- Code: [`code/`](../code/) (the download client, the Form 25 and 8-K readers, the payout table and the driver for the SEC phases, `code/sec_pilot.py`).

## 12. Words used

- **SEC (Securities and Exchange Commission):** the US regulator. **EDGAR:** its free public site of company filings.
- **Form 25:** the notice that a stock is being taken off an exchange. **Issuer:** the company whose shares are named in it.
- **8-K and its items:** the report a company files when something material happens. Each report lists item numbers saying what kind: 1.03 bankruptcy or receivership; 2.01 completion of an acquisition or disposition of assets; 3.01 notice of delisting; 3.03 material change to shareholder rights; 5.01 change in control.
- **CIK:** the SEC's number for a filing company. **Accession number:** the SEC's number for one filing.
- **Master index:** the SEC's quarterly list of every filing, with company, form type and date.
- **Provenance:** the record inside a result file of which commit, code, roster and rules produced it.
- **Evidence class:** which of the seven groups the filings put a stock in. **Confirmed or name-only:** whether the notice was opened and checked, or the stock was matched by name alone.
- **Literature fill:** a published average loss used when the true figure is missing; not a measurement of these stocks.
- **Permno:** CRSP's permanent number for one stock, which survives ticker changes. **CRSP:** a university database of US stock prices that keeps stocks that later stopped trading.
- **Dead stock:** a stock that stopped trading before the end of the data. **Holdout:** the later period (2020 to 2024) the first test's rule was not looked at on until it was run once.
- **bps (basis points):** hundredths of a percent; 18.3 bps is 0.183 percent of the value of each trade.
