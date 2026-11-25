# Dead-stock open cases: a second search for the 39 stocks with no filing evidence

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-data-integrity` (reopened for new files only; the earlier phases' files are untouched)
**Rules written first:** [2026-11-24-open-cases-brief.md](2026-11-24-open-cases-brief.md), committed before any Phase 3 code existed and amended once, before any run, after an independent cold read.
**Question:** The first test held 141 stocks that later stopped trading. An earlier phase of this study found a filing about an acquisition for 102 of them and nothing for the other 39. Can a different way of finding the company (a search by name among every company and every kind of filing in the SEC's quarterly lists, not only among the exchange's removal notices) turn up evidence for the 39?

## 1. In short

- **The count with filing evidence grew by one stock, from 102 to 103 of 141.** The new one is ALPHARMA INC-A (last traded 2008-12-29). The company we matched to it by name filed an 8-K (the report a company files when something important happens) carrying one of the three item numbers the SEC uses for an acquisition, a change of control or a change in shareholder rights. Its identity rests on the name alone (a name-cleaning rule written after the first run of the earlier phase), so it is a lead, not a confirmed match. The filings give no payment amount.
- **Four stocks have an announced merger vote but no completion.** The matched company filed a merger-vote notice (the document shareholders receive before voting on a merger; form code `DEFM14A`) within a year before its last trading day: ABGENIX (2006), VERITAS DGC (2007), SPINNAKER EXPLORATION (2005) and MAVERICK TUBE (2006). They are kept apart (class `deal_forms_only`) and **set no loss figure**. The search only checks that such a form was filed by that company in that window; it does not read which deal the form is about.
- **Two name matches were rejected, and what happened to those two stocks is not settled.** For JAZZ PHARMA (last traded 2012-01-17) and SPRINT NEXTEL (2013-07-10) the rule found a company with the same name that filed an annual or quarterly report more than 180 days after the last trading day, so it rejected the match as a different or surviving company. That may mean the stock simply changed name or legal form and kept existing, or that the name matched a different company; the files do not say which, and we did not check further. The first test treated both as dead stocks.
- **The rest stay open.** 20 of the 26 name searches found no company; the 2 above were rejected; 8 stocks still have no company name in our data; and 4 stocks (PREMCOR, VINTAGE PETROLEUM, PHARMION, AMERICAN NATIONAL INSURANCE) matched a company with neither an acquisition item nor a merger-vote or tender-offer form.
- **No bankruptcy item turned up.**
- **What this means for the rerun of the first test.** The rerun will set a loss figure only for stocks that have one in the table: 103 stocks at 0.0 (0% lost, every holder paid the last price: the brief's assumed scenario, not a measured payment). The other 38 have no figure, including the 4 merger-vote stocks and the 2 rejected matches. They will keep the old loss readings (1.0 = 100% lost, 0.5 = 50%, 0.0 = 0%). Only ALPHARMA INC-A is newly covered compared with the earlier table. The first test's best case (everything at 0.0) is unchanged by construction; the rerun has not been done.

## 2. What was done

- The rules in the brief were applied to the 26 stocks with no exchange removal notice (rule 1: search by company name), the 5 stocks the earlier phase matched to a company but found no acquisition item for (rule 2: the company stays as it was matched, and only the merger-vote search is added), and the 8 stocks with no company name (rule 7).
- The search read the 80 quarterly lists of the SEC (the US securities regulator) that were already saved on disk, from 2005 quarter 2 to 2025 quarter 1: 20,784,033 filing lines, nothing downloaded for them. A name matches when it equals the roster name (our list of the 141 stocks, with names from the McGill data set in our database, cut to 16 characters) after the same name-cleaning the earlier phase used; the name rule v1 is tried first and the later rule v1.1 only when v1 finds nothing.
- A first attempt stopped on one line of the 2016 first quarter with no company name (a quirk in the SEC's list: a fund's filing under a valid company number with a blank name). That is the only such line in the 80 lists. The reader was changed to keep a blank name (and never to match it), and the run was repeated from the start. This was a correction of the reader made after a run stopped; no matching or loss rule was changed after seeing results.
- New downloads: only the 8-K report lists of the 4 stocks matched by name (4 downloads and 4 further reads of saved copies; none failed). The contact line the SEC requires went out only through an environment variable and appears in no file.
- **The 8 stocks with no name (rule 7).** The read-only name lookup used by the stock list (McGill names, and the release-1 table that maps tickers to stock numbers) returns no row at all for their stock numbers (PERMNO, the price database's identifier for a stock: 87283, 75900, 92552, 92182, 93247, 16544, 64899, 16939). No name could be found without guessing, so all 8 stay without one; the SEC lists carry no stock numbers and cannot name them. This probe was run outside the driver; the recorded query is in `_hand_checks.txt` section 14 (two selects, one on `kaggle_chars.mcgill_of(2)` and one on `ref.symbol_map` for release 1, filtered to those eight numbers).

## 3. Results

### Classes before and after

| Class | Before | After |
|---|---:|---:|
| acquired_or_merged (an acquisition-type item found) | 102 | 103 |
| deal_forms_only (merger vote or tender offer announced; no loss figure) | 0 | 4 |
| no_8k_found (matched company, no acquisition item, no deal form) | 5 | 4 |
| no_filing_match (no company found by name, or the match was rejected) | 26 | 22 |
| name_unknown (no company name in our data) | 8 | 8 |
| bankruptcy_filed, delisted_other, ambiguous_match | 0 | 0 |

Moves: SPINNAKER and MAVERICK TUBE from no_8k_found to deal_forms_only; ABGENIX and VERITAS DGC from no_filing_match to deal_forms_only; ALPHARMA INC-A from no_filing_match to acquired_or_merged; AMERICAN NATIONAL INSURANCE from no_filing_match to no_8k_found. Nothing else changed class.

### Every stock matched by name (rule 1)

| Roster name | Company found | Name rule | Last trading day | Outcome |
|---|---|---|---|---|
| ABGENIX INC | ABGENIX INC (company number 1052837) | v1 | 2006-03-31 | merger-vote notice; no loss figure |
| VERITAS DGC INC | VERITAS DGC INC (28866) | v1 | 2007-01-11 | merger-vote notice; no loss figure |
| ALPHARMA INC-A | ALPHARMA INC (730469) | v1.1 | 2008-12-29 | acquisition-type item; loss 0.0 assumed; lead only |
| AMERN NATL INS | AMERICAN NATIONAL INSURANCE CO (904163) | v1 | 2022-05-24 | no acquisition item and no deal form found |
| JAZZ PHARMA | company number 1232524 (name and rule not kept in the output) | n/a | 2012-01-17 | rejected: filed an annual or quarterly report later |
| SPRINT NEXTEL | company number 101830 (name and rule not kept in the output) | n/a | 2013-07-10 | rejected: filed an annual or quarterly report later |

### Stocks the earlier phase already matched (rule 2)

SPINNAKER EXPLORATION and MAVERICK TUBE (the second with a checked notice) now carry a merger-vote notice and no loss figure. PREMCOR, VINTAGE PETROLEUM and PHARMION carry neither an acquisition item nor a deal form and stay at no_8k_found. None of the five changed company or confidence.

### What the filing lines show (controller hand check)

These are the forms on each company's own lines in the SEC list, by exact code and date; they show that the forms exist, not that they concern these stocks' deals.

| Roster name | Merger-vote or tender forms | Later removal-type forms |
|---|---|---|
| ABGENIX | `DEFM14A` 2006-02-09 | `15-12G` (ends the duty to report) 2006-04-03 |
| VERITAS DGC | `DEFM14A` 2006-12-01 | `15-12B` 2007-01-12; `25-NSE` (the exchange's removal notice) 2007-01-30 |
| SPINNAKER | `DEFM14A` 2005-11-10 | `15-12B` 2005-12-14; `25` 2006-01-06 |
| MAVERICK TUBE | `DEFM14A` 2006-09-01 | `25-NSE` 2006-10-06; `15-12B` 2006-10-17 |
| ALPHARMA | `SC TO-T/A` (a tender offer: an offer to buy shares from holders) 2008-11-24 to 2008-12-30 | `25-NSE` 2009-01-06; `15-12B` 2009-01-16 |

## 4. What this cannot tell you

- **Identity is by name only.** A different company with the same cleaned name that was alive near the exit and stopped filing annual and quarterly reports afterwards could still match. The pairs above are the reader's check.
- **A deal form is an announcement.** It does not say the deal finished or what holders were paid.
- **The still-filing rejection can be wrong either way.** It could reject the true company; it cannot confirm that a stock survived.
- **One listed form was left out of the search.** The brief says the deal forms include "their amendments"; the code's list has `/A` versions of three of the four form types but not `DEFM14A/A`. A check of the nine matched companies found no such line in the window, so no result changed; the omission is disclosed rather than corrected after the run.
- **20 stocks found no company at all**, many because the roster name is a cut-off or abbreviated form (UNITED DEFENSE, LYONDELL, VARIAN SEMI, SANTANDER CONS and others) that no rule here maps to the filer's full name. This run did not use a ticker or CUSIP lookup (ticker and CUSIP are other stock identifiers) or a hand-made name list. Other methods may exist; this run only shows that the fixed name rules were not enough.

## 5. Files

`_payouts_v2.csv` and `_payouts_v2.json` (141 rows: the 21 original columns plus `phase`, `name_rule`, `reason`, `still_filing_check`, `deal_form_accessions`; the 102 earlier evidence rows carry their original values); code `code/sec_name_index.py`, `code/open_cases.py`, `code/open_cases_run.py` and their tests; hand checks in `_hand_checks.txt` section 14. Reproduce (needs the `SEC_USER_AGENT` contact variable, the saved folder, and a clean working tree); the name probe of section 2 is a separate read-only query:

    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/open_cases_run.py --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
