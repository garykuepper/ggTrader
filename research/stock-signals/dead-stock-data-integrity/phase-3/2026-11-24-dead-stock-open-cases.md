# Dead-stock open cases: a second search for the 39 stocks with no filing evidence

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-data-integrity` (reopened for new files only; Phases 1 and 2 are untouched)
**Rules written first:** [2026-11-24-open-cases-brief.md](2026-11-24-open-cases-brief.md), committed before any Phase 3 code existed and amended once, before any run, after an independent cold read. Every term used below is explained there.
**Question:** For the 39 dead stocks that Phase 2 could not tie to a filing about an acquisition, does a different way of finding the company (looking it up by name among every filer in the SEC's quarterly lists, not only among exchange notices) turn up evidence?

## 1. In short

- **Evidence grew by one stock, from 102 to 103 of 141.** The newly found one is ALPHARMA INC-A (2008): its company filed the SEC's acquisition-completed report items in the weeks around its last trading day. Its identity rests on the company name alone (the name rule written after the first run, v1.1), so it is a lead, not a confirmed match.
- **Four more stocks now show an announced deal, but no completion.** A merger-vote notice (the form shareholders receive before voting on a merger) was filed by the matched company close to the last trading day for ABGENIX (2006), VERITAS DGC (2007), SPINNAKER EXPLORATION (2005) and MAVERICK TUBE (2006). These four are kept apart (class `deal_forms_only`) and **set no loss figure**: an announced merger is not a payment.
- **Two stocks are probably not dead at all.** JAZZ PHARMA (last price 2012-01-17) and SPRINT NEXTEL (2013-07-10) match a company that kept filing annual and quarterly reports for years afterwards under the same SEC number. The rule rejects them (`company_still_filing`). This points to a change of company name or legal form in our price data, not a stock that stopped. The filings do not say which; the first test treated both as dead stocks, so its harsher readings (loss 1.0 and 0.5) wrote them down.
- **The rest stay open.** 20 of the 26 name searches found no company, 2 were rejected (above), 8 stocks still have no name, and 4 stocks (PREMCOR, VINTAGE PETROLEUM, PHARMION and AMERICAN NATIONAL INSURANCE) matched a company with neither an acquisition item nor a deal form. Details in section 3.
- **The first test's best case cannot change.** All 103 stocks with evidence are at the 0.0 reading, which was already the best reading the first test tried. Nothing was rerun.
- **No bankruptcy turned up.** Again none of the matched companies carried the bankruptcy item.

## 2. What was done

- The rules in the brief were applied to the 26 stocks with no Form 25 match (rule 1), the 5 stocks Phase 2 matched but found no acquisition item for (rule 2: their company is not searched again), and the 8 unnamed stocks (rule 7).
- The search used the 80 quarterly lists already saved on disk (2005 quarter 2 to 2025 quarter 1; 20,784,033 filing lines, all read; nothing downloaded for them). A first attempt stopped at one line of the 2016 first quarter with no company name (a real quirk in the SEC's list: a fund's filing under the right number but a blank name). That is the only such line in the 80 lists; the reader was changed to keep a blank name, and the run was repeated from the start.
- The only new downloads were the 8-K report lists of the 4 stocks matched by name (4 downloads, 4 more reads of saved copies). Nothing failed. The contact line the SEC requires went out only through the environment variable and appears in no file.
- **Unnamed stocks (rule 7).** Our McGill names and the release-1 ticker map hold no row at all for the 8 unnamed stocks' PERMNO numbers (the price database's stock numbers: 87283, 75900, 92552, 92182, 93247, 16544, 64899, 16939). No name could be found without guessing, so all 8 stay unnamed. The SEC lists cannot name them: they carry no stock number.

## 3. Results

### Classes before and after

| Class | Before | After |
|---|---:|---:|
| acquired_or_merged (an acquisition-type item found) | 102 | 103 |
| deal_forms_only (a merger vote or tender offer announced; no loss figure) | 0 | 4 |
| no_8k_found (matched company, no acquisition item, no deal form) | 5 | 4 |
| no_filing_match (no company found by name, or rejected) | 26 | 22 |
| name_unknown (no company name in our data) | 8 | 8 |
| bankruptcy_filed | 0 | 0 |

Moves: 2 stocks moved from no_8k_found to deal_forms_only (Spinnaker, Maverick Tube); 1 stock from no_filing_match to acquired_or_merged (Alpharma); 2 from no_filing_match to deal_forms_only (Abgenix, Veritas DGC); 1 from no_filing_match to no_8k_found (American National Insurance).

### Every stock found by name (rule 1)

| Roster name | Company found | Name rule | Last trading day | Outcome |
|---|---|---|---|---|
| ABGENIX INC | ABGENIX INC | v1 | 2006-03-31 | merger-vote notice; no loss figure |
| VERITAS DGC INC | VERITAS DGC INC | v1 | 2007-01-11 | merger-vote notice; no loss figure |
| ALPHARMA INC-A | ALPHARMA INC | v1.1 | 2008-12-29 | acquisition item; loss 0.0 (assumed scenario); lead only |
| AMERN NATL INS | AMERICAN NATIONAL INSURANCE CO | v1 | 2022-05-24 | no acquisition item and no deal form found |
| JAZZ PHARMA | company 1232524 | v1 | 2012-01-17 | rejected: still filing |
| SPRINT NEXTEL | company 101830 | v1 | 2013-07-10 | rejected: still filing |

### Stocks Phase 2 already matched (rule 2)

SPINNAKER EXPLORATION and MAVERICK TUBE (the second with a checked notice) now carry a merger-vote notice and no loss figure. PREMCOR, VINTAGE PETROLEUM and PHARMION carry neither an acquisition item nor a deal form and stay at no_8k_found.

### Hand checks (the controller read the SEC index lines)

For ABGENIX, VERITAS DGC, SPINNAKER and MAVERICK TUBE the filer's own lines show a merger-vote notice (`DEFM14A`, dated 2006-02-09, 2006-12-01, 2005-11-10 and 2006-09-01), followed within weeks by a deregistration form and, for three of them, the exchange's removal notice (`15-12B` or `25-NSE`). For ALPHARMA the lines show a tender offer (`SC TO-T/A` forms through 2008-12-30) and the removal notice on 2009-01-06. These look like the takeovers they are named for. This is a check of the filing lines, not of what holders were paid.

## 4. What this cannot tell you

- **Identity is by name only.** A different company with the same cleaned name can still match if it was alive near the exit and stopped filing annual and quarterly reports afterwards. The pairs above are the reader's check; for the four matches they look right.
- **A deal form is an announcement.** It says a merger vote or tender offer was announced, not that it finished or what was paid. The report does not read which stock or deal the form concerns.
- **The still-filing rejection can be wrong.** It could reject the true company, for example one that keeps filing after a deal. The two rejections are listed above so they can be read.
- **The best case is unchanged and the harsher readings are not measured.** 103 stocks sit at 0.0 as an assumption; the filings give no payment.
- **20 stocks found no company at all**, many because the roster name is a cut-off or abbreviated form (UNITED DEFENSE, LYONDELL, VARIAN SEMI, SANTANDER CONS and others) that no rule here maps to the filer's full name. A ticker or CUSIP lookup (EDGAR offers no free one) or a hand-curated name list would be needed; neither is used.

## 5. Files

`_payouts_v2.csv` and `_payouts_v2.json` (141 rows: the 21 original columns plus `phase`, `name_rule`, `reason`, `still_filing_check`, `deal_form_accessions`; the 102 evidence rows carry their original values); code `code/sec_name_index.py`, `code/open_cases.py`, `code/open_cases_run.py` and their tests; hand checks in `_hand_checks.txt` section 14. Reproduce (needs `SEC_USER_AGENT`, the saved folder, and a clean working tree):

    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/open_cases_run.py --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
