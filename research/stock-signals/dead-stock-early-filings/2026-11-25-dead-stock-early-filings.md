# Dead-stock early filings: what the SEC's filings say about the stocks that stopped trading in 2000 to 2004

**Date:** 2026-11-25
**Study:** `research/stock-signals/dead-stock-early-filings` (plan frozen before any code: [2026-11-25-dead-stock-early-filings-brief.md](2026-11-25-dead-stock-early-filings-brief.md); five dated notes at its end record what changed)
**Question:** The first test's momentum rule held stocks that later stopped trading, and what holders of such stocks received is unknown for 2000 to 2004. For every stock in the first test's list that stopped trading from 2000 to 2004, what do the SEC's free filings say about how it ended?

## 1. In short

- **The filings say little for these years. Only 8 of the 251 stocks receive an acquisition-type or bankruptcy label.** Seven carry an acquisition-type filing (an 8-K current report with the item "changes in control of registrant", "acquisition or disposition of assets" or the later "completion of an acquisition") and one, Adelphia Communications, carries a bankruptcy filing. Two more (Conectiv and PanAmSat) also have an acquisition-type notice but stay labelled "kept filing", so they receive no suggested figure. 18 further stocks have only a notice of removal from an exchange (Form 25).
- **For 33 of the rule's held stocks the answer is mostly nothing.** Of the 33 dead stocks the first test's rule held in this window: 1 has an acquisition-type filing (IMC Global), 1 kept filing quarterly reports for years (Coastal Corp), 17 have a company but no relevant filing, 11 have no company found, and 3 match more than one company.
- **None of the 33 held stocks received a bankruptcy label in this search.** The one bankruptcy-labelled stock (Adelphia) is not among them. The study did not measure what holders received, so it can neither identify nor exclude cases where holders received nothing.
- **What the search can and cannot see.** It reads filings made by the company itself. The search did not read the buyer's filings, merger proxy statements or tender-offer documents, and its name matching is imperfect; both may explain the missing events, but their contributions were not measured. No stock from 2000 or 2001 is labelled with a removal notice (Form 25); the earliest such label is from 2002 (whether such notices of those years exist in the SEC index was not checked). 
- **The searches needed three repairs along the way** (section 2), each recorded before the full run. The names in our price data are short forms ("MIDAMERICAN ENRG"), 77 of the 251 cut off at 16 characters. After the repairs at least one company was found for 169 of the 238 named stocks (71 percent); 29 of those match more than one company (24 match two, 4 three and 1 twenty-one) and are left unresolved.
- **This does not change the earlier results.** No strategy was rerun and no payment amount was read. A label is the SEC's classification of a filing, not what holders received.
- **There is no go or no-go label.** The owner decides what, if anything, to do next (section 5).

## 2. What was done

- **The list.** Every stock of the first test's list (common stock on the NYSE or Nasdaq, price of at least $5, among the 1,000 largest) that was usable on a month-end decision day and whose price history ended in that month's holding period, with no later spell under the same company number and not a row cut off at the end of the CRSP file: 251 stocks (251 expected, from the momentum-deaths study), 33 held by the rule (33 expected). 13 have no company name in our data.
- **Company names** come from the McGill table in our database. The raw CRSP table has no name column.
- **The search.** The SEC's quarterly index files (every quarter from 1999 Q1 to 2006 Q4, 32 files) were cached on the thesix drive and scanned in two passes (name discovery, then collection by company number). For each named stock the index was searched for a company with a matching name (cleaning and abbreviation rules in notes 3 to 5 of the brief, copied from the earlier 2005 to 2024 study with a few added short forms). The filings of the matched company (found by its CIK, the SEC's company number, in every quarter, whatever name they were filed under) from 30 days before to 180 days after the stock's last trading day were read: 8-K current reports and Form 25 notices, by their headers, 243 files downloaded (89 further requests were served from the cache). Requests were limited to five a second and carried the owner's contact line, which is not written in any file of this study.
- **The labels**, in order of precedence: `ambiguous_match` (two or more companies have the stock's name: nothing is attributed to any), `bankruptcy`, `evidence_unresolved` (a filing could not be used; none occurred), `kept_filing` (the company filed 10-K or 10-Q reports more than 180 days after the stock's last day, so the stock may not have ended: a data break, or a bought company that keeps filing for its debt; such a company gets no figure even if an acquisition notice exists), `acquisition_type`, `delisting_notice_only`, `no_event_found`; `no_company_match` and `name_unknown` for stocks without a company. The suggested loss is 0.0 (the last price in full, as a scenario) only for `acquisition_type`.
- **Checks.** A company must have filed something on or before the stock's last day to count; an unreadable, truncated or mismatched filing header makes a stock `evidence_unresolved` rather than "nothing found"; a missing quarterly index stops the run. Every label count in this report is traceable to the per-stock table.
- **Three pilots and five notes.** A 20-stock pilot found a company for 6 stocks (the first matching rule was too strict), then 16 (after the matching repair, but with many false "kept filing" labels caused by ownership filings made by other parties and by a company created in 2004 that reuses a name), then 15 (after the kept-filing and company-age repairs). "Found a company" counts stocks with at least one candidate company, including 1, 4 and 4 ambiguous matches in the three pilots. The pilot's three versions are kept as `_pilot1.json`, `_pilot2.json` and `_pilot.json` (with their label files). The full run followed on the third version. I had read the 20 pilot outcomes before writing the matching repairs, and the brief says so.
- **Independent reviews and build.** Gemini 4 Neon built the code (six task units in five dispatches, plus fix rounds) and Codex Sol reviewed each (the reviews found a number of serious defects, among them quiet failures when a quarterly index or a header was missing, cross-company mix-ups and a form-type test that was too loose). The identified defects were fixed before the full run; several repairs followed the pilots. The full run took 288 seconds. Raw numbers: [_early.json](_early.json) and [_early_labels.csv](_early_labels.csv).

## 3. Results

**All 251 stocks:**

| Label | Stocks | Of which held by the rule |
|---|---:|---:|
| Bankruptcy | 1 | 0 |
| Acquisition-type filing | 7 | 1 |
| Kept filing periodic reports | 18 | 1 |
| Removal notice only | 18 | 0 |
| Company found, no relevant filing | 96 | 17 |
| Company matches more than one | 29 | 3 |
| No company found | 69 | 11 |
| No company name in our data | 13 | 0 |
| **Total** | **251** | **33** |

**By the year the stock stopped trading** (acquisition-type, bankruptcy, kept filing, removal notice only, no relevant filing, ambiguous, no company, no name; total):
- 2000: 0, 0, 3, 0, 54, 7, 19, 4; 87
- 2001: 0, 0, 9, 0, 28, 8, 19, 5; 69
- 2002: 3, 1, 2, 10, 3, 5, 6, 0; 30
- 2003: 0, 0, 3, 2, 6, 4, 12, 1; 28
- 2004: 4, 0, 1, 6, 5, 5, 13, 3; 37

All acquisition-type and removal-notice-only labels fall in 2002 to 2004, none in 2000 or 2001.

**How sure the matches are.** Of the 140 stocks with a single matched company (every label except the no-company, no-name and ambiguous ones), 20 were confirmed by a filing's own header (same company number and form as the index) and 120 rest on the name alone. By exchange (NYSE, Nasdaq, no exchange recorded because the name is missing): 174, 64 and 13 stocks. The acquisition-type labels are 7 on the NYSE and 0 on Nasdaq.

**The acquisition-type and bankruptcy cases:** Willamette Industries, RGS Energy Group, Apogent Technologies and National Processing (changes in control; all four checked); Security Capital Group and Evergreen Resources (name only); IMC Global (the one held by the rule, checked); Adelphia Communications (bankruptcy item, checked).

**The 18 stocks that kept filing** include Coastal Corp (held by the rule), Union Carbide, Hertz, Houghton Mifflin, PanAmSat, Dole Food and Household International. The index records their periodic filings more than 180 days after the last trading day, in some cases about six years later. That fits a company bought by a parent that still reports because of its public debt, and also fits a data break. The filings cannot tell the two apart, so these stocks have no suggested figure.

## 4. What this does not say

- **A label is not a payment.** An acquisition-type filing does not say that holders received cash, or the last price, or anything in particular. This study read no amounts.
- **The weak result may largely reflect what was read, but this was not measured.** The search used each company's own 8-K and Form 25 filings. Merger proxy statements, tender-offer responses and the buyer's own filings were not read, and no 2000 or 2001 stock carries a removal-notice-only label. The study does not say why.
- **Name matching still misses.** 69 named stocks found no company (a changed name, a spelling the rules do not cover and a company created after the stock's last day are possible reasons; their shares were not measured), and 29 matched several companies (possibly a parent and its subsidiaries, or an old and a new company of the same name). Ambiguous stocks are not resolved by their event.
- **251 stocks over five years.** The counts by year and exchange are small, and the held-by-rule subset is 33.
- **Same list and window as the earlier studies:** this adds no new market sample and no new return.

## 5. Files and how to reproduce

- Brief with five dated notes: [2026-11-25-dead-stock-early-filings-brief.md](2026-11-25-dead-stock-early-filings-brief.md)
- Results: [_early.json](_early.json), [_early_labels.csv](_early_labels.csv); pilots `_pilot1*`, `_pilot2*`, `_pilot*`.
- Code: [code/](code/); driver [code/run.py](code/run.py); roster [code/early_roster.py](code/early_roster.py); names [code/early_names.py](code/early_names.py); SEC wrapper and header parser [code/early_sec.py](code/early_sec.py); name matching [code/early_match.py](code/early_match.py); classification [code/early_classify.py](code/early_classify.py); engine copies `code/ef_*.py` and the SEC client copy with their hash records.
- To reproduce, from the repository root with a clean tracked tree and the owner's contact line in `~/.config/ggtrader/sec_contact` (the SEC answers are cached in `/media/thesix/ggTrader-data/sec-early-window/`, so a rerun makes few or no new requests): `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-early-filings/code/run.py --mode full --release-id 1 --crsp-load 1`.
