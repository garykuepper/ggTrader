# Closing the membership database's open checks with the official press releases

**Date:** 2026-11-09. **Status:** evidence note. It changes no code, no reference run and no stored list; it records what the S&P Dow Jones Indices (S&P DJI) releases and two company releases say about the five open checks in Appendix C6 of the [membership database design](../../../docs/superpowers/specs/2026-11-07-membership-database-design.md), and one data gap found on the way.

## Summary

- **All five checks are closed.** The list (vintage 3) agrees with the official sources on every point checked. No vintage needs to change.
- **Check 1 (anything missed in the gap).** S&P DJI published 25 releases between 2026-07-20 and 2026-11-06. Four carry S&P 500 changes inside or next to the gap, and every row is already in vintage 2 or 3. The other 21 are about other indices or products.
- **Check 2 (did the Kenvue removal slip).** No. The release says Kenvue leaves the S&P 500 before the open on 2026-11-04. No later release changes that date, and the SPY fund holdings for 2026-11-05 agree with our list.
- **Check 3 (Vylor announced earlier?).** No release before 2026-10-01 mentions Vylor. The release is dated 2026-10-01 and says Vylor was added that day when Corteva completed the spin-off.
- **Check 4 (rename dates).** Both match the company releases: SATS to ECHO on 2026-06-24 (EchoStar, announced 2026-06-22), BK to BNY on 2026-05-21 (BNY). Our list has exactly those dates.
- **Check 5 (old bars under the reused ticker `P`).** The price table holds no old Pandora bars. The series under `P` runs 2018-01-02 to 2026-09-21 at prices of about 15 to 118 dollars, which is the history of the company now called Everpure (formerly Pure Storage), not Pandora (a single-digit stock that left the market in 2019).
- **New finding, not a list error: the price table is missing or stale for 13 of the 503 current members** (table below). It affects research on recent dates, not the membership list.

## Check 1: every S&P DJI release in the gap

Source: the S&P DJI news page on PR Newswire (`prnewswire.com/news/s%26p-dow-jones-indices/`), 25 releases from 2026-07-20 to 2026-11-06, each fetched and its table parsed with `membership_press`. Releases whose table has S&P 500 rows:

| Release (published) | S&P 500 rows (effective date) | In our list |
|---|---|---|
| Ferguson, 2026-07-31 | FERG in, EA out (2026-08-05) | before the gap, in vintage 1 or 2 |
| Reddit, 2026-08-13 | RDDT in, AVB out (2026-08-18) | vintage 2 |
| Bloom Energy, Illumina, Everpure, 2026-09-04 | BE, ILMN, P in; TAP, TTD, BLDR out (2026-09-21) | vintage 3 |
| Vylor, 2026-10-01 | VYLR in (2026-10-01); CTVA, WBD out and TWLO in (2026-10-06) | vintage 3 |
| Okta, 2026-10-30 | OKTA in, KVUE out (2026-11-04) | vintage 3 |

The SmallCap 600, MidCap 400, Case-Shiller, Kaiko, Kalshi, Quantix, Pantera and Lincoln releases have no S&P 500 row. The Paramount Skydance row of vintage 3 (SKYD in, PSKY out on 2026-10-06) comes from the company's completion release, not from an S&P DJI release, as recorded when vintage 3 was loaded.

## Checks 2 and 3

The 2026-10-30 release says the changes take effect "prior to the opening of trading on Wednesday, November 4". The only release in the window with "Updated Effective Date" in its title is about Huron Consulting Group (SmallCap 600). The 2026-10-01 release was published after the close on 2026-10-01 and no earlier S&P DJI release names Vylor.

## Check 4: the two renames

- EchoStar: "The new ticker will begin trading on June 24, 2026" (company release of 2026-06-22). Our list: SATS to 2026-06-24, ECHO from 2026-06-24.
- BNY: "begin trading under the new ticker symbol, BNY, on the New York Stock Exchange effective May 21, 2026" (company release). Our list: BK to 2026-05-21, BNY from 2026-05-21.

The one-day difference between the two upstream datasets (check 4 in C6) is therefore settled for both: vintage 3 is right.

## Check 5 and the price-table gap

Read-only queries on the `ohlcv` table (295,251,075 rows, unchanged). Members of the latest list with no bars, or with a last bar before 2026-11-03 (SPY's last bar is 2026-11-05):

| Ticker | Last bar in `ohlcv` | Member since |
|---|---|---|
| BE, FERG, HONA, RDDT, SKYD, VMRK, VYLR | none at all | recent additions |
| ECHO, FLEX | 2026-05-18 | ECHO renamed 2026-06-24; its real bars sit under SATS (last bar 2026-11-04) |
| MRVL, OKTA, P, TWLO | 2026-09-21 | added by the September and October changes |

`BNY` is fine: it holds the full Bank of New York Mellon history to 2026-11-04. So are the other 490 members.

**What this means.** A backtest or screen that reads the frozen tape (or the database read-only) and asks for these 13 tickers on recent dates gets nothing or stale prices. The membership list itself is correct. The live trader is not affected by this: it fetches equities on demand (`CachedYFinanceLoader`) and is off in any case. Backfilling the 13 tickers is a database write to the price table, which this note did not do; it needs the owner's decision.

## Evidence

Pages were fetched on 2026-11-09 from PR Newswire (the copies of the S&P DJI releases that carry the same text as the press.spglobal.com copies cited in the vintage 3 events file; their sha256 differ because the page wrappers differ). sha256 of the bytes fetched:

- Ferguson 302840407 `7758fa0e40dd8c2eb14c86e8f748edbb4cf4fe2418fb0603cd2f1b465e9e99ba`
- Reddit 302851432 `95dadfd7b8e4ca3a16e5061f0c5fcf7798fd0db85196276da665e212bb68773a`
- Bloom Energy et al. 302870517 `740d6851a231dbb3b78c1b9cc80bb1c12a46fd88e79cdd9e7b2bbae35c3499d4`
- Vylor 302896675 `9ef7f7d5430709e110aed127a9b2791604ca02d160d16123d0257105dbedad50`
- Okta 302925405 `ae24cba8a3a5614ec17454609d4ed82c6375c7e055ba36a8fc7aca0b915cd68a`
- News page listing `0af519455ae87cfa08b05943b15f6e313a69fc58ab572d3afba111ef364c78dd`

Company releases: [EchoStar](https://echostar.gcs-web.com/news-releases/news-release-details/echostar-changing-stocker-ticker-sats-echo-marking-companys-next), [BNY](https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html). The weekly job now archives the same S&P DJI pages every Saturday in `universe.source_snapshots`.

## Update: the price-table gap was filled (2026-11-09, evening)

With the owner's go, `scripts/backfill_missing_members.py` (insert-only, reviewed by Sol and Opus; see the changelog) filled the 13 tickers up to 2026-11-05:

- **Inserted 19,491 rows:** BE 2,083; FERG 4,236; HONA 101; RDDT 660; SKYD 5,263; VMRK 6,752; VYLR 26 (all whole history); ECHO 119 and FLEX 119 (2026-05-19 on); MRVL, OKTA, P, TWLO 33 each (2026-09-22 on).
- **Refreshed 4 rows:** the 2026-09-21 bar of MRVL, OKTA, P and TWLO, which had been captured mid-session (price 0.19% to 1.31% off the complete bar, volume 31% to 60% lower). The old rows are in `data/exports/ohlcv_backfill/backfill_20261109T224708Z.csv` (git-ignored, this machine only), the keys actually inserted in `..._inserted.csv`.
- **Checked afterwards:** no weekend or non-midnight timestamp and no non-positive close among the 33,718 rows of the 13 tickers; SPY, AAPL, MSFT, SATS and BNY unchanged; every one of the 503 current members now has bars through at least 2026-11-03.
- **Known limits:** SKYD carries PSKY's history and VMRK carries EQR's (the same precedent as BNY and BK), so a screen that takes "every symbol with bars" would count those companies twice; the membership mask avoids this. MRVL's new bars sit 0.02% below its stored history (an ex-dividend date after its last stored bar). Other mid-session bars deeper in history (for example MRVL 2026-07-24, 0.55% off) were not touched.
- **FERG volume caveat (found by the second Sol review).** Of the 4,236 FERG rows written, 2,079 (2010-01-05 to 2021-03-05) have zero volume: yfinance has no volume for the period before Ferguson's US listing. The prices are fine. Do not use FERG's volume (liquidity filters, volume-weighted measures) before 2021-03-08.
- **Second Sol review of the tool and the data:** all 19,491 inserted keys exist exactly once and match the planned keys; the four rewritten bars equal a fresh download; the other five control symbols are unchanged; no jump above 25% at any old/new seam (22 jumps above 25% exist in the new rows, none at a seam, not individually authenticated); SKYD matches PSKY within 0.47% and VMRK matches EQR within 3.7% on overlapping days (a different adjustment basis, so avoid mixing the two series in one calculation). The review's code findings are fixed (see the changelog); a rerun of the tool on the same 13 tickers now plans zero inserts.
