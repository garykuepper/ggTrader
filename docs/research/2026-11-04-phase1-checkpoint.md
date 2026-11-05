# Phase 1 checkpoint and proposed Phase 2 re-run list

**Date:** 2026-11-04 · **Branch:** `investigation/root-cause-2026-11-04` · **Type:** checkpoint
for the owner (diagnosis only; nothing was re-run for this document) · **Design:**
`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` (section 6, Phase 2 and
the early-exit rule; section 10) · **Plan task:** Task 12 of
`docs/superpowers/plans/2026-11-04-root-cause-investigation.md`

Abbreviations, expanded once: SPY = SPDR S&P 500 ETF Trust (the benchmark); WFO = walk-forward
optimization (fit on a past window, test on the next, repeat); OOS = out of sample; PIT = point
in time (index membership as known on each day); DSR = deflated Sharpe ratio (a significance
gate); MDE = minimum detectable effect (the smallest true effect a test finds with 80%
probability); CAGR = compound annual growth rate; pp = percentage points; bp = basis points
(0.01%); TR = total return (dividends reinvested); EOD = end of day; ET/PT = US Eastern/Pacific
time; DB = the project's TimescaleDB database; RSP = Invesco S&P 500 Equal Weight ETF; SSO =
ProShares Ultra S&P 500 (2x daily); A23 = the owner-adopted 70/30 SPY/SSO allocation study;
NW = Newey-West (a standard error that allows autocorrelated returns); BTC/ETH = bitcoin/ether;
"replica" = the 2026-10-19 production-rule replica backtest.

This document has seven parts: (1) coverage against the inventory, (2) the status of each
lens, (3) findings carried forward in plain language, (4) the proposed Phase 2 list, (5) whether
Phase 2 can be skipped, (6) a usage note, (7) the owner decision block. It adds no new
judgement about any strategy. It does not recommend strategies, deployments or what to believe.

## Guard check and an incident during this task

- **Guard check (plan Task 12, step 1).** The lead ran `scripts/guard_snapshot.sh` before this
  task and found no difference from `docs/research/_investigation_guard_baseline_2026-11-04.txt`.
  I re-ran it after the incident below (2026-11-04 23:5x PST): still no difference (intraday
  observer timers, observer worktree HEAD, `ggtrader_live` container state, `data/` git status).
- **Incident (my error; reported, not hidden; corrected after review on 2026-11-05).** To read
  driver options I ran a shell loop of `scripts/<name>.py --help` over twelve driver scripts.
  `scripts/pead_correlation_check.py` has no argument parser, ignored `--help` and started its
  body. It ran for about two minutes (about 23:47 to 23:49:45 PST on 2026-11-04) before I stopped
  it. Its body calls `load_ohlcv` from a warm-up start 700 calendar days before 2015-06-01 (400 bars x
  1.6 + 60 days; its log shows fetches starting 2013-07-01) to "today" (`scripts/pead_correlation_check.py:25-39`), and
  `load_ohlcv` reads and writes through the database cache loader (`src/ggTrader/lab/data.py:57-110`;
  Lens 6 noted the writes at `docs/research/2026-11-04-lens6-implementation-correctness.md:347-348`).

  **What it wrote (established with SELECT-only queries):**
  - **It definitely wrote to the database.** The `symbol_inception` table carries a write
    timestamp (`recorded_at`, set by the loader at
    `src/ggTrader/data/live/cached_yfinance_loader.py:151-168`). 19 rows were rewritten at exactly
    2026-11-04 23:48:18 PST, for APC, ARES, CAM, CSRA, CZR, EMC, GDDY, LITE, NE, PARA, PCL, POM,
    QRVO, SBNY, SEDG, SPLS, STI, VEEV and XYZ.
  - **Each of those 19 symbols was refetched over its full range and upserted.** The loader sends
    a symbol to a full-range fetch when it has no cached bars, or when its first cached bar is more
    than 7 days after the requested start and does not match its recorded inception
    (`src/ggTrader/data/live/cached_yfinance_loader.py:329-345`). The full fetch is then written
    with `INSERT ... ON CONFLICT ... DO UPDATE` over the whole fetched range (lines 257-263, called
    at lines 355-368), and the new inception dates are recorded. All 31,967 `ohlcv` rows the 19
    symbols now hold (yfinance, daily; sum of the 19 per-symbol row counts) lie inside the fetched
    range (earliest 2013-10-16), so up to 31,967 rows were rewritten on today's yfinance adjustment
    basis. The 12:45 PT production run's incremental refresh does not rewrite whole histories, so
    this was not the same kind of write the cron makes daily.
  - **Other rewrites cannot be ruled out or sized.** PostgreSQL's table statistics show every
    `ohlcv` chunk from 2026-07-22 to 2026-11-04 auto-analyzed between 23:48:57 and 23:49:09 PST.
    That fits the loader's incremental refresh (from about 2026-08-11 for other symbols, because
    of the stale symbol EA; Lens 6 lines 256-271), but `ohlcv` has no write timestamp, so which
    other rows changed cannot be read back.
  - **`ohlcv_no_data` was not touched:** its latest `checked_at` is 2026-10-19 18:35 PST
    (173 rows). No file under `data/` or `results/` changed. Nothing live was touched: no order,
    no container, no observer, no broker call. The guard snapshot is unchanged.

  **A lens figure did change: PARA.**
  - Lens 1 measured PARA closes from 72.20 to 113,900 over its S&P 500 membership
    (2022-02-17 to 2025-07-23), near $100,000 through 2023 Q3, about 25,700 in 2024 Q1 and about
    42 in 2025 Q3, median volume 9 (`docs/research/2026-11-04-lens1-backtest-validity.md:37-41, 208-210`).
    Lens 2 quotes the same figures (`docs/research/2026-11-04-lens2-verdict-audit.md:381`).
  - The database now shows, over the same membership window: maximum close 3,935.11, minimum
    5.96, median volume 8. Quarterly maxima are about 3,386-3,935 from 2022 Q1 to 2023 Q4,
    887.9 in 2024 Q1 and 57.6 in 2025 Q3. The two highest reference points both shrank by the
    same factor (113,900 / 3,935.11 = 28.9; 25,700 / 887.9 = 28.9). The low end did not follow that
    factor (Lens 1's minimum 72.20 against 5.96 now), so this is not a uniform rescaling.
  - PARA is still the wrong security (prices in the thousands, median volume 8-9 shares), but the
    PARA figures in Lens 1, Lens 2 and finding 1 of section 3b below can no longer be reproduced
    from the database.
  - **The other 18 symbols:** I searched every lens report, the Phase 0 audit, the inventory report
    and the lens JSON files for each symbol. Only SBNY is cited, in Lens 1's list of members whose
    maximum/minimum close exceeds 50x in 2021-01-31..2026-04-30
    (`docs/research/2026-11-04-lens1-backtest-validity.md:414-416`); the database still reproduces
    that (ratio 766). No lens cites a price, volume or return for the other 17.

  **My earlier check did not test the affected set.** The first version of this section said "no
  lens figure was found to have changed", based on SPY, HAS and VZ bars. None of those is among
  the 19 rewritten symbols. That check was mine, and it was relied on in the first version of this
  checkpoint and my task report.

  **Consequences, stated plainly:**
  - The investigation's read-only rule was broken.
  - Phase 1 evidence for these 19 symbols is no longer exactly reproducible from the database
    (for PARA, demonstrably not).
  - Every Phase 2 stock item would run on the post-incident tape.
  - The write cannot be undone from inside this investigation. Whether to restore the 19 symbols
    from a backup is an owner decision. What exists, established read-only: the table
    `ohlcv_yf_preshift_backup_20260820` (bars stamped one day early, the pre-migration state, ending
    mid-August 2026; it holds PARA, SBNY and ARES rows) and the file
    `backups/ohlcv_yf_preshift_20260820.sql.gz` (same date; contents not opened). Whether either is
    a suitable restore source, and whether any later backup exists, is not established.
  - **Live trader exposure.** The live trader's universe is the S&P 500 point-in-time membership as
    of the run date (`src/ggTrader/paper/signal_runner.py:60-77`, `universe_members_asof`). By the
    membership file, 5 of the 19 are current members: ARES, GDDY, LITE, VEEV and XYZ, so their
    rewritten bars are inside the live signal window (365 days). The paper ledger shows two ARES
    trades, the last on 2026-10-22. Whether the rewrite changes any live signal is unknown (not
    compared).

  **Consequence for Phase 2:** 29 scripts under `scripts/` call `load_ohlcv` or the cached loader.
  Through them, a run can write (a) full-history rewrites of `ohlcv` for symbols with no cache or a
  changed inception, (b) incremental refreshes of recent bars, (c) `symbol_inception` rows and
  (d) possibly `ohlcv_no_data` rows. Decision 0 in section 7 asks about this.

## 1. Coverage against the inventory

Method: a script in the session scratchpad,
/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/t12/coverage.py,
loads the 144 inventory ids from `docs/research/_investigation_inventory.json` and searches each
lens report and each lens JSON for the exact id string. It also checks that every inventory row
with a verdict has a Lens 2 class. Output:
/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/t12/coverage_out.json.

| Check | Result |
|---|---|
| Inventory ids | 144 (108 stocks, 11 cross-asset, 25 crypto), all unique |
| Ids appearing in at least one lens report or lens JSON | **144 of 144** |
| Lenses per id, counting reports and JSON | min 4, median 6, max 8 |
| Lenses per id, counting markdown reports only | min 2, median 3, max 6 |
| Ids in no lens markdown report | **0** |
| Ids whose only markdown mentions are the full tables of Lens 2 and Lens 9 | 40 (all stocks) |
| Inventory rows with a verdict | 114 |
| Verdict rows with a Lens 2 class | **114 of 114**; 0 without a class; 0 Lens 2 ids outside the inventory |
| Rows without a verdict (operations, design, infrastructure) | 30; they carry Lens 1 and Lens 9 states and are not classified by Lens 2 |

No id is missing, so no id needed a fallback "unverifiable, prose only" label to count as
covered. Lenses 3 and 6 never cite ids by name in their reports (Lens 6 writes its 90 verdicts
to `docs/research/_lens6_verdicts.json`; Lens 3 is about the live account, not inventory rows).
The 40 rows mentioned only in the Lens 2 and Lens 9 tables are still classified there and in
the Lens 1, 5 and 7 JSON files. Coverage is complete under success criterion 1 of design
section 8 in this sense: every row has a classification or an "unknown, with reason" from each
lens that owns it. 103 of the 144 have no raw artifact (Lens 7), so their verification method
is "taken from prose".

**Trust-state summary (counts only, copied from the lens JSON files; no new judgement):**

| Measure (source) | Stocks (108) | Cross-asset (11) | Crypto (25) | Total |
|---|---|---|---|---|
| Lens 2 class (`_lens2_sensitivity.json`, 114 verdict rows) | inconclusive 77, unreliable GO 9, unreliable NO 2 | inconclusive 11 | inconclusive 15 | inconclusive 103, unreliable GO 9, unreliable NO 2, reliable 0 |
| Lens 1 tape state at run (`_lens1_tape_state.json`) | contaminated 50, clean 15, unknown 43 | contaminated 5, clean 3, unknown 3 | unknown 25 | contaminated 55, clean 18, unknown 71 |
| Lens 6 code matches spec (`_lens6_verdicts.json`, 90 rows) | yes 13, partly 28, no 14, unverifiable 18 | yes 1, unverifiable 7 | partly 1, unverifiable 8 | yes 14, partly 29, no 14, unverifiable 33 |
| Lens 7 recompute match (`_lens7_recompute.json`, 41 rows with a raw file) | yes 27, no 1, cannot locate 5 | yes 5 | partial 1, cannot locate 2 | yes 32, no 1, partial 1, cannot locate 7 |
| Lens 7 pre-registration (114 verdict rows) | pre-registered 13, amended 4, late 1, no brief 40, not applicable 30 | pre-registered 5, no brief 4, not applicable 2 | no brief 4, not applicable 11 | 18 / 4 / 1 / 48 / 43 |
| Lens 9 MDE status (`_lens9_power.json`) | computed 4, not computable from artifact 29, no raw artifact 75 | computed 2, not computable 3, no artifact 6 | computed 2, not computable 1, no artifact 22 | computed 8, not computable 33, no artifact 103 |

Lens 8 (crypto) adds pointers, not new classes: it located gitignored run folders for three
crypto rows the upstream lenses treated as artifact-less, and found one inventory pointer
aimed at the wrong run (`docs/research/2026-11-04-lens8-crypto.md:463-530`). It states that no
crypto row's Lens 2 class would change under the written rules (lines 499-509).

## 2. Lens status table

Review rounds come from the ledger in the SDD workspace
(`.superpowers/sdd/2026-11-04-root-cause-investigation/progress.md`, not tracked in git). Every
lens took one review, one fix round and one re-review that came back clean. "Unknown items"
counts the ❓ lines in each report's Done checklist.

| Lens | Report | Commits (add, correction) | Review rounds | Done status | Unknown items |
|---|---|---|---|---|---|
| 1 Backtest validity | `docs/research/2026-11-04-lens1-backtest-validity.md` | 77cecdab, 6576811e | 2 (1 fix round) | Done by the design's rule: every question answered or marked unknown with a cheapest test. Unknown: adjustment-seam extent beyond VZ and SPY; PARA's effect on the replica and core; day-shift direction per July event study (lines 464-485) | 3 |
| 2 Verdict audit | `docs/research/2026-11-04-lens2-verdict-audit.md` | 1b6c60d6, 3dfacd83 | 2 (1 fix round) | Done. Unknown: whether strict bars rejected small real effects (no SE or MDE for most rows); how much each of 18 revisions moved each verdict; PARA's effect; simpler-test outcome for 48 gated rows; crypto rows inconclusive on data (lines 1168-1191) | 5 |
| 3 Live parity | `docs/research/2026-11-04-lens3-live-parity.md` | c145d1ed, a825594b | 2 (1 fix round) | Done. Unknown: partial-bar signal parity; Alpaca's EOD valuation of AVB; rejected orders outside 09-25 (lines 469-482) | 3 |
| 4 Feasibility and ceiling | `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md` | bad6d855, 18cac0da | 2 (1 fix round) | Done. Unknown: crypto ceiling (Lens 8 was not yet committed; it now is, 392158e5); confidence intervals for leverage, cross-asset, French-screen and century rows; replica after-tax (lines 545-564) | 3 |
| 5 Coverage gaps | `docs/research/2026-11-04-lens5-coverage-gaps.md` | e14494e2, 9769a79d | 2 (1 fix round; Sonnet review) | Done. Unknown: external crowding per family; crypto detail (deferred to Lens 8, now committed) (lines 554-567) | 2 |
| 6 Implementation correctness | `docs/research/2026-11-04-lens6-implementation-correctness.md` | 36acba57, 659e3417 | 2 (1 fix round) | Done. Unknown: code-versus-spec for 17 rows not reached; size of the return effect of live-versus-backtest differences (lines 369-392) | 2 |
| 7 Research accuracy | `docs/research/2026-11-04-lens7-research-accuracy.md` | debcfaaa, cc6cb579 (plus lead spec correction 5cf88940) | 2 (1 fix round) | Done. Unknown: 103 prose-only rows; two minor citations; pre-registration before 2026-09-10 (lines 353-367) | 3 |
| 8 Crypto | `docs/research/2026-11-04-lens8-crypto.md` | 6f5b0725, 392158e5 | 2 (1 fix round; Sonnet review) | Done. Unknown: full-range BTC beta; funding-carry Sharpe convention; cause of the run-2 BTC benchmark mismatch; April-May runs not reproducible (tape wiped) (lines 625-638) | 4 |
| 9 Power and regime | `docs/research/2026-11-04-lens9-power-and-regime.md` | 21664be7, 7fe1ceb4 | 2 (1 fix round) | Done. Unknown: MDE for 33 rows whose artifact lacks a t-statistic or mean excess; whether 2021-2026 verdicts hold in another regime (lines 638-650) | 2 |

Total unknown items across the nine Done checklists: **27**. Two of them (Lens 4 and Lens 5
crypto) are now answerable by reading Lens 8 and need no re-run. One (April-May crypto runs) no
re-run can settle, because the tape they used was wiped on 2026-05-05.

## 3. Findings carried forward, in plain language

Nothing here is new. Each line points to the lens report that states it. "Reported" means the
lead already told the owner in chat (ledger: seven confirmed live-path facts after the Lens 6
review and seven after the Lens 3 review; the ledger records the count, not the wording, so the
"reported" list below is taken from the task brief).

### 3a. Live-path facts already reported to the owner

From Lens 6:
- The paper trader runs parameter values that no backtest fold ever picked (two are outside the
  search grid) (`docs/research/2026-11-04-lens6-implementation-correctness.md:31-33`).
- It computes signals at 12:45 PT from today's unfinished price bar (lines 34-46).
- It fills on the same day, about 15 minutes before the close. The walk-forward backtests fill at
  that day's final close and the replica fills a day later (lines 34-46, table row 2 at line 239).
- It holds at most 3 stocks, chosen alphabetically when more signals fire (lines 47-50).
- It sizes each buy at 3.3% of account value; the walk-forward backtests size at 3% of remaining
  cash (lines 47-50, table row 5).
- The snapshot value adds back a dividend accrual, and that inflated value becomes the next day's
  starting point for the 3% daily-loss halt, so the halt fires at about 2.7% (lines 57-69).

From Lens 3:
- The gap between the snapshot value and cash plus positions equals the code's dividend accrual
  ($320.24 since 10-30), and Alpaca's paper account never paid any of those dividends
  (`docs/research/2026-11-04-lens3-live-parity.md:16-24`).
- Two accrual rows are over-stated by $5.86 in total (AEP $5.76, SPY $0.10) (lines 22-24, 295-300).
- Alpaca paper never applied MNST's 2-for-1 split, so the 09-29 sale credited half the
  position's value, about -0.85 pp of the account (lines 44-46).
- From 07-15 to 07-30 the trader ran in dry-run by mistake and placed no orders (table row 11 at
  line 331).
- AVB (merged into EQR) is held at a price frozen since 08-17, about $746; Alpaca's end-of-day
  equity appears to value it near zero (hypothesis) (row 9 at line 329; lines 409-418).

### 3b. Live-path and data-integrity findings in the lens reports not yet in the reported list

| # | Finding, plain language | Where | Owner decision needed to look further? |
|---|---|---|---|
| 1 | The S&P 500 member PARA is the wrong security in the price table for its whole membership (prices up to 113,900, median volume 9 shares, as measured before the 2026-11-04 incident; not reproducible from the database now, see the incident section). The replica, which Lens 2 classes inconclusive (formerly rated by Phase 0 as the most trusted stock result), bought and sold it 3 times and its taint check did not flag it. Effect on the result: not measured. | `docs/research/2026-11-04-lens1-backtest-validity.md:37-41, 208-214` | Yes: measuring it needs a replica re-run (P2-16) |
| 2 | Stored prices mix dividend-adjustment levels. The price cache never re-adjusts older bars, so where a later download overlaps an earlier one a dividend shows up as a fake one-day move (VZ: -1.7% on 2026-07-20 against -0.2% in yfinance; SPY bars through 07-24 sit 0.25% high). Live signals read this same table, so a fake move can reach a live indicator. How many such seams exist was not measured. | Lens 1 lines 42-46, 238-254, 330-332 | No for a read-only count (P2-09); any repair is outside this investigation |
| 3 | In the price table AVB has carried EQR's price since 2026-07-20. | Lens 1 lines 47, 218-219 | Noted only |
| 4 | 14 index members in 2021-2026 have no prices in the table (for example ABC, ANTM, VIAC), so those member-days are missing from every stock backtest. | Lens 1 lines 48-50, 220-229 | Noted only |
| 5 | The 2010-2020 price data has identity faults (5,595 daily moves of 40% or more across 217 symbols), which is why there is no clean holdout. | Lens 1 lines 51-52 | Yes: a clean holdout is a multi-day item (P2-28) |
| 6 | The index membership file ends on 2026-06-02; some old tickers (FB, INFO, DISCA) now carry other securities (outside their membership windows); 9,567 bars close below $0.05 and 18 have a high below the low. | Lens 1 lines 230-237, 268 | Noted only |
| 7 | Some paper-versus-SPY comparisons used SPY's price return while the paper side included dividends. | Lens 1 lines 62-66, 333-334 | Noted only |
| 8 | The day's unfinished 12:45 PT bar is overwritten in the price table only by accident: a stale symbol (EA, last bar 2026-08-10) pulls every refresh back to 2026-08-11. Without it, partial bars would stay in the research tape. | Lens 6 lines 256-271 | Noted only |
| 9 | Live compares a partial day's volume with a 20-day average of full days for the volume voter, and needs only 60 bars of history where the backtests default to 400. | Lens 6 table rows 6 and 8 (lines 243, 245) | Noted only (P2-14 measures the partial-bar part) |
| 10 | The lab's "stability-aware live parameters" function has no consumer; live always uses class defaults. | Lens 6 lines 285-287 | Noted only |
| 11 | Alpaca's paper account has never credited a dividend, split, merger, interest or any other cash flow since funding (one $100,000 funding journal on 2024-06-21). | Lens 3 lines 286-294 | Noted only |
| 12 | 44 orders filled at the next day's open (after-close runs before 06-27 and on 07-30, and the 09-29 manual liquidations), costing about 47 bp against the prior close (estimate). Three trading days had no run (07-29, 09-16, 09-23). | Lens 3 lines 50-54, table rows 3 and 7 (lines 323, 327) | Noted only |
| 13 | Rejected orders do not appear in Alpaca's order history (all 212 listed orders are filled), so rejections cannot be counted from the broker. | Lens 3 table row 13 | No for a log count (P2-07) |
| 14 | The dividend accrual takes share counts from the ex-date snapshot, so a holding sold in an ex-date run would be missed. Nobody has checked whether that happened. | Lens 3 lines 456-458 | No for a read-only check (P2-08) |
| 15 | 43.7% of the "Binance.US" 4-hour crypto bars are copies of Kraken bars (a legacy backfill script cloned them), so that tape mixes two venues. | `docs/research/2026-11-04-lens8-crypto.md:52-53, 288-298` | Noted only |
| 16 | Delisted Binance.US coins were deleted from the table on 2026-06-07, and the coin universe was picked by current trading volume, so crypto backtests carry survivorship bias. | Lens 8 lines 327-339 | Noted only |
| 17 | About 100 stock tickers (AAPL, SPY and others), 52 coin symbols without a quote currency and 7 un-split `XBT` pair codes sit under the crypto venue labels. They do not enter a `-USD` universe. | Lens 8 lines 54-55, 316-326 | Noted only |
| 18 | The current lab annualizes crypto Sharpe with 252 days, which understates a 7-day crypto Sharpe by a factor of 0.83. No published crypto figure is affected. | Lens 8 lines 56-59 | Noted only |
| 19 | The crypto live tables hold 1-2 positions from 2026-06-04 with no matching order rows; their origin is not established. | Lens 8 lines 442-450 | Noted only (crypto is intentionally off) |
| 20 | One inventory pointer is wrong: the crypto "textbook reset" row points to the zero-fee run, not the realistic-fee run its claim is about. | Lens 8 lines 478-481 | Noted only |
| 21 | The position-sizing report's number comes from an earlier run with no raw file; the raw file's numbers were never reported. | `docs/research/2026-11-04-lens7-research-accuracy.md:20-22` (summary table) | Noted only |

## 4. Unknowns that only a re-run or more data could settle (proposed Phase 2 list)

**How to read this list.** Each item is an unknown recorded by a lens, a test that would settle
it, and its cost. Items are ordered by cost, cheapest first. This order is not a ranking of what
should be done. Nothing on this list runs until the owner approves it by id.

**Rules that apply to every item:**
- Every backtest or re-run goes through `scripts/run_capped.sh` with the 6 GB cap and swap off;
  at most two runs at once. A kill at the cap (exit 137) is reported, not retried.
- Windows are pinned. Production runs at max-leverage 1.0; the lab default is 2.0. Only blend
  runs use that setting; each item says whether it applies.
- Outputs go to new files (a scratch path or a new `docs/research/_p2_*.json`), never over a
  committed artifact.
- Free data only. No paid data. No orders, no live-container or observer contact. Broker access,
  where named, is read-only (allowed for Lens 3 by design section 3).
- **Peak memory has not been recorded for any driver** (no artifact stores it), so memory is
  "unknown" for every item; the 6 GB cap bounds it.
- **Database writes.** Items marked "DB-cache writes" (tagged "writes to the price cache" in
  section 7) use drivers that load prices through
  `load_ohlcv` and so write to the `ohlcv` table (see the incident above). They need decision 0.
- Runtimes come from `elapsed_sec` fields in the artifacts where they exist
  (`docs/research/_audit_reruns_20261019.json`: xs_momentum 2,230 s, dual_momentum 1,877 s, FOMC
  44 s; `docs/research/_pit_rebaseline_20260923.json`: core 1,479 s, ensemble_ic 1,348 s,
  ensemble_kelly 652 s; `docs/research/_month_end_treasury_results.json`: about 16 s;
  `docs/research/_cross_asset_sleeve_results.json`: about 25 s; momentum screen 0.07 s);
  otherwise they are estimates and labelled so.

**Independence.** All items are independent of each other except: P2-17 (sleeve versus RSP) and
P2-21 (after-tax replica) need the per-trade and sleeve-return output from P2-16; P2-22 overlaps
P2-19 for the PIT core rows. P2-01 and P2-06 both use the frozen French archives but do not
depend on each other.

**Not settleable by any re-run** (listed so they are not mistaken for gaps in this list): the
window behind the 1.12 Sharpe (it used "now" as its end date); the 2026-07-13 raw output behind
1.14; the April-May 2026 crypto runs (their tape was wiped and re-imported on 2026-05-05;
`docs/research/2026-11-04-lens8-crypto.md:636`); whether SPY's stray rows existed before
2026-06-20 (the table has no write timestamps; `docs/research/2026-11-04-lens1-backtest-validity.md:454-455`).

### Summary table

| Id | Unknown settled | Cost (wall clock) | Could change a recorded verdict or class? | Owner flag |
|---|---|---|---|---|
| P2-01 | Regime split of per-fold returns (partial) | minutes | No | No |
| P2-02 | MDE for the three French screens | minutes | Yes, in principle (Lens 2 class of 3 rows) | No |
| P2-03 | Month-end Treasury WFO under the DSR fix | about 1 min compute | Yes (Lens 2 class; recorded NO could be questioned) | Yes (DB-cache writes) |
| P2-04 | 2.0x SSO arm at higher funding cost | minutes (estimate) | No verdict; changes a Lens 4 ceiling condition | No |
| P2-05 | Alpaca's EOD valuation of AVB | about 15 min | No | No (read-only broker read) |
| P2-06 | French top-decile big-cap momentum after 1994 | about 15 min after a download | No recorded verdict; narrows the momentum NO's scope | Yes (where to store a new download) |
| P2-07 | Rejected orders outside 09-25 | 20-30 min | No | No |
| P2-08 | Dividend accrual: missed holdings and missing dividends | about 30 min | No | No |
| P2-09 | Adjustment-seam extent in the pinned window | about 30 min | Yes, in principle (Lens 1 tape states feed Lens 2) | No |
| P2-10 | Code check of the corrected FOMC event study | 20-40 min | Yes (its only failing Lens 2 condition) | No |
| P2-11 | Day-shift direction for each July event study | about 70 min | No class; changes the bias direction recorded | Yes (over 1 hour) |
| P2-12 | Crypto code reads (carry Sharpe, run-2 benchmark, Keltner/StochRSI rules) | 1-1.5 h | No | Yes (over 1 hour) |
| P2-13 | Crypto recompute of located artifacts; stage-1 fee-tier gate replay | about 1 h | No class (Lens 8 says so); Lens 7 recompute states | Yes (about 1 hour) |
| P2-14 | Partial-bar effect on live decisions | about 1 h (estimate) | No | Yes (about 1 hour) |
| P2-15 | Lens 7 re-check of 9 unchecked rows and 2 citations | about 1 h | No | Yes (about 1 hour) |
| P2-16 | Replica with PARA excluded, plus per-trade and sleeve output | about 1 h (estimate; not recorded) | Lens 1 tape state of the replica; not its Lens 2 class | Yes (re-run, DB-cache writes, driver change) |
| P2-17 | Stock sleeve versus equal-weight S&P 500 (RSP) | 15-30 min after P2-16 | No recorded verdict; tests Phase 0 hypothesis 2 | Yes (depends on P2-16) |
| P2-18 | A23 funding and distribution tax-character records | about 1 h | No investigation verdict (A23 is out of scope) | Yes (out-of-scope topic) |
| P2-19 | PIT core, ensemble_ic, ensemble_kelly under the DSR fix, with stored excess series | about 1 h (recorded 3,479 s) | Yes (the 0.59 figure and three Lens 1 states) | Yes (re-run, DB-cache writes) |
| P2-20 | Crowding and decay mapping (McLean and Pontiff) | 1-2 h | No | Yes (over 1 hour) |
| P2-21 | After-tax replica with short-term lots | 2-4 h after P2-16 | No (the closure ends the test at the pre-tax fail) | Yes (over 1 hour, driver writing) |
| P2-22 | Stored excess series for the remaining artifacts without a t-statistic | 2-4 h total | Yes, in principle (Lens 9 MDE feeds Lens 2 power) | Yes (over 1 hour, DB-cache writes for WFO drivers) |
| P2-23 | Pre-registration before 2026-09-10 (private transcripts) | 1-2 h | No class; changes Lens 7 pre-registration states | Yes (private records outside the repo) |
| P2-24 | Code check of 16 remaining unexamined rows | 5-11 h | Yes, in principle (Lens 6 feeds Lens 2 code condition) | Yes (over 1 hour) |
| P2-25 | July event studies on the corrected tape, pinned | 5-20 h | Yes (Lens 1 states; sign of the recorded NOs) | Yes (over 1 hour, driver writing, DB-cache writes) |
| P2-26 | Non-gated arms and one-change re-runs for gate revisions | tens of hours | Yes (Lens 2 gate-machinery and freedom unknowns) | Yes |
| P2-27 | Crypto like-for-like fee-tier re-run and full-range BTC beta | hours to a day+ | Not a class under Lens 2's rules | Yes (restores removed code in a scratch worktree) |
| P2-28 | Clean 2010-2020 holdout | days | Yes (regime unknown; replica holdout citability) | Yes |
| P2-29 | Longer frozen paper trading with an input archive | months to years | Settles parity, not alpha | Yes (needs a live change this investigation does not make) |

Under one hour: P2-01 to P2-10 (**10 items**). P2-11 is about 70 minutes.

### Item details

Each item gives: the unknown (with its source), method, pinned window and config, expected
runtime, memory, data, what it could change and in which direction, what result would change
nothing, the owner flag, and a pre-registration draft ready to commit before running.

**P2-01. Regime split of per-fold returns (partial check).**
- Unknown: whether 2021-2026 WFO verdicts depend on 2022 (`docs/research/2026-11-04-lens9-power-and-regime.md:571-579, 648`).
- Method: read-only Python over the committed artifacts that hold `fold_rows`
  (`docs/research/_pit_rebaseline_20260923.json`, `docs/research/_audit_reruns_20261019.json`,
  `docs/research/_wfo_fold_attribution_results.json`); split OOS fold returns into folds ending
  in 2022 and the rest; report means and counts. Driver would need writing: a 30-line script.
- Window/config: the artifacts' own pinned windows (2021-01-31 to 2026-04-30); no leverage setting.
- Runtime: minutes. Memory: unknown, small (capped at 6 GB, as in the rules above). Data: committed JSON only.
- Could change: no verdict or class (folds overlap and are not independent, per Lens 9). It adds
  a description of how much of each OOS result sits in 2022.
- Changes nothing if: the split is similar across both groups.
- Owner flag: no.
- Pre-registration: "Question: what share of each row's OOS fold return falls in folds ending in
  2022? Window: as stored. Comparator: the same row's other folds. Use: descriptive only; no
  verdict or class changes on this result."

**P2-02. MDE for the French momentum, factor and net-issuance screens.**
- Unknown: their power in pp/yr (Lens 2 lines 50-60 and 1185-1186; Lens 9 lines 555-561; Lens 4
  lines 560-562).
- Method: recompute the monthly net excess series from the frozen archives in
  `data/external/french/` (read-only) with each screen's registered construction, and store the
  annualized mean excess next to its NW t. The drivers `scripts/momentum_french_screen.py`,
  `scripts/factor_french_screen.py` and `scripts/net_issuance_screen.py` run offline but do not
  write the mean excess; driver would need a small addition (write `ann_mean_excess_pct` and the
  SE per hypothesis), run with `--out` pointing to a new file. Then MDE = (t-bar + 0.84) x SE.
- Window/config: the registered windows (factor 1963-07 to 2026-04; momentum 1927 onward with the
  post-1994 bar; net issuance as registered); no leverage.
- Runtime: minutes (the momentum screen recorded 0.07 s). Memory: unknown, small (capped at 6 GB, as in the rules above). Data: frozen
  free French files already on disk.
- Could change: Lens 2's power condition for three rows that fail only that condition (lines
  56-58). If an MDE is at or below the row's bar, that row could move from inconclusive to
  reliable NO; if above, it stays inconclusive. Lens 2's indicative note puts the momentum MDE
  near 2.5 pp/yr over 1927-2026 against a +1.0 bar (lines 67-76), which points to "stays".
- Changes nothing if: each MDE exceeds its registered bar.
- Owner flag: no (offline, no DB).
- Pre-registration: "Question: what is the MDE (t >= 3, 80% power) in pp/yr for each registered
  hypothesis of the three screens? Window: as registered. Comparator: the registered market or
  size-matched comparator. Decision rule fixed now: MDE <= bar meets Lens 2's power condition R5;
  MDE > bar leaves the row inconclusive. No recorded verdict changes."

**P2-03. Month-end Treasury WFO under the DSR fix.**
- Unknown: whether its NO-GO holds on current code; its only failing Lens 2 condition is that its
  WFO predates the DSR units fix (Lens 2 line 61; Lens 1 line 163).
- Method: `scripts/run_capped.sh .venv/bin/python scripts/month_end_treasury_wfo.py --force --out <new file>`.
- Window/config: pinned in the driver, 2021-01-31 to 2026-04-30; 1 bp costs as stored; single
  overlay funded from SPY, no blend, so max-leverage does not apply.
- Runtime: about 16 s of compute recorded (`wfo_elapsed_s` 11.4 plus primary 4.5). Memory:
  unknown (capped at 6 GB, as in the rules above). Data: DB tape for IEF and SPY. **DB-cache
  writes** (the driver uses `load_ohlcv`).
- Could change: Lens 1 state (contaminated to clean on the DSR item). If the NO holds, the row
  meets Lens 2's data condition and could move from inconclusive to reliable NO; if the overlay now
  beats its SPY funding leg, the recorded NO-GO is in question. Either direction is possible.
- Changes nothing if: the re-run reproduces the stored figures and Lens 2's other conditions are
  re-checked as unchanged; then only the Lens 1 tag changes.
- Owner flag: yes (DB-cache writes).
- Pre-registration: "Question: on current code (after 09f9f40), does the month-end IEF overlay's
  CAGR trail SPY on 2021-01-31..2026-04-30 at 1 bp, as stored (-0.08 pp/yr)? Comparator: SPY
  funding leg as in the artifact. Verdict rule: the original pre-registered bar in its brief;
  no looser bar."

**P2-04. 2.0x SSO arm at higher funding cost.**
- Unknown: the 2.0x arm's gap holds only at the window's 1.64% average funding; no sensitivity run
  exists (`docs/research/2026-11-04-lens4-feasibility-and-ceiling.md:58-75`).
- Method: `scripts/run_capped.sh .venv/bin/python scripts/static_leverage_aftertax.py --frozen-inputs data/external/a23_inputs --main-only --rf-offset 0.025 --arms "SPY buy-and-hold,static 2.0x (SSO) no rebalance" --out <new file>`,
  and the same with `--rf-offset 0.015`.
- Window/config: driver defaults 2006-06-21 to 2026-09-29, spread 0.75%; no lab max-leverage
  setting (leverage is the SSO weight).
- Runtime: minutes (estimate; not recorded). Memory: unknown (capped at 6 GB, as in the rules above). Data: frozen checksummed inputs on
  disk; no DB.
- Could change: the Lens 4 ceiling condition for the 2.0x arm (lower at higher funding, in
  principle). No inventory verdict: A23 is out of scope (design section 2) and this does not
  revisit the owner's adoption.
- Changes nothing if: not applicable; it fills a missing figure.
- Owner flag: no.
- Pre-registration: "Question: what is the 2.0x SSO arm's pre-tax and after-tax CAGR gap versus SPY
  TR on 2006-06-21..2026-09-29 with T-bill funding raised 1.5 and 2.5 points? Comparator: SPY TR
  on one consistent basis. Use: Lens 4 ceiling conditions only."

**P2-05. Alpaca's EOD valuation of AVB.**
- Unknown: Lens 3 lines 409-418 and 481.
- Method: read-only Alpaca portfolio history at 15-minute resolution for one day around 16:00 ET,
  compared with the position list; no order or configuration call.
- Window/config: one trading day, pinned in advance (for example 2026-11-03).
- Runtime: about 15 min. Memory: n/a. Data: read-only broker record.
- Could change: no verdict; turns the AVB mechanism from hypothesis to confirmed or not.
- Changes nothing if: the equity step at 16:00 ET equals AVB's value, which confirms the hypothesis.
- Owner flag: no (read-only broker reads are allowed by design section 3).
- Pre-registration: "Question: does Alpaca's EOD equity drop by AVB's position value (about $746)
  at the close on <date>? Comparator: the position list at 15:45 ET. Use: Lens 3 Q6 only."

**P2-06. French top-decile big-cap momentum after 1994.**
- Unknown: whether a concentrated big-cap momentum portfolio differs from the top-30% proxy the
  screen used (`docs/research/2026-11-04-phase0-trust-audit.md:35, 108`; Lens 2 line 58).
- Method: the frozen file `6_Portfolios_ME_Prior_12_2` has no decile split. A free French
  library file with size-by-momentum quintiles or momentum deciles would need downloading and
  freezing with a checksum, then the registered post-1994 calculation run on it. Driver would
  need writing (a small variant of `scripts/momentum_french_screen.py`).
- Window/config: 1994-01 to the last month in the file; no leverage.
- Runtime: about 15 min after the download. Memory: unknown, small (capped at 6 GB, as in the rules above). Data: free French library.
- Could change: no recorded verdict (the registered screen is closed). A gap above +2 pp/yr would
  narrow the scope of the momentum NO (Phase 0's threshold); below it, nothing changes.
- Changes nothing if: the post-1994 gap is below +2 pp/yr.
- Owner flag: yes, only for where the new file is stored (`data/` is off-limits in this
  investigation; a `results/` or scratch location would be used unless the owner says otherwise).
- Pre-registration: "Question: post-1994 net CAGR gap and NW t of the top big-cap momentum
  portfolio versus the French market. Comparator: French market. Threshold fixed now: +2 pp/yr,
  descriptive only; it does not reopen the registered verdict."

**P2-07. Rejected orders outside 2026-09-25.**
- Unknown: Lens 3 line 482 (Alpaca keeps no record of rejected orders).
- Method: read-only search of the run logs in `/home/flynn/logs/paper_trade_*.log` for rejection
  messages, counted per day.
- Window: 2026-06-20 to 2026-11-04. Runtime: 20-30 min. Memory: n/a. Data: local logs.
- Could change: no verdict; adds a count to Lens 3's unmodelled-behaviour table.
- Changes nothing if: the only rejections are the 6 already known on 09-25.
- Owner flag: no.
- Pre-registration: "Question: how many order rejections do the run logs record per day from
  06-20 to 11-04? Use: Lens 3 row 13 only."

**P2-08. Dividend accrual: holdings sold in an ex-date run, and dividends missing.**
- Unknown: Lens 3 lines 456-460.
- Method: SELECT-only reads of `paper_snapshots`, `paper_trades` and `paper_dividend_accruals`,
  checked against free yfinance dividend histories for every held symbol.
- Window: 2026-06-20 to 2026-11-04. Runtime: about 30 min. Memory: n/a. Data: DB (read-only),
  yfinance (free).
- Could change: no verdict; the count of accrual errors (currently 2) may rise.
- Changes nothing if: no further errors are found.
- Owner flag: no.
- Pre-registration: "Question: is any entitled dividend missing from the accruals, or any
  non-entitled one included, beyond AEP and SPY? Use: Lens 3 snapshot-gap section only."

**P2-09. Adjustment-seam extent inside the pinned window.**
- Unknown: Lens 1 lines 448-450 and 471-472.
- Method: one fresh yfinance download (`auto_adjust=True`) for the 597 members with tape, compared
  with SELECT-only DB closes; flag dates where the DB/yfinance ratio breaks. Driver would need
  writing; it must call yfinance directly and not through `load_ohlcv` (which writes to the DB).
- Window: 2021-01-31 to 2026-04-30. Runtime: about 30 min (Lens 1 estimate). Memory: unknown (capped at 6 GB, as in the rules above).
  Data: free yfinance; DB read-only.
- Could change: Lens 1 tape states for stock rows (unknown or clean to contaminated where a seam
  falls on a traded name), which feeds Lens 2's data condition. Direction for any verdict:
  unknown; a seam is a misdated move, not a bias in one direction.
- Changes nothing if: no seams fall inside the window apart from those already known.
- Owner flag: no.
- Pre-registration: "Question: how many symbol-dates in 2021-01-31..2026-04-30 show a break in the
  DB/yfinance close ratio larger than 0.1%? Use: Lens 1 tape state only."

**P2-10. Code check of the corrected FOMC event study.**
- Unknown: Lens 6 did not reach it (`docs/research/2026-11-04-lens6-implementation-correctness.md:321-335`);
  it is that row's only failing Lens 2 condition (Lens 2 lines 60 and 335).
- Method: read `scripts/fomc_clustered_event_study.py` and the strategy code against its method
  section and brief; hand-replay one month on three symbols with the Lens 6 replay pattern
  (SELECT-only reads).
- Window: one pinned month inside 2012-01-01..2026-04-30. Runtime: 20-40 min. Memory: unknown (capped at 6 GB, as in the rules above).
- Could change: Lens 6 verdict for `cross-asset-fomc-drift-corrected-harness`; if "yes", Lens 2's
  rules would make it the first reliable NO; if "no" or "partly", it stays inconclusive.
- Changes nothing if: the result is "partly" or "unverifiable".
- Owner flag: no.
- Pre-registration: "Question: does the code match the brief's event definition, clustering and
  entry timing, and does a hand replay of one month match the code? Use: Lens 6 verdict and
  Lens 2 code condition only."

**P2-11. Day-shift direction for each July event study.**
- Unknown: Lens 1 lines 376-381, 456-457, 484-485.
- Method: read each study's event join (PEAD, insider, congress, short interest, short volume,
  index deletion, headline sentiment) against the day-shifted tape description.
- Runtime: about 10 min per study, about 70 min total. Memory: n/a. Data: code only.
- Could change: the recorded bias direction (lookahead or delay) for those rows in Lens 2's table;
  not their class (they also lack artifacts). It also settles Phase 0's claim that two bugs
  favoured these studies.
- Changes nothing if: not applicable; it fills a missing direction.
- Owner flag: yes (just over 1 hour).
- Pre-registration: "Question: for each study, did the one-day label shift give lookahead or delay?
  Use: Lens 1 and Lens 2 bias-direction fields only."

**P2-12. Crypto code reads.**
- Unknowns: funding-carry Sharpe convention; cause of the run-2 BTC benchmark mismatch; Keltner and
  StochRSI rules against their 2026-04 descriptions (Lens 8 lines 576-578, 586-589, 633-635;
  Lens 6 lines 194-197).
- Method: `git show 82931a40^:<path>` reads of the removed carry engine, the pipeline benchmark
  loader and the entry classes; no checkout into the main tree.
- Runtime: 1-1.5 h. Memory: n/a. Data: git history.
- Could change: no class (Lens 8 lines 499-509); Lens 6 verdicts for the two entry rows could move
  from unverifiable to yes/partly/no.
- Changes nothing if: conventions match the reports.
- Owner flag: yes (over 1 hour).
- Pre-registration: "Questions: which annualization does the carry engine use; which series and end
  bar does the run-2 benchmark use; do the two entry classes match their descriptions? Use: Lens 6
  and Lens 8 fields only."

**P2-13. Crypto recompute of located artifacts and a stage-1 fee-tier gate replay.**
- Unknowns: Lens 7 recompute states for the crypto rows whose artifacts Lens 8 located (Lens 8
  lines 491-498); fee-tier effect beyond the stored tiers (Lens 8 lines 553-559).
- Method: read the gitignored run folders under `results/research/` named in Lens 8 (for example
  `results/research/archive/research_20260501_010648`), recompute stated figures; replay stage-1
  gates from the stored `wfo_stats_snapshot.json` at alternative fee levels.
- Runtime: about 1 h. Memory: unknown (capped at 6 GB, as in the rules above). Data: local files.
- Could change: Lens 7 recompute status (prose-only to recomputed). Not a Lens 2 class (Lens 8).
- Changes nothing if: figures match the reports.
- Owner flag: yes (about 1 hour).
- Pre-registration: "Question: do the stored per-worker curves and logs reproduce each stated
  crypto figure, and how many stage-1 gates pass at 0.10%, 0.25% and 0.40% per side? Use: Lens 7
  and Lens 8 fields; no verdict changes."

**P2-14. Partial-bar effect on live decisions.**
- Unknown: Lens 6 lines 179-184 and 384-386; Lens 3 lines 390-394 and 480.
- Method: rebuild the 15:45 ET partial bar for the last 30 sessions from free yfinance 5-minute
  data (available for about 60 days back), recompute the live-default ensemble on partial and final
  bars, and count differing buy and sell decisions. Driver would need writing; it must read prices
  directly (not through `load_ohlcv`) and must not touch the intraday observer.
- Window: the 30 most recent sessions at run time, pinned by date when registered. Live defaults
  (min_agree 2, 3-name cap); no leverage.
- Runtime: about 1 h (estimate). Memory: unknown (capped at 6 GB, as in the rules above). Data: free yfinance.
- Could change: no verdict; sizes a live-versus-backtest difference.
- Changes nothing if: decisions match on every session.
- Owner flag: yes (about 1 hour).
- Pre-registration: "Question: on sessions <d1>..<d30>, how many ensemble buy or sell decisions
  differ between the 15:45 ET bar and the final close? Use: Lens 3/Lens 6 parity only."

**P2-15. Lens 7 re-check of 9 unchecked rows and 2 citations.**
- Unknown: Lens 7 lines 343-345 and 360.
- Method: read each of the 9 rows' artifact keys against the report; read the two citations from
  free sources.
- Runtime: about 1 h. Memory: n/a.
- Could change: Lens 7 recompute status for up to 9 rows; Lens 2 recompute condition in principle.
- Changes nothing if: all match.
- Owner flag: yes (about 1 hour).
- Pre-registration: "Question: do the 9 rows' artifact values equal the reported numbers? Use:
  Lens 7 fields only."

**P2-16. Replica re-run with PARA excluded, writing per-trade and sleeve output.**
- Unknown: PARA's effect on the replica (Lens 1 lines 451-453, 475-476; Lens 2 lines 399-403,
  1188). The same run can write the sleeve return series needed by P2-17 and the trade list needed
  by P2-21.
- Method: `scripts/run_capped.sh .venv/bin/python scripts/production_replica_audit.py --window primary --out <new file> --curves <new file>`
  after a small driver change: the replica has no symbol-exclusion option
  (`src/ggTrader/lab/replica.py` checks symbols but has no exclude list), and it writes weights,
  not per-symbol P&L or trades. Driver would need writing: an exclusion list and per-trade and
  sleeve-return outputs, on this branch only (research code, not the paper trader).
- Window/config: primary 2021-01-31 to 2026-04-30 (pinned in the driver); live defaults, 3-name
  cap, 3.3% sizing, next-bar fills, 5 bp costs; the replica models the live book without lab
  leverage (no max-leverage setting).
- Runtime: about 1 h (Lens 1 estimate; the artifact records none). Memory: unknown (capped at 6 GB, as in the rules above). Data: DB tape.
  **DB-cache writes.**
- Could change: Lens 1 tape state of the replica (unknown to clean on PARA if immaterial).
  Not its Lens 2 class: the replica also fails Lens 2's power condition (MDE 4.24 > +1.0 pp/yr), so
  it stays inconclusive whatever the result. It could change the condition attached to the
  replica's book-level reading ("rules out above about 4.2 pp/yr if PARA is immaterial").
- Tape note: PARA's bars were rewritten on 2026-11-04 (incident section). A re-run of the
  unchanged replica now would read the post-incident PARA series, not the one the 2026-10-19 run
  traded. The other 18 rewritten symbols would also be read in their rewritten form, so even
  the PARA-excluded arm is not an exact like-for-like comparison with the stored result.
- Changes nothing if: the CAGR gap and t move by less than their rounding.
- Owner flag: yes.
- Pre-registration: "Question: with PARA excluded (and nothing else changed), what are the primary
  window's CAGR gap versus SPY TR and NW t? Comparator: the stored primary arm (-0.64 pp/yr, t
  -0.69). Use: Lens 1 tape state and the Lens 2 condition text; the recorded NO-GO and closure are
  not revisited."

**P2-17. Stock sleeve versus equal-weight S&P 500 (RSP).**
- Unknown: whether sizing or selection limits the sleeve (Phase 0 hypothesis 2;
  `docs/research/2026-11-04-phase0-trust-audit.md:106`).
- Method: compare the sleeve return series from P2-16 with RSP total return (free yfinance), NW t
  on the daily difference. Driver would need writing (short).
- Window: 2021-01-31 to 2026-04-30. Runtime: 15-30 min after P2-16. Memory: unknown, small (capped at 6 GB, as in the rules above).
- Could change: no recorded verdict. Lens 9's sleeve-scale illustration (MDE about 45 pp/yr at
  t >= 3) implies low power, so a non-significant result is the expected outcome under most effect
  sizes.
- Changes nothing if: |t| < 2.
- Owner flag: yes (depends on P2-16).
- Pre-registration: "Question: sleeve daily return minus RSP TR, annualized mean and NW t over the
  pinned window. Threshold fixed now: t > 2 read as 'sizing, not selection, limits the book'
  (Phase 0's wording), descriptive only."

**P2-18. A23 funding and distribution tax-character records.**
- Unknown: Phase 0 line 112. A23 is out of scope for this investigation (design section 2; the
  allocation has a scheduled review on 2027-09-30).
- Method: free ProShares records of SSO's distributions and tax character, and its published
  financing terms. No private account records.
- Runtime: about 1 h. Memory: n/a.
- Could change: no investigation verdict.
- Changes nothing if: not applicable.
- Owner flag: yes (out-of-scope topic; listed only because Phase 0 listed it).
- Pre-registration: not drafted unless the owner brings the topic into scope.

**P2-19. PIT core, ensemble_ic and ensemble_kelly under the DSR fix, with stored excess series.**
- Unknown: the "corrected" Sharpe 0.59 predates the DSR units fix, and a failed DSR check changes
  which parameters a fold deploys, so its OOS returns would change (Lens 1 lines 25-35, 363-368);
  no SPY-excess t is stored, so no MDE (Lens 9 lines 555-565).
- Method: `scripts/run_capped.sh .venv/bin/python scripts/ic_kelly_rebaseline_wfo.py --out <new file>`
  on current code, with a small addition to store each strategy's OOS daily excess series and
  `excess_stats` from `src/ggTrader/lab/significance.py`. Driver would need that addition.
- Window/config: 2021-01-31 to 2026-04-30 (pinned via `scripts/anchor_fix_reproduction_wfo.py`);
  gated WFO, single strategy, leverage 1.0 as in the original run (no blend, so the 2.0 default
  does not apply).
- Runtime: about 1 h (recorded 1,479 + 1,348 + 652 s). Memory: unknown (capped at 6 GB, as in the rules above). Data: DB tape.
  **DB-cache writes.**
- Could change: the recorded figures 0.59 / 0.41 / 0.50 in either direction; Lens 1 states
  (contaminated to clean on DSR); Lens 9 MDE for three rows. Lens 2 classes could stay
  inconclusive on power.
- Changes nothing if: figures reproduce within rounding.
- Owner flag: yes.
- Pre-registration: "Question: on current code, what are each strategy's OOS Sharpe, CAGR, CAGR gap
  versus SPY TR and NW t on 2021-01-31..2026-04-30? Comparator: stored 2026-09-23 figures and SPY TR.
  Verdict rule: original bars; no looser bar."

**P2-20. Crowding and decay mapping.**
- Unknown: external crowding per family (Lens 5 lines 491-496, 562-563).
- Method: map each of the 16 tested published stock anomalies to McLean and Pontiff's appendix
  (free PDF) and record publication year against each test window.
- Runtime: 1-2 h. Memory: n/a.
- Could change: no verdict.
- Changes nothing if: not applicable; it fills a description.
- Owner flag: yes (over 1 hour).
- Pre-registration: "Question: which of the 16 are in McLean and Pontiff's 97 predictors, and was
  each test window after publication? Use: Lens 5 description only."

**P2-21. After-tax replica with short-term lots.**
- Unknown: Lens 4 lines 493-497 and 563-564.
- Method: feed the P2-16 trade list and dividends into `TaxLotBook`
  (`src/ggTrader/lab/tax_lots.py`, a library with no current consumer). Driver would need writing.
- Window/config: as P2-16; tax brackets 24/15 and 37/20 as in Lens 4.
- Runtime: 2-4 h after P2-16 (Phase 0 estimate). Memory: unknown (capped at 6 GB, as in the rules above).
- Could change: no recorded verdict. The closure ends the replica's test at the pre-tax fail
  ("after_tax: Not run: primary pre-tax bar failed"), so no after-tax result can change it. It
  fills the Lens 4 after-tax figure for the replica.
- Changes nothing if: any result (for the verdict).
- Owner flag: yes.
- Pre-registration: "Question: replica primary after-tax CAGR gap versus SPY held and taxed at the
  end, at 24/15 and 37/20. Use: Lens 4 only."

**P2-22. Stored excess series for the remaining artifacts without a t-statistic.**
- Unknown: Lens 9's 33 "not computable" rows minus those in P2-02 and P2-19 (core+SPY blend,
  cross-asset sleeve and holdout, A23 arms, century leverage and others named in
  `docs/research/_lens9_power.json`); Lens 4's missing confidence intervals.
- Method: re-run each driver to a new file with a stored monthly excess series and NW t:
  `scripts/core_spy_blend.py --rerun --out <new file>` (a linear k-scaled blend of the PIT core
  and SPY with a 5% cash reserve, `scripts/core_spy_blend.py:1-24`; it has no max-leverage
  setting, so that parameter does not apply), `scripts/cross_asset_sleeve_wfo.py --force --out <new file>`,
  `scripts/static_leverage_aftertax.py --frozen-inputs data/external/a23_inputs --out <new file>`,
  `scripts/century_leverage.py` (frozen Shiller inputs). Each needs a small output addition.
- Windows: as pinned in each driver (2021-01-31..2026-04-30; 2006-06-21..2026-09-29; century as
  stored).
- Runtime: 2-4 h total (estimate; allocation drivers ran in seconds, WFO drivers 10-40 min).
  Memory: unknown (capped at 6 GB, as in the rules above). **DB-cache writes** for the two WFO drivers; none for the frozen-input drivers.
- Could change: Lens 9 MDEs and Lens 4 confidence intervals; through them, Lens 2's power condition
  in principle (classes could stay inconclusive on other conditions).
- Changes nothing if: MDEs exceed each row's bar.
- Owner flag: yes.
- Pre-registration: "Question: for each artifact, the annualized mean excess versus its stored
  comparator, NW t and MDE at t >= 3. Use: Lens 4 and Lens 9 fields; no verdict changes on these
  alone."

**P2-23. Pre-registration before 2026-09-10.**
- Unknown: Lens 7 lines 296-301 and 362-364.
- Method: search the owner's private session transcripts for each pre-September study's brief text
  and timestamp. These are outside the repo and are private records.
- Runtime: 1-2 h. Memory: n/a.
- Could change: Lens 7 pre-registration states for up to 48 rows; not a class by itself.
- Changes nothing if: no briefs are found.
- Owner flag: yes (private records).
- Pre-registration: "Question: for each of the 48 rows without a brief, does a timestamped pass
  bar exist before its result? Use: Lens 7 fields only."

**P2-24. Code check of the 16 remaining unexamined rows.**
- Unknown: Lens 6 lines 321-335 and 379-380 (17 rows; FOMC is P2-10).
- Method: read each strategy file against its report's method section and hand-replay one month on
  three symbols (SELECT-only), as Lens 6 did.
- Runtime: 20-40 min each, 5-11 h total. Memory: unknown (capped at 6 GB, as in the rules above).
- Could change: Lens 6 verdicts and Lens 2's code condition for those rows; classes could stay
  inconclusive on other conditions (most of these rows also lack artifacts or predate fixes).
- Changes nothing if: verdicts come back "partly" or "unverifiable".
- Owner flag: yes.
- Pre-registration: "Question per strategy: do rules, lookbacks, timing and PIT use match the
  report, and does a one-month replay match the code? Use: Lens 6 verdicts."

**P2-25. July event studies on the corrected tape, pinned.**
- Unknown: Lens 1 contamination and drifting windows for PEAD, insider, congress, short interest,
  short volume, index deletion, headline sentiment, MAX and related July rows (Lens 1 lines
  310-313; Lens 7 lines 277-282; Phase 0 line 109).
- Method: driver would need writing: one pinned driver per study that runs the corrected harness
  and writes JSON, on the pattern of `scripts/audit_reruns_wfo.py`. The existing
  `*_correlation_check.py` scripts use "today" as their end date and must not be used as they are.
  Event inputs come from tables already backfilled in the DB (read-only).
- Window/config: 2021-01-31 to 2026-04-30 (and each study's long window where it had one, for
  example PEAD from 2015-06); single-strategy WFO, no blend.
- Runtime: 30-60 min each per Lens 7, 1-3 h each per Phase 0; 5-20 h total. Memory: unknown (capped at 6 GB, as in the rules above).
  **DB-cache writes.**
- Could change: Lens 1 states (contaminated to clean), Lens 7 (prose-only to recomputable), and the
  sign of each recorded NO. Lens 2 classes would likely stay inconclusive on power.
- Changes nothing if: each re-run is a NO at a similar size.
- Owner flag: yes.
- Pre-registration: "Question per study: on current code and tape, OOS CAGR gap versus SPY TR and
  NW t on 2021-01-31..2026-04-30. Comparator: SPY TR. Verdict rule: the study's original bar where
  written; otherwise reported descriptively."

**P2-26. Non-gated arms and one-change re-runs for gate revisions.**
- Unknowns: simpler-test outcome for 48 gated rows; how much each of 18 revisions moved each verdict
  (Lens 2 lines 1031-1044 and 1187-1189).
- Method: a fixed-rule (no WFO) arm per gated candidate on the pinned window, as the replica did
  for the ensemble; and one-change re-runs reverting one revision at a time. Driver would need
  writing.
- Window: 2021-01-31 to 2026-04-30; leverage per the original config (1.0 where production, noted
  per run).
- Runtime: tens of hours (10-40 min per run, many runs). Memory: unknown (capped at 6 GB, as in the rules above). **DB-cache writes.**
- Could change: Lens 2's gate-machinery and researcher-freedom answers; recorded NOs where a
  fixed-rule arm differs.
- Changes nothing if: fixed-rule arms agree with the gated verdicts.
- Owner flag: yes.
- Pre-registration: drafted per run before it starts; the bar is the original study's.

**P2-27. Crypto like-for-like fee-tier re-run and full-range BTC beta.**
- Unknowns: Lens 8 lines 553-559 and 629 (full-range beta needs a combined full-range curve that
  was never stored).
- Method: restore the legacy pipeline from `82931a40^` into a scratch worktree (not the main
  checkout) and re-run one configuration varying only the fee tier, writing a combined curve.
  The tape has changed since (Binance.US deletions; Kraken 4h re-import), so results are on
  today's tape, not the original runs'.
- Runtime: hours to a day or more. Memory: unknown (capped at 6 GB, as in the rules above).
- Could change: Lens 8's fee answer; not a Lens 2 class under its written rules (Lens 8 lines
  499-509).
- Changes nothing if: the fee ordering matches the stored sweep.
- Owner flag: yes (crypto is intentionally off; restoring removed code is a separate decision).
- Pre-registration: drafted per run; benchmark BTC buy-and-hold over the same window.

**P2-28. Clean 2010-2020 holdout.**
- Unknown: regime dependence of the 2021-2026 verdicts; the replica holdout is non-citable on a
  faulty tape (Lens 9 lines 571-577 and 648; Lens 1 lines 51-52, 273-275).
- Method: build an identity-checked 2010-2020 tape from free data (flag implausible levels and
  moves of 40% or more not tied to known corporate events), then
  `scripts/production_replica_audit.py --window holdout --out <new file>`. Free yfinance lacks many
  delisted names, so a free rebuild risks survivorship gaps.
- Window: the replica's registered holdout window. Runtime: days. Memory: unknown (capped at 6 GB, as in the rules above). **DB-cache
  writes** unless the tape is built outside the DB.
- Could change: the replica holdout from non-citable to citable; the regime unknown.
- Changes nothing if: the holdout fails both registered bars again.
- Owner flag: yes.
- Pre-registration: the replica brief's holdout bars, unchanged.

**P2-29. Longer frozen paper trading with an input archive.**
- Unknown: live-versus-backtest parity per decision (Lens 3 lines 390-394). Lens 3 shows paper
  trading cannot detect sleeve alpha in any practical time (23-40 years for 10 pp/yr; lines
  335-376).
- Method: archive each run's input bars and signals beside the snapshot. That is a change to the
  live trader, which this investigation does not make.
- Runtime: calendar time. Memory: n/a.
- Could change: parity findings only.
- Owner flag: yes (live change outside this investigation).
- Pre-registration: not drafted here.

## 5. Can Phase 2 be skipped?

Design section 6 allows skipping Phase 2 only if **both** hold: Phase 1 confirmed every Phase 0
hypothesis, **and** no lens raised an unknown that could change a verdict. Neither holds.

**Phase 0 hypotheses (design section 4) against the lenses:**

| Phase 0 hypothesis | Phase 1 result |
|---|---|
| 1. No real stock-picking edge; replica "Reliable NO, high trust" | Not confirmed as rated. Lens 2: replica inconclusive (MDE 4.24 > 1.0 bar; PARA traded unflagged); 0 reliable NOs in the whole record (`docs/research/2026-11-04-lens2-verdict-audit.md:27-42, 360-403`) |
| 2. Live book about 84% SPY; sleeve too small to move it | Consistent: replica 85.33% SPY, 9.37% stocks (Lens 2 line 373); sleeve-scale MDE about 45 pp/yr (Lens 9 lines 49-54) |
| 3. Early beat-SPY figures from five bugs; corrected 0.59 | Partly: the bug signatures are gone, but 0.59 itself predates the DSR (deflated Sharpe ratio significance gate) units fix, which changes OOS returns too; "five bugs alone" not confirmed (Lens 1 lines 357-370) |
| 4. WFO NO-GOs weak; pass bars unreachable on five years | "Unreachable" confirmed; "t >= 3 needs IR near 1.3" partly overturned (1.62-1.68 at 80% power) (Lens 9 lines 58-66) |
| 5. Paper shortfall mostly configuration churn (-1.9 / -2.6 / +1.4) | Reproduces only on Phase 0's own method; corrected -3.02 / -2.28 / +1.60; idle cash -2.46 pp is the largest part (Lens 3 lines 25-49) |
| 6. Snapshot excess fits the dividend add-back, unproven | Confirmed (Lens 3 lines 16-24) |
| 7. 2x SPY out of reach; "+10 to +15 pp needed" | Target is 9.57 pp/yr on 2021-2026 but 0.77 to 3.54 on longer windows; no sound stock-selection result after 1994 reaches its window's gap (Lens 4 lines 34-90) |

**Unknowns that could change a recorded verdict or class (from section 4):** P2-02 (French screens'
power condition), P2-03 (month-end Treasury on current code), P2-09 (how many "adjustment seams", fake one-day price moves where old and new dividend
adjustments meet, sit in the price table; this feeds tape states), P2-10 (FOMC code check, the row's only failing condition), P2-19 (the 0.59 point-in-time S&P 500 core and
the IC/Kelly figures under the DSR significance-gate fix), P2-22 (power for artifacts without a t-statistic), P2-24
(16 unexamined code rows), P2-25 (sign of the July event-study NOs), P2-26 (gated versus fixed-rule
verdicts), P2-28 (clean holdout). P2-16 (PARA) changes the replica's tape state and its stated
condition but not its class.

**The lead's conclusion:** Phase 2 is **not** skipped, because unknowns exist that could change a
recorded verdict or class (listed above). Whether any of them is run is the owner's choice. If none
is approved, the Phase 3 synthesis is written with these items marked "unknown, re-run not run"
(design section 11, item 2).

## 6. Usage note

From the ledger (`.superpowers/sdd/2026-11-04-root-cause-investigation/progress.md`) and the SDD
workspace files, Tasks 1 to 11:

| Dispatch kind | Count | Notes |
|---|---|---|
| Implementer dispatches (one per task or sub-task) | 14 | Tasks 1-5, 6a, 6b, 7, 8, 9, 10a, 10b-i, 10b-ii, 11. Code tasks 1-3 per the plan's Sonnet role; lens and judgement tasks Opus (the Task 6a report confirms Opus). Plus 1 Opus dispatch for Lens 5 that failed on the usage limit (below) |
| Fix rounds | 13 | Task 2: 1; Task 4: 1; Task 5: 2; one each for 6a, 6b, 7, 8, 9, 10a, 10b-i, 10b-ii, 11 |
| Review rounds | 27 | One review diff per round in the workspace; every task ended with a clean review. Reviews for Lens 5 and Lens 8 ran on Sonnet and Lens 4's on Opus per the ledger; the ledger does not record the model of the others |
| Haiku extraction passes | 28 from the ledger (20 inventory, 4 Lens 1, 4 Lens 5); 3 more not recorded in the ledger | The workspace also holds extraction outputs for Lens 7 (2 files) and Lens 8 (1 file); the ledger does not record those passes, so their count is inferred from files only |
| This checkpoint (Task 12) | 1, plus 1 fix round | Run on Opus 5.5 with no subagents; the commit trailer reads Sonnet 5.5 because the task brief set that line |

The agent cannot see the owner's plan usage or remaining quota. One Opus dispatch (Lens 5) hit the
five-hour usage limit once, at about 21:50 PST on 2026-11-04; it was retried after the reset at
22:20 PST, and nothing was lost (the Haiku extraction for Lens 5 had already completed). The
owner may wish to check the Claude usage settings before approving Phase 2 items; the item costs
in section 4 are compute and wall-clock estimates, not quota estimates.

## 7. Owner decision block

The owner is asked to approve, by id, which of these items run. **Default if there is no answer:
approve none.** Nothing runs until the owner approves.

- **Decision 0 (applies to every item tagged "writes to the price cache" below):** 29 scripts
  under `scripts/` load prices through `load_ohlcv` or the cached loader. Run through them, an item
  can (a) rewrite a symbol's whole price history in the `ohlcv` table when the symbol has no cached
  bars or its first cached bar does not match its recorded listing date (this is what rewrote 19
  symbols, PARA among them, on 2026-11-04), (b) refresh recent bars, (c) write `symbol_inception`
  rows and (d) possibly write `ohlcv_no_data` rows. The owner can accept these writes for the
  approved items, or require a read-only price path for them (which may need driver changes and
  can change which symbols load). Default: neither; tagged items do not run.
- **Restoring the 19 rewritten symbols** (incident section) is a separate owner decision; it is not
  a Phase 2 item and nothing here restores them.
- Items under one hour: P2-01, P2-02, P2-03 (writes to the price cache: runs only if Decision 0 is
  approved), P2-04, P2-05, P2-06, P2-07, P2-08, P2-09, P2-10.
- Items of about one to two hours: P2-11, P2-12, P2-13, P2-14, P2-15, P2-16 (writes to the price
  cache: runs only if Decision 0 is approved), P2-17 (needs P2-16, so it also depends on Decision
  0), P2-18 (out-of-scope topic), P2-19 (writes to the price cache: runs only if Decision 0 is
  approved), P2-20, P2-23 (private records).
- Items of several hours or more: P2-21 (needs P2-16, so it also depends on Decision 0), P2-22 (its
  two WFO drivers write to the price cache: those parts run only if Decision 0 is approved), P2-24,
  P2-25 (writes to the price cache: runs only if Decision 0 is approved), P2-26 (writes to the price
  cache: runs only if Decision 0 is approved), P2-27 (whether the restored legacy crypto pipeline
  writes to the database is not established; treated as needing Decision 0), P2-28 (writes to the
  price cache unless the tape is built outside the database: runs only if Decision 0 is approved),
  P2-29 (needs a live change outside this investigation).

Approved items would run at most two at a time, each under the 6 GB cap, with its pre-registration
committed before it starts, and usage reported after each batch. Nothing runs until the owner
approves.
