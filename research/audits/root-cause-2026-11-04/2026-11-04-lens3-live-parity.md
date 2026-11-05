# Lens 3: live versus backtest parity (paper trader)

**Date:** 2026-11-04. **Branch:** `investigation/root-cause-2026-11-04`.
**Scope:** read-only diagnosis. No recommendations, fixes or deployments. This lens supplies
evidence to the owner's paper-parity plan (`docs/working_task_list.md`, Priority 1). It does
not write or replace that plan.
**Raw results:** `docs/research/_lens3_snapshot_gap.json`.

Abbreviations, expanded once: WFO = walk-forward optimization (the lab's backtest harness);
OOS = out of sample; TR = total return (dividends reinvested); pp = percentage points;
bp = basis points (0.01%); EOD = end of day; ET/PT = US Eastern/Pacific time; DB = the
TimescaleDB database; SD = standard deviation; MDE = minimum detectable effect.

## Summary

1. **The snapshot gap is the dividend accrual, and the broker never paid those dividends.
   The gap-equals-accrual part is not independent; only "never credited" has a broker
   source.** From 2026-08-20 to 2026-11-04, the daily gap
   `value - (cash + positions)` in `paper_snapshots` equals the cumulative
   `paper_dividend_accruals` total to within $6.59 on every day, and to the cent on most
   days. It is $320.24 from 10-30 onward. Alpaca records no dividends: there are zero DIV
   activities since 2026-05-01, and the only non-trade activity since 08-15 is FEE. Live
   broker cash equals snapshot cash on 11-04 ($5,177.86). Two accrual rows are over-stated,
   by $5.86 in total: AEP ($5.76, bought in the ex-date run) and SPY ($0.10, shares added
   on the ex-date).
2. **Phase 0's era figures (-1.9 / -2.6 / +1.4 pp) reproduce exactly, but only with its own
   method.** That method compares the stored snapshot value (which includes the dividend
   accrual and the MNST split correction) at about 15:45 ET with SPY *price* closes from
   the `ohlcv` tape. The tape includes a dividend-adjustment seam and the partial 11-04
   bar. That method gives -1.92 / -2.60 / +1.37 pp. The broker-consistent value against
   SPY total return over the same moments gives **-3.02 / -2.28 / +1.60 pp**. Alpaca EOD
   equity against SPY total-return closes gives **-3.69 / -2.32 / +1.40 pp** (the last era
   ends at the 11-03 close). The total since 06-24 is **-3.76 pp** on the broker-consistent
   basis (-3.24 pp by Phase 0's method).
3. **Where the shortfall sits.** This is an arithmetic decomposition and an estimate.
   Measured shortfall over the whole period: **-3.76 pp**. Components:

   | Component | pp |
   |---|---|
   | Idle cash before the SPY sweep | **-2.46** |
   | Stock sleeve | -0.88 |
   | SPY sweep sleeve (the uncredited 09-18 SPY dividend) | -0.15 |
   | Residual | -0.29 |

   About -0.85 pp of the stock sleeve's -0.88 is one broker artifact. Alpaca paper never
   applied MNST's 2-for-1 split, so the 09-29 sale credited half the position's value. By
   configuration era:
   - Cash dominates before 08-21.
   - The blend-era stock picks lost about -3.3 pp over E3 and E4 (07-30 to 09-22): -0.99
     in E3 and -2.33 in E4. The MNST hit is not in either era; it lands in E6.
   - The frozen-era stock sleeve gained +1.6 pp.
4. **Fills before the close match the backtest's close-price assumption on average:**
   +0.3 bp notional-weighted for stocks (n = 106, SD 39 bp per fill) and -0.2 bp for SPY
   (n = 52). The exceptions are 44 orders that filled at the next open. They come from
   after-close runs before 06-27 and on 07-30, and from the 09-29 manual liquidations.
   They cost about 47 bp against the prior close (about $290, estimate).
5. **Frozen paper trading has almost no power to detect anything at sleeve size.** The
   3-name sleeve's active return against SPY has an annualized SD of 17-22%. Detecting a
   sleeve excess of 10 pp/yr at 80% power takes about 5,800-10,000 trading days (23-40
   years). Detecting 1 pp/yr takes over 2,000 years. In a 3-month window the MDE is about
   +96 to +126 pp/yr.

## Falsifiers

Written before running the Lens 3 queries. Lens 6 had already reported that the code and
DB match for the gap (`docs/research/2026-11-04-lens6-implementation-correctness.md`), so
H-gap was partly supported in advance.

- **H-era (Phase 0 item 8, `docs/research/2026-11-04-phase0-trust-audit.md:75-77`):** the
  paper shortfall vs SPY by era is -1.9 pp (06-24 to 08-21), -2.6 pp (08-21 to 09-30) and
  +1.4 pp (09-30 to 11-04). Each of these would overturn it:
  - Recomputed on a broker-consistent value against SPY total return over the same
    dates, any era moves by more than 0.5 pp or changes sign.
  - Two independent value sources (DB snapshot and Alpaca EOD equity) disagree with each
    other by more than 0.5 pp.
  - Git history and the changelog do not support the era boundaries.

  **Verdict against these rules.** The first era (06-24 to 08-21) is **overturned**. On
  the broker-consistent basis it moves 1.10 pp (-1.92 to -3.02), past the 0.5 pp limit.
  The two value sources also disagree by 0.67 pp (U -3.02, Alpaca EOD -3.69). The second
  era moves 0.32 pp (-2.60 to -2.28) and the third 0.23 pp (+1.37 to +1.60). Their two
  sources agree within 0.2 pp, so **both survive**. The signs are unchanged, and git and
  the changelog support the boundaries. The JSON records
  `phase0_era_figures_survive_correction: false` because the first era fails.
- **H-era-cause ("most paper shortfall is config churn, not signal"):** overturned if the
  decomposition puts most of the shortfall in the stock sleeve's own picks rather than in
  idle cash, broker artifacts and transition events.
- **H-gap (Phase 0, `docs/research/2026-11-04-phase0-trust-audit.md:114-115`):** "snapshot
  value exceeds cash plus positions by $135-$320 every day from 09-15 to 11-04; fits the
  dividend add-back but is not proven to be the only cause." Each of these would overturn
  it:
  - On any day since 08-20, the gap differs from the cumulative accrual by more than $10.
  - The gap exists before the accrual feature shipped (2026-08-19).
  - Alpaca shows credited dividends. That would mean the add-back double counts, and the
    gap is not a pure reporting overlay.
- **H-frozen ("+1.96% vs SPY +0.2% in the frozen window is noise"):** overturned if the
  frozen window has enough statistical power to detect an effect of the observed size.

## Steelman

This is the strongest evidence that live paper trading tracks the backtested strategy:

- **Rule code is right.** Lens 6 hand-computed the live-default ensemble on 10 symbols for
  March to May 2025. The code's entries and exits match exactly (the first record in
  `docs/research/_lens6_replays.json`, `hand_result == code_result`).
- **Price-level parity at the fill.** The WFO fills at the signal bar's close. Live fills
  about 14-15 minutes before the close, and on average at that close:
  - stocks: +2.5 bp mean, +0.3 bp notional-weighted (n = 106);
  - SPY: -1.9 bp mean (n = 52).

  All 212 Alpaca orders from 06-23 to 11-04 show status `filled`. None is partial.
- **Accounting reconciles.** The snapshot gap is fully explained (residual of $6.59 or
  less). Snapshot cash equals broker cash. The SPY sweep sleeve tracked SPY price with
  about zero error, apart from the dividend the broker never paid.
- **The frozen configuration behaves as specified.** From 09-30 the book holds 4 to 5
  names (including the untradable AVB and the core-era names it started with). The trader
  makes 1 or 2 stock trades a day at the 3.3% size. The sweep holds cash near the 5%
  reserve, and cash averaged 5.1% of the account.
- **The frozen era is positive.** The stock sleeve added +1.6 pp over SPY in five weeks.
  The account beat SPY's total return on every basis: +1.60 pp broker-consistent, +1.40
  pp at EOD.

The steelman covers mechanics, not edge. Most of the shortfall came from eras that no
longer run (idle cash, the blend, missed exits). Those eras say little about the current
configuration in either direction.

## Data and value bases

Four value series, all in the JSON under `value_bases`:

| Basis | Definition | Use |
|---|---|---|
| V | `paper_snapshots.portfolio_value` as stored = broker value + split correction + cumulative dividend accrual (`src/ggTrader/paper/trader.py:1056`, saved at `src/ggTrader/paper/trader.py:1089`) | Phase 0's basis; reference only |
| B | cash + sum of snapshot position `market_value` = V minus the accrual (split-corrected positions) | Decomposition (economic view of MNST) |
| **U** | cash + sum of `qty * current_price` = the broker's own figure at run time, without accrual or split correction | **Primary basis for era returns** |
| EOD | Alpaca portfolio-history 1D equity (end of day) | Independent cross-check, aligned to the close |

**Why U is primary.** The task asks for the value the broker would report. U equals the
Alpaca `portfolio_value` the trader read (`src/ggTrader/paper/trader.py:1030`) before it
added any corrections. Neither U nor EOD contains the uncredited dividends, so both
understate a real broker's total return by about 0.31% of the account over the period.
V adds the dividends back and is closer to the economic value. Results on all bases are
reported.

**SPY total return.** SPY price comes from three sources: yfinance daily closes, Alpaca
IEX (Investors Exchange feed) daily closes, and the snapshot's own SPY `current_price` or
Alpaca IEX 1-minute bars at the snapshot minute. One dividend falls in the window: $1.889
per share, ex-date 2026-09-18. yfinance gives $1.889 and the Alpaca corporate-actions rate
in `paper_dividend_accruals` gives $1.888834. It is reinvested at the 09-18 close.

The `ohlcv` tape is not used for SPY return, because of a seam. SPY bars for 08-25 to
09-17 are dividend-adjusted (×0.99752 vs yfinance raw), while the surrounding bars are
raw. Across 08-21 to 09-30 the tape therefore yields price return (-0.404%), not total
return (-0.157%). This is the Lens 1 mixed-adjustment pattern
(`docs/research/2026-11-04-lens1-backtest-validity.md`).

**DB caveats.**
- Rows before 08-22 had their `run_date` shifted +1 day by the 2026-08-22 migration. The
  06-24 row was created 06-23 17:10 PT, after the close.
- MNST snapshot rows for 08-11 to 09-22 were restated on 2026-09-22, including
  `portfolio_value` (changelog 2026-09-22 and the memory note).
- The DB is therefore not the original record for those rows. U removes the restatement.

## Era boundaries

Era dates are snapshot run dates. A run at about 15:45 ET on date D sets the book that is
held from D onward.

| Era | Start → end | What changed | Evidence |
|---|---|---|---|
| E1 | 06-24 → 07-15 | Core ensemble, 3.3% size per buy, 60-68% idle cash | `08252f3c` (06-20, paper CLI and cron); first snapshot 06-24 all cash; `src/ggTrader/paper/risk.py:18` `position_pct 0.033` |
| E2 | 07-15 → 07-30 | Unintended dry-run freeze: the blend was wired 07-13 (`08414ff6`, `19e4a066`) but cron did not pass `--live` | `58d4a03f` (07-29) "flip cron wrapper to --live, end unintended dry-run freeze"; no `paper_trades` rows between 07-15 and 07-30; snapshot cash constant at 66,267.89 |
| E3 | 07-30 → 08-21 | 3-sleeve blend trading live, no sweep | 07-30 run: 16 buys of about $1K (sleeve sizing), filled at the 07-31 open; `docs/changelog.md` 2026-06-27→08-17 summary |
| E4 | 08-21 → 09-22 | Blend plus SPY cash sweep (5% reserve) | `docs/changelog.md` 2026-08-22 (later), `CASH_SWEEP_ENABLED=true`; first sweep buy of about $58K on 08-24 |
| E5 | 09-22 → 09-28 | Standalone S&P 500 core revert plus sweep | `d4b69340`; deployed about 23:00 PT on 09-22 (memory note, core revert deployed 2026-09-22); the 09-23 run crashed (`833125e5`) |
| E6 | 09-28 → 09-30 | Missed-exit catch-up, `MAX_POSITIONS=3`, 15 legacy and 4 orphan manual liquidations, 365-day lookback | `fe5c4a72` (09-28 16:51 PT); `17b0971f` deployed 09-29 about 17:15 PT (`docs/changelog.md` 2026-09-29) |
| E7 | 09-30 → 11-04 | Frozen configuration | Memory note on the 365-day deploy (frozen about 3 months); `docs/research/2026-11-02-paper-freeze-readiness.md` |

There are no snapshots for 07-29, 09-16 and 09-23. Each of those days had no completed
run.

## Era attribution

### Per era (U vs SPY TR at the snapshot moment; EOD cross-check)

| Era | Account U | SPY TR | **Shortfall U** | Shortfall B | Shortfall V | EOD acct | EOD SPY TR (yf / IEX) | **Shortfall EOD** |
|---|---|---|---|---|---|---|---|---|
| E1 | +1.57% | +2.78% | **-1.21** | -1.21 | -1.21 | +1.61% | +2.89 / +2.88% | **-1.29** |
| E2 | +0.27% | -1.63% | **+1.90** | +1.90 | +1.90 | -0.19% | -1.74 / -1.74% | **+1.55** |
| E3 | -0.37% | +3.34% | **-3.71** | -2.76 | -2.69 | -0.71% | +3.24 / +3.24% | **-3.95** |
| E4 | -0.94% | +1.31% | **-2.26** | -2.32 | -2.26 | -1.08% | +1.25 / +1.27% | **-2.33** |
| E5 | -0.95% | -1.16% | **+0.21** | +0.17 | +0.17 | -0.84% | -1.01 / -1.03% | **+0.17** |
| E6 | -0.26% | +0.00% | **-0.27** | -1.11 | -1.10 | -0.58% | -0.39 / -0.41% | **-0.19** |
| E7 | +1.80% | +0.20% | **+1.60** | +1.60 | +1.77 | +0.32%* | -1.09 / -1.04%* | **+1.40*** |
| All 06-24→11-04 | +1.07% | +4.84% | **-3.76** | -3.76 | -3.45 | -1.48%* | +3.09 / +3.09%* | **-4.57*** |

\* The EOD series ends at the 11-03 close, because 11-04 was still trading.

How the bases differ:
- **U vs B (E3, E6):** this is MNST. U drops about $951 at 08-11, when the broker kept the
  pre-split share count. B carries the full value until the 09-29 sale credited half
  (-$874).
- **EOD vs U (from 08-18):** these are Alpaca's daily equity minus the snapshot's broker
  value U (cash + quantity × price), on the same date.
  - Before 08-11: the offset is -$435 to +$177. That is consistent with the 15:45 ET
    snapshot being taken 15 minutes before the close.
  - 08-11 to 08-17: -$84 to +$48, so about $0.
  - From 08-18: -$456 to -$1,107, mostly near -$750. It is -$748 on 11-03.

  **AVB alone fits.** The merged AVB holding is about $746, and its price has been frozen
  at 184.06 since 08-17. The hypothesis is that Alpaca's EOD series values it at about
  zero, while the live `portfolio_value` includes it at 184.06. Alpaca's `last_equity` for
  11-03 ($100,941.02) is about $748 below the 11-03 snapshot U. This mechanism is
  **unproven**.

  MNST cannot contribute: U already carries MNST at the broker's unsplit quantity, and
  the offset is about $0 from 08-11 to 08-17. The larger gap, -$613 to -$1,976, appears
  only when EOD is measured against the *stored* value V. V also includes the MNST split
  correction (about $1,000 until 09-28) and the dividend accrual ($74-$320). Series
  against U: JSON `broker_eod_offset`.

### Phase 0 boundaries

| Era | Phase 0 | Phase 0 method reproduced | **U vs SPY TR** | EOD vs SPY TR | Verdict |
|---|---|---|---|---|---|
| 06-24 → 08-21 | -1.9 | -1.92 | **-3.02** | -3.69 | **Overturned** under the pre-registered rule. It reproduces on Phase 0's method, but moves 1.10 pp on U (the MNST split, which U carries at the unsplit quantity from 08-11) and 1.77 pp on EOD (AVB valued at about 0 from 08-18). The sources disagree by 0.67 pp |
| 08-21 → 09-30 | -2.6 | -2.60 | **-2.28** | -2.32 | **Survives** (moves 0.32 pp). Phase 0's SPY figure is price, not TR, because of the tape seam (+0.25 pp) |
| 09-30 → 11-04 | +1.4 | +1.37 | **+1.60** | +1.40 (to 11-03) | **Survives** (moves 0.23 pp) |
| Headline since 06-24 | +1.39% vs +4.63% | +1.386% vs +4.628% | +1.07% vs +4.84% TR | -1.48% vs +3.09% (to 11-03) | The headline uses V and SPY price on the partial 11-04 bar |

Phase 0's method was: V, at about 15:45 ET, against `ohlcv` SPY daily closes (06-24 close
733.24, partial 11-04 bar 767.17).

### Decomposition (B basis, estimate)

Method (JSON `decomposition_B_basis_estimate`):
- Between consecutive snapshots, position P&L is quantity × price change, or sale
  proceeds minus the starting value if the position was sold.
- Cash earns 0. No INT (interest) activity exists at Alpaca.
- Each component's excess is its P&L minus its starting value × SPY TR over the interval.
- Components are summed, not compounded, and divided by the era's starting B.
- Before 08-24, SPY return per interval uses yfinance close-to-close, while positions are
  priced at about 15:45 ET. **Those intervals are estimates.**

| Era | Avg cash / stock / SPY weight | Cash drag | Stock sleeve | SPY sweep | Residual | Sum | Measured B |
|---|---|---|---|---|---|---|---|
| E1 | 68% / 32% / 0 | -2.08 | +0.87 | 0 | -0.12 | -1.33 | -1.21 |
| E2 | 64% / 36% / 0 | +1.11 | +0.90 | 0 | 0.00 | +2.01 | +1.90 |
| E3 | 57% / 43% / 0 | -1.61 | -0.99 | 0 | -0.02 | -2.63 | -2.76 |
| E4 | 8% / 36% / 57% | +0.09 | -2.33 | -0.15 | -0.01 | -2.40 | -2.32 |
| E5 | 5% / 34% / 61% | +0.06 | +0.12 | 0 | -0.01 | +0.17 | +0.17 |
| E6 | 5% / 22% / 73% | 0.00 | -0.98 | 0 | -0.12 | -1.11 | -1.11 |
| E7 | 5% / 10% / 85% | -0.01 | +1.61 | 0 | -0.01 | +1.59 | +1.60 |
| **All** | 31% / 29% / 40% | **-2.46** | **-0.88** | **-0.15** | -0.29 | -3.78 | -3.76 |

Reading the table:
- **Stock sleeve:** E6's -0.98 is almost all the MNST sale at half value (-$874, about
  -0.85 pp). This is a paper-broker artifact. *Assumption, not tested here:* a live
  brokerage account would have applied the split. Alpaca paper recorded no SPLIT activity
  (see "Snapshot gap").
- **SPY sweep:** E4's -0.15 equals the uncredited 09-18 SPY dividend (83.17 shares ×
  $1.889 = $157).
- **Residual:** P&L from fill to snapshot on positions opened in the same run, plus fees
  and rounding.
- **Excluding MNST**, the stock sleeve over the whole period is about 0 pp: E1/E2 +1.8,
  E3/E4 -3.3 (blend era, 28-30 names), E5-E7 +1.6 (excluding MNST).

**H-era-cause, partly supported.** Idle cash is the largest single piece (-2.5 of -3.8
pp). Broker artifacts add about -1.0 pp (MNST, the uncredited SPY dividend). The blend-era
stock sleeve (E3 to E4) still lost about 3.3 pp against SPY. That loss is strategy
behaviour under a configuration that included missed exits, not "churn". With 3-30 names
over 4.5 months, signal and configuration cannot be separated cleanly.

## Snapshot gap

Daily series from 2026-08-19: JSON `snapshot_gap.series`. Each row holds the gap, the
cumulative accrual at the snapshot's `created_at`, the residual, the Alpaca DIV credited
(always 0), and the split correction.

- The gap is 0.00 on 08-19, before the accrual feature ran. It is $74.06 on 08-20 and
  08-21: the three backfilled rows MPWR + VZ + PNR = 5.05 + 54.51 + 14.50. It then steps
  up with each new accrual: $104.61, $124.82, $134.74, $137.97, $148.72, $156.49 and
  $320.24. The 10-30 step adds SPY $157.09 and GL $6.66.
- The residual (gap minus accrual) is 0.00 on most days. Its range is -2.30 to +6.59. Most
  non-zero values fall on trading days. They are consistent with the seconds between the
  trader's `get_account` and `get_positions` calls, but this is not proven.
- **Alpaca DIV series:** empty. `get_account_activities_by_type(DIV, after=2026-05-01)`
  returns `[]`. Non-trade activities since 08-15 are FEE only (REG, TAF and CAT fees, a
  few cents each). This matches the 2026-08-18 note in
  `src/ggTrader/paper/dividend_check.py`. I re-ran the reviewer's check with
  `get_account_activities` for types DIV, DIVCGL, DIVCGS, DIVFEE, DIVFT, DIVNRA, DIVROC,
  DIVTW, DIVTXEX, CGD, SPLIT, MA, REORG, REO, NC, SPIN, JNLC, JNLS, INT, CSD and CSW, from
  2024-06-01. It returns one entry: the 2024-06-21 $100,000 JNLC funding journal. The
  account has never been credited a dividend, a split, a merger, interest or any other
  cash flow.
- **Accrual errors.** `qty_held_at_ex_date` uses the snapshot at or *on* the ex-date
  (`src/ggTrader/paper/dividend_check.py:55-76`). A run on the ex-date therefore counts
  shares bought after that day's open, which are not entitled to the dividend:
  - AEP: 6.067 shares bought in the 08-10 run, ex-date 08-10, $5.76 accrued.
  - SPY: 0.051 extra shares from the 09-18 sweep, about $0.10.

  The other 17 rows have the same quantity in the prior day's snapshot.
- **Verdict:** confirmed. The gap is the code's cumulative uncredited-dividend accrual
  (`src/ggTrader/paper/trader.py:1056`). The broker paid nothing.
  - **"Gap = accrual"** is confirmed from the DB accrual table and the code at
    `src/ggTrader/paper/trader.py:1056`. These are not independent: the same code writes
    both the accrual and the snapshot.
  - **"Never credited"** is the only part with an independent broker source: empty
    dividend activity and the activity re-check above.
  - Broker cash equalled snapshot cash ($5,177.86) on 11-04. That was checked on one day
    only. The consequence for the daily-loss halt (about 2.7% instead of 3%) is Lens 6's
  finding and is not re-derived here.

## Unmodelled behaviours

This builds on Lens 6's parity table (rows 1-15 in
`docs/research/2026-11-04-lens6-implementation-correctness.md`). Figures are measured
unless marked "estimate".

| # | Behaviour | Modelled by WFO? | Paper evidence (this lens) |
|---|---|---|---|
| 1 | Signals from the partial 12:45 PT bar | No (final close) | **Not quantifiable from stored data.** Decision-time bars are not preserved, and partial bars are overwritten later (Lens 6). The readiness review marks "No missed exits" unknown per session |
| 2 | Fill about 15 minutes before the close | Close + 5 bp | Pre-close fills vs same-day close: stocks n = 106, mean +2.5 bp, SD 38.7 bp, notional-weighted +0.3 bp; SPY n = 52, -1.9 bp, SD 11.6 bp |
| 3 | Fills at the next open | No | 44 orders ($61K): 06-25 and 06-29 (13:30 PT cron, before the 06-27 change), 07-31 (07-30 run at 18:37 PT), 09-29 (manual liquidations). Cost vs prior close: 47 bp notional-weighted, about $289 in total, about 0.28 pp (estimate; the reference close is a counterfactual) |
| 4 | Position count | No cap in WFO | Mean names excluding SPY: E1 9.7, E3 28.5, E4 30, E5 30, E7 4.3 (cap 3 plus AVB and starting names). The changelog (2026-09-28) calls about 30 live vs about 3 in the backtest "the main driver" of the exposure gap |
| 5 | Idle cash | Partly: WFO sizes 3% of cash, and its book also sat about 61% in cash (memory note, 2026-06-24) | Cash drag vs SPY: -2.46 pp over the period, almost all before 08-21 (estimate) |
| 6 | SPY cash sweep | No | 52 SPY fills. Tracks SPY price; the only loss vs TR is the uncredited dividend (-0.15 pp). 32 sweep sells and 20 sweep buys ($173K notional) |
| 7 | Missed exits and catch-up | First exit bar after entry | 6 catch-up exits on 09-29 (AEE, HAS, HON, MNST, PNR, VZ); 3 trading days with no run (07-29, 09-16, 09-23) |
| 8 | Manual liquidations and orphans | No | 09-29: 15 `manual_legacy_liquidation` ($10,358) and 4 `manual_orphan_liquidation` ($3,560, the MidCap names H/HGV/WLK/WWD, outside the S&P 500 universe). Open-fill cost vs prior close about $112 (estimate) |
| 9 | Untradable holding (AVB, merged into EQR) | No | Held since 08-10; price frozen at 184.06 since 08-17; $746 (0.7% of the account). Valued at about zero in EOD equity (hypothesis); excluded from slots since 09-30 |
| 10 | Broker corporate-action gaps | Adjusted data | MNST 2:1 split never applied by Alpaca paper; sold 09-29 for $868.78 vs about $1,743 corrected, so -$874 realized (-0.85 pp). 19 dividends ($320.24) never credited |
| 11 | Dry-run freeze | No | E2 (07-15 to 07-30): no orders; the book was held static |
| 12 | Catastrophe stop and halts | No | One stop fire: KNF, 09-25, $755 |
| 13 | Rejected orders | No | 6 of 8 buys rejected on 09-25 (`docs/changelog.md` 2026-09-25). They do not appear in Alpaca order history: all 212 listed orders are `filled`. Rejections on other days cannot be counted from the broker |

## Detectable effect

**Question:** a frozen sleeve of about 10% of the account. How many trading days are needed
to detect an excess return of X pp/yr over SPY with 80% power?

**Formula:** N_days = 252 × ((z_0.975 + z_0.80) × σ_ann / X)², with z_0.975 + z_0.80 = 1.960
+ 0.842 = 2.802.

**Inputs (from the DB):**
- σ_ann is the annualized SD of the sleeve's daily active return (sleeve return minus SPY
  TR) between consecutive snapshots.
  - Frozen era (25 intervals): σ_daily 1.414%, so σ_ann **22.4%**.
  - Whole period (89 intervals): σ_daily 1.080%, so σ_ann **17.1%**.
- The account-level active SD in the frozen era is 2.59%/yr, consistent with a 10% sleeve
  plus a sweep that tracks SPY closely.

| X (sleeve excess, pp/yr) | Days (σ 22.4%) | Years | Days (σ 17.1%) | Years | Days if X is account-level (sleeve 10X, σ 22.4%) |
|---|---|---|---|---|---|
| 1 | 996,166 | 3,953 | 581,321 | 2,307 | 9,962 |
| 2 | 249,041 | 988 | 145,330 | 577 | 2,490 |
| 5 | 39,847 | 158 | 23,253 | 92 | 398 |
| 10 | 9,962 | 39.5 | 5,813 | 23.1 | 100 |

**MDE:**
- Over a 63-day (3-month) window, the MDE is 2.802 × σ_ann × √(252/63): **+126 pp/yr**
  (σ 22.4%) or **+96 pp/yr** (σ 17.1%).
- Over 252 days it is +63 or +48 pp/yr.

**Assumptions:**
- Daily active returns are independent and roughly normal. Fat tails in a 3-name sleeve
  make this optimistic.
- The test is two-sided at α = 0.05.
- σ is estimated from small samples (25 and 89 intervals).
- The sleeve composition (3 names at 3.3%) stays fixed.
- SPY TR is the benchmark.

**H-frozen.** The frozen-era sleeve's mean active return is +0.60%/day, with t = 2.14 over
25 intervals. Taken at face value that is about +150 pp/yr, which is not plausible as a
persistent effect. The window was picked after the fact from several era cuts. Over the
whole period t = 0.73. The window cannot detect any realistic effect, so **H-frozen
("noise") stands**: the frozen result is not evidence either way.

## Questions and answers

1. **What does the paper trader do that the tests did not model?** Answered. See
   "Unmodelled behaviours":
   - partial-bar signals;
   - next-open fills on some days;
   - about 30 names during E3 to E5;
   - the SPY sweep;
   - missed exits and the catch-up;
   - 19 manual liquidations;
   - AVB;
   - broker corporate-action gaps (MNST split, uncredited dividends);
   - the dry-run freeze;
   - rejected orders.

   The partial-bar effect on *which* names are traded is **unknown**, because decision-time
   bars were not stored. The cheapest test is to save each run's input bars and signals
   alongside the snapshot and compare them with a final-close replay.
2. **Do fills before the close behave as the backtest's fill assumptions do?** Answered
   for price: yes on average (+0.3 bp stocks, -0.2 bp SPY against the same-day close),
   with ±39 bp noise per stock fill. The replica's next-bar-close assumption was not
   compared. Signal-level parity is unknown (see item 1).
3. **How much of the paper shortfall does each configuration era explain?** Answered. See
   the per-era table and the decomposition: -3.76 pp in total on the broker-consistent
   basis. Phase 0's -1.9 / -2.6 / +1.4 reproduce only on its own method. Corrected: -3.02
   / -2.28 / +1.60 (U) and -3.69 / -2.32 / +1.40 (EOD).
4. **What explains the snapshot-value discrepancy?** Answered and confirmed. It is the
   cumulative uncredited-dividend accrual. The DB accrual table and the snapshot gap share
   one writer (`trader.py:1056`), so that match is not independent; only "never credited"
   has an independent broker source (Alpaca DIV is empty and cash matches, checked on one
   day). Two rows are over-accrued by $5.86.
5. **How long would frozen paper trading need to run to detect an effect of a given size
   in a sleeve of about 10%?** Answered: 23-40 years for 10 pp/yr at the sleeve level.
   See the table.
6. **Why is EOD equity below the snapshot?** Partly answered. Against the broker value U,
   the offset is about $0 from 08-11 to 08-17. From 08-18 it is -$456 to -$1,107, mostly
   near -$750 (-$748 on 11-03). AVB alone (about $746, price frozen since 08-17) fits,
   valued at about zero in Alpaca's EOD series. MNST does not contribute, because U already
   carries it at the unsplit quantity. Against the stored value V the gap is -$613 to
   -$1,976. V adds the MNST split correction (about $1,000) and the dividend accrual
   ($74-$320). The mechanism is unproven, because Alpaca's EOD valuation rule was not
   inspected. The cheapest test is portfolio history at 15-minute resolution on one day
   around 16:00 ET, compared with the position list.

## Verification

Commands and outputs. Scratch scripts are in the session scratchpad, under
`l3/analyze.py` and `l3/build_json.py`.

- DB (SELECT-only, read-only session, statement timeout): full `paper_snapshots` (91
  rows), `paper_trades` (205), `paper_dividend_accruals` (19, total $320.24),
  `paper_pending_orders` (0), and SPY `ohlcv` 06-22 to 11-04 (96 rows, interval `1d`).
- Alpaca read-only tools:
  - `get_account_info`: paper account confirmed; identifier not recorded. Cash 5,177.86;
    `last_equity` 100,941.02.
  - DIV activities: `[]`. Fix round: 21 activity types since 2024-06-01 return only the
    2024-06-21 $100,000 JNLC funding entry.
  - Non-trade activities since 08-15: FEE only.
  - `get_portfolio_history` 1D, 06-20 to 11-04: 95 points; cash flows FEE only.
  - `get_all_positions`: 5 names; the position sum matches `long_market_value` within
    $1.89.
  - `get_orders` status=all, 06-20 to 11-04: 212 orders, all `filled`.
  - `get_stock_bars` SPY: 1Day IEX, plus 1Min IEX at 07-15, 08-21 and 11-04 around
    19:44-20:48 UTC. The SIP (consolidated) feed returned 403.

  No order, cancel, replace, close, watchlist or configuration tool was called.
- yfinance: SPY daily closes and dividends (06-18 $1.904, 09-18 $1.889); daily closes for
  85 traded symbols.
- Cross-checks:
  - SPY TR from yfinance vs Alpaca IEX closes agree within 0.05 pp on every era.
  - The `ohlcv` tape matches yfinance except 08-25 to 09-17 (×0.99752) and the partial
    11-04 bar.
  - The Phase 0 reproduction gives V vs `ohlcv`: -1.92 / -2.60 / +1.37 and +1.386% vs
    +4.628%.
  - Decomposition sums match the measured B shortfall within 0.13 pp in every era.
- Gate: `scripts/check_lens_report.py` on this file.

**Unverified:**
- The partial-bar signal effect.
- Alpaca's EOD valuation rule for AVB (hypothesis only).
- The mirror-image accrual error. The accrual takes its share count from the snapshot on
  the ex-date (`src/ggTrader/paper/dividend_check.py:55-76`), so a holding sold in an
  ex-date run would be missed even though it was entitled. No such case was searched for.
- The fill-cost statistics (n = 106, +0.3 bp; n = 52 SPY; 44 next-open fills at 47 bp)
  and the 212-order total were not independently re-verified by the review.
- Rejected-order counts beyond 09-25.
- Whether any dividend on a held name is missing from the accruals. The corporate-actions
  feed was not re-queried per holding.
- The pre-08-24 SPY timing in the decomposition (close-to-close vs 15:45 ET snapshots).
- One EOD outlier on 07-30 (-$435 vs U; Alpaca repeats the 07-28 equity for 07-29).
- The pre-08-22 `run_date` shift and the 09-22 MNST restatement mean the DB is partly
  restated. U removes the MNST part only.

## Done checklist

- ✅ Falsifiers written before the Lens 3 queries (H-gap partly pre-supported by Lens 6, as stated)
- ✅ Steelman
- ✅ Era boundaries from git, changelog and memory notes, with evidence
- ✅ Per-era returns vs SPY TR on the broker-consistent basis, with a second source (Alpaca EOD) and a second SPY source (IEX)
- ✅ Phase 0's -1.9 / -2.6 / +1.4 compared: reproduced on its method, revised on the corrected bases
- ✅ Within-era decomposition (estimate, labelled)
- ✅ Snapshot gap vs accruals vs Alpaca DIV, confirmed by two sources; both series recorded in the JSON
- ✅ Unmodelled behaviours with paper evidence (fill times, next-open fills, liquidations, AVB, MNST, rejections)
- ✅ Detectable-effect formula, inputs and table
- ❓ Partial-bar signal parity: unknown (decision-time bars not stored)
- ❓ Alpaca EOD valuation of AVB: unknown (hypothesis consistent with data)
- ❓ Rejected orders outside 09-25: unknown (no broker record of rejections)
