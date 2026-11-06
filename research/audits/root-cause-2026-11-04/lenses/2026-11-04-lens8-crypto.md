# Lens 8: crypto

**Date:** 2026-11-04. **Branch:** `investigation/root-cause-2026-11-04`.
**Scope:** read-only diagnosis for the root-cause investigation
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`, section 5, Lens 8).
Crypto is parked by the owner's choice; that is not treated as a defect here. This lens
says what was tried, what the evidence shows and what is unknown. It does not propose
strategies or say what to try.
**Raw results:** `research/audits/root-cause-2026-11-04/lenses/_lens8_tape_checks.json` (tape counts, benchmark table,
annualization, all 25 crypto inventory rows, live-evidence counts).
**Inputs:** inventory `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json` (25 `crypto-` rows); Lens 1
`research/audits/root-cause-2026-11-04/lenses/_lens1_tape_state.json` (deferred crypto tape checks to this lens); Lens 2
`research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`; Lens 6 `research/audits/root-cause-2026-11-04/lenses/_lens6_verdicts.json`; Lens 9
`research/audits/root-cause-2026-11-04/lenses/_lens9_power.json`; Lens 5 `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md`;
an unreviewed Haiku history pull (32 studies, used as leads and checked where relied on);
`docs/archive/edge_search_report_2026-06-08.md`; `docs/changelog.md`; git tree at
`82931a40^` (the commit before the 2026-06-16 removal); the local, gitignored run folders
under `results/research/`; TimescaleDB (SELECT only, 30-120 s statement timeouts, symbol and
venue filters); yfinance daily closes as an external cross-check.

Abbreviations, expanded once: WFO = walk-forward optimization; OOS = out of sample;
IS = in sample; B&H = buy and hold; YTD = the pipeline's "YTD" window, which is in fact the
trailing 12 months, not calendar year to date; pp = percentage points; RT = round trip (buy
plus sell); DD = maximum drawdown; PIT = point in time; OLS = ordinary least squares
regression; DST = daylight saving time; UTC = Coordinated Universal Time; ETF = exchange-traded
fund; IBIT = iShares Bitcoin Trust ETF; PF_XBTUSD = Kraken Futures BTC/USD linear perpetual;
ccxt = the exchange-access library the legacy code used; EW = equal weight.

## Summary

1. **No crypto result beat buy-and-hold BTC over any three-year window** (the one-year
   funding-carry window, +0.45% vs BTC -23%, beat BTC but is not like-for-like: the strategy is
   delta-neutral). Every portfolio with a recoverable artifact lost to BTC over its three-year
   range by 5 to 44 pp per year (ratio of cumulative returns -0.19 to 0.82; the negative value is
   edge-search run 3, which lost money while BTC gained). The only wins over BTC are
   trailing-12-month ("YTD") windows in which BTC fell 17-40%. For the 2026-04-01, 04-02 and
   05-01 runs those windows sit inside the WFO folds that chose the coins and parameters, so
   they are in-sample. For edge-search runs 2 and 3 about 7 of the 12 months (2025-11-01 to
   2026-06-08) are the 20% holdout, after selection: over that holdout stretch run 2 returned
   -9.4% and run 3 -18.9% vs BTC -42%, i.e. they lost less than BTC out of sample in a falling
   market. In the full 12-month regression (not the holdout alone) their BTC beta was low and the
   intercept not significant (beta 0.10 and 0.20; t = 0.43 and -0.44), and both lost
   to BTC over three years.
2. **The steelman (2026-05-01: 48.46% vs BTC -21.34%) is real in the artifact and does not
   survive scrutiny as evidence of an edge.** It is a one-year in-sample replay on the pre-repair
   tape (about 55 of its 366 days, 15%, fall in the 2026-03-08..05-05 period of time-shifted
   bars, later wiped), using a universe picked by 2026 trading volume. BTC beta is 0.26-0.31
   (depending on how 4h bars are aligned to days) and does not explain the gain; the leftover
   (alpha) has t = 1.51-1.56, which is not significant. Over the same run's three-year range it
   lost to BTC (137.6% vs 167.1%).
3. **The current tape is clean on the equity bug classes** (no day-level duplicates, no off-grid
   or shifted timestamps, daily labels match yfinance) **but has crypto-specific defects:**
   43.7% of Binance.US 4h bars match the Kraken bar at the same time in open, close and volume
   (copied); delisted Binance.US
   coins were deleted; about 100 equity tickers, 52 quote-less coin symbols and 7 un-split
   `XBT` pair codes sit under the crypto venues. Universe construction was not point-in-time.
4. **Annualization:** the current lab (`src/ggTrader/lab/metrics.py:10-29`) uses 252 days, which
   understates a daily 7-day crypto Sharpe by a factor of 0.83. **No published crypto number is
   affected:** all of them came from the removed legacy pipeline, whose Sharpe uses 365 days
   (recomputed and matched below). The one crypto-labelled lab run is uncited.
5. **"Fees dominate" (2026-06-08) is directionally supported but its headline comparison is
   confounded.** A controlled cost sweep on 7 coins does show costs killing gate passes (36 →
   14 → 13 → 0). The +13.8% → -10% flip, however, compares different universes, venue tapes
   and selected coins, and both runs charged 0.3% per side of slippage that the "0.02% vs 0.25%"
   framing leaves out.
6. **Live crypto left almost no record.** The `trades` table is empty and `orders` holds one
   dry-run row. The only realized-PnL figure (+$9.88 over about 5 days on Kraken, April 2026) is
   prose whose records and script were deleted.

## Falsifiers

Written before the evidence was read. Each states what would have shown the opposite of the
working view (no crypto edge; tape fit for purpose).

- **F1 (beat BTC):** any crypto strategy whose cumulative return over its full stated window
  exceeds BTC B&H over the same window on the same tape, net of its own costs. Stronger form:
  the same holds in a window not used to select coins or parameters.
- **F2 (alpha not BTC beta):** a regression of strategy daily returns on BTC daily returns with
  a positive intercept at t ≥ 2 in a window not used for selection.
- **F3 (fees not decisive):** a same-universe, same-tape run in which changing only the fee
  rate leaves the verdict unchanged.
- **F4 (clean tape):** zero day-level duplicate bars, zero off-grid timestamps, daily labels
  matching an external source at lag 0, BTC/ETH gaps under 0.1% of days, no cross-venue
  copying, and no deleted delisted symbols.
- **F5 (PIT universe):** universe code that ranks coins by volume known at each historical
  date, or a stored universe file dated at the start of each window.
- **F6 (annualization matters to published numbers):** a published crypto Sharpe that was
  computed with a 252-day factor on 7-day data.
- **F7 (live evidence exists):** order or trade rows in the database for live crypto, or a
  surviving trade log, sufficient to compute a live return.

Outcome: F1 is not met over any three-year window. It holds only in its weak form, for
trailing windows in which BTC fell: in-sample for the 04-01, 04-02 and 05-01 runs, and partly
out of sample (the post-2025-11-01 holdout) for edge-search runs 2 and 3, where the strategy lost
less than BTC. F2 is not met (best t = 1.56). F3 is not met. F4 is met for the equity bug classes and fails on cross-venue copying
and deleted delisted symbols. F5 is not met. F6 is not met. F7 is not met.

## Steelman

The strongest evidence for a crypto edge is the Kraken-era WFO pipeline in April-May 2026:

| Run | Window | Strategy | BTC B&H (tape) | ETH B&H (tape) |
|---|---|---:|---:|---:|
| `research_20260501_010648` "YTD" (the 48.46% headline, promoted to live) | 2025-05-01..2026-05-01 | **+48.40%** | -17.46% | +26.97% |
| `research_20260401_225306` "YTD" | 2025-04-02..2026-04-02 | +27.38% | -20.90% | +9.65% |
| `research_20260402_123248` "YTD" (deployed 2026-04-02) | 2025-04-03..2026-04-02 | -10.65% | -19.88% | +12.78% |

The numbers in the artifacts are real: the 48.40% total return re-derives from the equity curve
stored in `results/research/archive/research_20260501_010648/plots/combined_portfolio_ytd_dashboard.html`
(48.72% from the first stored point, 48.40% from the 1,000 start value), and the report's
Sharpe 1.2378 re-derives at 1.2438. In that window the strategy beat both BTC and ETH in raw
terms, with low BTC exposure (beta 0.26-0.31), so it was not simply levered BTC.

How well it survives:

- **Window and selection.** The "YTD" window lies inside WFO folds 5-8 (test periods
  2025-03-27..2026-04-30), and the coins and parameters were chosen by IS and OOS robustness
  scores across all folds (`results/research/archive/research_20260501_010648/research_report.md`,
  "WFO Fold Timeline"; selection_reason `wfo_robustness`). The report's own line is "WFO-selected
  parameters replayed on the YTD window, no re-optimization": the optimizer had already seen it.
  So this is an in-sample replay.
- **Full range.** The same run lost to BTC over its three-year range: 137.60% vs 167.09% (ratio
  0.82, -5.3 pp per year). The 2026-04-02 run lost 53.89% vs 140.97% (ratio 0.38).
- **Single regime.** All three "YTD" results beat BTC in one BTC drawdown year (the 04-02 one
  by losing less). When BTC rose, the
  strategies trailed it.
- **Universe.** The coin list was ranked by trading volume at run time (2026), then backtested
  from 2023. It includes coins that only became large later (PENGU, FARTCOIN, HYPE, TRUMP, MON,
  RAVE in `results/research/archive/research_20260501_010648/top_ccxt_volume.json`). That is
  look-ahead in coin selection.
- **Tape.** All three runs predate the 2026-05-05 repair of the timestamp bug
  (`docs/changelog.md:2058-2068`): the live writer stored bars 7 hours early from 2026-03-08, about
  346 off-grid 4h bars per symbol by early May, and all 1.7M crypto 4h rows were then wiped and
  re-imported. Sized from the changelog dates, the affected stretch (2026-03-08 to the run date) is
  about 26 of 366 days (7%, ~156 4h bars) of the 04-01 and 04-02 "YTD" windows, 55 of 366 days
  (15%, ~330 bars) of the 05-01 "YTD" window, and 2-5% of each three-year window. The changelog
  also attributes two earlier phantom ADA gaps (ending 2025-04-01 and 2025-11-29) to the same
  shift, so some symbols may have been affected earlier; how many is unknown. The three-year BTC
  comparisons rest mostly on unaffected data, and the stated BTC figures sit within 0.1-2.7 pp of
  today's clean tape after costs, so the "lost to BTC over three years" result does not depend on
  the bad bars. The "YTD" strategy figures could; that is not checkable because the old tape no
  longer exists.
- **Alpha.** OLS of daily strategy returns on BTC: beta 0.263 (standard error 0.041), intercept
  +50.1% per year simple, t = 1.51, R² 0.10, with BTC re-sampled onto the equity curve's own 4h
  timestamps and both taken at the last bar of each UTC day. Taking BTC's daily last bar directly
  from the tape gives beta 0.314, intercept +50.4%, t = 1.56 (an independent reviewer got 0.314,
  standard error 0.040). The beta is alignment-sensitive within 0.26-0.31; the conclusion is not. Adding an EW basket of the run's own coins: intercept
  +42.0% per year, t = 1.39, R² 0.26. Neither intercept is significant, and both are in-sample.
- **Fees.** Net of 0.4% fee + 0.3% slippage per side (1.4% RT), so the cost assumption is, if
  anything, conservative for Kraken taker.
- **Live fills.** Promoted to live on 2026-05-01; the live trader then had entry droughts
  (sizing unwired, BTC regime gate) and moved venue on 2026-05-23 without trading. No live
  return exists to compare.

Verdict on the steelman: the artifact supports "beat BTC in one falling-BTC year, in sample, on
a partly time-shifted tape".
It does not support "beat BTC".

## What was tried

All 25 inventory crypto rows plus the Haiku pull, reconciled. Lens columns come from the
committed upstream files. In the Lens 2 column, "inconclusive" for STATUS rows is a label, not a
test result (`research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` status_note); "absent" means the row is not
in that file. In the Lens 1 column, every crypto row is "unknown" because Lens 1 deferred crypto
tape state to this lens. "Local artifact" means a gitignored run folder exists under
`results/research/`.

| Id / study | Universe | Venue / costs per side | Benchmark used | Strategy | Headline | Window | Verdict | L2 | L6 | Evidence state |
|---|---|---|---|---|---|---|---|---|---|---|
| crypto-pipeline-runs-2026-03 (28 runs) | 1-30 coins from symbols files | Kraken; not stated per run | EW B&H of the run's coins | legacy pipeline, TA entries | all 28 runs below EW B&H (e.g. BTC-only +4.15% total vs 74% B&H CAGR) | unpinned | none | absent | absent | prose only; run folders not found |
| crypto-wfo-research-snapshots-kraken (03-29, 04-01, 04-02 runs) | 22-33 coins, live top volume | Kraken 0.4% + 0.3% | BTC B&H, SPY | per-coin WFO tournament | 04-02: CAGR 15.46%, YTD -10.66% vs BTC -20.92% | 2023-04-03..2026-04-02 | deployed | absent | unverifiable | local artifact (03-29 run not found) |
| crypto-config-tweaks-reverted-2026-04-02 | 32 → 26 coins | Kraken | none | metric weights | reverted | 2026-04-02 | reverted | inconclusive | absent | prose only |
| crypto-kraken-live-bot-pnl | live | Kraken live | none | live engine | +$9.88 realized, 24 round trips | ~5 days to 2026-04-06 | "modestly profitable" | inconclusive | unverifiable | prose only; CSVs and script deleted |
| crypto-wfo-tournament-robustness | per coin | Kraken | none | keltner/stoch_rsi entries | robustness 0.99 (TAO) | 2026-04 | none | absent | unverifiable | prose only |
| crypto-wfo-recalibration-2026-05-01 | 30 coins (top-25 file + live top-50) | Kraken 0.4% + 0.3% | BTC B&H, SPY | per-coin WFO | YTD 48.46% vs BTC -21.34% | 2023-05-01..2026-04-30 | promoted to live | absent | unverifiable | local artifact `results/research/archive/research_20260501_010648` (Lens 6 said not found) |
| crypto-regime-filters | 22 coins live | Kraken live | none | BTC/altcoin gates | 17 days zero entries | 2026-05 | disabled | absent | unverifiable | prose only |
| crypto-universal-trailing-stops | live | Kraken live | none | exit rule | none | 2026-05-06 | none | absent | partly | code at 82931a40^ |
| crypto-diversity-cap-disabled-2026-05-07, crypto-trade-frequency-gate-2026-05-07, crypto-strategy-usage-analyzer-2026-05-07, crypto-wfo-overfitting-param-stability | research universe | Kraken | none | gate and tooling changes | configuration counts | 2026-05-07..09 | disabled / NO-OP | inconclusive or absent | absent | prose only; no return claim |
| crypto-live-ohlcv-tz-bug | all crypto | n/a | n/a | data bug | ~346 off-grid bars per symbol | 2026-03-08..05-05 | fixed | absent | unverifiable | fixed in current tape (checked below) |
| crypto-wfo-textbook-reset | 7 coins (BTC, ETH, TRX, DOGE, XMR, DASH, ADA) | Kraken: 0, 0.2+0.3, 0.25+0.3, 0.4+0.3 | none | textbook gates | 36 / 14 / 13 / 0 gate passes of 231; 0 coins in every variant | 2023-05-02..2026-05-01 | closed | inconclusive | unverifiable | local artifacts; inventory's raw file `research_20260511_193048` is the frictionless run, the realistic-fee run is `research_20260511_162755` |
| crypto-venue-migration-path-d | BTC, ETH, DOGE (TRX deferred) | Kraken → Binance.US | none | 22 textbook combos | decision | 2026-05-12 | defer TRX | inconclusive | absent | prose only; combos never traded |
| crypto-funding-carry-phase4 | BTC spot + PF_XBTUSD | Kraken spot/futures | none stated | delta-neutral funding carry | +0.45%, Sharpe 4.44, 8 trades | 2025-05-15..2026-05-15 | "directionally credible, not numerically trustworthy" | inconclusive | absent | prose only for results; input data still in DB; code at 82931a40^ |
| crypto-binance-us-live-trader | 15-19 coins | Binance.US live | none | legacy params | never traded | 2026-05-23..06-16 | on hold | inconclusive | unverifiable | DB snapshots only (see Live evidence) |
| cross-sectional momentum harness (Haiku only) | Binance.US 4h | Binance.US | none | CrossSectionalMomentum | no result recorded | 2026-06-06 | tooling | n/a | n/a | local folders `results/research/cross_sectional_research_*`, no headline |
| crypto-edge-search-2026-06-08 (3 runs) | top-11 / all-51 / top-100 | Binance.US 0.02% + 0.3%; Kraken 0.25% + 0.3% | BTC B&H, SPY | per-coin WFO cascade | +13.8% CAGR vs BTC +38.8%; -10% on Kraken | 2023-06-09..2026-06-08 | no deployable edge | inconclusive | absent | local artifacts `results/research/research_20260608_083103` etc. |
| crypto-execution-venue-fees | n/a | Alpaca, Kraken, Binance.US | n/a | memo | fee table | 2026-07-06 | n/a | absent | absent | memo, `research/crypto/execution-venues/2026-07-06-crypto-execution-venues.md` |
| crypto-perp-funding-carry-candidate | PF_XBTUSD | Kraken Futures | n/a | feasibility | ~1 year of funding history | 2026-07-20 | infeasible | inconclusive | absent | feasibility only |
| crypto-tsmom-btc, crypto-spot-etf-flow-persistence, crypto-defi-tokenized-equity-arb, crypto-web-research-batch-2026-10-25 | literature | n/a | n/a | candidates | none | 2026-07..10 | untested / parked | inconclusive | absent | never tested |
| crypto-ibit-overnight-session | IBIT ETF | US equity listing | none | descriptive split | overnight +186% vs intraday -43% | 2024-01-11..2026-10-23 | closed | inconclusive | absent | artifact `research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json` |

Reconciliation notes: the Haiku pull lists the 28 pipeline runs, the 03-29 and 04-01 Kraken runs,
the three textbook-reset fee variants and the edge-search runs individually; the inventory
groups them. The Haiku pull omits the 2026-05-01 20:52 and 20:58 two-coin runs found locally
(`research_20260501_205240`: 98.63% vs BTC 162.02% stated; `research_20260501_205849`: 90.48%),
which also lost to BTC over the full range. Nothing in the Haiku list contradicts the inventory.

## Benchmark comparison

Method: BTC and ETH B&H from the TimescaleDB `ohlcv` 4h close, first to last bar inside the
window, on the venue the run used (`kraken_spot` or `binanceus_spot`), gross of costs. yfinance
`BTC-USD`/`ETH-USD` daily closes are a cross-check (within 0-4.5 pp of the tape in every window).
"Ratio" is the spec section 1 measure: strategy cumulative return divided by BTC cumulative
return over the same window; it is undefined when BTC's return is zero or negative, so the
growth-factor ratio (1 + strategy) / (1 + BTC) is given as well. pp/yr is the CAGR difference.
Strategy figures are as stated in each run's `research_report.md`, net of that run's costs.

| Run (venue) | Window | Basis | Strategy | BTC | ETH | Ratio vs BTC | Growth ratio | pp/yr vs BTC | Beat BTC / ETH |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 0401_225306 full (Kraken) | 2023-04-01..2026-03-31 | IS replay, net | 52.66% | 138.95% | 15.15% | 0.38 | 0.64 | -18.6 | no / yes |
| 0401_225306 YTD | 2025-04-02..2026-04-02 | IS replay, net | 27.38% | -20.90% | 9.65% | n/a | 1.61 | +48.3 | yes / yes |
| 0402_123248 full (deployed) | 2023-04-03..2026-04-02 | IS replay, net | 53.89% | 140.97% | 15.65% | 0.38 | 0.64 | -18.6 | no / yes |
| 0402_123248 YTD | 2025-04-03..2026-04-02 | IS replay, net | -10.65% | -19.88% | 12.78% | n/a | 1.12 | +9.3 | yes / no |
| 0501_010648 full (live 05-01) | 2023-05-01..2026-04-30 | IS replay, net | 137.60% | 167.09% | 22.22% | 0.82 | 0.89 | -5.3 | no / yes |
| 0501_010648 YTD (48.46% headline) | 2025-05-01..2026-05-01 | IS replay, net; labelled CAGR | 48.40% | -17.46% | 26.97% | n/a | 1.80 | +65.9 | yes / yes |
| edge run 2 full (Binance.US tape) | 2023-06-09..2026-06-08 | replay, net; IS to 2025-11-01, then 20% holdout (OOS) | 47.32% | 141.94% | -6.85% | 0.33 | 0.61 | -20.5 | no / yes |
| edge run 2 YTD | 2025-06-08..2026-06-08 | replay, net; ~5 months IS, ~7 months holdout (OOS) | 3.38% | -39.09% | -32.23% | n/a | 1.70 | +42.5 | yes / yes |
| edge run 3 full (Kraken) | 2023-06-09..2026-06-08 | replay, net; IS to 2025-10-31, then holdout (OOS) | -27.13% | 139.93% | -8.07% | -0.19 | 0.30 | -43.9 | no / no |
| edge run 3 YTD | 2025-06-08..2026-06-08 | replay, net; ~5 months IS, ~7 months holdout (OOS) | -21.15% | -39.77% | -32.76% | n/a | 1.31 | +18.6 | yes / yes |
| Funding carry A (Kraken) | 2025-05-15..2026-05-15 | delta-neutral; prose only | 0.45% | -23.09% | -10.36% | n/a | 1.31 | +23.6 | yes / yes (not like-for-like) |
| Funding carry B (proxy) | 2023-01-01..2025-05-15 | proxy; prose only | 3.45% | 528.0% | 113.6% | 0.007 | 0.16 | -115.8 | no / no |

Negative ratio: edge run 3 lost money (-27.13%) while BTC gained (+139.93%), so the ratio of
cumulative returns is negative (-0.19).

Holdout split of the edge-search "YTD" windows (same extracted curves and tape). The WFO folds
end 2025-11-01; the edge report treats the rest as true OOS for the passers
(`docs/archive/edge_search_report_2026-06-08.md:28-38`).

| Run | Part | Strategy | BTC | ETH |
|---|---|---:|---:|---:|
| edge run 2 | 2025-06-08..2025-11-01 (IS) | +14.24% | +4.08% | +54.65% |
| edge run 2 | 2025-11-01..2026-06-08 (holdout, OOS) | -9.44% | -41.94% | -55.95% |
| edge run 3 | 2025-06-08..2025-11-01 (IS) | -2.75% | +4.03% | +54.21% |
| edge run 3 | 2025-11-01..2026-06-08 (holdout, OOS) | -18.92% | -42.30% | -56.21% |

Out of sample, both books lost money but lost less than BTC and ETH while BTC fell 42%. In the
full 12-month regression (not the holdout alone), BTC exposure was low (beta 0.10 and 0.20) and
the intercept was not significant (t = 0.43 and -0.44).
Both books lost to BTC over the three-year range.

Not computable: the 28 March pipeline runs (windows unpinned, folders not found; the record
shows each below its own EW B&H, `docs/archive/pipeline_run_history.md`, 28 negative "Excess
CAGR" lines); the Kraken live PnL (a dollar figure with a mid-window capital injection,
`docs/changelog.md:2389-2391`); the textbook reset (no portfolio formed).

**BTC beta.** Daily OLS of each combined "YTD" equity curve (extracted from the run's
`combined_portfolio_ytd_dashboard.html`) on daily BTC returns from the same venue tape:

| Run | Beta (s.e.) | Intercept, simple % per year | t | R² | Same-beta BTC + cash, total |
|---|---|---:|---:|---:|---:|
| 0401_225306 | 0.166 (0.023) | +28.3 | 1.53 | 0.13 | -2.2% |
| 0402_123248 | 0.416 (0.031) | -1.3 | -0.05 | 0.33 | -6.6% |
| 0501_010648 | 0.263 (0.041) | +50.1 | 1.51 | 0.10 | -4.0% |
| edge run 2 | 0.104 (0.032) | +11.0 | 0.43 | 0.03 | -4.3% |
| edge run 3 | 0.195 (0.033) | -11.7 | -0.44 | 0.09 | -8.1% |

BTC beta does not explain the "YTD" outperformance: the books held little BTC risk (beta
0.10-0.42), and a BTC-plus-cash mix at the same beta lost 2-8% in each window. The residual is
not significant (largest t = 1.53; 1.56 for 05-01 under the alternative alignment). The 04-01,
04-02 and 05-01 windows are in-sample; the edge-search windows are about 7/12 holdout. Bucketing:
BTC re-indexed to each curve's 4h timestamps, last bar of each UTC day, simple daily returns, OLS
with intercept, n = 364-365. For 05-01 the beta is 0.263 under this bucketing and 0.314 when BTC's
daily last bar is taken directly from the tape (intercept +50.4%, t 1.56; matches an independent
reviewer's 0.314, s.e. 0.040), so read the betas as ±0.05 alignment-sensitive. Full-range regressions were not
run: only per-worker full-range curves are stored, not the combined portfolio.

**Benchmark-figure discrepancies (stated vs tape).** For the Kraken runs the stated BTC figure is
within 0.1-2.7 pp of the tape once the run's 1.4% RT cost is applied (e.g. 136.55% stated vs
140.97% tape gross, 137.6% net). The largest gap is the 05-01 "YTD" window: -21.32% stated vs
-17.46% gross, about -18.6% net. It closes when the start bar is aligned with the equity curve's
first point (2025-05-01 12:00): BTC -20.1% gross, about -21.2% net.
Edge-search run 2 states BTC +167.21% (DD -49.89%) for 2023-06-09..2026-06-08; the Binance.US tape
gives +141.94% (DD -51.66%) for that window but +169.71% (DD -49.82%) if the series ends on
2026-06-01. The report's BTC benchmark therefore appears to end about a week before its stated end
date, while the strategy curve runs to 2026-06-08 16:00. Run 3 states +128.86% vs a tape +139.93%.
Cause not established. The errors make BTC look better in run 2 and worse in run 3; neither changes
the "lost to BTC" outcome.

## Tape integrity

All queries are SELECT-only on `ohlcv`, restricted to `venue in ('kraken_spot','binanceus_spot')`
and the named intervals. Counts are in `research/audits/root-cause-2026-11-04/lenses/_lens8_tape_checks.json` under `tape`.

**Coverage.** `-USD` symbols with daily bars: 627 (623 Kraken, 53 Binance.US). Kraken 1d, 12h and
1h end 2025-12-31 / 2026-05-18; Kraken 4h (161 symbols) and Binance.US 4h (77) end 2026-06-08 16:00;
Binance.US 1d ends 2026-06-08. BTC on Kraken also has 1m/5m/15m to 2025-12-31.

**(a) Duplicates.** The primary key `ohlcv_pkey` is UNIQUE on (timestamp, symbol, interval, venue),
so exact duplicates cannot exist. Day-level duplicates (the SPY bug class: two daily bars on one
calendar date) are **0** on both venues, across all `-USD` daily rows (385,914). No symbol-day has
more than six 4h bars. Crypto-specific duplication does exist: **94,721 of 216,818 Binance.US 4h
bars (43.7%, 48 symbols) are identical in open, close and volume to the Kraken bar at the same
time**. For BTC and ETH the copies cover 86% of 2023 Q3, all of 2023 Q4 to 2024 Q4 and 55% of 2025 Q1.
The source is the legacy script backfill_binanceus_from_kraken.py (at `82931a40^`, now deleted),
which cloned Kraken candles into `binanceus_spot`. So the "Binance.US" research tape is a mix of
venues. Separately, 2,442 Binance.US 4h bars on 29 symbols differ from Kraken by more than 5%. For
BTC and ETH this is concentrated in June-July 2023, when Binance.US prices sat up to 20% below
Kraken (a real venue episode, not a write bug).

**(b) Timestamp shift.** Daily bars: 100% at 00:00 (332,069 Kraken, 53,845 Binance.US). 4h bars:
0 off-grid on either venue. BTC/ETH 1h on Kraken: 0 off-grid. External label check: the tape's
daily BTC close matches yfinance at lag 0 (median gap 0.018% Kraken, 0.034% Binance.US). At ±1 day
the gap jumps to 1.17-1.25%, so there is no day shift. Internally, the daily close equals the
20:00 4h close on 100% of Kraken days and 99.76% of Binance.US days. The equity-tape day-shift bug
class is absent from the current crypto tape. History: from 2026-03-08 to 2026-05-05 the live
writer shifted crypto bars by 7 hours (`docs/changelog.md:2058-2068`). The repair wiped and
re-imported the 4h tape, so every run before 2026-05-05 used data that no longer exists.

**(c) Gaps.** BTC daily: one missing day (2024-03-31) on both venues. ETH daily: none. BTC 4h:
2-3 single bars. ETH 4h on Kraken: 64 missing bars, including a 52-bar (about 9-day) run in May
2026. The sample of 10 others (SOL, XRP, DOGE, ADA, TRX, SUI, PEPE, LINK, ZEC, XMR) has 0-125
missing 4h bars on Kraken over 2023-01..2026-06. Binance.US daily TRX misses 105 days and ADA 49.
Across venues, 179 of 623 Kraken and 21 of 53 Binance.US daily symbols have a gap; 85 and 9 miss
more than 5% of their span.

**(d) Symbol normalization.** No legacy-prefixed `-USD` symbols exist (`XXBT`, `XBT`, `XETH`,
`XXDG`, `XDG` and the other `SYMBOL_MAPPING` keys: 0 rows). The ranker at
`82931a40^:scripts/update_universe_ccxt.py:126-127` applies `SYMBOL_MAPPING` from
`src/ggTrader/data/core/constants.py` to the ccxt base before ranking, and so does the listings
code (`82931a40^:src/ggTrader/data/core/venue_listings.py:63-65`). ccxt already returns unified
codes, so this is a no-op in practice. Outside the `-USD` universe, malformed rows remain:
7 un-split `XBT` pair codes (XBTCHF, XBTDAI, XBTEURC, XBTEUROP, XBTUSD1, XBTUSDQ, XBTUSDR; 71,383
rows), 17 other un-split codes (for example ETHCHF, USDTCHF, MATICPOL), 52 quote-less base symbols
on Kraken 4h (for example `ETH`, `AAVE`; 38,153 rows, 2025-11-24..2026-04-01), 2 `BTC` rows on
Binance.US, and about 100 equity tickers (AAPL, SPY and others; 25,783 daily rows) stored with
venue `kraken_spot`. None of these match a `-USD` query, so they would not enter a `-USD`
universe. They are tape hygiene defects, not ranking defects.

**(e) Survivorship.** Universe construction was not point-in-time. The ranker pulled live ccxt
tickers at run time, intersected them with the venue's current listings snapshot, ranked by
current 24h or 7/30-day volume and then backtested the previous three years
(`82931a40^:scripts/update_universe_ccxt.py:95-200`). Coins that died between 2023 and the run
date could not be selected, and coins that became large later were selected from the start of
the window. In the tape, 48 of 623 Kraken daily symbols end more than 30 days before the venue's
last bar (MATIC, FTM, EOS, WAVES and others), so dead-coin history survives on Kraken. On
Binance.US, 9 delisted or renamed symbols (about 15k rows) were deleted on 2026-06-07 (memory
note project_venue_availability_registry_2026-06-07), and 0 of 77 Binance.US 4h symbols now end
early. 421 of 623 Kraken daily symbols start after January 2023. The coin universe is
survivorship-biased in both the selection code and the Binance.US tape.

**(f) Venue/interval coverage per symbol.** Per-symbol spans are in the JSON (`per_symbol_1d`,
`per_symbol_4h`). Bad-price checks on daily bars: 0 null or non-positive closes, 0 bars with
high < low.

Tape verdict: clean on the equity bug classes (duplicates, day shift, large BTC/ETH gaps). Not
clean overall: the Binance.US series is 44% Kraken copies, it has had delisted coins removed, and
the universe was chosen with hindsight. Runs before 2026-05-05 used a tape that was later replaced.

## Fees

What was assumed (per side, from each run's stored config): Kraken-era runs fee 0.4% + slippage
0.3% (1.4% RT); textbook-reset variants 0 / 0.2+0.3 / 0.25+0.3 / 0.4+0.3; edge-search run 1 and 2
(Binance.US) 0.02% + 0.3% (0.64% RT); run 3 (Kraken maker) 0.25% + 0.3% (1.10% RT). The venue memo
(`research/crypto/execution-venues/2026-07-06-crypto-execution-venues.md:16-36`, written after the edge search) lists
Kraken 0.25%/0.40% and Binance.US 0%/0.02% maker/taker, with at least 0.2% altcoin slippage on
Binance.US. The fee inputs match; the memo's slippage floor is lower than the 0.3% the runs charged.

What is sound:
- A controlled sweep exists in the textbook reset: same 7 coins, same 231 combinations, same
  window (2023-05-02..2026-05-01), same tape, with only costs varied. Aggregate-gate passes from
  the worker logs: **36 frictionless, 14 at 0.5%/side, 13 at 0.55%/side, 0 at 0.7%/side.** Cost
  level does decide gate survival on this set. (`docs/changelog.md:1646` calls the 0.5%/side run
  "0.5% RT"; the stored config is 0.2% + 0.3% per side, i.e. 1.0% RT.)
- Turnover is high enough for costs to matter: edge run 2 made 167 trades over three years on
  two coins, run 3 made 172.
- **Inventory pointer is mis-pointed.** Row crypto-wfo-textbook-reset cites
  `results/research/research_20260511_193048/wfo_stats_snapshot.json` as the raw artifact for
  "zero combos pass under realistic fees". That run is the frictionless one (stored config FEES
  0, SLIPPAGE 0; 36 gate passes). The realistic-fee run (0.4% + 0.3% per side; 0 passes) is
  `results/research/research_20260511_162755`. Both claims recompute from the worker logs, but
  only the frictionless run has a full `wfo_stats_snapshot.json`.

What is confounded or missing:
- The headline flip (+13.8% on Binance.US → -10.0% on Kraken) changes four things at once:
  universe (51 vs 100 coins), venue tape (Binance.US, 44% Kraken copies, vs Kraken), the coins
  selected (DOGE+TRX vs DOGE+SUI) and the fee. It is not a fee-only comparison.
- In the Binance.US runs the 0.3% slippage is 94% of the per-side cost (0.32%). The 2026-05-12
  venue decision describes Binance.US as "0.04% RT … effectively frictionless"
  (`docs/changelog.md:1631-1646`), but the backtests charged 0.64% RT. The textbook-reset sweep
  never ran a 0.32%/side variant, so the expectation of "close to the full 36" was never tested
  on the controlled set. The edge search's Binance.US runs passed 13/300 and 22/1462 gate
  combinations and 0 or 2 coins.
- No run on the Binance.US tape varied only the fee. No maker-vs-taker run at the same tape
  exists beyond the 0.25% maker variant.
- Quote currency: only `-USD` pairs were tested. The Binance.US USD series diverged from Kraken
  by up to 20% in mid-2023, so the quote and venue choice moves the price series itself, not only
  the cost. USDT-quoted pairs exist on Kraken in the tape (for example `BTC-USDT`) but were never
  in a research universe; Binance.US USDT pairs are not in the tape.
- Even at zero cost, no portfolio was formed: 0 coins survived the full cascade in the
  frictionless textbook run. Costs are not the only binding constraint.

Fees verdict: "costs matter a great deal at this turnover" is supported. "Fees dominate", as
measured by the +13.8% vs -10% flip, is confounded, and "Binance.US is effectively frictionless"
conflicts with the cost model the backtests used. Venue tier and quote currency were not isolated.

## Scope tested and not tested

- **Long-only spot:** every directional crypto backtest and the live engine. The legacy
  `Direction.SHORT` enum (`82931a40^:src/ggTrader/core/signal.py:20`) is used only to label
  exchange fills.
- **Shorting (directional):** never tested. Alpaca BTC/USD is not shortable or marginable
  (`research/WEB_RESEARCH_CANDIDATES.md`, 2026-10-25 batch). Binance.US offers no perpetual futures or swaps (same file, section 4).
- **Perpetual futures / funding carry:** tested once. FundingCarryBTC Phase 4 (long BTC spot,
  short PF_XBTUSD) ran on one year of real funding data (+0.45%, 8 trades) and on a basis proxy
  for 2023-2025 (`docs/archive/phase4_funding_carry.md:43-47`); the author marks the proxy as not
  quotable. The results are prose only, but the inputs are still in the DB (`funding_rates`
  PF_XBTUSD 8,816 rows, 2025-05-15..2026-05-18; `perp_ohlcv` PF_XBTUSD 1h 36,400 rows,
  2022-03-23..2026-05-18) and the code is at `82931a40^` (strategies/carry/funding_carry.py), so
  this study is the most reconstructable of the crypto set. The later candidate was marked
  infeasible for history length (crypto-perp-funding-carry-candidate). Cash-and-carry with dated
  futures existed as code (strategies/carry/cash_and_carry.py at `82931a40^`); no result for it
  was found in the record.
- **Never tested:** long/short cross-sectional crypto, BTC time-series momentum (A16, literature
  only), on-chain or flow signals (A25-A27), options, any crypto test longer than about 3.4 years
  (own tape starts 2023-01-01).

## Annualization

- **Current lab:** `src/ggTrader/lab/metrics.py:10-29` sets `TRADING_DAYS_PER_YEAR = 252` for
  volatility, Sharpe and Sortino. CAGR uses calendar days / 365.25 (line 16), so CAGR is correct.
  For a daily series that trades 7 days a week, the reported Sharpe, Sortino and volatility equal
  the correct values × √(252/365) = **0.831**; the correct Sharpe is the reported one × 1.204.
  `src/ggTrader/lab/data.py:38-52` defines a daily crypto config with a BTC benchmark, so the path
  exists. Gates that threshold Sharpe would be stricter for crypto by the same factor.
- **Legacy pipeline (all published crypto numbers):** `82931a40^:src/ggTrader/core/metrics.py:11-14`
  annualizes with vectorbt's `year_freq` (365 days; installed vectorbt 0.28.5 reports `365 days`)
  divided by the bar frequency (4h → 2,190 periods per year). Recomputing Sharpe from the
  extracted 4h equity curves with √2190 reproduces the reports: 1.2438 vs 1.2378 (05-01),
  -0.2318 vs -0.2317 (04-02), 0.2607 vs 0.2577 (edge run 2), 1.2830 vs 1.2827 (04-01).
- **Per row:** crypto-wfo-research-snapshots-kraken, crypto-wfo-recalibration-2026-05-01,
  crypto-edge-search-2026-06-08, crypto-wfo-textbook-reset: **does not affect**.
  crypto-pipeline-runs-2026-03: **does not affect** (same legacy path; inferred, not recomputed).
  crypto-funding-carry-phase4: **unknown** (separate Phase 4 engine; its Sharpe convention was not
  checked). crypto-ibit-overnight-session: **does not affect** (total returns only).
- **Lab crypto runs:** `lab_runs` holds one `market='crypto'` run (`xs_momentum_8a792c5c`,
  2023, created 2026-06-16) with stock-style parameters (benchmark SPY, fees 0). It is cited
  nowhere in `docs/`. If its data were 7-day, its Sharpe would be affected; it carries no verdict.

Annualization verdict: a real latent defect in the current lab for crypto, with no effect on any
published crypto figure. The research-process audit's caution
(`research/audits/research-process/2026-11-01-research-process-audit.md:72-77`) is confirmed as "traceable risk, not
a bad historical verdict".

## Live evidence

| Source | What exists |
|---|---|
| `trades` table | 0 rows |
| `orders` table | 1 row: a dry-run buy, 2026-06-05 |
| `live_balance_snapshots` | 4,681 rows, 2026-05-23..2026-06-10; 3,151 with 0 positions, 1,530 with 1-2 positions from 2026-06-04 (no matching order rows; likely the stale local-state positions noted on 2026-06-05; not resolved) |
| `live_positions_snapshot` | 4,681 rows; circuit breaker flagged on 211, 2026-06-04 23:18..2026-06-05 16:57 PT |
| `kraken_ledger` | 9 deposits, 8 withdrawals, 2025-04-14..2026-06-06 (cash movements only) |
| `equity_curves` | 0 rows |
| Local files | `data/live` holds only a dashboard folder; Kraken-era trade CSVs and the investigate_live_trades.py script are gone |

Balances were not read into this report. Why live crypto barely traded, from the record:
Kraken-era entries were blocked for about three weeks by sizing not wired after a refactor and
for 17 days by the BTC regime gate (`docs/changelog.md` 2026-05-05 and 2026-05-06 entries). After
the move to Binance.US on 2026-05-23 (with legacy parameters), a sizing floor blocked every entry
until 2026-06-04 (`docs/changelog.md:1552-1559`). An IP-allowlist `-2015` error crash-looped the
container on 2026-06-03 (about 372 restarts, per the memory note), and the daily-loss breaker
tripped on 2026-06-04. The engines were removed on 2026-06-16 (commit `82931a40`). The only live
performance figure, +$9.88 over about 5 days on Kraken in April 2026, cannot be checked.

## Upstream implications

Upstream files are not edited here. This section states what Lens 8's findings mean for them.

**Located artifacts.** Gitignored run folders exist for rows the upstream lenses treated as
artifact-less: crypto-wfo-recalibration-2026-05-01 (`results/research/archive/research_20260501_010648`),
crypto-wfo-research-snapshots-kraken (`results/research/archive/research_20260402_123248` and
`results/research/archive/research_20260401_225306`; the 03-29 run was not found) and
crypto-edge-search-2026-06-08 (`results/research/research_20260607_215008`,
`results/research/research_20260608_083103`, `results/research/research_20260608_093714`). Each
holds `research_report.md`, worker result JSON with the stored config, worker logs with gate
verdicts, `top_ccxt_volume.json` and plot HTML with the combined equity curve. Folders dated
2026-05-07..05-09 also exist under `results/research/archive/` and may back the 05-07..05-09
configuration rows; they were not opened.

**Inventory raw_artifact pointers.** Of the 3 crypto rows with a pointer, 1 is mis-pointed:
crypto-wfo-textbook-reset points at the frictionless run (`research_20260511_193048`, 36 passes)
for a claim about the realistic-fee run (`research_20260511_162755`, 0 passes). The two IBIT
pointers (`research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json`) are
correct. Three rows have a null pointer although an artifact exists (the three rows above).

**Lens 6** (`research/audits/root-cause-2026-11-04/lenses/_lens6_verdicts.json`). spec_matches_code can stay "unverifiable" for
crypto-wfo-recalibration-2026-05-01 and crypto-wfo-research-snapshots-kraken because the code was
removed and was not compared with a spec here. Its evidence text "Run artifacts not found in
results/" is wrong for the recalibration row. The numbers-true question is now answerable, and
this lens verified: the 05-01 "YTD" total 48.40% from the stored equity curve; Sharpe re-derived
for the 04-01, 04-02, 05-01 and edge run-2 "YTD" curves within 0.006; stored fees and slippage
per run; and gate-pass counts 36/14/13/0 and 13/22/13 from worker logs. Not verified: the
full-range combined returns (only per-worker curves are stored).

**Lens 7** (`research/audits/root-cause-2026-11-04/lenses/_lens7_recompute.json`). Rows whose prose-only or cannot-locate status
could change, because a raw artifact exists: crypto-wfo-recalibration-2026-05-01,
crypto-wfo-research-snapshots-kraken, crypto-edge-search-2026-06-08 (now prose-only) and
crypto-wfo-textbook-reset (now cannot_locate; the gate counts recompute from the logs of
`research_20260511_193048` and `research_20260511_162755`). Possibly also
crypto-diversity-cap-disabled-2026-05-07, crypto-trade-frequency-gate-2026-05-07,
crypto-strategy-usage-analyzer-2026-05-07 and crypto-wfo-overfitting-param-stability (archive
folders of those dates exist, not opened). One recompute discrepancy relates to a benchmark,
not a strategy figure: the run-2 BTC benchmark (+167.21% stated vs +141.94% tape).

**Lens 2** (`research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`). No crypto row's class would change under
its written rules. R2 states that crypto rows do not meet the data condition, and Lens 8's
findings would not meet it either: the NO rows (crypto-edge-search-2026-06-08,
crypto-wfo-textbook-reset) carry universe survivorship, which favours the strategy, but also
cross-venue copying (a tape identity fault, direction unknown), so not every item favours the
strategy. R3 stays unmet (Lens 6 unverifiable or no row). R5 stays unmet (Lens 9 has no MDE).
Under R4 the recompute basis for edge-search moves from "prose-only (flagged)" (met) to an
artifact that Lens 7 has not recomputed (not met until it does). That does not change
"inconclusive". For the two GO rows (crypto-funding-carry-phase4, crypto-kraken-live-bot-pnl)
no universe-survivorship item applies (BTC only, live), and R3/R4 remain unmet, so they stay
inconclusive. No row becomes "unreliable NO" by R6: the strategy figures recompute.

**Lens 1** (`research/audits/root-cause-2026-11-04/lenses/_lens1_tape_state.json`). All crypto rows are "unknown" there. On
Lens 8's evidence:
- Rows run between 2026-03-08 and 2026-05-05 (crypto-wfo-research-snapshots-kraken 04-01/04-02,
  crypto-config-tweaks-reverted-2026-04-02, crypto-wfo-recalibration-2026-05-01,
  crypto-kraken-live-bot-pnl, crypto-wfo-tournament-robustness) used a tape with a time-shift
  fault covering 7-15% of the trailing windows and 2-5% of three-year windows. That is the
  day_shift class (direction unknown); the tape no longer exists.
- Rows run after the re-import (crypto-wfo-textbook-reset, crypto-diversity-cap-disabled-2026-05-07
  onward, crypto-edge-search-2026-06-08) used the repaired Kraken 4h tape, which is clean on the
  equity bug classes today. Edge-search runs 1-2 also used the Binance.US series (cross-venue
  copies: tape identity fault) and, from 2026-06-07, the tape with delisted Binance.US symbols
  deleted. All universe-ranked rows carry universe survivorship (favours the strategy).
- Whether the tape at each run date equals today's tape is not established (later edits
  include the 2026-06-07 deletions and Binance.US backfills), so "clean" cannot be asserted for
  any crypto row. Their tape_state stays "unknown", with these items attached.

## Questions and answers

1. **What was tried, with what universe, fees, venue and benchmark?** Long-only spot per-coin
   WFO tournaments of technical-analysis entries and exits on 4h bars. Universes of 1-100 coins
   ranked by live volume. Kraken (0.4% + 0.3% per side) through May 2026, then Binance.US
   (0.02% + 0.3%) and a Kraken-maker variant (0.25% + 0.3%). Benchmarked against BTC B&H and SPY,
   or EW B&H in March. Also one delta-neutral funding-carry study. See "What was tried".
2. **Did any crypto strategy beat buy-and-hold BTC or ETH?** BTC: not over any three-year window
   (ratio -0.19 to 0.82, -5 to -44 pp per year). The one-year funding-carry window beat BTC
   (+0.45% vs -23%) but is delta-neutral, so not like-for-like. Yes in five trailing-12-month
   windows when BTC fell: in-sample for the 04-01, 04-02 and 05-01 runs; for edge runs 2 and 3
   about 7 of the 12 months are the post-selection holdout, where the books lost 9-19% vs BTC
   -42% (low beta, intercept t 0.43 / -0.44). ETH: the full-range results beat ETH in four of five cases because
   ETH returned between -8% and +22% over three years; the 2026-05-01 "YTD" beat ETH (+48.4% vs
   +27.0%) in-sample.
3. **Does BTC beta explain apparent alpha?** No. Betas are 0.10-0.42, and same-beta BTC-plus-cash
   lost money in every window. The remaining intercepts are not significant (t ≤ 1.56) and the
   largest ones are in-sample. Full-range beta is not computable (no combined full-range curve stored).
4. **Is the 2026-06-08 "fees dominate" finding sound and complete?** Partly. A controlled
   7-coin sweep shows costs eliminating gate passes (36 → 0). The headline +13.8% → -10% comparison
   is confounded by universe, tape and coin selection, and both runs carry a 0.3% slippage that
   the fee framing omits. Not complete: no fee-only variation on the Binance.US tape, no 0.32%/side
   variant on the controlled set, and no portfolio formed even at zero cost.
5. **Do venue fee tiers and quote-currency choice change it?** Fee tier changes gate counts
   (36/14/13/0). Venue changes the price series (44% copied, with a 20% divergence episode in
   2023). Quote currency was never varied (USD only). Tier effects beyond the $0 tier: unknown,
   never tested; the record lacks a like-for-like re-run that varies only the tier. Cheapest test:
   a gate replay of the stored `wfo_stats_snapshot.json` files (stage-1 gates only; portfolio
   effects are not covered by it).
6. **Long-only spot only? Shorting, perpetuals, funding carry?** Directional work was long-only
   spot. Directional shorting was never tested. Funding carry on PF_XBTUSD was tested once, for one
   year (prose results, data and code recoverable). Cash-and-carry exists as code without a found
   result.
7. **Do the equity bug classes affect the crypto tape?** Not in the current tape: 0 day-level
   duplicates, 0 off-grid bars, daily labels match yfinance at lag 0, BTC/ETH daily gaps ≤ 1 day.
   They did affect the pre-2026-05-05 tape (7-hour shift) used by the April-May runs. A
   crypto-specific duplication class exists instead: Binance.US bars copied from Kraken.
8. **Were XBT/XETH symbols normalized before ranking?** Yes for the `-USD` research universe
   (ranker applies `SYMBOL_MAPPING`; no prefixed `-USD` rows in the tape). Un-split `XBT*` pair
   codes and quote-less symbols remain outside it.
9. **Is the coin universe survivorship-biased?** Yes. Live-volume, current-listing selection was
   applied to three years of history, and delisted Binance.US history was deleted. Kraken keeps
   48 dead-coin series.
10. **Does the 252-day annualization distort crypto statistics?** In the current lab, yes:
    Sharpe, Sortino and volatility are understated by a factor of 0.831 on daily 7-day data. No
    published crypto number is affected (legacy path uses 365 days, verified). Funding-carry
    Sharpe convention: unknown; cheapest test is reading
    `82931a40^:src/ggTrader/strategies/carry/funding_carry.py` and its engine.
11. **Why did live crypto barely trade, and what live evidence exists?** Sizing defects, the BTC
    regime gate, the venue move with legacy parameters, the IP-allowlist crash loop and the
    breaker. The DB holds 0 trades, 1 dry-run order and 4,681 snapshots. No live return can be
    computed.
12. **Steelman: is there evidence an edge existed?** The strongest result (05-01) is in-sample,
    in one falling-BTC year, on a partly time-shifted (15% of the window) since-replaced tape,
    with a hindsight universe and an insignificant intercept. The only OOS evidence (edge-search
    holdout) is losing less than a falling BTC at low beta, without a significant intercept.
13. **Why did the stated BTC benchmark in edge-search run 2 not match the tape?** Unknown. The
    numbers fit a BTC series ending around 2026-06-01. The cheapest test is reading the benchmark
    loader at `82931a40^` (pipeline report code) for its data source and end bar.

## Verification

Commands and queries run (scratch scripts under
`/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/l8/`;
DB access via `scripts/run_capped.sh .venv/bin/python ro.py`, read-only session,
`SET statement_timeout` 30-120 s):

- `select venue, interval, count(distinct symbol), count(*), min(timestamp), max(timestamp) from ohlcv where symbol like '%-USD' and venue in (...) and interval in ('1d','4h','12h','1h') group by 1,2` → coverage table. An unfiltered all-interval variant hit the 120 s timeout and was cancelled by the server.
- Day-level duplicates: `select venue, symbol, timestamp::date, count(*) ... interval='1d' group by 1,2,3 having count(*)>1` → 0 rows; 4h `having count(*)>6` → 0 rows.
- Hour distribution of daily bars and `mod(extract(hour),4)<>0 or minute<>0` on 4h → all on grid.
- Prefixed-symbol query on `split_part(symbol,'-',1)` in the `SYMBOL_MAPPING` keys → 0 rows; no-dash symbol listing → `nodash.txt`.
- Per-symbol spans (`per_symbol_span_1d`, `_4h`) → gap and survivorship counts (`analyze.py`).
- BTC/ETH 1d and 4h closes for both venues → cross-venue identity and divergence, B&H per window; cross-venue join on 4h `-USD` (216,818 Binance.US bars, 94,721 identical).
- yfinance `BTC-USD`/`ETH-USD` 2023-01-01..2026-06-10 → lag-0/±1 label check and B&H cross-check (`yf_check.py`).
- Equity curves extracted from plotly HTML (`extract_eq.py`), daily OLS on BTC (`beta.py`) and on BTC plus an EW basket of the run's coins (`basket.py`).
- Stored configs read from `worker_*_results.json` (fees, slippage, symbols, windows); gate verdicts counted with `grep -c ': PASS'` over `[Gates]` lines in worker logs.
- Code read with `git show 82931a40^:<path>` and `git grep 82931a40^`.
- Live tables: counts only from `orders`, `trades`, `live_balance_snapshots` (num_positions), `live_positions_snapshot`, `kraken_ledger` (type counts), `equity_curves`, `funding_rates`, `perp_ohlcv`, `lab_runs`, `lab_summary`.

What remains unverified:
- The April-May run results cannot be re-derived from today's tape (wiped and re-imported
  2026-05-05). Only internal consistency (equity curve vs report) was checked.
- Equity curves come from the plot HTML, not a raw returns file. The first stored point of the
  2026-05-01 curve is 997.84, not 1,000.
- Full-range BTC beta: not computable.
- The funding-carry figures and the March pipeline runs: prose only. The pipeline run folders
  were not found.
- The 1-2 positions in live snapshots from 2026-06-04 have no matching order rows; their origin
  was not established.
- The run-2 BTC benchmark end bar: cause unknown.
- The Haiku pull was used as leads. Every figure in this report was re-read from the primary
  file, artifact or query, not from the pull.
- Not re-run: no backtest or WFO (Phase 2 is owner-gated).

## Done checklist

- ✅ What was tried (universe, fees, venue, benchmark), all 25 inventory rows reconciled with the Haiku pull.
- ✅ Beat BTC/ETH: computed for 12 windows from the tape with a yfinance cross-check; 3 marked not computable with reasons.
- ✅ BTC beta and residual alpha for the 5 stored "YTD" curves; ❓ full-range beta (no combined full-range curve stored).
- ✅ Fees audit: assumptions, controlled sweep, confounds, missing variants.
- ✅ Scope tested vs not (long-only spot, shorting, perps, funding carry).
- ✅ Tape (a) duplicates, (b) timestamp shift, (c) gaps, (d) symbol normalization, (e) survivorship, (f) coverage; counts in JSON.
- ✅ Annualization: current lab affected (×0.831); published legacy numbers not affected (recomputed); ❓ funding-carry Sharpe convention (separate engine not read).
- ✅ Live evidence counts (no balances).
- ❓ Cause of the edge-search run-2 BTC benchmark mismatch (benchmark loader not read).
- ❓ April-May runs on the pre-fix tape: not reproducible from current data; the affected share is sized (7-15% of the trailing windows, 2-5% of the three-year windows).
- ✅ Upstream implications for Lenses 1, 2, 6 and 7, and the inventory pointer check (1 mis-pointed of 3).
- ✅ In-sample vs OOS stated per window (edge-search trailing windows are about 7/12 holdout).
