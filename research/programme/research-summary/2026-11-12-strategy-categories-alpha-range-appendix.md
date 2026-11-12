# Appendix: every strategy, with its numbers

Companion to the [strategy-category summary](2026-11-12-strategy-categories-alpha-range.md). Generated from the verified table `_inputs/category_inputs_2026-11-12.csv`; nothing here is hand-typed.

## Appendix A: every strategy with a comparable return

Gap = candidate minus benchmark, in basis points (bps) a year. Range = gap plus or minus two standard errors, only where the source gives a t-statistic. Costs and tax follow each row's own basis, so they are not all the same. "Legacy" = computed on the old engine, not re-run. The FOMC drift row has no gap because its source says the comparison is inappropriate.

| Category | Strategy | Variant | Benchmark | Candidate % | Benchmark % | Gap (bps) | Range (bps) | Engine |
|---|---|---|---|---:|---:|---:|---|---|
| Stock-selection screens | stocks-low-accruals | full | French market (Mkt-RF + RF), not SPY | 11.40 | 10.83 | +57 | -67 to +181 | n/a |
| Stock-selection screens | stocks-low-accruals | since_1994 | French market (Mkt-RF + RF), not SPY | 11.48 | 10.87 | +61 | -120 to +242 | n/a |
| Stock-selection screens | stocks-low-accruals | spy_overlap | SPY (60-month diagnostic) | 13.11 | 13.02 | +10 | -113 to +132 | n/a |
| Stock-selection screens | stocks-high52 | full | French market total return (Mkt-RF + RF) | 12.48 | 10.80 | +169 | -143 to +480 | n/a |
| Stock-selection screens | stocks-high52 | since_1994 | French market total return (Mkt-RF + RF) | 9.86 | 10.81 | -95 | -376 to +187 | n/a |
| Stock-selection screens | stocks-rd-exploratory | full | French market, not SPY | 18.20 | 12.01 | +618 | 0 to +1,237 | n/a |
| Stock-selection screens | stocks-rd-exploratory | since_1994 | French market, not SPY | 14.06 | 10.81 | +324 | -248 to +897 | n/a |
| Stock-selection screens | stocks-monthly-reversal | full | French market (Mkt-RF + RF), not SPY | 8.79 | 10.83 | -204 | -836 to +428 | n/a |
| Stock-selection screens | stocks-monthly-reversal | since_1994 | French market (Mkt-RF + RF), not SPY | 7.76 | 10.87 | -311 | -1,191 to +569 | n/a |
| Stock-selection screens | stocks-low-variance | full | French market (Mkt-RF + RF), not SPY | 8.78 | 10.83 | -205 | -408 to -2 | n/a |
| Stock-selection screens | stocks-low-variance | since_1994 | French market (Mkt-RF + RF), not SPY | 8.60 | 10.87 | -227 | -557 to +103 | n/a |
| Stock-selection screens | stocks-accounting-score | full | French market total return (Mkt-RF + RF) | 15.23 | 11.17 | +406 | +47 to +765 | n/a |
| Stock-selection screens | stocks-accounting-score | since_1994 | French market total return (Mkt-RF + RF) | 9.72 | 10.81 | -109 | -669 to +450 | n/a |
| Stock-selection screens | stocks-accounting-score | descriptive_candidate_vs_big_value.full | large-value comparator in JSON descripti | 15.23 | 13.06 | +217 | -56 to +490 | n/a |
| Stock-selection screens | stocks-expected-growth | full | Global-q broad market (R_MKT + R_F), not | 13.02 | 10.94 | +208 | +56 to +361 | n/a |
| Stock-selection screens | stocks-expected-growth | since_1994 | Global-q broad market (R_MKT + R_F), not | 13.41 | 10.78 | +264 | +58 to +469 | n/a |
| Stock-selection screens | stocks-earnings-consistency | full | French market total return, not SPY | 11.93 | 10.80 | +113 | -71 to +297 | n/a |
| Stock-selection screens | stocks-earnings-consistency | since_1994 | French market total return, not SPY | 10.75 | 10.81 | -6 | -38 to +27 | n/a |
| Stock-selection screens | pairs-stat-arb | OOS (corrected-data rerun reported unchanged in register) | SPY | -2.60 | 15.20 | -1,780 | no uncertainty measure | legacy |
| Stock-selection screens | short-volume-ratio | OOS, free-data FINRA short-volume proxy | SPY | 1.10 | 17.00 | -1,590 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-short-interest | OOS, free-data version | SPY | 3.80 | 13.00 | -920 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-index-deletion-fade | OOS, S&P 500 SP500 WFO | SPY | 6.00 | 15.20 | -920 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-max-effect | OOS, two runs (5.0 shown; 5.1 also reported) | SPY | 5.00 | 15.20 | -1,020 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-insider-cluster-buy | matched window (2021-01 to 2026-04) | SPY | 7.60 | 13.00 | -540 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-congress-trades | matched window (2021-01 to 2026-04) | SPY | 6.10 | 13.00 | -690 | no uncertainty measure | legacy |
| Stock-selection screens | headline-sentiment | pilot, 50 symbols | SPY | 4.80 | 22.90 | -1,810 | no uncertainty measure | legacy |
| Stock-selection screens | stocks-pead | matched window, pead@sp500 | SPY | 12.30 | 13.00 | -70 | no uncertainty measure | legacy |
| Stock-selection screens | idio-vol | OOS WFO, 17 folds, 6 combos | SPY | 10.20 | 13.00 | -280 | no uncertainty measure | legacy |
| Stock-selection screens | xs-momentum-corrected-replay | corrected replay, excess CAGR over SPY | SPY buy-and-hold | not stated | — | +437 | -711 to +1,584 | legacy |
| The live design and its variants | lens2-stocks-sp500-equity-research-doc-trials | archived research-doc run (invalidated) | SPY buy-and-hold | 14.35 | 15.15 | -80 | no uncertainty measure | legacy |
| The live design and its variants | midcap | MidCap 400 reversion, after survivorship haircut | MDY (MidCap 400) | 15.00 | 9.10 | +590 | no uncertainty measure | legacy |
| The live design and its variants | production-replica | primary: lag 1, 5 bp, T-bills (daily-bar replica) | SPY buy-and-hold (same window) | 14.19 | 14.83 | -64 | no uncertainty measure | legacy |
| The live design and its variants | position-sizing-regimes | Regime P blend (status quo) vs SPY row labelled same window | SPY buy-and-hold | 4.97 | 15.89 | -1,092 | no uncertainty measure | legacy |
| The live design and its variants | stocks-sp500-core-anchor-fix-reproduction | post-anchor-fix SP500 core, 2021-01-31 → 2026-04-30 | SPY (this run window) | 7.80 | 13.00 | -520 | no uncertainty measure | legacy |
| The live design and its variants | stocks-sp500-core-corrected-tape-rebaseline | corrected-tape rebaseline (same run as anchor-fix reproducti | SPY (this run window) | 7.80 | 13.00 | -520 | no uncertainty measure | legacy |
| The live design and its variants | stocks-sp500-core-pit-rebaseline | PIT membership rebaseline, pinned window | SPY | 3.10 | 13.00 | -990 | no uncertainty measure | legacy |
| The live design and its variants | stocks-ensemble | PIT ensemble run, closed NO-GO (same table) | SPY | 3.10 | 13.00 | -990 | no uncertainty measure | legacy |
| Allocation and leverage | core-spy-blend | k=1 as backtested (~8% stocks / 87% SPY) | SPY buy-and-hold (same table) | 12.96 | 12.96 | 0 | no uncertainty measure | legacy |
| Allocation and leverage | cross-asset-sleeve | static 80/20, 1 bp (pinned) | SPY buy-and-hold | 13.60 | 14.70 | -110 | no uncertainty measure | legacy |
| Allocation and leverage | three-sleeve-blend | post-anchor-fix 3-sleeve blend, leverage 1.0 | SPY (same window) | 4.76 | 13.00 | -824 | no uncertainty measure | legacy |
| Allocation and leverage | century-leverage | primary 24%/15% bracket, 30% SSO mix | 0% SSO mix (unlevered SPY) | 10.71 | 9.47 | +124 | no uncertainty measure | n/a |
| Allocation and leverage | commodity-trend | OOS, 14 single-commodity ETFs, 12-1 momentum with vol filter | SPY | 1.00 | 13.80 | -1,280 | no uncertainty measure | legacy |
| Allocation and leverage | fx-hedge-overlay | dynamic carry+value+trend hedge, ETF pairs | static 50/50 FX hedge (not SPY) | 6.00 | 12.40 | -640 | no uncertainty measure | legacy |
| Allocation and leverage | leveraged-trend | trend-filtered SP500 (SSO/UPRO) | SPY | 4.00 | 15.30 | -1,130 | no uncertainty measure | legacy |
| Allocation and leverage | month-end-treasury | 80% SPY + 20% rotating to IEF, 1 bp | SPY buy-and-hold (pinned window, full sp | 14.60 | 14.70 | -10 | no uncertainty measure | legacy |
| Allocation and leverage | static-leverage | 70/30 SPY/SSO (1.3x static), 24%/15% bracket | SPY (after tax, same window) | 12.00 | 10.40 | +160 | no uncertainty measure | n/a |
| Allocation and leverage | static-leverage | A24 sleeve rider, after-tax 24%/15% | static 70/30 comparator (same exposure) | 11.73 | 12.00 | -27 | no uncertainty measure | n/a |
| Allocation and leverage | treasury-curve | dynamic vs static 100% SHY (decisive comparator) | static 100% SHY (the source rejects SPY  | 0.80 | 1.50 | -70 | no uncertainty measure | legacy |
| Allocation and leverage | cross-asset-sleeve | trend 80/20 (252d), 1 bp (pinned) | SPY buy-and-hold | 13.50 | 14.70 | -120 | no uncertainty measure | legacy |
| Event and calendar timing | overnight-gap | OOS WFO, 17 folds | SPY | -0.20 | 13.00 | -1,320 | no uncertainty measure | legacy |
| Event and calendar timing | turn-of-month-rotation | A14 SPY turn-of-month 8-day window, else cash (descriptive i | SPY buy-and-hold | 2.10 | 15.10 | -1,300 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-edge-search | all-51 universe, Binance.US, min-vol $0 | BTC buy-and-hold | 13.80 | 38.80 | -2,500 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-pipeline | 20-coin baseline (pipeline_20260322_092410) | equal-weight buy-and-hold | 2.18 | 26.80 | -2,462 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-pipeline | alt ranks 9–16, ATR default, 8 coins (pipeline_20260322_1209 | equal-weight buy-and-hold | 3.30 | 8.01 | -471 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-pipeline | top-5 default universe, ATR-only (pipeline_20260322_104314) | buy-and-hold (5-coin) | 28.96 | 54.51 | -2,555 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-recalibration | trailing YTD window, 30 coins, promoted to live parameters ( | BTC | 48.46 | -21.34 | +6,980 | no uncertainty measure | legacy |
| Crypto (parked) | crypto-snapshots | 04-02 run, YTD | BTC | -10.66 | -20.92 | +1,026 | no uncertainty measure | legacy |

## Appendix B: studies with no comparable annual return

These studies report a diagnostic, an event test, a feasibility note or a parked or unresolved result. Nothing here is a rating.

| Category | Strategy | What the source reports instead |
|---|---|---|
| Stock-selection screens | stocks-factor-screen | no comparable return (A2 CAGR differences only) |
| Stock-selection screens | stocks-momentum | no comparable return (A2 CAGR differences only) |
| Stock-selection screens | stocks-net-issuance | no comparable return (A2 CAGR difference only) |
| Stock-selection screens | xs-momentum-2026-06 | no comparable return (June claim; conflicting comparisons) |
| Stock-selection screens | dual-momentum | no comparable return (filter never fired) |
| The live design and its variants | bb-reversion | no comparable return (WFO Sharpe only) |
| The live design and its variants | conviction-bb | no comparable return (inactive) |
| The live design and its variants | ema-cross | no comparable return (historical claim) |
| The live design and its variants | exit-rule-sweep | no comparable return (NO-GO) |
| The live design and its variants | lens2-stocks-three-pct-position-size | no comparable return (claim only) |
| The live design and its variants | live-revert-blend-to-core | no comparable return (blend reverted to core) |
| The live design and its variants | macd-divergence | no comparable return (Sharpe only) |
| The live design and its variants | mtf-reversion | no comparable return (ensemble Sharpe only) |
| The live design and its variants | six-indicator-ensemble | no comparable return (degradation vs five-voter) |
| The live design and its variants | ml-entry-gate-falsified | no comparable return (anti-predictive gate) |
| The live design and its variants | stocks-nasdaq100-single-core | no comparable return (rejected vs voided baseline) |
| The live design and its variants | stocks-wfo-tournament-sp500-monthly | no comparable return (clear NO-GO, no figures in cited JSON row) |
| The live design and its variants | stocks-wfo-policy-synthetic | no comparable return (synthetic P_halt test) |
| The live design and its variants | trailing-stops | no comparable return (NO-GO) |
| The live design and its variants | vix-entry-throttling | no comparable return (historical NO-GO) |
| Allocation and leverage | equity-diversification-sleeves | no comparable return |
| Allocation and leverage | leveraged-rotation | SP500 leveraged rotation, 2019+ span-union universe |
| Event and calendar timing | fomc-drift | event strategy (NOT COMPARABLE to SPY) |
| Crypto (parked) | crypto | no comparable return (bundled config test) |
| Crypto (parked) | crypto-funding-carry | no comparable return (real-funding carry run) |
| Crypto (parked) | crypto-ibit-overnight-session | no comparable return (descriptive split) |
| Crypto (parked) | crypto-kraken-live-bot | no comparable return (live P&L claim) |
| Crypto (parked) | crypto-wfo-textbook-reset | no comparable return (closure) |
