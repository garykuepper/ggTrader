# Early-decades test registration

## 0. Registered parameters

- eval_start: 1969-02-03
- eval_end: 1998-07-31
- max_leverage: 1.0
- benchmark: CRSP market return (stand-in), then SPY
- costs: 0.183% per side, sensitivity 0.5% per side
- family_size: 18
- seed: none

This registration records the intended test before its stock-selection results are calculated. The owner must settle the comparison before the early window is scored. The 2020-01-02 through 2024-12-31 regression must run first and match the first test's saved cells exactly.

The test window is 1969-02-03 through 1998-07-31. Each month, the rule ranks stocks by their return over the prior year, leaving out the most recent month, and buys the top 50 equally. Universe A takes common shares (CRSP share codes 10 and 11) on NYSE or Nasdaq (exchange codes 1 and 3), with absolute price at least $5 and a nonmissing market value; the top 1,000 by market value are selected using each stock’s previous-month final row. Market value is absolute price times shares outstanding; the code does not require shares to be positive. Eligibility uses information available at the time and includes stocks that later stopped trading.

The three arms are: A, the original monthly rule; B, the same rule rebalanced at the first decision whose next-session execution is strictly more than one calendar year after the previous execution, with the long-term tax rate only when a holding actually exceeds one year; and C, the monthly rule excluding stocks with a one-day rise of at least 20% in the prior 252 trading days, using only data before the pick date.

Trading costs are 0.183% per purchase or sale, with 0.5% per side as a sensitivity test. Tax is 24% on short-term gains and 15% on long-term gains. Three readings cover missing final prices for stocks that die: loss 0.0 (the headline), 0.5 (middle), and 1.0 (worst). The full set has 3 arms × 2 cost levels × 3 recovery readings = 18 variants.

The proposed comparison uses SPY, the S&P 500 fund, from the frozen `market.spy_daily_yahoo_1993_1999` table for 1993-02-01 through 1998-07-31. For 1969-02-03 through 1993-01-29 it uses CRSP `vwretd`, a market return weighted by company size, as a stand-in, not a fund. Show both sub-periods and the combined period. The comparison starts at the 1969-02-03 close, so that date’s return is set to zero. The stand-in includes the close on 1993-01-29; SPY returns begin on 1993-02-01 using its 1993-01-29 close as the starting price. The comparison is bought and held without trading costs, with tax on its final sale.

The data have no volume and no final return at death. Many early prices are quote midpoints, the middle of the quoted bid and ask, rather than trades; only about 1,203 NYSE common stocks qualify on 1969-02-03. Results are reported as found, with no tuning on this window, and all variants are shown.

The 1968–1999 CRSP market-return completeness check found 8,058 dates and no null values on 2026-11-26; see the [research agenda](../../programme/next-research-agenda-2026-11-26.md) and [database measurements](../../../docs/database-inventory-2026-11-26-measurements.txt). The driver checks the comparison and stock-tape dates against the NYSE trading calendar before scoring.

A bounce between quoted bid-ask midpoints can wrongly trigger arm C. A sale to fund a January tax bill may realize a short-term gain even in arm B. Tax funding sales and other early exits can occur between scheduled rebalances, so the actual tax lot holding period controls the rate.

The dirty-check and linked-Yahoo drift check run before the claim. The claim files are
`_early-A.claim`, `_early-B.claim` and `_early-C.claim` in this study folder; they record the time,
commit, command and output path. After a crash before any scoring, the owner may remove the
relevant claim after checking that no result was produced; keep it if scoring may have occurred.

The period breakdown follows the continuous strategy account, including taxes actually paid,
without a hypothetical sale at the first boundary. The comparison assumes a sale at each reported
period end. Therefore a period-end strategy value can include unrealized gains and deferred tax;
the sub-period readings do not represent two separately liquidated accounts. Realized gain totals
are sale-time gains net of losses, before later wash-sale adjustments.

For arm C, a reopening price is compared with the last observed price across at most ten missing
trading sessions. The data preparation can join a gap of one to ten sessions when its other
continuity checks pass; a longer gap starts a separate price-history column. Comparison Sharpe
and deepest fall use its pre-tax daily curve; growth and yearly growth include tax once at the
end of the reported period. The raw output includes that pre-tax daily curve.
