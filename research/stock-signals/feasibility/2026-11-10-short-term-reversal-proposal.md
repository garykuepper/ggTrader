# Proposed monthly reversal screen

**Status: APPROVED for the historical screen.** The owner approved this specified screen in the conversation on 2026-11-11 UTC. This is a local record, not an independent timestamp. No paper or live trading is authorized.

## Question and inputs

Compare one long-only portfolio: `BIG LoPRIOR` from the `Average Value Weighted Returns -- Monthly` section of the [French 25 size-by-short-term-reversal portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_25_port_form_sz_pr_1_0.html). It holds a large-size group with low returns in the previous month and is re-formed monthly. The comparator is the French broad-market total return, `Mkt-RF + RF`, **not SPY**. This is a portfolio proxy; it does not select S&P 500 stocks or test a hold-winners exit.

The locally frozen portfolio ZIP has SHA-256 `1c4070b54f42c3e4cf58420f6f7e460c08de886e32574ddb71c17bfa2c606260`; the market ZIP has SHA-256 `b03d39e4bb253dda0f14893db3b78e64b40b8d6549a0fc0fd4111dae84c98618`. Both are in `ggTrader-refs/reversal-20261111` outside the repository. The provider returned both same-hash files on 2026-11-11 UTC with last-modified times 14 seconds apart; this does not prove identical underlying vintages. The [driver](code/reversal_screen.py) refuses other hashes and can verify inputs with `--check-only` without parsing return tables. If either input must change, amend the proposal before viewing results.

## Proposed fixed rules

- **Windows:** July 1963–April 2026 and January 1994–April 2026. These match the earlier accruals screen's endpoints and do not constitute an unseen holdout.
- **Costs:** Deduct 1% per year from the candidate only, divided equally across months; report a 2% stress. These are illustrative, not measured costs or taxes. Reversal's monthly turnover could make them inaccurate.
- **Full-history gate:** A 12-lag Newey–West test of monthly arithmetic candidate-minus-market excess after the 1% assumption must have t ≥ 3 and one-sided p ≤ 0.00135.
- **Since-1994 gate:** Candidate annualized compound growth after the 1% assumption minus the market's must be at least +1.0 percentage point per year. Both gates must pass. Missing months, duplicate observations or undefined inference mean unresolved.
- Report terminal wealth, growth, month-end drawdown and the stress result descriptively. Do not change the cell, windows, costs, benchmark or thresholds after viewing results.

## Limits and next action

The 52-week-high and prior daily mean-reversion results remain separate and unchanged. This candidate was considered after many previous searches; no family-wide correction across those searches or independence from related ideas is claimed. The published portfolio does not establish point-in-time stock identity, executable next-session fills, realistic turnover costs or after-tax performance. A pass would not authorize paper or live trading; a miss would not rule out a smaller effect.

The owner approved this candidate before the historical run. The specification and driver must be committed before the calculation. Input and code hashes identify the files used but do not provide an independent witness to when returns were viewed. The production freeze remains in force.
