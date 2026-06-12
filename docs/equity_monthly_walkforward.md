# Equity Research: Rolling Monthly Walk-Forward (S&P 500)

> **Date:** 2026-06-10
> **Status:** Infrastructure complete and leak-checked; first full run in progress
> **Replaces:** `docs/archive/sp500_equity_strategy_research.md` (results invalid —
> selection bias, parameter leakage, survivorship bias) and
> `docs/archive/refactor_vectorbt_simplification.md` (cited nonexistent vbt APIs)
> **Commits:** `4e7af64`, `6a0acc6`, `63714e0` (refactor) · `f8c7efe`, `ee705e0` (research infra)

## 1. What this is

An honest out-of-sample backtest of the strategy library on US equities, built so
that *every* decision the backtest makes at time T uses only data ≤ T:

- **Point-in-time universe** — S&P 500 membership as of each selection date
  (fja05680/sp500 history, 2,712 snapshots 1996 → 2026-06-02, committed at
  `data/universe/sp500_constituents_history.csv.gz`). No "today's index applied
  to the past."
- **Full strategy tournament** — every stock is evaluated on all entry × exit
  combos from the registries (11 entries × 3 exits = 33 combos), not a
  pre-chosen strategy. The WFO judges the best combo per stock per month.
- **Selection inside the loop** — at each month-end T, run the per-stock WFO on
  `data.loc[:T].tail(504)` (2-year trailing window), rank by OOS robustness,
  pick the top 50, freeze entry/exit/params, then trade *forward* one month.
  Stitch the monthly forward returns into one equity curve.
- **Leak-checked** — `--leak-check` re-runs a selection with all post-T data
  removed and asserts identical picks. **PASSES.**

### Honest limitations (quantified, not hand-waved)

- **Residual data survivorship:** yfinance lacks many delisted tickers (52 of
  668 union members failed to download in the v1 run — VIAC, FRC, etc.).
  Membership is point-in-time, but a stock with no data can't be selected.
  Per-month `coverage.json` records exactly who was missing.
- **Month-boundary close:** forward simulation implicitly flattens at month
  ends. Bounded impact given the 2–10 day target hold, but it slightly
  understates trades that would have spanned the boundary.
- **Fees/slippage:** `FEES=0.0` (commission-free assumption), `SLIPPAGE=0.0005`.

## 2. How to run

```bash
source .venv/bin/activate

# smoke test (~minutes): 20 stocks, ~6 months, 2x1 strategies
python -u scripts/sp500_monthly_walkforward.py --quick

# verify no lookahead in the selection layer
python -u scripts/sp500_monthly_walkforward.py --quick --leak-check

# full run (unattended; checkpointed + resumable per month)
nohup python -u scripts/sp500_monthly_walkforward.py --jobs 4 --run-id sp500_monthly_v1 \
    > results/monthly_wf/full_run_v1.log 2>&1 &
```

Key defaults (`MonthlyHarnessConfig`): eval 2021-01-31 → present (64 monthly
selection dates), `lookback_bars=504`, `n_splits=5`, `test_ratio=3.0`,
`top_n=50`, `max_position_pct=0.02`, all registry strategies, detailed grid
book. Narrow with `--entries/--exits`, throttle with `--refit-every N`.

Checkpoints land at `results/monthly_wf/<run-id>/month=YYYY-MM/`
(`selections.json`, `month_returns.parquet`, `coverage.json`); a killed run
resumes where it left off. Final outputs: `summary.json` +
`equity_curve.parquet`.

There is also a simpler per-stock tournament CLI (`scripts/equity_wfo_research.py`,
package `ggTrader.research.equity_wfo`) for ranking stocks/strategies on a fixed
window — its combined-portfolio output is labeled **IN-SAMPLE smoke test** and
must not be quoted as an OOS result.

## 3. Refactor that made it possible (measured deltas)

The legacy scalar signal path was deleted; everything now runs through the
vectorized registry path. Validated with deterministic before/after snapshots
(synthetic OHLCV, seeded):

| Snapshot | Result |
|---|---|
| S1 — vectorized grid (psar_adx + atr_trailing) | **Bit-identical** through all phases |
| S2 — legacy scalar path | 21 → 22 trades; fully attributed to two *bug fixes* (below) plus gap-adjusted stop fills replacing close-price fills |
| S3 — full WFO loop, ema_cross variant | Now actually runs ema_cross (previously silently ran psar_adx) |

Bugs fixed by deleting the legacy path:

1. **Wrong-strategy replay:** `FastBacktest._generate_signals` called
   `SignalFactory` unconditionally and `**kwargs` swallowed unknown params — every
   legacy-path call site (WFO test folds, frozen replay, smoke/holdout,
   portfolio optimizer) evaluated psar_adx with defaults regardless of
   `ENTRY_STRATEGY`.
2. **DMP/DMN mismap:** vbt's `from_pandas_ta("adx")` wrapper mismapped the
   directional-movement columns (pandas_ta ground truth DMP=49.4 vs legacy path
   16.1). The vectorized `IndicatorPrecomputer` is correct.

Other refactor outcomes: `vbt_patches.py` monkey-patching replaced with explicit
copies + raw profit-factor/expectancy computation in `core/metrics.py`
(`safe_portfolio_stats()` for `pf.stats()` callers); disk indicator cache
removed (in-memory FIFO-bounded shared cache kept — it's what makes the
33-combo sweep tractable); `avg_holding_days` added to `get_stats()`. Test
suite: 260 passed, 3 pre-existing failures unchanged from the pre-refactor
baseline.

## 4. Results — run `sp500_monthly_v1`

> **Status: COMPLETE** (started 2026-06-10, finished 2026-06-11; ~26 min/month
> on 4 cores, 64 months). **Verdict: clear NO-GO** — the strategy was nearly
> flat over 5+ years while SPY doubled, and lost on every risk-adjusted metric.

Universe/data: 1,919 rows × 616 symbols (1d), 64 selection dates
2021-02-26 → 2026-05-29, 52 delisted tickers unavailable. 63 traded months.

| Metric | Strategy | SPY |
|---|---|---|
| Total return | +0.88% | +103.83% |
| CAGR | 0.17% | 14.46% |
| Sharpe | 0.12 | 0.89 |
| Sortino | 0.15 | 1.22 |
| Ann. volatility | 1.49% | 16.88% |
| Max drawdown | −4.30% | −24.50% |
| Monthly hit rate vs SPY | 0.365 | — |

- **Holding-days distribution** (goal: 2–10 day average): p25 2.6 / median 4.9
  / p75 9.4 — squarely in the target band, so the signals do trade; they just
  don't make money after fees.
- **Avg monthly selection turnover:** 0.83 — the tournament re-picks ~83% of
  the book every month, i.e. selections are unstable month to month.
- **Combo tournament winners** (selection counts, top of 25 combos that ever
  won a slot): `rsi_reversal+fixed_sl_tp` 663, `supertrend_flip+fixed_sl_tp`
  472, `bbands_mean_reversion+fixed_sl_tp` 421, `donchian_breakout+fixed_sl_tp`
  250, `macd_cross+fixed_sl_tp` 240. Mean-reversion entries and fixed
  stop/target exits dominate — consistent with the crypto edge-search finding —
  but `psar_adx` (the old in-sample crypto favorite) won only 2 slots in 3,200,
  confirming its earlier results were hindsight bias.
- **Reading:** ann. vol of 1.49% with 2% per-position caps means the portfolio
  was mostly in cash — per-stock WFO gates rejected most candidates most months,
  and what passed didn't carry out-of-sample. Selection robustness in-sample
  does not transfer to the forward month (hit rate 0.365 < coin flip).

Quick-mode sanity run (10 stocks, 6 months — mechanics check only, not an
estimate): monthly hit rate 0.6 vs SPY, median hold ~15.6d, turnover 0.44.

### Momentum baselines over the same window (run 2026-06-12)

Same harness, same 64 selection dates, `--strategy xs_momentum` (12-1
cross-sectional momentum, top-50 equal weight, always invested):

| Metric | xs_momentum | SPY |
|---|---|---|
| Total return | +125.98% | +104.52% |
| CAGR | 16.69% | 14.51% |
| Sharpe | 0.82 | 0.89 |
| Sortino | 1.13 | 1.22 |
| Ann. volatility | 21.83% | 16.89% |
| Max drawdown | −22.38% | −24.50% |
| Monthly hit rate vs SPY | 0.49 | — |

Avg exposure 1.00, avg monthly turnover 0.29. Beats SPY on raw return by
taking ~30% more volatility — **fails the §5 risk-adjusted test** (Sharpe and
Sortino both below SPY). A textbook result: the equity momentum premium is
roughly market-like after 2021–26.

`dual_momentum` produced **identical** results: with top-50 of ~500 names, the
absolute-momentum filter never fired — checkpoint audit shows the *minimum*
selected 12-1 momentum across all 64 months was +12.2% (October 2022, the bear
low). The absolute filter only matters with a much smaller book or applied at
the portfolio level (Antonacci-style, vs T-bills); as a stock-level filter on
a top-50 S&P 500 book it is vacuous.

Checkpoints: `results/monthly_wf/sp500_xs_momentum/`,
`results/monthly_wf/sp500_dual_momentum/`.

## 5. Decision rule

Deploy consideration requires the **full-run OOS** equity curve to beat SPY on
risk-adjusted terms (Sharpe/Sortino) with a drawdown the account can tolerate,
across regimes (2021 melt-up, 2022 bear, 2023-25 recovery). In-sample numbers,
including §4's quick run and anything from `equity_wfo_research.py`'s combined
validation, do not count.
