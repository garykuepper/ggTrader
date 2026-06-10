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

> **Status: RUNNING** (started 2026-06-10, ~26 min/month on 4 cores, ~64
> months). Fill this section from
> `results/monthly_wf/sp500_monthly_v1/summary.json` when complete.

Universe/data: 1,919 rows × 616 symbols (1d), 64 selection dates
2021-02-26 → 2026-05-29, 52 delisted tickers unavailable.

| Metric | Strategy | SPY |
|---|---|---|
| CAGR | _TBD_ | _TBD_ |
| Sharpe | _TBD_ | _TBD_ |
| Sortino | _TBD_ | _TBD_ |
| Max drawdown | _TBD_ | _TBD_ |
| Monthly hit rate vs SPY | _TBD_ | — |

- **Holding-days distribution** (goal: 2–10 day average): _TBD_
- **Avg monthly selection turnover:** _TBD_
- **Combo tournament winners** (selection counts): _TBD — early checkpoints
  show `rsi_reversal+fixed_sl_tp` dominating, with `bbands_mean_reversion`,
  `supertrend_flip`, `adx_filtered_rsi` also winning slots; consistent with the
  crypto edge-search finding that mean-reversion is where the life is._

Quick-mode sanity run (10 stocks, 6 months — mechanics check only, not an
estimate): monthly hit rate 0.6 vs SPY, median hold ~15.6d, turnover 0.44.

## 5. Decision rule

Deploy consideration requires the **full-run OOS** equity curve to beat SPY on
risk-adjusted terms (Sharpe/Sortino) with a drawdown the account can tolerate,
across regimes (2021 melt-up, 2022 bear, 2023-25 recovery). In-sample numbers,
including §4's quick run and anything from `equity_wfo_research.py`'s combined
validation, do not count.
