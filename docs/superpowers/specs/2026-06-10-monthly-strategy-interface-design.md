# Design: Strategy-Agnostic Monthly Walk-Forward Harness

> **Date:** 2026-06-10
> **Status:** Approved (conversation 2026-06-10)
> **Motivation:** The honest monthly walk-forward harness
> (`research/monthly_walkforward.py`) is hard-wired to one paradigm — the
> per-stock WFO strategy tournament — which the interim `sp500_monthly_v1`
> results show has no out-of-sample edge (Sharpe −0.53 vs SPY 0.73 through
> 37/64 months, hit rate 0.35, ~9% average exposure). The harness itself
> (point-in-time universe, leak check, checkpointing, SPY benchmark) is the
> durable asset. Generalize it so new paradigms — starting with
> cross-sectional momentum and dual momentum — are ~50-line plug-ins, each
> honestly testable in about an hour.

## 1. Goals

- A `MonthlyStrategy` interface: *given data ≤ T and the point-in-time
  eligible universe, produce next-month selections; then simulate the forward
  month.*
- The existing WFO tournament becomes plug-in #1 with **identical behavior**
  (existing checkpoints stay readable; `sp500_monthly_v1` stays comparable).
- Two new plug-ins: `CrossSectionalMomentum` (12-1 momentum, top-N, equal
  weight, fully invested) and `DualMomentum` (same + absolute-momentum cash
  fallback).
- Harness hardening: `select()` receives data already truncated to ≤ T (a
  buggy strategy cannot peek by accident); per-month diagnostics gain
  `avg_exposure` (mean fraction of capital deployed).

Non-goals: position carry across month boundaries (known, documented
limitation); intraday data; ML strategies; changing the running
`sp500_monthly_v1` process or its outputs.

## 2. Interface

New module `src/ggTrader/research/monthly_strategies.py`:

```python
class MonthlyStrategy(Protocol):
    name: str

    def select(
        self,
        asof: pd.Timestamp,
        ohlcv: pd.DataFrame,        # MultiIndex (symbol, field); index <= asof GUARANTEED
        eligible: list[str],        # PIT members with >= min_history_bars of data
    ) -> list[dict]:                # JSON-able records; must include "symbol"
        ...

    def simulate(
        self,
        ohlcv: pd.DataFrame,        # full data (forward month included)
        selections: list[dict],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> tuple[pd.Series, dict]:    # daily returns for (asof, month_end], diagnostics
        ...
```

The select/simulate split is what makes the generic leak check possible:
`leak_check` calls `strategy.select` twice (full vs `ohlcv.loc[:asof]`) and
asserts identical JSON.

## 3. Components

### 3.1 Harness (`monthly_walkforward.py`, slimmed)

Keeps: data loading (`fetch_stock_ohlcv`), PIT membership + eligibility filter
(`min_history_bars`) + `coverage_stats`, month-end selection dates, refit
cadence (`refit_every_n_months`), per-month checkpoints
(`selections.json` / `month_returns.parquet` / `coverage.json`), resume logic,
equity stitching, `benchmark_vs_spy`, turnover/holding-days/combo summaries,
`summary.json` + `equity_curve.parquet`.

Changes:
- `run_monthly_walkforward(cfg, strategy)` takes a `MonthlyStrategy`; the
  month loop calls `strategy.select(asof, ohlcv.loc[:asof], eligible)` and
  `strategy.simulate(ohlcv, selections, asof, month_end)`.
- Eligibility computation moves harness-side (it is universe hygiene, shared
  by all strategies).
- `leak_check(cfg, strategy)` is strategy-generic.
- Diagnostics gain `avg_exposure`: mean over forward-month days of
  (invested asset value / portfolio value), computed by each strategy's
  simulate (helpers provided) and surfaced in `coverage.json` diagnostics and
  the run summary.

### 3.2 Plug-in #1: `WfoTournamentStrategy`

Current behavior moved verbatim: `select` = `select_for_month` body (process
pool over `_select_worker`, full entry×exit tournament, top-N by OOS
robustness, trailing-window `avg_holding_days` replay); `simulate` =
`simulate_forward_month` (signals warmed up on `warmup_bars` pre-T bars,
entries masked to the forward month, one cash-shared `Portfolio.from_signals`,
2% sizing). Config knobs unchanged (entries/exits/grid_book/n_splits/
test_ratio/top_n/max_position_pct/warmup_bars/n_jobs).

### 3.3 Plug-in #2: `CrossSectionalMomentum`

- `select`: momentum score = close[T−skip] / close[T−lookback] − 1 with
  defaults lookback=252, skip=21 (the standard 12-1 formulation avoiding
  short-term reversal). Require non-NaN closes at both ends. Rank eligible
  descending, take top-N (reuses `cfg.top_n`, default 50), equal weight
  1/N each. Records: `{"symbol", "weight", "momentum"}`.
- `simulate`: shared `simulate_hold_weights()` helper — buy target weights at
  the first bar after `asof` (`vbt.Portfolio.from_orders`, `targetpercent`
  sizing, one rebalance), hold to `month_end`, return daily returns sliced to
  the month. Fees/slippage from `STOCK_BASE_CONFIG`.

### 3.4 Plug-in #3: `DualMomentum`

`CrossSectionalMomentum.select`, then any pick whose own momentum score < 0
has its weight reassigned to cash (weight dropped; not renormalized). All
negative → empty selection → flat month. ~20 lines; same simulator.

### 3.5 CLI (`scripts/sp500_monthly_walkforward.py`)

- `--strategy {wfo_tournament,xs_momentum,dual_momentum}` (default
  `wfo_tournament` — fully backward compatible).
- `--mom-lookback 252`, `--mom-skip 21`.
- Default `run_id` becomes `sp500_<strategy>` when not given.
- `--quick` and `--leak-check` work for every strategy.

## 4. Error handling

Unchanged conventions: per-symbol failures inside a strategy log and skip;
an empty selection month yields an empty returns series (flat month,
`n_positions=0` recorded); the harness raises only if **no** month produced
returns. Strategy `select` exceptions abort the run (they indicate bugs, not
data issues).

## 5. Testing

- **Unit (new `tests/test_monthly_strategies.py`):**
  - momentum ranking: synthetic data with a deterministic winner ranks first;
    the skip window is actually skipped (price move inside the last `skip`
    bars does not change the score);
  - `simulate_hold_weights`: returns ≈ hand-computed weighted buy-and-hold
    returns on synthetic data; exposure ≈ sum of weights;
  - dual momentum: all-negative momentum month → empty selection / flat
    month; mixed signs drop only the negatives.
- **Equivalence:** `WfoTournamentStrategy` under the `--quick` config
  reproduces the selections of the pre-refactor code (fixture from an
  existing quick-run checkpoint).
- **Leak check:** generic; exercised per strategy in `--quick` mode.
- Full suite `pytest -m 'not integration'` stays green (baseline: 260 passed,
  3 pre-existing failures).

## 6. Compatibility with the running v1 process

The running `sp500_monthly_v1` process loaded its code at start; editing these
files does not affect it. Its checkpoint format is preserved (the
`WfoTournamentStrategy` selection records are identical), so resume and final
summary work regardless of when the refactor lands.
