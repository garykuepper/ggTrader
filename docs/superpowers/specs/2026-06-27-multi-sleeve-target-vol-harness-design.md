# Multi-Sleeve Target-Vol Research Harness — Design

**Date:** 2026-06-27
**Status:** Approved (brainstorming)
**Author:** garykuepper + Claude

## Purpose

Validate, under honest walk-forward gating, that blending weakly-correlated
single-stock reversion sleeves and scaling the blend to a target volatility
beats the current live large-cap engine on a risk-adjusted basis.

This is the gate-honest successor to the idealized analysis in
`scripts/portfolio_blend.py`. That script blended **fixed-parameter** (no
anchor/halt) equity curves and used **full-period** volatility for risk parity —
both in-sample shortcuts. The June 27 correlation study using it found:

- SP500 ↔ MidCap400 = 0.70, **MidCap400 ↔ Nasdaq100 = 0.35** (weakly correlated)
- MidCap+Nasdaq 50/50 blend: Sharpe 1.39, MaxDD −3.97%, vol 4.20% — best
  risk-adjusted of any combination, vs SP500 core 1.15 / −8.66% / 6.82%.

The low blend volatility (4.2%) means it can be **levered up** toward the core's
risk (~6.8% vol) to recover CAGR while keeping a shallower drawdown. This harness
measures whether that result survives (a) real per-sleeve gating and (b)
out-of-sample, rolling volatility estimation.

## Scope

**In scope:** an offline research harness (CLI script + a pure, unit-tested
allocation module) that runs three fixed sleeves (`sp500`, `midcap400`,
`nasdaq100`) through their own gated WFO, combines them with a rolling
inverse-volatility weighting scaled to a target vol (leverage allowed, capped),
and prints a comparison report.

**Out of scope (YAGNI / follow-ups):**
- Live/paper wiring (the allocator that sizes real orders) — separate phase.
- Generalized N-sleeve framework — three sleeves are parameters, not a plugin system.
- Transaction-cost / borrow-cost modelling on the leverage — flagged in the report as a known omission.
- Russell 2000 — its snapshot is a 200-name sample, ~29% backfilled (58/200); needs full PIT membership + OHLCV backfill before inclusion.

## Architecture

Two new units, clear separation of concerns:

### 1. `src/ggTrader/lab/allocation.py` (new — pure functions, no I/O)

Reusable, fully unit-tested overlay math. No DB, no WFO, no printing.

- `trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series`
  Rolling annualized realized volatility: `std(daily returns over window) * sqrt(252)`.
  Uses only trailing data (pandas `.rolling(window)`), so value at date *t*
  excludes *t*'s own future.
- `inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]`
  Risk-parity weights `w_i = (1/vol_i) / Σ(1/vol_j)`, summing to 1.0. Guards
  against zero/NaN vol (drop or floor).
- `target_vol_scale(blend_trailing_vol: float, target_vol: float, max_leverage: float = 2.0) -> float`
  Exposure multiplier `clip(target_vol / blend_trailing_vol, 0.0, max_leverage)`.
- `combine_sleeves(sleeve_returns: pd.DataFrame, target_vol: float = 0.068, window: int = 60, max_leverage: float = 2.0, rebalance: str = "ME") -> tuple[pd.Series, pd.DataFrame]`
  Orchestrates the overlay over the aligned daily-returns frame:
  - At each rebalance date (month-end, pandas `"ME"`), compute per-sleeve
    trailing vol and inverse-vol weights, form the provisional blended return
    series, compute its trailing vol, and derive the target-vol scale.
  - Hold those weights + scale until the next rebalance.
  - Returns `(blended_daily_returns, diagnostics_df)` where diagnostics carries
    per-rebalance weights, blend vol, and applied leverage.
  - **Warmup:** until `window` daily observations exist, use equal weights at
    scale 1.0 (documented, no look-ahead).

### 2. `scripts/multi_sleeve_research.py` (new — orchestration)

Mirrors the conventions of `midcap_research.py` / `portfolio_blend.py`.

1. Load OHLCV for the union of the three sleeve universes + benchmarks (SPY).
2. For each sleeve, run its **gated** WFO and take the OOS equity curve.
3. Align the three OOS curves on common dates → daily returns frame.
4. Call `allocation.combine_sleeves(...)`.
5. Print the report.

### 3. `src/ggTrader/lab/wfo.py` (targeted change)

`run_wfo()` currently returns only the printed table string, so the gated
per-fold OOS equity curve is unrecoverable by callers. Introduce a structured
return:

```python
class WfoResult(NamedTuple):
    oos_equity: pd.Series       # continuous gated OOS curve (anchor/halt applied)
    fold_results: list[dict]    # existing per-fold dicts
    live_params: dict           # recommended live params
    table: str                  # formatted table (still printed by run_wfo)
```

`run_wfo` keeps printing the table (back-compatible behaviour) but returns
`WfoResult`. Existing callers that use the return value as the table string
(`midcap_research.py` via `parse_table`) are updated to read `result.table`.

## Data Flow

```
for sleeve in [sp500, midcap400, nasdaq100]:
    result = run_wfo(sleeve, ... gated ...)        # anchor + halt applied
    curves[sleeve] = result.oos_equity
returns_df = align(curves).pct_change().dropna()   # common dates
blended, diag = combine_sleeves(
    returns_df, target_vol=0.068, window=60, max_leverage=2.0, rebalance="ME")
report(sp500_core=curves["sp500"], sleeves=curves, blended=blended, diag=diag)
```

## Report

Markdown performance table (CAGR / Sharpe / Sortino / Annual Vol / Max DD):

- **S&P 500 (gated core)** — the baseline the blend must beat.
- Each sleeve standalone (gated): SP500, MidCap400, Nasdaq100.
- **Inverse-vol + target-vol blend** (the candidate).
- SPY buy-and-hold cross-reference.

Plus:
- Pairwise OOS correlation matrix of the three gated sleeve return streams.
- Realized leverage stats from diagnostics: average and max applied scale (how
  much margin the result actually needs).
- **Honesty caveats** printed in the report: (a) curves are gate-honest but
  contain no borrow/transaction cost on leverage; (b) `target_vol` is a fixed
  a-priori constant (default 0.068 = observed SP500 core vol, for equal-risk
  comparison), not re-fit per period.

## Target-Vol & Leverage Decisions

- **Weighting:** rolling inverse-volatility (risk parity), recomputed monthly
  from 60-day trailing realized vol.
- **Target vol:** default 0.068 annualized (matches the live SP500 core's
  realized vol so the comparison is equal-risk); CLI-overridable.
- **Leverage:** allowed, capped at 2.0× (Alpaca Reg-T supports 2×); CLI-overridable.
- **Rebalance:** monthly (month-end).

## Testing

`tests/lab/test_allocation.py` (TDD, pure unit tests):
- `trailing_realized_vol`: known-input annualization; NaN during warmup.
- `inverse_vol_weights`: lower-vol sleeve gets higher weight; weights sum to 1;
  zero/NaN-vol guard.
- `target_vol_scale`: scales up when blend vol < target; clipped at `max_leverage`;
  floored at 0.
- `combine_sleeves`: warmup uses equal-weight scale 1.0; **look-ahead check** —
  weights/scale applied on date *t* depend only on returns strictly before the
  rebalance, never future data; diagnostics frame shape/contents.

`run_wfo` change covered by existing `tests/lab/test_wfo.py` plus an assertion
that `WfoResult.oos_equity` is a non-empty aligned series.

The orchestration script is thin and validated by an end-to-end run.

## Success Criteria

1. `allocation.py` unit tests pass, including the look-ahead guard.
2. `multi_sleeve_research.py` runs end-to-end and prints the report.
3. The report answers the headline question: at equal risk to the gated SP500
   core, does the levered 3-sleeve blend beat it on CAGR **and** drawdown after
   real gates? (Either verdict is a valid, decision-useful result.)
