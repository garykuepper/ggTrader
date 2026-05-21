# Phase 4: Funding-rate arb on Kraken (FundingCarryBTC)

**Date:** 2026-05-18
**Status:** complete

## Why funding-arb instead of cash-and-carry

Reality check before any code: Kraken Futures lists **zero linear dated
quarterlies**. Their dated quarterly product line (`FI_*`) is inverse
(BTC-margined, BTC-P&L). Their linear product line (`PF_*`) is
**perpetuals only**. Cash-and-carry on real Kraken data requires either
pivoting venue or pivoting strategy. The user chose strategy pivot:
funding-rate arbitrage on `PF_XBTUSD` (linear perp).

This IS the canonical Kraken-tradeable "carry" trade. The math:
long spot + short perp (equal notional, hedged) → P&L = realized funding
collected by the short side, net of fees and any imperfect-hedge drift.

## Data backfill — what we actually got

| Dataset | Endpoint | Window |
|---|---|---|
| `funding_rates(PF_XBTUSD)` | `/derivatives/api/v4/historicalfundingrates` | 2025-05-15 → 2026-05-18 (**API hard-caps at ~1y**) |
| `perp_ohlcv(PF_XBTUSD, 1h)` | `/api/charts/v1/trade/.../1h` | 2022-03-23 (listing) → 2026-05-18 |
| `ohlcv(BTC-USD, 1h)` (spot) | already in TimescaleDB + Coinbase gap-fill | 2023-01-01 → 2026-05-18 |

**Spot 2022 history is unavailable** — Kraken's spot OHLC REST endpoint
only serves the most recent ~720 candles, Binance is geo-blocked, and
Coinbase doesn't reach pre-2023 either (didn't try other venues since
the rest of the chain works from 2023 onward).

Call counts / elapsed:
- Perp 1h chart: 17 paginated calls, **19.7s** wall (≥2000 bars/call, 250ms throttle)
- Funding history: 1 call, **1.9s** wall
- Matview refresh: **46.2s** wall (~80k rows × window functions)
- Spot gap-fill (Coinbase): 11 paginated calls, ~5s wall

Reproducible by re-running `scripts/backfill_kraken_futures.py`. Snapshot
date is logged each run.

## The three backtests

| | Window | Years | Return | Sharpe | Max DD | Trades | Fees | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **A. Real funding** | 2025-05-15 → 2026-05-15 | 1.00 | **+0.45 %** | **4.44** | -0.05 % | 8 | $111.88 | canonical |
| **B. Basis proxy** | 2023-01-01 → 2025-05-15 | 2.37 | +3.45 % | 7.34 | -0.10 % | 22 | $375.48 | older regime |
| **C. Basis proxy (overlap)** | 2025-05-15 → 2026-05-15 | 1.00 | +0.76 % | 8.64 | -0.02 % | 4 | $51.08 | validates B |

Numbers are net of trade fees and include funding accrual (per-bar
carry hook in the backtest engine).

## Proxy-credibility analysis (overlap window)

Pairwise statistics on 8,812 hourly bars where both real funding
(`funding_apr_30d`) and basis proxy (`premium_apr_30d`) exist:

| Metric | Value |
|---|---:|
| Correlation (Pearson) | **+0.983** |
| Mean bias (basis − funding) | **+1.76 % APR** |
| RMSE | 2.17 % APR |
| Trade-event overlap (A ∩ C) | **0 of 8 / 4** |

So:
- **The basis proxy tracks real funding extremely tightly** (ρ = 0.98).
- **But it sits ~1.76% APR higher on average.** With the 8% APR entry
  threshold the basis-proxy strategy hits the bar earlier and stays in
  longer than the real-funding strategy. Even at 98% correlation, the
  resulting strategy timings diverge enough that **zero entry/exit
  events line up** between A and C.
- That's why C books $0.76 %/yr while A books $0.45 %/yr: C stays in
  position longer (4 trades = 1 cycle vs A's 8 trades = 2 cycles) and
  the basis premium overstates the carry available.

## Regime breakdown of B (basis-proxy, 2.4 years)

| Year | Return | Max DD | Days |
|---|---:|---:|---:|
| 2023 | +0.66 % | -0.07 % | 365 |
| 2024 | +2.64 % | -0.10 % | 366 |
| 2025 (partial, Jan–May) | +0.11 % | -0.07 % | 135 |

2024 (the ETF spike year) dominates the basis-proxy 3-year P&L. Outside
that regime, the strategy roughly breaks even. **The funding-arb edge on
Kraken is regime-dependent and concentrated in contango spikes** — not a
steady all-weather yield.

## My call on the basis-proxy result

**Treat B (the 3-year basis-proxy result) as *directionally* credible but
not numerically trustworthy.** Specifically:

- ✅ **Use it to assess regime persistence** (does the strategy work in
  2023 chop? 2024 ETF spike? both?). The signal is there in both years,
  concentrated in 2024.
- ❌ **Do not use B's +3.45 % return as a forecast.** It overstates carry
  by ~1.76 % APR × 65 % time-in-position ≈ 1.15 % per year of inflated
  return. The "real" expected return for a 2023-2025 funding-arb run
  would be closer to +1.5 – 2 % over 2.4 years.
- The Sharpes (7-8) on B and C are also inflated because the strategy
  collects a smooth carry stream while the small denominator
  (daily-bar P&L volatility) understates real execution volatility.

**The 1-year real-funding result (A: +0.45 %, Sharpe 4.44) is the only
number I'd quote to a fund seed conversation.** Sharpe is high but
absolute return is anemic — Kraken's funding regime in this window
was modest. The strategy works; the carry just isn't deep on Kraken
right now.

## Data quality / anomalies surfaced

1. **Funding API hard-cap at ~1 year.** Not documented, discovered
   empirically. Multiple param names tried (`from`, `to`, `since`,
   `before`) — none honored. Cannot get 2022-2024 funding from the
   public API.
2. **Spot 1h coverage discontinuous in early 2026.** Kraken's spot OHLC
   REST only serves the most recent ~720 candles regardless of `since`.
   Coinbase gap-fill backfilled Jan-Apr 2026; older 2022 gap remains
   (no usable source found).
3. **Coverage mismatch between sources.** `basis_series` matview
   inner-joins spot ∩ perp on (ts, interval), so the basis-proxy series
   exists only where BOTH spot and perp 1h bars exist. ~30k rows
   2023-01-01 → 2026-05-18.
4. **Basis-vs-funding sign convention confirmed.** Late-2025 funding
   averaged +0.0073 per hour absolute → +6 % APR. Sign convention:
   positive funding ⇒ longs pay shorts ⇒ short perp receives. Test
   `test_sign_convention_positive_funding_means_longs_pay` asserts this
   against Nov 2025 real data.
5. **Basis annualization gotcha.** Kraken's `fundingRateCoefficient` is
   24, meaning the funding mechanism converges the basis over 24h, not
   1h. Naïve `basis × 24 × 365` overstates by 24×. Correct annualization
   is `basis × 365`, validated against real funding (correlation 0.98).
   The matview DDL documents this.

## Architectural feedback (Phase 3.5 abstractions, applied to funding)

Same protocol as the Phase 3 feedback — what felt awkward this pass.

### 1. The backtest engine had no carry/funding hook  — added in this pass

Phase 3.5's engine modeled price P&L only. For a funding-arb strategy,
**price P&L is essentially zero by design** (the hedge is the whole
point) — the entire return is from per-bar funding accrual. I had to
extend `run_backtest` with a `position_carry_fn` callable. The CLI
detects whether the strategy exposes a `.position_carry` method and
wires it through.

**Suggestion:** carry/funding should be a first-class engine concept,
not an optional callable. Promote it to a `Strategy.position_carry`
abstract method with a default returning 0. Engines always call it; most
strategies inherit zero.

### 2. The "daily bar" engine cadence is wrong for hourly strategies

`FundingCarryBTC` declares `timeframe="1h"` but the engine iterates
`freq="1D"`. The mismatch silently works because the 30d-smoothed signal
barely moves intra-day, but a faster signal (1h funding, scalping) would
miss everything. I papered over it by accruing daily carry as
`signal / 365` — fine for a smooth signal, broken for spiky ones.

**Suggestion:** engine should iterate at `Strategy.timeframe`, not a
hard-coded daily cadence. Trivial change; defer until a strategy
genuinely needs sub-daily resolution. Document the daily-bar limitation
in `run_backtest`'s docstring meanwhile.

### 3. Hysteresis state lives on the Strategy instance, breaking re-entrancy

`FundingCarryBTC._in_position` and `_consecutive_negative` are instance
attributes mutated inside `generate_signals`. That works for a single
backtest pass but breaks if you reuse a strategy instance for a second
pass, or run two backtests in parallel against one instance. The
Phase 3.5 `Strategy` ABC has nothing to say about strategy state.

**Suggestion:** either make Strategy explicitly per-pass-stateful (with
a `reset()` method and a contract that instances aren't shared) or
externalize state into a `StrategyState` parameter the engine passes
into `generate_signals`. The latter is the textbook "pure function"
shape; the former is pragmatic. Pick one and put it in the ABC.

### 4. FeatureStore returns DataFrame but engine only wants one row

`feature_store.get_at(name, instruments, ts)` returns a `pd.Series`
indexed by the feature's column labels. For pair features the label is
`"spot|perp"` and the consumer has to know the convention. For
single-instrument features it's the bare symbol. The strategy ends up
with the slightly grim `row[pair_column_label([spot, perp])]` lookup.

**Suggestion:** add a thin helper to FeatureStore:
`get_scalar(name, instruments, ts) -> Decimal` for the by-far-most-common
case where the strategy wants one number. Pair/universe features can
still return DataFrame via `get`. Cuts ~3 lines of label arithmetic per
strategy.

### 5. The TimescaleFeatureStore in-memory cache is footgun-y

I added a per-instance dict cache keyed by `(feature_name,
instrument-tuple)` to avoid re-querying the DB on every `get_at`. But
it only hits if the request window is contained in a previously-fetched
window. For the per-bar `get_at(ts, ts)` access pattern, that's never
true — every call is a new query. **The cache as written gives no
benefit in the backtest's actual usage**. I left it because fixing it
properly requires a range-merge cache (or pre-loading the entire
backtest window upfront, which is what real implementations do).

**Suggestion:** add an explicit `prefetch(name, instruments, start, end)`
method to FeatureStore. Backtest engine calls it once at the start with
the full window. Subsequent `get_at` calls are pure DataFrame slices.
Drop the (mis-)cache.

### 6. mid_price came out clean

The Phase 3.5 prescription "mid_price as a standard feature" worked
exactly as designed. The same engine code that prices BTC spot also
prices PF_XBTUSD perp, with no special-casing — `_table_and_symbol_for`
dispatches by `asset_class` inside `features/price.py`. This is the kind
of payoff the hexagonal refactor was supposed to deliver. ✓

### 7. Strategy timeframe / feature timeframe mismatch is silent

The strategy declares `timeframe="1h"` and `required_features=["funding_apr_30d"]`.
The feature is itself a 30-day rolling average of hourly funding. There's no
checking that the engine's iteration cadence, the strategy's nominal timeframe,
and the feature's smoothing window are consistent. Right now they happen to be
because I tuned each one by hand. A `validate(self) -> list[str]` method on
Strategy that runs at backtest-start time would catch incoherence early.

## Files created / modified

**New (Phase 4):**
- `src/ggTrader/data/store/migrations/001_kraken_futures.sql` — schema
- `src/ggTrader/data/sources/kraken_futures.py` — ingester (REST-based)
- `src/ggTrader/features/derivatives.py` — `funding_apr`, `basis_premium_apr`
- `src/ggTrader/features/price.py` — `mid_price` (spot + perp dispatch)
- `src/ggTrader/features/timescale_store.py` — `TimescaleFeatureStore`
- `src/ggTrader/strategies/carry/funding_carry.py` — `FundingCarryBTC` + `SpotPerpUniverse`
- `src/ggTrader/config/strategies/funding_carry_btc_real.yaml`
- `src/ggTrader/config/strategies/funding_carry_btc_basis.yaml`
- `scripts/backfill_kraken_futures.py`, `scripts/backfill_spot_coinbase.py`,
  `scripts/backfill_spot_btc.py`, `scripts/phase4_comparison.py`
- `tests/unit/test_funding_carry.py` (13 tests)
- `tests/integration/test_funding_carry_backtest.py` (1 test)
- `tests/integration/test_kraken_futures_ingest.py` (3 tests)
- `docs/phase4_funding_carry.md` (this file)

**Modified:**
- `src/ggTrader/backtest/vectorized.py` — added `position_carry_fn`
  parameter + per-bar carry accrual block before MTM
- `src/ggTrader/cli/cmd_backtest_strategy.py` — dispatch to
  TimescaleFeatureStore for FundingCarryBTC; wires `position_carry`;
  honest data-source footer

**Not touched:**
- Phase 3.5 CashAndCarryBTC + synthetic feature store (regression-tested)
- Live trading code
- Existing TA indicator pipeline

## Test results

- Unit (Phase 4): 13/13 ✓
- Integration (Phase 4): 4/4 ✓ (1 backtest + 3 ingester / sign-convention)
- Phase 3.5 regression: 15/15 ✓
- Full suite: 246 passed / 2 pre-existing failures unrelated to Phase 4
- mypy on Phase 4 modules: ✓ no issues

## Next followups (not done this pass)

1. **Engine: hourly bar cadence** — let the engine iterate at `Strategy.timeframe`.
2. **Engine: position carry as first-class Strategy method** — drop the callable.
3. **FeatureStore: `prefetch(window)` + `get_scalar()`** — fix the dead cache,
   tighten the per-bar access ergonomics.
4. **Strategy: state hygiene** — pick "reset()" vs "external StrategyState".
5. **Sizer abstraction** — still deferred. FundingCarryBTC's fixed-notional
   sizing works, but a vol-target sizer is the natural next layer.
