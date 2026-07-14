# Live 3-Sleeve Blend on Paper Trading — Design

## Context

The July 13 research verdict (`roadmap.md` §3) found that the SP500+MidCap400+
Nasdaq100 blend, run at the leverage the live account can actually use (1.0x,
matching the live account's unlevered flat-3% sizing), beats the SP500-core
baseline on a matched eval window: Sharpe **1.14** vs **0.97**, MaxDD **-5.39%**
vs **-11.0%**, for a CAGR concession (9.93% vs 13.4%). That result was a GO
for the *research* question. It changes nothing about live trading by itself
— the live paper account (`ggt paper`, cron 12:45 Mon-Fri) still runs the
single-universe SP500 `EnsembleSignal` exclusively, with zero code paths
connecting it to `lab/blend.py`.

This spec scopes what it takes to make the validated blend the thing that
actually trades on that account.

## Decisions already made

- **Replace the current live paper account** with the blend, rather than
  running a second account in parallel.
- **Recompute the overlay's trailing-vol input from OHLCV daily** (reusing
  the lab's `simulate_signals()`), not from the live account's own executed
  P&L — matches the research mechanism exactly and has no bootstrap gap.
- **Per-sleeve dollar budget** for position sizing (`portfolio_value ×
  sleeve_weight × scale`, split across that sleeve's buy signals), not a
  flat-3%-per-position approximation — faithful to what was actually
  measured, at the cost of more code.

## Non-goals

- No change to global risk guards (drawdown halt, daily loss limit, overall
  `max_positions`, per-symbol concentration cap) — these are portfolio-level
  and nothing about adding sleeves changes their rationale.
- No re-implementation of the NDH/DSR gates or WFO circuit-breaker in the
  live path. Confirmed: `signal_runner.py` never re-runs gates today — a
  single already-vetted parameter set is hardcoded once, not re-validated
  live. The blend's trailing-vol simulation follows the same pattern: raw
  vol estimation, no gating.
- No attempt to make individual position sizing exactly reproduce
  return-space blending in a mathematically airtight way beyond the
  "rebalance monthly at current portfolio value" invariant below — this is a
  reasonable-but-not-separately-backtested implementation of the validated
  return-stream result, and the spec says so plainly rather than overclaiming
  equivalence.

## Architecture

### 1. Multi-sleeve signal generation

Replace `signal_runner.py`'s single-universe `generate_signals()` with a
version that, for each of `sp500`, `midcap400`, `nasdaq100`:
- Builds that universe's point-in-time member list (existing
  `sp500_members_asof`-style helper, extended to the other two universes —
  `eligible_at()`/`universe_members_asof()` in `lab/data.py` already cover
  this).
- Runs the existing `EnsembleSignal.select()` / `.to_targets()` unchanged,
  using the **same hardcoded default params** the current live account uses
  (see Invariant 1 below) — no per-sleeve parameter divergence.
- Tags resulting buys/sells with their source sleeve.

### 2. Trailing-vol overlay computation

**Invariant 1 (load-bearing):** the monthly trailing-equity-curve simulation
for the overlay must replay each sleeve's signals using the *identical* fixed
`EnsembleSignal` defaults that `signal_runner.py` actually trades — not a
fresh WFO parameter search. Live trading already runs on hardcoded defaults,
not a periodically-refreshed WFO winner; the overlay's vol estimate has to
describe the strategy that is actually running, or it silently decouples
from reality. Concretely: both call sites should construct `EnsembleSignal`
from the same single source of truth (a shared default-params constant), and
this equivalence should be covered by a test that asserts the params
constructed in the trailing-curve simulation match what `signal_runner.py`
instantiates.

Mechanism: on a rebalance date, for each sleeve, pull trailing ~90 calendar
days of OHLCV, run the sleeve's signals, and simulate via the existing
`lab.simulate.simulate_signals()` to get a synthetic equity curve — the same
tool the research blend used, just re-run incrementally in production
instead of once over history. Feed the three curves into the existing, pure,
already-tested `allocation.py` functions unchanged:
`trailing_realized_vol()` → `inverse_vol_weights()` → `target_vol_scale()`
(capped at `max_leverage=1.0`, matching the validated deployable result).

### 3. Rebalance cadence and state

Monthly rebalance (`rebalance="ME"`, matching `combine_sleeves`), not daily.
A persisted "last rebalance date" + the resulting weights/scale live in the
paper-trading DB (extending `persist.py`'s existing schema). On a new
rebalance date, recompute; other days reuse the last-computed values.

**Fallback:** if the trailing-OHLCV fetch fails or is incomplete on a
rebalance date, fall back to the previous month's weights/scale rather than
raising — this must be an explicit, tested code path, not an unhandled
exception. Log/notify (existing `TelegramNotifier`) when the fallback fires,
since it's a signal something's degraded.

### 4. Position sizing

**Invariant 2:** `sleeve_budget = current_portfolio_value × sleeve_weight ×
scale`, computed fresh using the portfolio value *at the rebalance moment* —
not held constant from account inception. This is what makes the live
mechanism a genuine "fixed-weight-between-rebalances" portfolio matching what
`combine_sleeves` modeled; using a stale portfolio value would silently drift
from the validated construction over time.

`risk.py`'s `position_notional()` becomes sleeve-aware: given a sleeve's
budget and its buy-signal count (capped by that sleeve's proportional share
of `max_positions` — `floor(weight_i × max_positions)`, minimum 1 slot for
any sleeve with `weight_i > 0`, remainder slots from rounding down assigned
to the highest-weight sleeve so the total never exceeds `max_positions`),
size each new position at `sleeve_budget / min(signal count, sleeve slot
cap)`. This explicitly overrides the separately-validated
flat-3%-per-position result — call this out in the PR/commit, not just bury
it as an implementation detail, since it's a real, accepted behavioral
change, not a strict refinement of the flat-3% finding. The existing
per-symbol concentration check (`check_concentration`, 5% cap) stays wired in
unchanged against whatever the new prospective notional is, so no position
can blow through the existing concentration limit regardless of sleeve math.

### 5. Pre-flight: shadow/dry-run burn-in

Since the account is being replaced (not run in parallel), there's no live
side-by-side comparison once this ships. **Correction from an earlier
assumption:** there is no existing dry-run mode to reuse — `cli/main.py`
today only has `lab`, `ingest`, `db`, and `paper` subcommands; a prior
`ggt signals` dry-run command referenced in project memory no longer exists
(likely retired in the June 15 lab-core rebuild). This needs to be built new:
a flag on `PaperTrader.run()` (or a new thin CLI entry point) that runs the
full multi-sleeve signal + sizing pipeline and Telegram-alerts the intended
trades, without calling `broker.submit_buy()`/`submit_sell()`. Run this for a
couple of weeks, comparing logged intended trades against expectations,
before switching the cron job over to submit real (paper) orders.

### 6. Pre-flight: account leverage check

One-line confirmation that the Alpaca paper account is a cash/unlevered
account (not margin-enabled), since `max_leverage=1.0` in the overlay assumes
this. Check via `AlpacaBroker`/account config before relying on it.

## What doesn't change

Order submission, fill polling/reconciliation, Telegram notification
formatting, and all halt/drawdown machinery in `trader.py` — this touches
signal generation, the overlay computation, and position sizing only.

## Testing

- Unit tests for the new sleeve-budget math (pure functions, same style as
  existing `tests/lab/test_allocation.py`).
- A test asserting Invariant 1: the trailing-curve simulation's
  `EnsembleSignal` construction uses the same params as
  `signal_runner.py`'s.
- A test asserting Invariant 2: sleeve budgets are computed from the
  portfolio value at rebalance time, not a cached/stale one.
- An integration test for the rebalance-failure fallback path (mock a failed
  OHLCV fetch on a rebalance date, assert previous weights are reused and a
  notification fires).
- An integration test mirroring `tests/paper/` structure: one
  `PaperTrader.run()` call correctly attributes buys to sleeves and sizes
  them proportionally to weight × scale.

## Deliverable

Code changes to `signal_runner.py`, `risk.py`, `trader.py` (minimal), and
`persist.py` (rebalance-state schema addition), plus new tests. Shipped
behind the existing dry-run mode first; cron only flips to real order
submission after the burn-in window and a manual go-ahead — not automatic.
