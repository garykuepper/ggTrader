# Kelly-Criterion Position Sizing (Equity Core)

**Status:** Design approved, not yet implemented.
**Date:** 2026-07-06

## Context

The equity selection book closed on 2026-06-28: entry-level ML/feature gating,
exit-rule sweeps, and IC-weighted voting were all tested in honest walk-forward
and rejected (see `docs/roadmap.md`). The deployed baseline is the 5-voter
ensemble on the SP500 core universe, flat 3% position size per trade
(1.12 Sharpe / 16.3% CAGR / -11% DD, OOS).

The only untested lever remaining on the equity side (per the roadmap's
"Future Research" row) is position sizing: Kelly-criterion or regime-based.
This spec covers **Kelly-criterion sizing** — regime-corrected exposure
scaling was considered and parked as a separate, later experiment (see
Alternatives Considered).

Current sizing implementations (flat 3%, kept in sync manually):
- Live/paper trader: `src/ggTrader/paper/risk.py` — `RiskConfig.position_pct = 0.033`,
  applied in `position_notional()`. Independent guardrails: `max_positions=30`,
  `max_concentration_pct=0.05`, `daily_loss_pct=0.03`, `max_drawdown_pct=0.15`.
- Research/lab: `src/ggTrader/lab/data.py` — `STOCK_BASE_CONFIG["SIGNAL_POSITION_SIZE"] = 0.03`,
  applied per-bar in `simulate.py` via `size_type="percent"`. A `scalar`
  multiplier hook already exists there (`base_size * scalar`), currently used
  for a vol-target scalar in an unrelated multi-sleeve experiment — this is
  the intended reuse point for the Kelly scalar.

No Kelly-criterion code exists anywhere in the repo today (`grep -rn kelly -i`
returns zero hits) — this is net-new.

## Goal

Test whether sizing trades by an estimated historical edge (Kelly criterion)
beats the flat-3% baseline on the SP500 core reversion strategy, in honest
walk-forward, using the same statistical bar that rejected the prior three
equity-side experiments.

## Mechanism

**Kelly fraction:** `f* = W - (1-W)/R`, where:
- `W` = win rate across closed trades
- `R` = average-win / average-loss ratio (payoff ratio)

**Pooling:** `f*` is computed once, pooled across *all* historical closed
trades of the ensemble strategy — not per-symbol. Per-symbol estimation was
considered and rejected: each symbol has far fewer trades, and the recent
IC-weighted-voting NO-GO (`ensemble_ic`, rejected 2026-06-28) failed for
exactly this reason (winning weights were noise, chosen in only 3/17 folds).
Pooling avoids repeating that mistake.

**Estimation window:** expanding, not rolling. At each walk-forward fold,
`f*` is recomputed from every trade closed strictly before that fold's
training cutoff. This matches how the existing WFO folds already work — no
new look-ahead risk — and avoids introducing a new tunable window-size
parameter that would add unnecessary sweep surface.

**Position sizing formula:**

```
position_size = min(k * f* * portfolio_value, max_concentration_pct * portfolio_value)
```

- `k` is the Kelly multiplier (swept, see below).
- `max_concentration_pct` reuses the existing live risk-guard ceiling (5%,
  from `paper/risk.py`) as a hard cap — this experiment can never size more
  aggressively than the live system already permits elsewhere.
- **Fallback:** if `f* <= 0` (no measurable historical edge) at any fold, size
  at the flat-3% baseline for that fold rather than sizing to zero or
  negative. Rationale: a strategy that goes dark on a temporary bad patch is
  worse for evaluation than one that reverts to the known-good baseline size.

**Sweep:** `k ∈ {0.25, 0.5, 1.0}` (quarter-, half-, and full-Kelly) as a
3-combo WFO grid over the same 17 SP500 folds used to validate the deployed
baseline. Full Kelly is known to be too aggressive in practice given
estimation error in `W`/`R`; sweeping lets the existing gates (not a
preselected guess) determine which fraction survives.

## Implementation shape

Follows the `ensemble_ic` precedent (`src/ggTrader/lab/strategies/ensemble_ic.py`):

1. **`src/ggTrader/lab/kelly.py`** — pure function(s) computing the expanding-window
   Kelly fraction from a trade-history series. Unit-testable in isolation:
   - zero trades (no history yet) → fallback behavior
   - negative edge (`f* <= 0`) → fallback to baseline sizing
   - zero-loss trades (division-by-zero guard on `R`)
2. **New strategy variant** registered in `STRATEGY_REGISTRY`
   (`src/ggTrader/lab/strategies/__init__.py`), e.g. `ensemble_kelly` — reuses
   the existing ensemble signal logic, only the sizing scalar differs. Wires
   into the existing `scalar` multiplier hook in `simulate.py` rather than
   duplicating ensemble entry/exit logic.
3. **Run path:** `ggt lab --strategy ensemble_kelly@sp500 --wfo`, using the
   existing sweep/WFO/gate/persistence pipeline (`cli.py`, `wfo.py`,
   `gates.py`, `persist.py`) unchanged. Same NDH (Neighborhood Density
   Hurdle) and DSR (Deflated Sharpe Ratio) gates as every prior experiment.
   Persisted to `lab_runs`/`lab_summary` like every prior experiment.

## Verdict bar

Same standard applied to the last three rejected experiments. This is a GO
only if **all** of:
- OOS Sharpe clears the 1.12 baseline
- Drawdown no worse than the baseline's -11%
- The winning `k` is stable across a *majority* of the 17 folds — not a
  fold-count fluke like `ensemble_ic`'s 3/17

Any other outcome is a NO-GO, documented in the roadmap the same way as the
prior three.

## Testing

TDD, per repo convention:
1. Unit tests for `kelly.py` first — zero-trades, negative-edge, and
   zero-loss-division edge cases before any WFO wiring.
2. Integration test asserting the expanding-window estimate is causal: fold
   N's `f*` must only reflect trades closed before fold N's training cutoff
   (mirrors the existing look-ahead tests for WFO folds).
3. Full test suite (currently 305 lab tests) must stay green.

## Alternatives considered

- **Regime-corrected exposure scaling** — reusing `src/ggTrader/lab/regime.py`
  with the scalar direction flipped (up in down/turbulent regimes, down in
  calm uptrends, per the June 24 finding that this reversion strategy's edge
  is inverted from naive trend-following intuition). Parked as a separate,
  later experiment rather than combined here, to keep this experiment's
  variable isolated and its result attributable to Kelly sizing alone.
- **Combined Kelly × regime scalar** — rejected for the same reason: conflates
  two untested mechanisms, making a result (positive or negative) hard to
  attribute to either one.
- **Per-symbol Kelly estimation** — rejected due to small-sample noise risk,
  as explained above.

## Out of scope

- Wiring this into the live/paper trader (`paper/risk.py`) — this spec covers
  the research/lab experiment only. Live deployment is a separate decision
  gated on a GO verdict here, per the project's "only invest real capital in
  systems that survive strict out-of-sample tests" policy.
- The regime-corrected exposure scaling experiment (see Alternatives).
