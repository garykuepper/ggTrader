# Leveraged/Inverse Index Rotation — Research Design

## Context

Every strategy shipped so far in this codebase picks individual stocks within
an equity universe (5-voter ensemble reversion, `xs_momentum`, `idio_vol`,
`ensemble_ic`, `overnight_gap`, ...). This is a structurally different
proposal: a market-timing strategy that trades a single leveraged or inverse
index ETF at a time — long a leveraged-long ETF, long an inverse ETF, or cash
— never picking among individual names.

The signal driving the rotation is a *repurposing* of the existing, validated
5-voter `EnsembleSignal`: instead of asking "which stocks look attractive
today," it asks "what fraction of the universe looks attractive today"
(breadth), and uses that fraction to time the index itself. That signal was
validated for stock selection, not index timing — this research answers
whether breadth carries any index-timing information at all. Given eight of
nine prior equity-side hypotheses in this project's history were rejected in
honest walk-forward, the realistic prior here is a coin flip, not a strong
expectation of success. It's worth testing precisely because the instrument
class (leveraged/inverse ETFs) and the signal's *job* (timing vs. picking)
are both genuinely new, not incremental tweaks to what's already been tried.

## Goal

For each of three universes (SP500, Nasdaq-100, Russell 2000), build and
gate-honestly walk-forward test a breadth-driven rotation strategy across
leveraged-long / inverse / cash, at both 2x and 3x leverage tiers, using real
historical ETF prices (so real daily-rebalancing volatility decay is
captured, not a synthetic multiplied return series). Produce a GO/NO-GO
verdict per universe plus a combined verdict, written up as a research report
and a roadmap.md entry.

## Non-goals

- **MidCap400**: excluded. Its 3x inverse ETF (MIDZ) was delisted in April
  2020 (data confirmed: 2,600 rows ending 2020-04-30 vs. ~4,155 for a full
  2010+ history), and no clean 2x mid-cap inverse alternative was found. Not
  worth testing with a broken/partial inverse leg.
- **Dow-30**: excluded. Real leveraged/inverse ETF data exists (UDOW/SDOW,
  DDM/DXD) but this lab framework has no existing point-in-time Dow-30
  constituent infrastructure (`UNIVERSE_CHOICES` is
  `sp500`/`nasdaq100`/`russell2000`/`midcap400` only) — adding Dow means
  building new constituent-list infra, out of scope for this pass.
- **Live paper trading wiring**: this plan is research only. Every other
  strategy in this codebase reached paper trading only after clearing a
  gate-honest WFO first; this follows the same order. A second Alpaca paper
  account has already been provisioned for a fast-follow live-wiring plan if
  and when this clears WFO — not part of this plan.
- **Simultaneous long+inverse hedge / volatility-harvesting decay-arbitrage
  mechanisms**: considered and explicitly not chosen — this is a
  single-position-at-a-time rotation strategy only.
- **Parameter sweeps beyond breadth thresholds, hysteresis, and leverage
  tier**: no exotic feature engineering (macro overlays, options-implied
  vol, etc.) in this pass.

## Universe → ETF pair mapping

Real price history confirmed for all six tickers below, full-history back to
2010 (TQQQ/SQQQ/UDOW/SDOW start Feb 2010; the rest start Jan 2010):

| Universe | 3x pair (long / inverse) | 2x pair (long / inverse) |
|---|---|---|
| sp500 | UPRO / SPXU | SSO / SDS |
| nasdaq100 | TQQQ / SQQQ | QLD / QID |
| russell2000 | TNA / TZA | UWM / TWM |

## Mechanism

**Rebalance cadence: monthly, not daily.** The existing weight-strategy WFO
harness (`rebalance_dates()` in `wfo.py`, shared with `idio_vol` and every
other weight-based strategy) only re-evaluates positions on the last trading
day of each month — hardcoded, not a parameter. Reusing it as-is (per this
spec's "no framework changes" scope) means this is a **monthly
regime-timing strategy**, not a fast tactical rotation: breadth is checked
once a month, and whatever state that implies is held for the following
month. This is treated as a feature, not just a constraint — frequent flips
are exactly what causes the worst decay on leveraged ETFs, so a monthly
cadence is arguably the lower-whipsaw-risk design anyway. Extending the
harness to daily cadence was considered and explicitly deferred as separate,
larger-scoped work that would touch shared code other strategies depend on.

**Signal (breadth).** As of each month-end rebalance date, run the
*unmodified* `EnsembleSignal(LabConfig())` — same construction as the
deployed core strategy, no separate params — against that universe's
point-in-time eligible constituents, vectorized across the strategy's full
data window in one pass (mirroring how `EnsembleSignal.to_targets()` already
computes its own entries/exits — no per-day loop needed). Compute
`breadth = (# stocks with an active buy signal that day) / (# stocks in the
breadth universe)`, using a fixed denominator (the full breadth-universe
size, not the count of stocks currently past warmup) — a deliberate
simplification that slightly understates breadth in the first ~1-2 years of
the window while few symbols have enough trailing history; each WFO fold's
own training window already begins well after this warmup period, so it
doesn't affect the eval folds.

**Rotation rule.** Three states, one held at a time, decided at each
monthly rebalance from that date's breadth value:
- `breadth > upper_threshold` → 100% long the universe's leveraged-long ETF
- `breadth < lower_threshold` → 100% long the universe's inverse ETF
- otherwise → cash

**Hysteresis / minimum hold.** A `min_hold_months` parameter requires N
consecutive monthly signals in the same direction before the position
actually flips, guarding against whipsaw between adjacent regime states.
Swept, not fixed.

**Leverage tier.** `{2x, 3x}` swept per universe, using that tier's real ETF
pair from the table above — not a synthetic leveraged-multiple of the
underlying index, since the whole point is to capture actual decay behavior.

## Architecture

A shared `_LeveragedRotationBase` in
`src/ggTrader/lab/strategies/leveraged_rotation.py` holding all breadth +
rotation + hysteresis logic, plus one thin subclass per universe
(`LeveragedRotationSp500`, `LeveragedRotationNasdaq100`,
`LeveragedRotationRussell2000`), each fixing its own class-level ETF-pair
constants (`PAIR_3X`, `PAIR_2X`). `target_kind = "weights"` (same protocol
`xs_momentum`/`idio_vol` already use — a full weight on one ETF, zero on the
other, simulated via the existing `Portfolio.from_orders` path; no new
`target_kind` needed).

Three subclasses rather than one universe-parametrized class because
`wfo.py` constructs strategy instances as `strategy_cls(cfg)` (bare, no
extra args — used for anchor-set computation) and `strategy_cls(combo_cfg,
**extra_kwargs)` (swept params only) in multiple places. Any constructor
argument that isn't part of `sweep_params()` has to come from a class
default, not a required parameter — so the universe binding lives on the
class, not the instance.

**Data flow:** the OHLCV frame passed into `run_wfo()` combines the
universe's constituent stocks (for breadth) *and* **all four** ETF tickers
for that universe (both leverage tiers), in one combined fetch. `universe_fn`
returns that fixed 4-ticker list regardless of `asof`/`past` — it can't be
leverage-tier-aware, because `universe_fn(asof, past)` is called once per
`(asof, combo)` pair without seeing which combo is active. Instead,
`select()` reads `self.leverage_tier` (its own swept param) to pick 2 of the
4 eligible tickers to actually target, and computes breadth from the *other*
columns present in `data` — the breadth-universe stocks, which are in the
combined frame but outside `eligible`. This requires zero changes to
`wfo.py` or the `Strategy` protocol.

**Registration:** all three subclasses added to `STRATEGY_REGISTRY`.
`sweep_params()` (shared via the base class) returns the grid:
`upper_threshold`, `lower_threshold`, `min_hold_months`, `leverage_tier`.

**Execution:** the generic `ggt lab --wfo` CLI path always derives `eligible`
from real point-in-time index membership (`equity_universe_between` +
`eligible_at`) and has no hook to override it — only `blend.py`'s
`run_blend()` threads a custom `universe_fn` into `run_wfo()`. Since this
strategy needs the fixed 4-ticker `universe_fn` described above, it gets its
own small orchestration script, `scripts/leveraged_rotation_research.py`
(same shape as `blend.py`'s `run_blend()`, not a new CLI mode): for each of
the 3 universes, loads the combined OHLCV (constituents + 4 ETF tickers),
calls `run_wfo()` directly with the fixed `universe_fn` and that universe's
subclass, gate-honest (NDH + DSR, same gates every other strategy in this
project uses) against its own SPY-benchmarked window starting 2010 (per your
call — this doesn't need to match the SP500 core's exact eval window; it's a
new baseline in its own right, apples-to-apples against SPY on the same
dates within each universe's own run).

## Verification

- Unit tests for `LeveragedRotationStrategy.select()`/`to_targets()`
  (breadth computation, threshold/hysteresis logic, correct ETF-pair
  selection per universe/tier) following this project's existing strategy
  test conventions (see `tests/lab/strategies/` for the pattern other
  strategies use).
- Full `pytest` suite green before any WFO run.
- Each of the 3×2 = 6 (universe × leverage tier) WFO runs must show its gate
  pass rate and per-fold parameter stability (the same "winner selected in
  only 3/17 folds = noise" tell that closed eight prior hypotheses in this
  project) — a result isn't reported as a GO unless the winning params are
  stable across a majority of folds, not just a headline Sharpe number.

## Deliverable

- `src/ggTrader/lab/strategies/leveraged_rotation.py` (base class + 3
  per-universe subclasses) + tests, registered in `STRATEGY_REGISTRY`.
- `scripts/leveraged_rotation_research.py` orchestration script (loads
  combined OHLCV per universe, runs `run_wfo()` with the fixed `universe_fn`,
  prints/persists a summary table across all 3 universes × 2 leverage tiers).
- One research report (following `docs/research/TEMPLATE`'s standard
  6-section structure) covering all three universes and both leverage
  tiers, with a GO/NO-GO verdict per universe and a combined verdict.
- A `roadmap.md` entry recording the result, win or lose — this project's
  established practice of documenting rejected hypotheses alongside
  accepted ones.
- No live trading changes. If a universe clears WFO, live wiring is a
  separate follow-up plan.
