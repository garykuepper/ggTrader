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

**Signal (breadth).** For a given universe and `asof` date, run the
*unmodified* `EnsembleSignal(LabConfig())` — same construction as the
deployed core strategy, no separate params — against that universe's
point-in-time eligible constituents. Compute
`breadth = (# stocks with an active buy signal) / (# eligible stocks)`.

**Rotation rule.** Three states, one held at a time:
- `breadth > upper_threshold` → 100% long the universe's leveraged-long ETF
- `breadth < lower_threshold` → 100% long the universe's inverse ETF
- otherwise → cash

**Hysteresis / minimum hold.** A `min_hold_days` parameter prevents
re-evaluating the rotation decision more often than every N trading days,
guarding against whipsaw (rapid flips are especially costly here because
leveraged ETFs compound daily-rebalancing decay on top of any trading
losses). Swept, not fixed.

**Leverage tier.** `{2x, 3x}` swept per universe, using that tier's real ETF
pair from the table above — not a synthetic leveraged-multiple of the
underlying index, since the whole point is to capture actual decay behavior.

## Architecture

New `LeveragedRotationStrategy` in
`src/ggTrader/lab/strategies/leveraged_rotation.py`, `target_kind="weights"`
(same protocol `xs_momentum`/`idio_vol` already use — a full weight on one
ETF, zero on the other, simulated via the existing `Portfolio.from_orders`
path; no new `target_kind` needed).

**Data flow:** the OHLCV frame passed to `select()` combines the universe's
constituent stocks (for breadth) *and* that universe's ETF pair, both tiers
(4 tickers) in one combined fetch. `universe_fn` for this strategy returns a
fixed `eligible = [long_ticker, inverse_ticker]` for the active leverage
tier — it does not vary by `asof` the way stock-universe membership does.
`select()` computes breadth from the constituent columns of `data` (present
in the frame but outside `eligible`) and returns a plan targeting one of the
two ETFs, or neither (cash). This requires zero changes to `wfo.py` or the
`Strategy` protocol — it's a normal weight-based strategy from the harness's
point of view, just with a data frame that mixes a large breadth-source
universe and a tiny 2-ticker tradeable universe.

**Registration:** added to `STRATEGY_REGISTRY` like every other strategy.
`sweep_params()` returns the grid: `upper_threshold`, `lower_threshold`,
`min_hold_days`, `leverage_tier`.

**Execution:** run the existing `--wfo` CLI once per universe
(`--strategy leveraged_rotation --universe sp500`, `--universe nasdaq100`,
`--universe russell2000`), each gate-honest (NDH + DSR, same gates every
other strategy in this project uses) against its own SPY-benchmarked window
starting 2010 (per your call — this doesn't need to match the SP500 core's
exact eval window; it's a new baseline in its own right, apples-to-apples
against SPY on the same dates within each universe's own run).

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

- `src/ggTrader/lab/strategies/leveraged_rotation.py` + tests, registered in
  `STRATEGY_REGISTRY`.
- One research report (following `docs/research/TEMPLATE`'s standard
  6-section structure) covering all three universes and both leverage
  tiers, with a GO/NO-GO verdict per universe and a combined verdict.
- A `roadmap.md` entry recording the result, win or lose — this project's
  established practice of documenting rejected hypotheses alongside
  accepted ones.
- No live trading changes. If a universe clears WFO, live wiring is a
  separate follow-up plan.
