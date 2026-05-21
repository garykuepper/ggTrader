# Phase 3 architectural feedback

Candid review of the Section 5 core abstractions after actually using them to
build CashAndCarryBTC end-to-end. Phase 3's whole point is this validation;
the items below are what felt awkward and what I'd change.

**Status (2026-05-17, post Phase 3.5):** items 1–8 are addressed. Item 9 was
docs-only (added to `strategies/base.py` docstring). Item 10 was already a
✅ confirmation rather than an action item.

## 1. Signal carries no quantity, but the strategy knows its sizing  — ✅ FIXED in Phase 3.5

Spec §5.4 puts `confidence` on Signal and §5.8 says the Sizer converts Signal
→ Order with a quantity. For cash-and-carry I had to stuff `notional_usd` as
a **string into `Signal.metadata`** so the backtest engine could compute
`quantity = notional / price`. Two issues:

- Decimal-via-string-in-dict defeats the type discipline we just put on
  Signal/Order.
- A fixed-notional strategy "knowing" its size is not unreasonable; forcing
  it to pretend it doesn't and routing through a separate Sizer just to
  re-inject the same number is ceremony.

**Suggestion:** add `target_notional_usd: Optional[Decimal]` to Signal as a
first-class field. Sizers consume it (or override it). Strategies that don't
know their sizing leave it None; Sizers fill it. This stops forcing
metadata-as-typed-payload.

## 2. FeatureStore.get(name, instrument, ts) is wrong shape for pair features  — ✅ FIXED in Phase 3.5

`basis_apr` is a property of a `(spot, future)` pair, not a single instrument.
I keyed it on the future and got away with it because the spot is implied.
But cross-sectional features (`rank_30d_return_in_universe`,
`pair_zscore_spread`, `beta_to_BTC`) will all have the same shape problem.

**Suggestion:** widen the signature to
`get(name, instruments: Instrument | list[Instrument], ts)`. Single-instrument
features pass through; multi-instrument features get a tuple key. Phase 2
should land this signature; if Phase 2 implements the narrow one we'll have
to widen later and break callers.

## 3. Pricer is a missing abstraction (or it's just a feature)  — ✅ FIXED in Phase 3.5

The spec does not name a "pricer" but the backtest engine cannot function
without one. I invented `Pricer = Callable[[Instrument, datetime,
FeatureStore], Decimal]` and a `synthetic_pricer` that *cheats* by knowing
how to compute future price from spot + basis + DTE.

**Suggestion:** delete the Pricer concept. Require `FeatureStore` to serve
`mid_price` (or `last_price`) per instrument as a standard feature. Backtest
engine reads `feature_store.get("mid_price", instrument, ts)`. The synthetic
case becomes: the SyntheticFeatureStore serves a synthesized `mid_price` for
the future from its internal basis model. Cleaner and one fewer Protocol.

## 4. Strategy purity + roll state = double fees in the backtest engine  — ✅ FIXED in Phase 3.5

The Strategy is a pure function emitting *target positions*. The backtest
engine sees "future A drops out, future B enters" and treats it as
`exit-A, entry-B` — two fee events. But for a cash-and-carry **roll**, the
realistic execution is a calendar spread, often executed atomically with one
fee schedule. My current engine **double-charges fees on rolls** vs. real
execution.

**Suggestion:** the engine needs a notion of *position continuity*. Either:
(a) Signal carries an optional `replaces_position_id`, or
(b) the engine recognizes a same-strategy same-direction-class swap (long
spot→long spot, short-future-A → short-future-B) as a roll and applies a
single-trip fee. (b) is leaky abstraction; (a) is cleanest. Phase 1b /
Phase 5 (PaperBroker) need to confront this.

## 5. round_trip_fee_apr in the strategy and taker_fee_bps on the instrument are two sources of truth  — ✅ FIXED in Phase 3.5

The CashAndCarry config has `round_trip_fee_apr: 0.025` (the strategy's
mental model of total cost; used in the entry threshold). The Instrument has
`taker_fee_bps` (used by the backtest engine for actual fees). Nothing
checks they're consistent. For a 90-day contract:
4 × (26+5) bps × 365/90 ≈ 5% APR — but my config says 2.5%. Easy to drift.

**Suggestion:** derive `round_trip_fee_apr` automatically from instrument
fees + average contract duration, or surface it as a derived feature. Don't
let the user type it.

## 6. YAML config → strategy boilerplate is ~50 lines per strategy  — ✅ FIXED in Phase 3.5

`strategies/carry/config.py` is 95 lines of Pydantic schemas mirroring the
YAML structure plus a `build_strategy` factory. Every new strategy needs the
same shape. The `strategy_class` field is in the YAML but I didn't actually
use it (hard-coded the import in the factory).

**Suggestion:** generic config loader that uses `strategy_class` to import
dynamically, plus a convention that strategies expose a
`from_config(config: BaseModel)` classmethod. New strategies just define the
Pydantic config schema and the classmethod — no factory function.

## 7. Strategy ABC mixes class-level attrs and instance-level attrs  — ✅ FIXED in Phase 3.5

`required_features` and `timeframe` work nicely as class attributes;
`strategy_id` and `universe` are instance-level (set in `__init__`).
Subclasses have to remember which is which. mypy is happy but readers
won't be.

**Suggestion:** make all four instance-level. Class attribute defaults are
unnecessary cleverness.

## 8. TradingCalendar dormant for crypto, but won't be for equities  — ✅ WIRED in Phase 3.5

The calendar Protocol is defined but unused in Phase 3 because crypto is
24/7. Equity strategies (Phase 4) will require: don't generate signals on
non-trading days; the backtest engine should respect `calendar.is_open(ts)`.
Flag this so we don't bolt it on awkwardly later.

**Suggestion:** the backtest engine should accept an optional
`TradingCalendar` and skip iterations where the calendar is closed. Pass it
from `Strategy.universe[0].calendar_id` resolved through a registry.

## 9. Universe.members(ts) returns a list, but the strategy iterates implicitly  — ✅ DOCUMENTED in Phase 3.5

CarryUniverse.members returns `[spot, active_future]`. The strategy does
**not** iterate this list — it asks for `universe.active_future(ts)` directly.
So the Universe abstraction is half-used. For cross-sectional strategies
(Phase 3.5+) where the universe IS the rebalancing set, `members(ts)` will
be load-bearing. For carry strategies, it's vestigial.

**Suggestion:** that's fine. Different strategies use different parts of the
Universe API. But document the two patterns (iterate-all vs. pick-one) so
future strategies don't get confused which to follow.

## 10. The seam where synthetic data plugs out is clean

This is the **good** news. `features/derivatives_synthetic.py` is one file,
imported only by the CLI command and the integration test. Replacing it with
a real `features/derivatives.py` + a TimescaleDB-backed FeatureStore changes
exactly the wiring in `cmd_backtest_strategy.py`. The strategy, the Universe,
the YAML, the backtest engine — none of those change. That's the
architectural validation Phase 3 was supposed to prove, and it does.

---

## Net assessment

The core abstractions hold. The four real issues to address before Phase
3.5+ are:

1. Signal needs `target_notional_usd` (don't stuff it in metadata)
2. FeatureStore needs multi-instrument feature key
3. Drop the Pricer concept; require `mid_price` as a standard feature
4. Backtest engine needs position-continuity for roll fee accounting

Items 5–9 are smaller; address them as they bite.
