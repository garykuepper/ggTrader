"""CashAndCarryBTC — long spot BTC + short equivalent-notional dated future.

Captures the annualized basis premium when futures trade at a contango premium
to spot. Position is hedged: a parallel move in BTC cancels between the two
legs, so P&L comes from the convergence of future → spot at expiry plus any
favorable change in basis.

Entry  : ``basis_apr(spot, active_future) > threshold_apr + round_trip_fee_apr``
Exit   : basis collapses below the same threshold, OR the active contract has
         rolled (handled by ``CarryUniverse.active_future`` advancing within
         ``roll_buffer_hours`` of expiry; the backtest engine then sees the
         old contract drop out of the target set as a roll).

Phase 3.5 changes:
- Uses ``Signal.target_notional_usd`` (typed) instead of metadata-string.
- Uses ``feature_store.get_at("basis_apr", [spot, future], ts)`` (pair feature).
- ``round_trip_fee_apr`` derived from instrument fees, not the YAML.

Margin / capital model — assumes Kraken Spot and Kraken Futures as independent
sub-accounts (no cross-margining). Documented and accepted; not modeled.

Hard constraints: imports only ``core/*``, ``strategies.base``, and the
``features.base`` Protocol. No ccxt. No broker imports. No pandas-ta.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from ggTrader.core.instrument import AssetClass, Instrument
from ggTrader.core.signal import Direction, Signal
from ggTrader.core.universe import Universe
from ggTrader.features.base import FeatureStore, pair_column_label
from ggTrader.strategies.base import Strategy


def _parse_iso_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def estimate_round_trip_fee_apr(
    spot: Instrument,
    future: Instrument,
    avg_contract_days: float = 90.0,
) -> Decimal:
    """Approximate amortized round-trip fees as an APR.

    Round trip = open spot + open future + close spot + close future.
    Each leg pays taker fees on entry and exit. Amortize across a typical
    contract duration to convert to APR.
    """
    bps_per_round_trip = (spot.taker_fee_bps + future.taker_fee_bps) * 2
    bps_decimal = bps_per_round_trip / Decimal("10000")
    annualization = Decimal("365") / Decimal(str(avg_contract_days))
    return bps_decimal * annualization


class CarryUniverse(Universe):
    """Dynamic universe: at each ts, returns [spot, active_future]."""

    def __init__(
        self,
        spot: Instrument,
        futures: list[Instrument],
        roll_buffer_hours: int = 24,
    ) -> None:
        if spot.asset_class is not AssetClass.CRYPTO_SPOT:
            raise ValueError("spot leg must be CRYPTO_SPOT")
        for f in futures:
            if f.asset_class is not AssetClass.CRYPTO_DATED_FUTURE:
                raise ValueError(f"{f.symbol} must be CRYPTO_DATED_FUTURE")
            if f.expiry is None:
                raise ValueError(f"{f.symbol} missing expiry")
        self._spot = spot
        self._futures = sorted(futures, key=lambda f: _parse_iso_utc(str(f.expiry)))
        self._roll_buffer = timedelta(hours=roll_buffer_hours)

    def is_dynamic(self) -> bool:
        return True

    def members(self, ts: datetime) -> list[Instrument]:
        active = self.active_future(ts)
        return [self._spot, active] if active is not None else [self._spot]

    def active_future(self, ts: datetime) -> Optional[Instrument]:
        threshold = ts + self._roll_buffer
        for f in self._futures:
            if _parse_iso_utc(str(f.expiry)) > threshold:
                return f
        return None

    @property
    def spot(self) -> Instrument:
        return self._spot

    @property
    def futures(self) -> list[Instrument]:
        return list(self._futures)


class CashAndCarryBTC(Strategy):
    """Cash-and-carry on BTC: long spot + short dated future."""

    def __init__(
        self,
        universe: CarryUniverse,
        threshold_apr: Decimal,
        notional_usd: Decimal,
        strategy_id: str = "cash_and_carry_btc",
        round_trip_fee_apr: Optional[Decimal] = None,
    ) -> None:
        if threshold_apr <= 0:
            raise ValueError("threshold_apr must be positive")
        if notional_usd <= 0:
            raise ValueError("notional_usd must be positive")
        super().__init__(
            strategy_id=strategy_id,
            universe=universe,
            required_features=["basis_apr", "mid_price"],
            timeframe="1d",
        )
        if round_trip_fee_apr is None:
            round_trip_fee_apr = estimate_round_trip_fee_apr(
                spot=universe.spot,
                future=universe.futures[0],
            )
        self.threshold_apr = threshold_apr
        self.notional_usd = notional_usd
        self.round_trip_fee_apr = round_trip_fee_apr

    @property
    def carry_universe(self) -> CarryUniverse:
        u = self.universe
        assert isinstance(u, CarryUniverse)
        return u

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "CashAndCarryBTC":
        """Build a CashAndCarryBTC from a YAML-derived dict.

        Allows the generic ``ggTrader.strategies.loader`` to instantiate this
        strategy without per-strategy factory boilerplate. Coerces YAML
        primitives into the Pydantic-strict types Instrument requires.
        """
        from ggTrader.core.instrument import AssetClass, Instrument, Venue

        def _to_instrument(spec: dict[str, Any]) -> Instrument:
            coerced = {
                "symbol": spec["symbol"],
                "asset_class": AssetClass(spec["asset_class"]),
                "venue": Venue(spec["venue"]),
                "base_currency": spec["base_currency"],
                "quote_currency": spec["quote_currency"],
                "tick_size": Decimal(str(spec["tick_size"])),
                "min_order_size": Decimal(str(spec["min_order_size"])),
                "maker_fee_bps": Decimal(str(spec["maker_fee_bps"])),
                "taker_fee_bps": Decimal(str(spec["taker_fee_bps"])),
                "calendar_id": spec["calendar_id"],
            }
            if "contract_multiplier" in spec:
                coerced["contract_multiplier"] = Decimal(str(spec["contract_multiplier"]))
            if spec.get("expiry"):
                coerced["expiry"] = spec["expiry"]
            if spec.get("venue_specific_id"):
                coerced["venue_specific_id"] = spec["venue_specific_id"]
            return Instrument(**coerced)

        spot = _to_instrument(raw["spot"])
        defaults = raw["futures_defaults"]
        futures = [_to_instrument({**defaults, **f}) for f in raw["futures"]]
        universe = CarryUniverse(
            spot=spot,
            futures=futures,
            roll_buffer_hours=int(raw.get("roll_buffer_hours", 24)),
        )
        rt_raw = raw.get("round_trip_fee_apr")
        rt = Decimal(str(rt_raw)) if rt_raw is not None else None
        return cls(
            universe=universe,
            threshold_apr=Decimal(str(raw["threshold_apr"])),
            notional_usd=Decimal(str(raw["notional_usd"])),
            strategy_id=str(raw.get("strategy_id", "cash_and_carry_btc")),
            round_trip_fee_apr=rt,
        )

    def generate_signals(self, ts: datetime, feature_store: FeatureStore) -> list[Signal]:
        universe = self.carry_universe
        active = universe.active_future(ts)
        if active is None:
            return []

        pair = [universe.spot, active]
        try:
            row = feature_store.get_at("basis_apr", pair, ts)
        except LookupError:
            return []

        basis_apr = Decimal(str(row[pair_column_label(pair)]))
        net_threshold = self.threshold_apr + self.round_trip_fee_apr
        if basis_apr <= net_threshold:
            return []

        edge_apr = basis_apr - self.round_trip_fee_apr
        metadata = {
            "basis_apr": str(basis_apr),
            "edge_apr_net_of_fees": str(edge_apr),
            "active_future_expiry": str(active.expiry),
        }

        return [
            Signal(
                ts=ts,
                instrument=universe.spot,
                direction=Direction.LONG,
                confidence=Decimal("1.0"),
                strategy_id=self.strategy_id,
                target_notional_usd=self.notional_usd,
                metadata=metadata,
            ),
            Signal(
                ts=ts,
                instrument=active,
                direction=Direction.SHORT,
                confidence=Decimal("1.0"),
                strategy_id=self.strategy_id,
                target_notional_usd=self.notional_usd,
                metadata=metadata,
            ),
        ]
