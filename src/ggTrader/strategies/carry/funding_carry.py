"""FundingCarryBTC — long spot + short linear perp, harvest funding.

Phase 4. The "Kraken-tradeable cash-and-carry equivalent": Kraken doesn't
list linear dated quarterlies, so the natural carry trade is funding-rate
arbitrage on PF_XBTUSD. When funding APR is sufficiently positive (longs
pay shorts), short the perp + long the spot in equal notional. The hedge
neutralizes BTC price exposure; P&L comes from collected funding.

Entry / exit are **hysteretic** (different thresholds going in vs out) so
the strategy doesn't churn around a single boundary. The signal source is
configurable so the same code runs against either real funding rates
(``signal_source: funding``) or the spot/perp basis proxy
(``signal_source: basis_proxy``). Phase 4 needs both for the side-by-side
comparison.

Universe: ``Pair(BTC-USD spot, PF_XBTUSD perp)``. No rolls (perp doesn't
expire). The optional ``negative_funding_exit_consecutive`` parameter
forces exit when funding flips negative for N consecutive bars, even if
the smoothed signal still reads above the exit threshold.

Hard constraints: imports only ``core/*``, ``features.base``,
``strategies.base``. No ccxt, no broker, no pandas-ta. <200 lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Optional

from ggTrader.core.instrument import AssetClass, Instrument
from ggTrader.core.signal import Direction, Signal
from ggTrader.core.universe import Universe
from ggTrader.features.base import FeatureStore, pair_column_label
from ggTrader.strategies.base import Strategy

SignalSource = Literal["funding", "basis_proxy"]


@dataclass(frozen=True)
class _SignalSpec:
    feature_name: str
    arity: Literal["single", "pair"]


_SOURCES: dict[SignalSource, _SignalSpec] = {
    "funding": _SignalSpec(feature_name="funding_apr_30d", arity="single"),
    "basis_proxy": _SignalSpec(feature_name="basis_premium_apr_30d", arity="pair"),
}


class SpotPerpUniverse(Universe):
    """Static pair: [spot, perp]. No roll mechanics; perp never expires."""

    def __init__(self, spot: Instrument, perp: Instrument) -> None:
        if spot.asset_class is not AssetClass.CRYPTO_SPOT:
            raise ValueError("spot leg must be CRYPTO_SPOT")
        if perp.asset_class is not AssetClass.CRYPTO_PERP:
            raise ValueError("perp leg must be CRYPTO_PERP")
        self._spot = spot
        self._perp = perp

    def is_dynamic(self) -> bool:
        return False

    def members(self, ts: datetime) -> list[Instrument]:
        return [self._spot, self._perp]

    @property
    def spot(self) -> Instrument:
        return self._spot

    @property
    def perp(self) -> Instrument:
        return self._perp


class FundingCarryBTC(Strategy):
    """Long spot + short perp when funding/basis is favorably positive."""

    def __init__(
        self,
        universe: SpotPerpUniverse,
        entry_threshold_apr: Decimal,
        exit_threshold_apr: Decimal,
        notional_usd: Decimal,
        signal_source: SignalSource = "funding",
        negative_funding_exit_consecutive: int = 0,
        strategy_id: str = "funding_carry_btc",
    ) -> None:
        if entry_threshold_apr <= exit_threshold_apr:
            raise ValueError("entry_threshold_apr must exceed exit_threshold_apr (hysteresis)")
        if notional_usd <= 0:
            raise ValueError("notional_usd must be positive")
        if signal_source not in _SOURCES:
            raise ValueError(f"signal_source must be one of {list(_SOURCES)}; got {signal_source}")
        spec = _SOURCES[signal_source]
        super().__init__(
            strategy_id=strategy_id,
            universe=universe,
            required_features=[spec.feature_name],
            timeframe="1h",
        )
        self.entry_threshold_apr = entry_threshold_apr
        self.exit_threshold_apr = exit_threshold_apr
        self.notional_usd = notional_usd
        self.signal_source = signal_source
        self._spec = spec
        self.negative_funding_exit_consecutive = negative_funding_exit_consecutive
        self._consecutive_negative: int = 0
        self._in_position: bool = False

    @property
    def pair_universe(self) -> SpotPerpUniverse:
        u = self.universe
        assert isinstance(u, SpotPerpUniverse)
        return u

    # ---- signal --------------------------------------------------------------

    def _read_signal(self, ts: datetime, feature_store: FeatureStore) -> Optional[Decimal]:
        u = self.pair_universe
        try:
            if self._spec.arity == "single":
                row = feature_store.get_at(self._spec.feature_name, u.perp, ts)
                value_raw = row[u.perp.symbol]
            else:
                row = feature_store.get_at(self._spec.feature_name, [u.spot, u.perp], ts)
                value_raw = row[pair_column_label([u.spot, u.perp])]
        except LookupError:
            return None
        if value_raw is None:
            return None
        try:
            return Decimal(str(value_raw))
        except Exception:  # noqa: BLE001
            return None

    # ---- core ----------------------------------------------------------------

    def generate_signals(self, ts: datetime, feature_store: FeatureStore) -> list[Signal]:
        signal = self._read_signal(ts, feature_store)
        if signal is None:
            return self._emit(ts, hold=self._in_position)

        if signal < 0:
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0

        if not self._in_position:
            if signal > self.entry_threshold_apr:
                self._in_position = True
                return self._emit(ts, hold=True, signal=signal)
            return self._emit(ts, hold=False)

        # Currently in position; decide whether to exit
        forced_negative_exit = (
            self.negative_funding_exit_consecutive > 0
            and self._consecutive_negative >= self.negative_funding_exit_consecutive
        )
        if signal < self.exit_threshold_apr or forced_negative_exit:
            self._in_position = False
            return self._emit(ts, hold=False)
        return self._emit(ts, hold=True, signal=signal)

    # ---- helpers -------------------------------------------------------------

    def _emit(
        self,
        ts: datetime,
        hold: bool,
        signal: Optional[Decimal] = None,
    ) -> list[Signal]:
        if not hold:
            return []
        u = self.pair_universe
        metadata: dict[str, Any] = {"signal_source": self.signal_source}
        if signal is not None:
            metadata["signal_apr"] = str(signal)
        return [
            Signal(
                ts=ts,
                instrument=u.spot,
                direction=Direction.LONG,
                confidence=Decimal("1.0"),
                strategy_id=self.strategy_id,
                target_notional_usd=self.notional_usd,
                metadata=metadata,
            ),
            Signal(
                ts=ts,
                instrument=u.perp,
                direction=Direction.SHORT,
                confidence=Decimal("1.0"),
                strategy_id=self.strategy_id,
                target_notional_usd=self.notional_usd,
                metadata=metadata,
            ),
        ]

    # ---- carry accounting -------------------------------------------------

    def position_carry(
        self,
        instrument: Instrument,
        direction: Direction,
        notional: Decimal,
        ts: datetime,
        feature_store: FeatureStore,
    ) -> Decimal:
        """Per-bar funding accrual hook for the backtest engine.

        Daily bar approximation: credit notional × (signal_apr / 365). Real funding
        accrues hourly; integrating 24 hourly samples is the right thing for hourly
        bars but the engine is currently daily — the daily-average APR ÷ 365 lines
        up with the integral over 24h to first order. Refine when the engine gets
        intra-day bars.

        Sign convention (Kraken): positive funding APR ⇒ longs pay, shorts receive.
        So a short perp position with positive funding gets a positive carry.
        """
        u = self.pair_universe
        # Carry applies only to the perp leg (spot doesn't pay/receive funding).
        if instrument.symbol != u.perp.symbol:
            return Decimal("0")
        try:
            if self._spec.arity == "single":
                row = feature_store.get_at(self._spec.feature_name, u.perp, ts)
                value_raw = row[u.perp.symbol]
            else:
                row = feature_store.get_at(self._spec.feature_name, [u.spot, u.perp], ts)
                value_raw = row[pair_column_label([u.spot, u.perp])]
        except LookupError:
            return Decimal("0")
        if value_raw is None:
            return Decimal("0")
        signal = Decimal(str(value_raw))
        daily_carry = notional * signal / Decimal("365")
        return daily_carry if direction is Direction.SHORT else -daily_carry

    # ---- YAML -------------------------------------------------------------

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "FundingCarryBTC":
        from ggTrader.core.instrument import AssetClass as AC
        from ggTrader.core.instrument import Instrument as I
        from ggTrader.core.instrument import Venue as V

        def _to_inst(spec: dict[str, Any]) -> I:
            coerced = {
                "symbol": spec["symbol"],
                "asset_class": AC(spec["asset_class"]),
                "venue": V(spec["venue"]),
                "base_currency": spec["base_currency"],
                "quote_currency": spec["quote_currency"],
                "tick_size": Decimal(str(spec["tick_size"])),
                "min_order_size": Decimal(str(spec["min_order_size"])),
                "maker_fee_bps": Decimal(str(spec["maker_fee_bps"])),
                "taker_fee_bps": Decimal(str(spec["taker_fee_bps"])),
                "calendar_id": spec["calendar_id"],
            }
            if spec.get("venue_specific_id"):
                coerced["venue_specific_id"] = spec["venue_specific_id"]
            return I(**coerced)

        universe = SpotPerpUniverse(spot=_to_inst(raw["spot"]), perp=_to_inst(raw["perp"]))
        return cls(
            universe=universe,
            entry_threshold_apr=Decimal(str(raw["entry_threshold_apr"])),
            exit_threshold_apr=Decimal(str(raw["exit_threshold_apr"])),
            notional_usd=Decimal(str(raw["notional_usd"])),
            signal_source=str(raw.get("signal_source", "funding")),  # type: ignore[arg-type]
            negative_funding_exit_consecutive=int(raw.get("negative_funding_exit_consecutive", 0)),
            strategy_id=str(raw.get("strategy_id", "funding_carry_btc")),
        )
