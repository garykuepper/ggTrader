"""Phase 3 unit tests: CashAndCarryBTC signal logic, threshold, roll behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ggTrader.core import AssetClass, Direction, Instrument, Venue
from ggTrader.strategies.carry import CarryUniverse, CashAndCarryBTC


def _spot() -> Instrument:
    return Instrument(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO_SPOT,
        venue=Venue.KRAKEN_SPOT,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("16"),
        taker_fee_bps=Decimal("26"),
        calendar_id="crypto_24_7",
    )


def _future(expiry: str, symbol: str | None = None) -> Instrument:
    return Instrument(
        symbol=symbol or f"BTC-USD-{expiry.replace('-', '')[2:]}",
        asset_class=AssetClass.CRYPTO_DATED_FUTURE,
        venue=Venue.KRAKEN_FUTURES,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("0.5"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("2"),
        taker_fee_bps=Decimal("5"),
        calendar_id="crypto_24_7",
        expiry=expiry,
    )


class _StubFeatureStore:
    """Phase 3.5 widened FeatureStore stub: serves basis_apr as a pair feature."""

    def __init__(self, basis_apr: float = 0.0) -> None:
        self.basis_apr = basis_apr
        self.calls: list[tuple[str, tuple[str, ...], datetime]] = []

    def _norm(self, instruments) -> list[Instrument]:
        if isinstance(instruments, Instrument):
            return [instruments]
        return list(instruments)

    def get(self, feature_name, instruments, start, end):  # pragma: no cover
        raise NotImplementedError

    def get_at(self, feature_name, instruments, ts):
        import pandas as pd

        insts = self._norm(instruments)
        self.calls.append((feature_name, tuple(i.symbol for i in insts), ts))
        if feature_name == "basis_apr" and len(insts) == 2:
            label = "|".join(i.symbol for i in insts)
            return pd.Series({label: self.basis_apr})
        raise KeyError(feature_name)


def _strategy(threshold: str = "0.10", fee_apr: str = "0", futures=None) -> CashAndCarryBTC:
    futures = futures or [_future("2026-06-26"), _future("2026-09-25"), _future("2026-12-25")]
    return CashAndCarryBTC(
        universe=CarryUniverse(spot=_spot(), futures=futures, roll_buffer_hours=24),
        threshold_apr=Decimal(threshold),
        notional_usd=Decimal("10000"),
        round_trip_fee_apr=Decimal(fee_apr),
    )


# ---- threshold logic --------------------------------------------------------


def test_below_threshold_emits_no_signals():
    strat = _strategy(threshold="0.10")
    fs = _StubFeatureStore(basis_apr=0.05)
    sigs = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    assert sigs == []


def test_at_threshold_emits_no_signals():
    strat = _strategy(threshold="0.10")
    fs = _StubFeatureStore(basis_apr=0.10)
    sigs = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    assert sigs == []


def test_above_threshold_emits_long_spot_short_future():
    strat = _strategy(threshold="0.10")
    fs = _StubFeatureStore(basis_apr=0.20)
    sigs = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    assert len(sigs) == 2

    by_symbol = {s.instrument.symbol: s for s in sigs}
    assert by_symbol["BTC-USD"].direction is Direction.LONG
    assert by_symbol["BTC-USD-260626"].direction is Direction.SHORT
    assert all(s.confidence == Decimal("1.0") for s in sigs)
    assert all(s.target_notional_usd == Decimal("10000") for s in sigs)
    assert all(s.metadata["basis_apr"] == "0.2" for s in sigs)


def test_round_trip_fee_increases_effective_threshold():
    strat = _strategy(threshold="0.10", fee_apr="0.05")
    fs = _StubFeatureStore(basis_apr=0.12)
    # 12% basis - 5% fees = 7% < 10% threshold → flat
    sigs = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    assert sigs == []

    fs.basis_apr = 0.20
    sigs = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    assert len(sigs) == 2


# ---- roll / universe logic --------------------------------------------------


def test_active_future_picks_nearest_beyond_roll_buffer():
    universe = CarryUniverse(
        spot=_spot(),
        futures=[_future("2026-06-26"), _future("2026-09-25")],
        roll_buffer_hours=24,
    )
    # Well before June expiry → June is active
    assert (
        universe.active_future(datetime(2026, 5, 1, tzinfo=timezone.utc)).symbol == "BTC-USD-260626"
    )
    # 1h before June 26 16:00 UTC → buffer crossed, rolls to September
    just_before_roll = datetime(2026, 6, 25, 17, 0, tzinfo=timezone.utc)
    assert universe.active_future(just_before_roll).symbol == "BTC-USD-260925"


def test_active_future_returns_none_when_no_contract_far_enough():
    expiry = "2026-06-26"
    universe = CarryUniverse(
        spot=_spot(),
        futures=[_future(expiry)],
        roll_buffer_hours=24,
    )
    # Inside the roll buffer for the last contract → no active future
    assert universe.active_future(datetime(2026, 6, 26, 13, 0, tzinfo=timezone.utc)) is None


def test_strategy_emits_no_signals_when_no_active_future():
    strat = _strategy(futures=[_future("2026-06-26")])
    fs = _StubFeatureStore(basis_apr=0.20)
    # Past last expiry → no contract → no signal regardless of basis
    sigs = strat.generate_signals(datetime(2026, 7, 1, tzinfo=timezone.utc), fs)
    assert sigs == []


def test_strategy_signals_switch_contracts_across_roll():
    strat = _strategy()
    fs = _StubFeatureStore(basis_apr=0.20)

    pre = strat.generate_signals(datetime(2026, 5, 1, tzinfo=timezone.utc), fs)
    post = strat.generate_signals(datetime(2026, 6, 27, tzinfo=timezone.utc), fs)

    pre_future = next(s for s in pre if s.direction is Direction.SHORT)
    post_future = next(s for s in post if s.direction is Direction.SHORT)
    assert pre_future.instrument.symbol == "BTC-USD-260626"
    assert post_future.instrument.symbol == "BTC-USD-260925"


# ---- validation --------------------------------------------------------------


def test_universe_rejects_wrong_asset_class_for_spot():
    perp = Instrument(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO_PERP,
        venue=Venue.KRAKEN_FUTURES,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("16"),
        taker_fee_bps=Decimal("26"),
        calendar_id="crypto_24_7",
    )
    with pytest.raises(ValueError, match="CRYPTO_SPOT"):
        CarryUniverse(spot=perp, futures=[_future("2026-06-26")])


def test_strategy_rejects_negative_threshold():
    with pytest.raises(ValueError):
        CashAndCarryBTC(
            universe=CarryUniverse(spot=_spot(), futures=[_future("2026-06-26")]),
            threshold_apr=Decimal("-0.10"),
            notional_usd=Decimal("10000"),
        )


# ---- YAML loader -------------------------------------------------------------


def test_yaml_config_loads_and_builds_strategy(tmp_path):
    from ggTrader.strategies.loader import build_strategy_from_yaml

    cfg_path = "src/ggTrader/config/strategies/cash_and_carry_btc.yaml"
    strat = build_strategy_from_yaml(cfg_path)
    assert strat.strategy_id == "cash_and_carry_btc"
    assert strat.threshold_apr == Decimal("0.10")
    # Phase 3.5 expiry ladder spans 2022-2027; mid-2026 active should be the
    # June 2026 contract.
    members = strat.universe.members(datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert members[0].symbol == "BTC-USD"
    assert members[1].symbol == "BTC-USD-260626"


def test_round_trip_fee_derived_from_instrument_fees():
    """Phase 3.5: when round_trip_fee_apr is omitted, it's computed from
    instrument taker fees + assumed 90-day contract duration."""
    strat = CashAndCarryBTC(
        universe=CarryUniverse(spot=_spot(), futures=[_future("2026-06-26")]),
        threshold_apr=Decimal("0.10"),
        notional_usd=Decimal("10000"),
    )
    # spot taker 26 bps + future taker 5 bps = 31 bps × 2 round trips = 62 bps
    # 62 bps / 10000 = 0.0062 × (365/90) ≈ 0.02514
    assert Decimal("0.024") < strat.round_trip_fee_apr < Decimal("0.026")
