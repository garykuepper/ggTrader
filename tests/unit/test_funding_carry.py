"""Phase 4 unit tests: FundingCarryBTC strategy + funding accrual hook."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from ggTrader.core.instrument import AssetClass, Instrument, Venue
from ggTrader.core.signal import Direction
from ggTrader.features.base import pair_column_label
from ggTrader.strategies.carry.funding_carry import (
    FundingCarryBTC,
    SpotPerpUniverse,
)


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


def _perp() -> Instrument:
    return Instrument(
        symbol="BTC-USD-PERP",
        asset_class=AssetClass.CRYPTO_PERP,
        venue=Venue.KRAKEN_FUTURES,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("1"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("2"),
        taker_fee_bps=Decimal("5"),
        calendar_id="crypto_24_7",
        venue_specific_id="PF_XBTUSD",
    )


class _StubFundingStore:
    """Minimal FeatureStore stub: serves funding_apr_30d as arity='single'."""

    def __init__(self, funding_apr: float = 0.0) -> None:
        self.funding_apr = funding_apr

    def get(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def get_at(self, feature_name, instruments, ts):
        if feature_name == "funding_apr_30d":
            inst = (
                instruments
                if not hasattr(instruments, "__iter__") or isinstance(instruments, Instrument)
                else list(instruments)[0]
            )
            return pd.Series({inst.symbol: self.funding_apr})
        raise KeyError(feature_name)


class _StubBasisStore:
    """FeatureStore stub: serves basis_premium_apr_30d as arity='pair'."""

    def __init__(self, basis_apr: float = 0.0) -> None:
        self.basis_apr = basis_apr

    def get(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def get_at(self, feature_name, instruments, ts):
        if feature_name == "basis_premium_apr_30d":
            insts = list(instruments)
            label = pair_column_label(insts)
            return pd.Series({label: self.basis_apr})
        raise KeyError(feature_name)


def _strategy(entry="0.08", exit="0.04", source="funding", neg_n=0) -> FundingCarryBTC:
    return FundingCarryBTC(
        universe=SpotPerpUniverse(spot=_spot(), perp=_perp()),
        entry_threshold_apr=Decimal(entry),
        exit_threshold_apr=Decimal(exit),
        notional_usd=Decimal("10000"),
        signal_source=source,  # type: ignore[arg-type]
        negative_funding_exit_consecutive=neg_n,
    )


# ---- threshold logic --------------------------------------------------------


def test_below_entry_threshold_no_signals():
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.05)
    out = s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)
    assert out == []


def test_above_entry_threshold_emits_long_spot_short_perp():
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.12)
    out = s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)
    assert len(out) == 2
    by_sym = {x.instrument.symbol: x for x in out}
    assert by_sym["BTC-USD"].direction is Direction.LONG
    assert by_sym["BTC-USD-PERP"].direction is Direction.SHORT
    assert all(x.target_notional_usd == Decimal("10000") for x in out)


def test_hysteresis_holds_position_between_thresholds():
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.12)
    s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)  # entry
    fs.funding_apr = 0.06  # below entry (8%) but above exit (4%) → hold
    out = s.generate_signals(datetime(2025, 6, 2, tzinfo=timezone.utc), fs)
    assert len(out) == 2  # still in position


def test_exit_below_exit_threshold():
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.12)
    s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)  # entry
    fs.funding_apr = 0.03  # below exit (4%) → exit
    out = s.generate_signals(datetime(2025, 6, 2, tzinfo=timezone.utc), fs)
    assert out == []


def test_negative_funding_consecutive_force_exit():
    s = _strategy(neg_n=3)
    fs = _StubFundingStore(funding_apr=0.12)
    s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)  # entry
    fs.funding_apr = -0.01  # negative but smoothed signal still above exit threshold
    # 2 consecutive negative bars → still holding
    s.generate_signals(datetime(2025, 6, 2, tzinfo=timezone.utc), fs)
    s.generate_signals(datetime(2025, 6, 3, tzinfo=timezone.utc), fs)
    # 3rd consecutive negative → force exit
    out = s.generate_signals(datetime(2025, 6, 4, tzinfo=timezone.utc), fs)
    assert out == []


def test_validates_hysteresis_band():
    with pytest.raises(ValueError, match="hysteresis"):
        FundingCarryBTC(
            universe=SpotPerpUniverse(spot=_spot(), perp=_perp()),
            entry_threshold_apr=Decimal("0.04"),
            exit_threshold_apr=Decimal("0.08"),  # exit > entry — invalid
            notional_usd=Decimal("10000"),
        )


# ---- carry accrual ----------------------------------------------------------


def test_carry_credits_short_perp_when_funding_positive():
    """Sign convention: positive funding → longs pay, shorts receive."""
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.073)  # 7.3% APR
    carry = s.position_carry(
        instrument=_perp(),
        direction=Direction.SHORT,
        notional=Decimal("10000"),
        ts=datetime(2025, 6, 1, tzinfo=timezone.utc),
        feature_store=fs,
    )
    expected_daily = Decimal("10000") * Decimal("0.073") / Decimal("365")
    assert carry > 0
    assert abs(carry - expected_daily) < Decimal("0.001")


def test_carry_debits_long_perp_when_funding_positive():
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.073)
    carry = s.position_carry(
        instrument=_perp(),
        direction=Direction.LONG,
        notional=Decimal("10000"),
        ts=datetime(2025, 6, 1, tzinfo=timezone.utc),
        feature_store=fs,
    )
    assert carry < 0


def test_carry_zero_for_spot_position():
    """Spot legs never accrue funding."""
    s = _strategy()
    fs = _StubFundingStore(funding_apr=0.10)
    carry = s.position_carry(
        instrument=_spot(),
        direction=Direction.LONG,
        notional=Decimal("10000"),
        ts=datetime(2025, 6, 1, tzinfo=timezone.utc),
        feature_store=fs,
    )
    assert carry == Decimal("0")


# ---- basis_proxy signal source ---------------------------------------------


def test_basis_proxy_uses_pair_feature():
    s = _strategy(source="basis_proxy")
    fs = _StubBasisStore(basis_apr=0.12)
    out = s.generate_signals(datetime(2025, 6, 1, tzinfo=timezone.utc), fs)
    assert len(out) == 2


def test_basis_proxy_carry_accrual():
    s = _strategy(source="basis_proxy")
    fs = _StubBasisStore(basis_apr=0.10)
    carry = s.position_carry(
        instrument=_perp(),
        direction=Direction.SHORT,
        notional=Decimal("10000"),
        ts=datetime(2025, 6, 1, tzinfo=timezone.utc),
        feature_store=fs,
    )
    assert carry > 0


# ---- YAML loader ------------------------------------------------------------


def test_yaml_loads_real_funding_config():
    from ggTrader.strategies.loader import build_strategy_from_yaml

    s = build_strategy_from_yaml("src/ggTrader/config/strategies/funding_carry_btc_real.yaml")
    assert isinstance(s, FundingCarryBTC)
    assert s.signal_source == "funding"
    assert s.entry_threshold_apr == Decimal("0.08")
    assert s.exit_threshold_apr == Decimal("0.04")


def test_yaml_loads_basis_proxy_config():
    from ggTrader.strategies.loader import build_strategy_from_yaml

    s = build_strategy_from_yaml("src/ggTrader/config/strategies/funding_carry_btc_basis.yaml")
    assert isinstance(s, FundingCarryBTC)
    assert s.signal_source == "basis_proxy"
