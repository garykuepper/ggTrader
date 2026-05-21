"""Phase 3.5 integration test: end-to-end CashAndCarryBTC backtest against
synthetic basis data. Validates the full pipeline (YAML → loader →
SyntheticFeatureStore → backtest engine → roll accounting → metrics)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from ggTrader.backtest.vectorized import run_backtest
from ggTrader.cli.cmd_backtest_strategy import _BroadcastFutureFeatureStore
from ggTrader.core.calendars import Crypto24x7Calendar
from ggTrader.features.derivatives_synthetic import (
    SyntheticBasisConfig,
    SyntheticFeatureStore,
)
from ggTrader.strategies.loader import build_strategy_from_yaml

pytestmark = pytest.mark.integration

CONFIG_PATH = Path("src/ggTrader/config/strategies/cash_and_carry_btc.yaml")


def _build_backtest():
    strategy = build_strategy_from_yaml(CONFIG_PATH)
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    spot = strategy.carry_universe.spot
    futures = strategy.carry_universe.futures
    inner = SyntheticFeatureStore(
        config=SyntheticBasisConfig(start=start, end=end, seed=42),
        spot_instrument=spot,
        future_instruments=futures,
    )
    feature_store = _BroadcastFutureFeatureStore(inner, spot, futures)
    result = run_backtest(
        strategy=strategy,
        feature_store=feature_store,
        start=start,
        end=end,
        starting_equity=Decimal("100000"),
        calendar=Crypto24x7Calendar(),
    )
    return strategy, result


def test_end_to_end_backtest_produces_coherent_pnl():
    strategy, result = _build_backtest()
    metrics = result.metrics()

    # equity curve coherent
    assert len(result.equity_curve) >= 365 * 3
    assert result.equity_curve.index.is_monotonic_increasing
    # at least some trades fired
    assert metrics["n_trades"] > 0
    # drawdown well-bounded for a market-neutral carry trade
    assert metrics["max_drawdown"] > -0.10
    # entries cleared the strategy's effective threshold
    entry_basis = [
        float(t.metadata["basis_apr"])
        for t in result.trades
        if t.reason == "entry" and "basis_apr" in t.metadata
    ]
    if entry_basis:
        threshold = float(strategy.threshold_apr + strategy.round_trip_fee_apr)
        assert min(entry_basis) >= threshold - 1e-9


def test_rolls_detected_and_charged_single_fee():
    """Phase 3.5: when CarryUniverse.active_future rolls and a position is held
    at that moment, the engine emits a RollEvent and charges one calendar-spread
    fee instead of two separate trade fees."""
    _, result = _build_backtest()
    # With the proper expiry ladder (2022-2027 quarterly contracts), at least
    # some rolls should fire across 3 years of basis cycles.
    assert len(result.rolls) > 0
    for roll in result.rolls:
        # spread_fee is bounded above by the fee on the larger leg only
        # (i.e. less than sum of two independent trade fees on each side)
        max_leg_notional = max(
            roll.old_price * roll.old_quantity, roll.new_price * roll.new_quantity
        )
        assert roll.spread_fee <= max_leg_notional * Decimal("0.0005") + Decimal("0.01")
        # symbols actually swap
        assert roll.old_symbol != roll.new_symbol


def test_signal_target_notional_is_typed():
    """Phase 3.5: trades carry target_notional_usd through Signal as a typed
    Decimal field, not a metadata-string. Failing this means the
    metadata-string hack regressed."""
    strategy, _ = _build_backtest()
    from datetime import datetime as _dt

    fs_inner = SyntheticFeatureStore(
        config=SyntheticBasisConfig(
            start=_dt(2022, 1, 1, tzinfo=timezone.utc),
            end=_dt(2022, 12, 31, tzinfo=timezone.utc),
            seed=42,
        ),
        spot_instrument=strategy.carry_universe.spot,
        future_instruments=strategy.carry_universe.futures,
    )
    fs = _BroadcastFutureFeatureStore(
        fs_inner, strategy.carry_universe.spot, strategy.carry_universe.futures
    )
    # Find a date where basis is comfortably above threshold so signals fire.
    for day in range(0, 365, 7):
        ts = _dt(2022, 1, 1, tzinfo=timezone.utc).replace(day=1) + __import__("datetime").timedelta(
            days=day
        )
        sigs = strategy.generate_signals(ts, fs)
        if sigs:
            assert all(s.target_notional_usd == Decimal("10000") for s in sigs)
            return
    pytest.fail("strategy never emitted a signal in 2022 — basis tuning regression?")
