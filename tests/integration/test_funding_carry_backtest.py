"""Phase 4 integration test: end-to-end FundingCarryBTC backtest against
real Kraken funding data in TimescaleDB. Skipped if DB unreachable."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ggTrader.backtest.vectorized import run_backtest
from ggTrader.core.calendars import Crypto24x7Calendar

pytestmark = pytest.mark.integration


def test_real_funding_run_produces_positive_carry_in_2025h2():
    try:
        from ggTrader.features.timescale_store import TimescaleFeatureStore

        fs = TimescaleFeatureStore()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"TimescaleDB unreachable: {exc}")
    from ggTrader.strategies.loader import build_strategy_from_yaml

    strat = build_strategy_from_yaml("src/ggTrader/config/strategies/funding_carry_btc_real.yaml")
    result = run_backtest(
        strategy=strat,
        feature_store=fs,
        start=datetime(2025, 5, 15, tzinfo=timezone.utc),
        end=datetime(2026, 5, 15, tzinfo=timezone.utc),
        starting_equity=Decimal("100000"),
        calendar=Crypto24x7Calendar(),
        position_carry_fn=strat.position_carry,
    )
    fs.close()
    m = result.metrics()
    # Positive funding regime over this window → expect non-negative carry
    assert m["total_return"] >= 0.0
    # Market-neutral hedge → tight drawdown
    assert m["max_drawdown"] > -0.05
    # Strategy fired at least one cycle
    assert m["n_trades"] >= 4
