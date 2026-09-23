"""Tests for cash_sweep.py's sizing math and env-var config."""

from __future__ import annotations

from unittest.mock import patch

from ggTrader.paper import cash_sweep
from ggTrader.paper.cash_sweep import (
    DEFAULT_BUY_TRIGGER_PCT,
    buy_trigger_pct,
    compute_sweep_buy,
)


class TestEnvConfig:
    def test_sweep_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CASH_SWEEP_ENABLED", raising=False)
        assert cash_sweep.sweep_enabled() is False

    def test_sweep_enabled_truthy_values(self, monkeypatch):
        for on in ("1", "true", "True", "yes", "on"):
            monkeypatch.setenv("CASH_SWEEP_ENABLED", on)
            assert cash_sweep.sweep_enabled() is True
        for off in ("0", "false", "no", ""):
            monkeypatch.setenv("CASH_SWEEP_ENABLED", off)
            assert cash_sweep.sweep_enabled() is False

    def test_sweep_symbol_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SWEEP_SYMBOL", raising=False)
        assert cash_sweep.sweep_symbol() == "SPY"
        monkeypatch.setenv("SWEEP_SYMBOL", "voo")
        assert cash_sweep.sweep_symbol() == "VOO"

    def test_reserve_pct_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SWEEP_CASH_RESERVE_PCT", raising=False)
        assert cash_sweep.reserve_pct() == 0.05
        monkeypatch.setenv("SWEEP_CASH_RESERVE_PCT", "0.1")
        assert cash_sweep.reserve_pct() == 0.1

    def test_min_clip_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SWEEP_MIN_CLIP_USD", raising=False)
        assert cash_sweep.min_clip_usd() == 500.0
        monkeypatch.setenv("SWEEP_MIN_CLIP_USD", "250")
        assert cash_sweep.min_clip_usd() == 250.0


class TestComputeSweepBuy:
    def test_buys_leftover_cash_above_reserve(self):
        # portfolio 100k, reserve 5% = 5000, cash 20000 -> sweep 15000
        action = cash_sweep.compute_sweep_buy(
            cash_after_strategy_orders=20_000.0, portfolio_value=100_000.0
        )
        assert action.side == "buy"
        assert action.notional == 15_000.0

    def test_below_min_clip_skips(self):
        # cash 5300, reserve 5000 -> target 300 < default min_clip 500
        action = cash_sweep.compute_sweep_buy(
            cash_after_strategy_orders=5_300.0, portfolio_value=100_000.0
        )
        assert action.side is None
        assert action.notional == 0.0

    def test_cash_below_reserve_no_buy(self):
        action = cash_sweep.compute_sweep_buy(
            cash_after_strategy_orders=1_000.0, portfolio_value=100_000.0
        )
        assert action.side is None

    def test_custom_reserve_and_min_clip(self):
        action = cash_sweep.compute_sweep_buy(
            cash_after_strategy_orders=10_000.0,
            portfolio_value=100_000.0,
            reserve_pct=0.02,
            min_clip=100.0,
        )
        # reserve = 2000, target = 8000
        assert action.side == "buy"
        assert action.notional == 8_000.0

    def test_zero_portfolio_value_is_noop(self):
        action = cash_sweep.compute_sweep_buy(
            cash_after_strategy_orders=10_000.0, portfolio_value=0.0
        )
        assert action.side is None
        assert action.notional == 0.0

    def test_buy_skipped_when_cash_above_reserve_but_below_trigger(self):
        """A small daily surplus just above the 5% reserve (but below the 8%
        trigger) must not fire a sweep buy -- this is the daily-churn bug."""
        action = compute_sweep_buy(
            cash_after_strategy_orders=6_500.0,  # 6.5% of 100k: > 5% reserve, < 8% trigger
            portfolio_value=100_000.0,
            reserve_pct=0.05,
            min_clip=500.0,
            buy_trigger_pct=0.08,
        )
        assert action.side is None

    def test_buy_fires_once_cash_exceeds_trigger_and_targets_reserve(self):
        """Once cash clears the 8% trigger, the buy still sweeps down to the
        5% reserve floor, same target math as before."""
        action = compute_sweep_buy(
            cash_after_strategy_orders=9_000.0,  # 9% of 100k: > 8% trigger
            portfolio_value=100_000.0,
            reserve_pct=0.05,
            min_clip=500.0,
            buy_trigger_pct=0.08,
        )
        assert action.side == "buy"
        assert action.notional == 4_000.0  # sweeps 9,000 down to the 5,000 reserve

    def test_buy_trigger_pct_env_var_default(self):
        assert buy_trigger_pct() == DEFAULT_BUY_TRIGGER_PCT

    @patch.dict("os.environ", {"SWEEP_BUY_TRIGGER_PCT": "0.10"})
    def test_buy_trigger_pct_env_var_override(self):
        assert buy_trigger_pct() == 0.10

    def test_inverted_thresholds_log_a_warning(self, caplog):
        """Finding 5 (2026-09-11 remediation review): if an operator sets
        `SWEEP_CASH_RESERVE_PCT` above `SWEEP_BUY_TRIGGER_PCT`, the
        dead-band silently inverts (no buy ever fires) -- must be visible
        in logs, not a silent misconfiguration."""
        with caplog.at_level("WARNING", logger="ggTrader.paper.cash_sweep"):
            compute_sweep_buy(
                cash_after_strategy_orders=9_000.0,
                portfolio_value=100_000.0,
                reserve_pct=0.10,
                min_clip=500.0,
                buy_trigger_pct=0.05,
            )
        assert any(
            "buy_trigger_pct" in record.getMessage() and "reserve_pct" in record.getMessage()
            for record in caplog.records
        )

    def test_equal_thresholds_also_log_a_warning(self, caplog):
        """The dead-band is empty (not just inverted) when the two
        thresholds are equal -- also worth flagging."""
        with caplog.at_level("WARNING", logger="ggTrader.paper.cash_sweep"):
            compute_sweep_buy(
                cash_after_strategy_orders=9_000.0,
                portfolio_value=100_000.0,
                reserve_pct=0.08,
                min_clip=500.0,
                buy_trigger_pct=0.08,
            )
        assert any("buy_trigger_pct" in record.getMessage() for record in caplog.records)

    def test_normal_thresholds_log_no_warning(self, caplog):
        with caplog.at_level("WARNING", logger="ggTrader.paper.cash_sweep"):
            compute_sweep_buy(
                cash_after_strategy_orders=9_000.0,
                portfolio_value=100_000.0,
                reserve_pct=0.05,
                min_clip=500.0,
                buy_trigger_pct=0.08,
            )
        assert caplog.records == []


class TestComputeSweepSellForFunding:
    def test_no_shortfall_no_sell(self):
        # cash 10000 covers prospective buys 3000 + reserve 5000 (100k * 5%)
        action = cash_sweep.compute_sweep_sell_for_funding(
            cash_available=10_000.0,
            portfolio_value=100_000.0,
            prospective_buy_notional=3_000.0,
            current_sweep_position_value=20_000.0,
        )
        assert action.side is None
        assert action.notional == 0.0

    def test_shortfall_sells_exact_amount_needed(self):
        # need 3000(buys) + 5000(reserve) = 8000, have 4000 cash -> shortfall 4000
        action = cash_sweep.compute_sweep_sell_for_funding(
            cash_available=4_000.0,
            portfolio_value=100_000.0,
            prospective_buy_notional=3_000.0,
            current_sweep_position_value=20_000.0,
        )
        assert action.side == "sell"
        assert action.notional == 4_000.0

    def test_shortfall_capped_at_sweep_position_value(self):
        # shortfall would be 8000, but the sweep position only holds 5000
        action = cash_sweep.compute_sweep_sell_for_funding(
            cash_available=0.0,
            portfolio_value=100_000.0,
            prospective_buy_notional=3_000.0,
            current_sweep_position_value=5_000.0,
        )
        assert action.side == "sell"
        assert action.notional == 5_000.0

    def test_no_sweep_position_is_noop(self):
        action = cash_sweep.compute_sweep_sell_for_funding(
            cash_available=0.0,
            portfolio_value=100_000.0,
            prospective_buy_notional=3_000.0,
            current_sweep_position_value=0.0,
        )
        assert action.side is None

    def test_zero_portfolio_value_is_noop(self):
        action = cash_sweep.compute_sweep_sell_for_funding(
            cash_available=0.0,
            portfolio_value=0.0,
            prospective_buy_notional=3_000.0,
            current_sweep_position_value=5_000.0,
        )
        assert action.side is None


class TestEstimateProspectiveBuyNotional:
    def test_single_sleeve_within_caps(self):
        total = cash_sweep.estimate_prospective_buy_notional(
            buys_by_sleeve={"sp500": ["AAPL", "MSFT", "NVDA"]},
            sleeve_notional={"sp500": 1_000.0},
            slot_caps={"sp500": 10},
            slots_available=10,
        )
        assert total == 3_000.0

    def test_capped_by_sleeve_cap(self):
        total = cash_sweep.estimate_prospective_buy_notional(
            buys_by_sleeve={"sp500": ["AAPL", "MSFT", "NVDA", "GOOG"]},
            sleeve_notional={"sp500": 1_000.0},
            slot_caps={"sp500": 2},
            slots_available=10,
        )
        assert total == 2_000.0

    def test_capped_by_global_slots(self):
        total = cash_sweep.estimate_prospective_buy_notional(
            buys_by_sleeve={"sp500": ["A", "B"], "nasdaq100": ["C", "D"]},
            sleeve_notional={"sp500": 1_000.0, "nasdaq100": 2_000.0},
            slot_caps={"sp500": 5, "nasdaq100": 5},
            slots_available=3,
        )
        # Global slots exhausted across sleeves in iteration order; total
        # candidates consumed never exceed slots_available.
        assert total in (3_000.0, 4_000.0, 5_000.0)
        # More precisely: whichever sleeve is consumed first takes up to 2,
        # the other takes what's left of the 3 slots. Since dict iteration
        # order is insertion order here, sp500 (2 candidates) goes first,
        # consuming 2 of 3 slots, leaving 1 for nasdaq100.
        assert total == 2_000.0 + 2_000.0

    def test_empty_buys_is_zero(self):
        total = cash_sweep.estimate_prospective_buy_notional(
            buys_by_sleeve={}, sleeve_notional={}, slot_caps={}, slots_available=10
        )
        assert total == 0.0
