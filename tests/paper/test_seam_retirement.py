"""The eight former private trader seams are gone; the step objects own the injection points.

Each test proves two things for one former seam: `PaperTrader` no longer has the private forward,
and replacing the named method of the step object that owns the body intercepts the run (the
spy delegates, so the recorded run must equal the untouched one).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from ggTrader.paper.corporate_actions import CorporateActions
from ggTrader.paper.execution import OrderExecutor
from ggTrader.paper.trader import PaperTrader
from tests.paper.test_seam_contract import _same_run, drive
from tests.paper.transcript import harness
from tests.paper.transcript.scenarios import SCENARIOS


def _spy_on(owner: type, method: str, scenario_name: str) -> list[tuple]:
    """Run the scenario twice (untouched, then with a delegating spy); return the spy's calls."""
    scenario = SCENARIOS[scenario_name]
    baseline = drive("new", scenario)
    calls: list[tuple] = []
    real = getattr(owner, method)

    def configure(_trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return real(self, *args, **kwargs)

        stack.enter_context(patch.object(owner, method, spy))

    intercepted = drive("new", scenario, configure)
    assert _same_run(baseline, intercepted)
    return calls


def test_submit_order_is_replaced_by_the_executor_submit_order():
    assert not hasattr(PaperTrader, "_submit_order")
    calls = _spy_on(OrderExecutor, "submit_order", "plain_buys_and_sells")
    assert {args[0] for args, _ in calls} == {"BUY", "SELL"}


def test_poll_orders_is_replaced_by_the_executor_poll_orders():
    assert not hasattr(PaperTrader, "_poll_orders")
    assert _spy_on(OrderExecutor, "poll_orders", "plain_buys_and_sells")


def test_reconcile_order_intents_is_replaced_by_the_executor_reconcile_order_intents():
    assert not hasattr(PaperTrader, "_reconcile_order_intents")
    assert _spy_on(OrderExecutor, "reconcile_order_intents", "plain_buys_and_sells")


def test_reconcile_pending_orders_is_replaced_by_the_executor_reconcile_pending_orders():
    assert not hasattr(PaperTrader, "_reconcile_pending_orders")
    assert _spy_on(OrderExecutor, "reconcile_pending_orders", "plain_buys_and_sells")


def test_flag_if_stale_is_replaced_by_the_executor_flag_if_stale():
    assert not hasattr(PaperTrader, "_flag_if_stale")
    assert _spy_on(OrderExecutor, "flag_if_stale", "legacy_pending_stale")


def test_missed_exits_is_replaced_by_the_corporate_actions_missed_exits():
    assert not hasattr(PaperTrader, "_missed_exits")
    assert _spy_on(CorporateActions, "missed_exits", "missed_exit_catch_up")


def test_compute_split_corrections_is_replaced_by_the_corporate_actions_split_corrections():
    assert not hasattr(PaperTrader, "_compute_split_corrections")
    assert _spy_on(CorporateActions, "split_corrections", "split_correction")


def test_accrue_dividends_is_replaced_by_the_corporate_actions_accrue_dividends():
    assert not hasattr(PaperTrader, "_accrue_dividends")
    assert _spy_on(CorporateActions, "accrue_dividends", "new_dividend_accrual")


def test_the_seam_container_and_the_trader_builder_are_gone():
    from ggTrader.paper import seams

    assert not hasattr(seams, "TraderSeams")
    assert not hasattr(PaperTrader, "_seams")


def test_step_objects_take_no_seams_argument():
    import inspect

    from ggTrader.paper.portfolio import PortfolioView

    assert "seams" not in inspect.signature(OrderExecutor.__init__).parameters
    load = inspect.signature(PortfolioView.load).parameters
    assert "seams" not in load
    assert "corporate_actions" in load
