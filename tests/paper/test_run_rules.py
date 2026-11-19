"""The seven order-flow rules of design 5.1, plus the characterizations the nets miss (Task 48).

Immutable behavioral set (hash-recorded at Task 51). Each test builds its input from an existing
scenario with ``dataclasses.replace`` (`rules_support.from_scenario`), runs it under every
implementation in `IMPLS` and asserts on the transcript's calls, never on a golden file. Targets:
``broker.submit_sell``, ``broker.submit_buy``, ``persist.create_order_intent``,
``persist.settle_order_intent``, ``notifier.send`` and ``lock.verify`` (which carries no client id).

The characterizations at the end are each measured on the untouched code and written as the
literal it measured: a call sequence, an exception type, a message, a cause chain.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from ggTrader.paper import catastrophe_stop
from ggTrader.paper.ledger import WRITE_METHODS
from ggTrader.paper.risk import RiskGuard
from ggTrader.paper.trader import PaperTrader
from tests.paper import rules_support as rs
from tests.paper.rules_support import IMPLS
from tests.paper.transcript.scenarios import SCENARIOS, _pending

# `IMPLS` names the implementations; the frozen copy exists exactly when it is listed.
assert ("legacy" in IMPLS) == hasattr(PaperTrader, "_run_legacy")

DRY_MEMBERSHIP = "fixture:sp500_v1_2026-06-02"
INVALID_QUANTITIES = [
    pytest.param(0.0, id="zero"),
    pytest.param(None, id="none"),
    pytest.param(math.nan, id="nan"),
    pytest.param(-10.0, id="negative"),
]
INVALID_QUANTITY_ERROR = (
    "Cannot submit risk-reducing SELL AAPL: invalid long quantity; new strategy buys blocked"
)
STOPPED_POSITION = {"cost_basis": 1000.0, "unrealized_pl": -400.0}  # -40%, past the -25% floor

pytestmark = pytest.mark.parametrize("impl_name", IMPLS)


def _position_without(key: str, **kwargs: float) -> dict:
    """A 10-share position at 150 whose record lacks `key`."""
    position = rs.pos(10.0, 150.0, **kwargs)
    del position[key]
    return position


def _position_with_quantity(qty: float | None, **kwargs: float) -> dict:
    """A position at 100 whose `qty` field is set to `qty` after the record is built."""
    position = rs.pos(10.0, 100.0, **kwargs)
    position["qty"] = qty
    return position


# =========================================================================== rule 1


def test_rule1_catastrophe_stopped_symbol_leaves_at_once_and_is_not_sold_twice_or_rebought(
    impl_name,
):
    """Scenario `catastrophe_stop`: NXPI breaches the floor and is also a sell and a buy signal.

    Pinned today by `TestCatastropheStopEnabled.test_stopped_symbol_not_also_sold_by_coincident_
    strategy_exit` and `test_stopped_symbol_not_rebought_by_coincident_buy_signal`
    (test_trader.py:2694, 2708). The stop's freed slot is counted once: the result lists NXPI as
    sold once and MSFT as the only buy. The second case shrinks NXPI to 1.4% of the book (as the
    second pinned test does): at the scenario's 6% the concentration cap would block the re-buy
    on its own and hide the `stopped_symbols` guard.
    """
    t = rs.run(impl_name, SCENARIOS["catastrophe_stop"])
    assert rs.submissions(t) == [("SELL", "NXPI", 100.0), ("BUY", "MSFT", 3300.0)]
    intents = rs.calls(t, "persist.create_order_intent")
    assert [(c["args"][2], c["args"][3], c["args"][6]) for c in intents] == [
        ("SELL", "NXPI", "catastrophe_stop"),
        ("BUY", "MSFT", None),
    ]
    assert t["result"]["sells"] == ["NXPI"]
    assert t["result"]["buys"] == ["MSFT"]

    small = rs.from_scenario(
        "catastrophe_stop",
        replies=rs.replies(
            rs.acct(), {"NXPI": rs.pos(20.0, 70.0, cost_basis=2000.0, unrealized_pl=-600.0)}
        ),
    )
    t = rs.run(impl_name, small)
    assert rs.submissions(t) == [("SELL", "NXPI", 20.0), ("BUY", "MSFT", 3300.0)]
    assert t["result"]["sells"] == ["NXPI"]
    assert t["result"]["buys"] == ["MSFT"]


# =========================================================================== rule 2


def _three_held_one_stopped() -> object:
    """MAX_POSITIONS 3; held: NXPI (stopped), EX (ordinary exit), FLAT; three buy signals."""
    return rs.from_scenario(
        "catastrophe_stop_at_position_cap",
        signals=rs.blend(["B1", "B2", "B3"], ["EX"]),
        replies=rs.replies(
            rs.acct(),
            {
                "FLAT": rs.pos(1.0, 100.0),
                "EX": rs.pos(1.0, 100.0),
                "NXPI": rs.pos(100.0, 70.0, cost_basis=10_000.0, unrealized_pl=-3000.0),
            },
        ),
    )


def test_rule2_slot_arithmetic_counts_regular_sells_not_all_sells(impl_name):
    """Slots count the ordinary sells only; the stop already left `strategy_positions`.

    Case 1 (three held, one stopped, one ordinary sell that fully fills, three buy signals):
    `len(strategy_positions)` is 2 after the stop, one ordinary sell gives a held count of 1 and
    `max_new_positions(1) = 3 - 1 = 2` free slots: two buys. Counting the stop again through
    `len(executed_sells) = 2` would give held count 0, three free slots and three buys (one
    over the cap once the flat holding is counted).
    Case 2 is scenario `catastrophe_stop_at_position_cap` (no ordinary sell): held count 2, one
    free slot, one buy; the double count would give two. Pinned today by
    `test_slot_math_counts_stop_exactly_once_at_position_cap` (test_trader.py:2770).
    """
    first = rs.run(impl_name, _three_held_one_stopped())
    assert rs.submissions(first) == [
        ("SELL", "NXPI", 100.0),
        ("SELL", "EX", 1.0),
        ("BUY", "B1", 3300.0),
        ("BUY", "B2", 3300.0),
    ]
    assert first["result"]["buys"] == ["B1", "B2"]
    assert first["result"]["sells"] == ["NXPI", "EX"]

    second = rs.run(impl_name, SCENARIOS["catastrophe_stop_at_position_cap"])
    assert rs.submissions(second) == [("SELL", "NXPI", 100.0), ("BUY", "MSFT", 3300.0)]


# =========================================================================== rule 3

# (status sequence of the AAPL sell, status in the message, settled (qty, price, status))
_TERMINAL_UNFILLED = [
    pytest.param(
        [
            {"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 150.0},
            {"status": "canceled", "filled_qty": 4.0, "filled_avg_price": 150.0},
        ],
        "canceled",
        (4.0, 150.0, "canceled"),
        id="canceled_after_a_partial_fill",
    ),
    pytest.param(
        [{"status": "rejected", "filled_qty": 0.0, "filled_avg_price": None}],
        "rejected",
        (0.0, None, "rejected"),
        id="rejected",
    ),
    pytest.param(
        [{"status": "expired", "filled_qty": 0.0, "filled_avg_price": None}],
        "expired",
        (0.0, None, "expired"),
        id="expired",
    ),
    pytest.param(
        [{"status": "filled", "filled_qty": 4.0, "filled_avg_price": 150.0}],
        "filled",
        (4.0, 150.0, "filled"),
        id="filled_with_a_short_quantity",
    ),
]


@pytest.mark.parametrize("sweep", [False, True], ids=["no_sweep", "sweep_on"])
@pytest.mark.parametrize(("fills", "status", "settled"), _TERMINAL_UNFILLED)
def test_rule3_a_terminal_sell_that_does_not_fully_fill_blocks_all_strategy_buys(
    impl_name, fills, status, settled, sweep
):
    """A sell of 10 AAPL ends terminal without filling 10: no strategy buy, the error is added.

    Scenarios `sell_partial_fill_canceled` (canceled after 4 of 10) and `sell_rejected_by_broker`
    (the rejected status; here without that scenario's drawdown halt, which would block the
    buys on its own). Pinned today by `test_exit_must_fill_to_free_a_position_slot`
    (test_trader.py:3785). The intent reaches `settle_order_intent`. With the sweep on, the sweep
    BUY is still submitted (Quirk 9: its condition is not the strategy buys').
    """
    env = {**rs.LIVE_ENV, **rs.SWEEP_ENV} if sweep else rs.LIVE_ENV
    scenario = rs.from_scenario("sell_partial_fill_canceled", env=env, fills={"AAPL": fills})
    t = rs.run(impl_name, scenario)
    assert ("BUY", "MSFT", 3300.0) not in rs.submissions(t)
    assert [symbol for _side, symbol, _qty in rs.submissions(t, "BUY")] == (
        ["SPY"] if sweep else []
    )
    message = f"SELL AAPL not fully filled ({status}); new strategy buys blocked"
    assert t["result"]["errors"] == [message]
    assert message in "\n".join(m for m in rs.messages(t) if "order(s) failed" in m)
    assert rs.intents_settled(t)[0] == (rs.client_id(1), *settled)
    assert t["result"]["buys"] == []


def test_rule3_a_sell_still_working_at_the_poll_deadline_raises_in_reconciliation_instead(
    impl_name,
):
    """Scenario `sell_partial_fill_working` (Quirk 10): the sell is still `partially_filled`.

    The reconcile after the sell poll raises before the "not fully filled" check can run: there
    is no settlement, no buy and no result, so no "not fully filled" error exists.
    """
    t = rs.run(impl_name, SCENARIOS["sell_partial_fill_working"])
    assert t["raised"] == "RuntimeError: Unresolved working orders block submissions"
    assert t["result"] is None
    failure = [m for m in rs.messages(t) if "Order recovery/pending orders failed" in m]
    assert len(failure) == 1
    assert rs.calls(t, "persist.settle_order_intent") == []
    assert rs.submissions(t, "BUY") == []
    assert not any("not fully filled" in m for m in rs.messages(t))
    # the notice came after the poll and before the lock was released
    notice = next(
        i for i, c in enumerate(t["calls"]) if "recovery/pending orders failed" in str(c["args"])
    )
    assert t["calls"][notice + 1]["target"] == "lock.exit"


# =========================================================================== rule 4


def _invalid_exit_scenario(kind: str, qty: float | None, cash: float, *, healthy_exit: bool):
    """AAPL with an invalid quantity is a strategy exit or a stop, with the sweep on.

    The account holds 50 SPY at 400 and `cash`. A stop needs the position to breach the floor,
    so the stopped AAPL carries a -40% record. With `healthy_exit` NVDA (5 shares) is a normal
    exit that still goes out.
    """
    breach = STOPPED_POSITION if kind == "stop" else {}
    env = rs.SWEEP_STOP_ENV if kind == "stop" else {**rs.LIVE_ENV, **rs.SWEEP_ENV}
    sells = ["NVDA"] if healthy_exit else []
    if kind == "exit":
        sells = ["AAPL", *sells]
    positions = {"AAPL": _position_with_quantity(qty, **breach), "SPY": rs.pos(50.0, 400.0)}
    if healthy_exit:
        positions["NVDA"] = rs.pos(5.0, 100.0)
    return rs.from_scenario(
        "strategy_exit_invalid_qty",
        env=env,
        signals=rs.blend(["MSFT"], sells),
        replies=rs.replies(rs.acct(100_000.0, cash), positions),
    )


@pytest.mark.parametrize("kind", ["exit", "stop"])
@pytest.mark.parametrize("qty", INVALID_QUANTITIES)
def test_rule4_an_invalid_exit_quantity_blocks_strategy_buys_and_the_sweep_buy(
    impl_name, qty, kind
):
    """An unusable exit quantity (0, None, NaN, negative), on a strategy exit and on a stop.

    Scenario `strategy_exit_invalid_qty`; pinned today by
    `test_invalid_exit_quantity_blocks_buys_but_allows_other_exits` and
    `test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy` (test_trader.py:3901). Cash is
    50,000, enough for a sweep BUY, so this fails if the invalid exit does not block it: not a
    single `submit_buy`, strategy or sweep.
    """
    t = rs.run(impl_name, _invalid_exit_scenario(kind, qty, 50_000.0, healthy_exit=True))
    assert rs.submissions(t, "BUY") == []
    assert rs.submissions(t, "SELL") == [("SELL", "NVDA", 5.0)]
    failed = [m for m in rs.messages(t) if "order(s) failed" in m]
    assert failed == [f"<b>❌ 1 order(s) failed:</b>\n{INVALID_QUANTITY_ERROR}"]
    if qty is None:
        # Measured: the order flow runs to its end as for the other values, then the daily
        # summary's own formatting of the `None` quantity raises (a defect candidate, preserved).
        assert t["result"] is None
        assert t["raised"] == (
            "TypeError: float() argument must be a string or a real number, not 'NoneType'"
        )
        # The valid NVDA exit settled normally before the failure.
        assert [(s[0], s[3]) for s in rs.intents_settled(t)] == [(rs.client_id(1), "filled")]
        assert rs.intents_settled(t)[0][1] == 5.0
        # The run got past the post-trade reads, then called the final daily summary with the
        # `None` quantity in its positions argument; the TypeError is raised from inside it.
        targets = rs.targets(t)
        last_order = max(
            i for i, name in enumerate(targets) if name == "persist.settle_order_intent"
        )
        tail = targets[last_order:]
        assert tail == [
            "persist.settle_order_intent",
            "persist.get_unresolved_order_intents",
            "persist.get_pending_orders",
            "notifier.trade_alert",
            "notifier.send",
            "notifier.send",
            "broker.get_account",
            "broker.get_positions",
            "persist.get_earliest_snapshot",
            "notifier.daily_summary",
            "lock.exit",
        ]
        (summary,) = rs.calls(t, "notifier.daily_summary")
        assert summary["args"][2]["AAPL"]["qty"] is None
        assert "persist.log_snapshot" not in targets
    else:
        assert t["result"]["errors"] == [INVALID_QUANTITY_ERROR]
        assert t["result"]["buys"] == []


@pytest.mark.parametrize("kind", ["exit", "stop"])
@pytest.mark.parametrize(("cash", "sweep_sale"), [(5000.0, None), (1000.0, 10.0), (50_000.0, None)])
def test_rule4_an_invalid_exit_does_not_block_a_reserve_restoring_sweep_sell(
    impl_name, cash, sweep_sale, kind
):
    """Quirk 8: the invalid exit empties the buys before the sweep estimate, not the sweep SELL.

    With cash below the reserve the sweep SELL goes out (cash 1,000: SELL SPY qty 10); with cash
    5,000 or 50,000 no sweep order goes out. Pinned today by
    `test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy` (its three parameter sets,
    test_trader.py:3901).
    """
    t = rs.run(impl_name, _invalid_exit_scenario(kind, -10.0, cash, healthy_exit=False))
    expected = [] if sweep_sale is None else [("SELL", "SPY", sweep_sale)]
    assert rs.submissions(t) == expected
    assert t["result"]["errors"] == [INVALID_QUANTITY_ERROR]


# =========================================================================== rule 5


@contextmanager
def _recorded_concentration_checks() -> Iterator[list[dict]]:
    """Wrap `RiskGuard.check_concentration`: record each call's arguments, call the real method.

    The positions argument is deep-copied at call time, so a later mutation by the trader
    cannot change what the test reads.
    """
    seen: list[dict] = []
    real = RiskGuard.check_concentration

    def recording(self, symbol, positions, portfolio_value, prospective_notional=0.0):
        seen.append(
            {
                "symbol": symbol,
                "positions": copy.deepcopy(positions),
                "portfolio_value": portfolio_value,
                "prospective_notional": prospective_notional,
            }
        )
        return real(self, symbol, positions, portfolio_value, prospective_notional)

    with patch.object(RiskGuard, "check_concentration", recording):
        yield seen


def test_rule5_orders_use_the_brokers_uncorrected_quantity(impl_name):
    """With a split correction on record the order is the broker's raw quantity.

    Scenario `split_correction`: the broker holds 20.80410098 MNST shares (the split-adjusted
    figure would be 41.608...). The corrected figures appear only in the daily summary (the
    `split_corrections` keyword and the doubled market value of a held name) and in the
    concentration check. Scenario `split_correction_held` shows the second: a recording wrapper
    around the real `check_concentration` sees the held MNST at its doubled market value, so the
    corrected positions (not `strategy_positions`, whose MNST value is the broker's) reach the
    check; the broker's raw quantity is preserved there. Pinned today by
    `TestSellExits.test_sell_submits_brokers_uncorrected_qty_not_split_adjusted`
    (test_trader.py:306).
    """
    t = rs.run(impl_name, SCENARIOS["split_correction"])
    assert rs.submissions(t) == [("SELL", "MNST", 20.80410098)]
    (summary,) = rs.calls(t, "notifier.daily_summary")
    assert summary["kwargs"]["split_corrections"] == {"MNST": 2.0}
    assert summary["kwargs"]["booked_equity"] == 100000.0

    held_scenario = SCENARIOS["split_correction_held"]
    with _recorded_concentration_checks() as checks:
        held = rs.run(impl_name, held_scenario)
    assert rs.submissions(held) == [("BUY", "NVDA", 3300.0)]
    raw = held_scenario.replies["get_positions"][0]["MNST"]
    (check,) = checks
    assert check["symbol"] == "NVDA"
    checked = check["positions"]["MNST"]
    assert checked["market_value"] == pytest.approx(2 * raw["market_value"])
    assert checked["market_value"] != raw["market_value"]
    assert checked["qty"] == raw["qty"]  # the broker's quantity is never split-adjusted
    (held_summary,) = rs.calls(held, "notifier.daily_summary")
    shown = held_summary["args"][2]["MNST"]["market_value"]
    assert shown == pytest.approx(2 * raw["market_value"])
    assert held_summary["kwargs"]["split_corrections"] == {"MNST": 2.0}


# =========================================================================== rule 6


def test_rule6_a_read_only_run_writes_nothing_and_records_what_it_skipped(impl_name):
    """Scenario `dry_run_read_only`: no write method reaches the ledger; the skips are listed.

    Pinned today by `TestReadOnlyDryRun.test_dry_run_is_read_only_by_default` and
    `test_pending_orders_are_neither_polled_nor_booked`.
    """
    t = rs.run(impl_name, SCENARIOS["dry_run_read_only"])
    expected = ["init_paper_schema", "refresh_benchmark_tape", "save_peak_value", "log_snapshot"]
    written = [
        c["target"] for c in t["calls"] if c["target"].removeprefix("persist.") in WRITE_METHODS
    ]
    assert written == []
    # Read-only also covers the two tape writes outside the ledger: the signal generator is told
    # not to persist its tape, and the benchmark tape refresh is never called.
    (signal_call,) = rs.calls(t, "signals.generate_core_signals")
    assert signal_call["kwargs"]["persist_tape"] is False
    assert rs.calls(t, "tape.refresh_benchmark_tape") == []
    assert t["skipped_writes"] == expected
    assert t["result"]["skipped_writes"] == expected


def test_rule6_the_same_run_live_does_write(impl_name):
    """The control: the same signals live write, so the test above would see a missing wrapper."""
    base = SCENARIOS["dry_run_read_only"]
    live = rs.from_scenario(
        "dry_run_read_only",
        dry_run=False,
        membership="db",
        persist={**base.persist, "get_pending_orders": []},
    )
    t = rs.run(impl_name, live)
    written = {c["target"].removeprefix("persist.") for c in t["calls"]} & WRITE_METHODS
    assert {
        "init_paper_schema",
        "save_peak_value",
        "log_snapshot",
        "create_order_intent",
    } <= written
    assert t["skipped_writes"] == []
    assert t["result"]["skipped_writes"] == []
    (signal_call,) = rs.calls(t, "signals.generate_core_signals")
    assert signal_call["kwargs"]["persist_tape"] is True
    assert len(rs.calls(t, "tape.refresh_benchmark_tape")) == 1


# =========================================================================== rule 7


def _failure_tail(t: dict) -> list[str]:
    """The targets from the first intent on, minus the harness's noise (clock, polls, attach)."""
    return [
        c["target"]
        for c in t["calls"]
        if c["target"].startswith(("persist.create", "lock.", "broker.submit", "notifier.send"))
    ]


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_rule7a_an_exception_on_submit_ends_the_run_after_the_intent_is_saved(impl_name, side):
    """Scenario `submit_error`: the first order raises on submit; nothing is submitted after it.

    The calls are `create_order_intent`, `lock.verify`, `submit_*` (raises), then the failure
    notice naming the stage, then the lock is released. Pinned today by
    `test_order_failure_stops_later_buys` and `test_run_stops_after_ambiguous_order_failure`
    (test_trader.py:2411, 3701). The SELL case fails the first sell of `plain_buys_and_sells`.
    """
    if side == "BUY":
        scenario = SCENARIOS["submit_error"]
        failing = "broker.submit_buy"
    else:
        base = SCENARIOS["plain_buys_and_sells"]
        scenario = rs.from_scenario(
            "plain_buys_and_sells",
            replies={**base.replies, "submit_sell": [RuntimeError("insufficient buying power")]},
        )
        failing = "broker.submit_sell"
    t = rs.run(impl_name, scenario)
    assert t["raised"] == "RuntimeError: insufficient buying power"
    assert t["result"] is None
    assert _failure_tail(t) == [
        "lock.enter",
        "persist.create_order_intent",
        "lock.verify",
        failing,
        "notifier.send",
        "lock.exit",
    ]
    assert rs.messages(t)[-1] == (
        f"Order {rs.client_id(1)} failed at submit; investigation required"
    )
    assert len(rs.calls(t, "persist.create_order_intent")) == 1
    assert len(rs.submissions(t)) == 1


def test_rule7b_a_rejected_order_settles_normally_and_other_exits_stay_eligible(impl_name):
    """A rejected exit settles and the other exit is still submitted; a rejected buy settles late.

    Exit case: scenario `sell_rejected_by_broker` (AAPL rejected, MSFT fills), pinned today by
    `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit` (test_trader.py:3752): both
    sells go out before any poll and `rejected` reaches `settle_order_intent`. Buy case: the
    first of the two buys of `plain_buys_and_sells` is rejected; the second is still submitted,
    and the rejection is settled at the reconcile after the buys.
    """
    t = rs.run(impl_name, SCENARIOS["sell_rejected_by_broker"])
    assert rs.submissions(t) == [("SELL", "AAPL", 10.0), ("SELL", "MSFT", 5.0)]
    first_poll = next(i for i, c in enumerate(t["calls"]) if c["target"] == "broker.get_order")
    last_submit = max(i for i, c in enumerate(t["calls"]) if c["target"] == "broker.submit_sell")
    assert last_submit < first_poll
    assert rs.intents_settled(t) == [
        (rs.client_id(1), 0.0, None, "rejected"),
        (rs.client_id(2), 5.0, 100.0, "filled"),
    ]

    base = SCENARIOS["plain_buys_and_sells"]
    buys = rs.from_scenario(
        "plain_buys_and_sells",
        fills={
            **base.fills,
            "NVDA": [{"status": "rejected", "filled_qty": 0.0, "filled_avg_price": None}],
        },
    )
    b = rs.run(impl_name, buys)
    assert rs.submissions(b, "BUY") == [("BUY", "NVDA", 3300.0), ("BUY", "GOOGL", 3300.0)]
    settled = {s[0]: s for s in rs.intents_settled(b)}
    assert settled[rs.client_id(2)] == (rs.client_id(2), 0.0, None, "rejected")
    assert settled[rs.client_id(3)] == (rs.client_id(3), 33.0, 100.0, "filled")
    index = {
        c["args"][0]: i
        for i, c in enumerate(b["calls"])
        if c["target"] == "persist.settle_order_intent"
    }
    last_buy = max(i for i, c in enumerate(b["calls"]) if c["target"] == "broker.submit_buy")
    assert index[rs.client_id(2)] > last_buy
    assert b["result"]["buys"] == ["NVDA", "GOOGL"]


def test_a_loss_halt_clears_the_buys_but_leaves_every_exit_eligible(impl_name):
    """Scenario `sell_rejected_by_broker` with both sells filling: only the drawdown halt blocks.

    Peak 120,000 against 100,000 is a drawdown halt: both exits go out and fill, NVDA is not
    bought, and the daily summary carries the halt. Pinned today by
    `test_loss_limit_allows_protective_exit_but_blocks_new_buys`. This is the test that fails a
    mutation dropping the clearing of the buys on a loss halt (SM15), which the rejected-exit
    case above hides behind its own "not fully filled" block.
    """
    scenario = rs.from_scenario("sell_rejected_by_broker", fills={})
    t = rs.run(impl_name, scenario)
    assert rs.submissions(t) == [("SELL", "AAPL", 10.0), ("SELL", "MSFT", 5.0)]
    assert t["result"]["buys"] == []
    assert t["result"]["sells"] == ["AAPL", "MSFT"]
    assert len(t["result"]["errors"]) == 1
    assert t["result"]["errors"][0].startswith("Max drawdown breached: 16.7%")
    assert any("HALTED buys; exits only" in m for m in rs.messages(t))


# =========================================================================== characterizations


def _live_exit(position: dict, *, dry: bool = False) -> object:
    return rs.from_scenario(
        "plain_buys_and_sells",
        dry_run=dry,
        membership=DRY_MEMBERSHIP if dry else "db",
        signals=rs.blend([], ["AAPL"]),
        replies=rs.replies(rs.acct(), {"AAPL": position}),
    )


def test_live_exit_reads_market_value_only_after_the_order_is_submitted(impl_name):
    """Quirk 11: a position record without `market_value` raises after the live order went out.

    Both the strategy exit and the catastrophe stop read `qty` before the submit and the book
    amount after it. `submit_sell` was called once and `attach_broker_order` ran, then
    `KeyError('market_value')` reaches the caller; no poll, no stop or sell notice follows.
    """
    exit_run = rs.run_raw(impl_name, _live_exit(_position_without("market_value")))
    assert repr(exit_run.error) == "KeyError('market_value')"
    assert exit_run.targets_before_release[-3:] == [
        "lock.verify",
        "broker.submit_sell",
        "persist.attach_broker_order",
    ]
    assert exit_run.targets.count("broker.submit_sell") == 1
    assert exit_run.targets.count("broker.get_order") == 0
    assert not any("sell" in m.lower() or "stop" in m.lower() for m in exit_run.messages())

    stop = rs.from_scenario(
        "catastrophe_stop",
        signals=rs.blend([], []),
        replies=rs.replies(
            rs.acct(), {"NXPI": _position_without("market_value", **STOPPED_POSITION)}
        ),
    )
    stop_run = rs.run_raw(impl_name, stop)
    assert repr(stop_run.error) == "KeyError('market_value')"
    assert stop_run.targets_before_release[-3:] == [
        "lock.verify",
        "broker.submit_sell",
        "persist.attach_broker_order",
    ]
    assert stop_run.targets.count("broker.submit_sell") == 1
    assert stop_run.calls("persist.create_order_intent")[0]["args"][6] == "catastrophe_stop"
    assert not any("Catastrophe stop" in m for m in stop_run.messages())


def test_dry_run_exit_never_reads_market_value(impl_name):
    """The same inputs in a dry run complete: the DRY RUN notice, no raise.

    This is what forbids a planner from reading `market_value` into an eager `OrderIntent`.
    """
    run = rs.run_raw(impl_name, _live_exit(_position_without("market_value"), dry=True))
    assert run.error is None
    assert run.result["sells"] == ["AAPL"]
    assert "<b>🔍 DRY RUN sell:</b> AAPL (qty 10.0)" in run.messages()

    stop = rs.from_scenario(
        "catastrophe_stop",
        dry_run=True,
        membership=DRY_MEMBERSHIP,
        signals=rs.blend([], []),
        replies=rs.replies(
            rs.acct(), {"NXPI": _position_without("market_value", **STOPPED_POSITION)}
        ),
    )
    stop_run = rs.run_raw(impl_name, stop)
    assert stop_run.error is None
    assert stop_run.result["sells"] == ["NXPI"]
    assert (
        "<b>🛑 DRY RUN catastrophe stop:</b> NXPI (qty 10.0, unrealized -40.0% breached "
        "-25.0% floor)" in stop_run.messages()
    )


def test_missing_qty_raises_before_any_order(impl_name):
    """A held exit candidate without `qty` raises `KeyError('qty')` in the validation loop."""
    scenario = rs.from_scenario(
        "plain_buys_and_sells",
        signals=rs.blend(["MSFT"], ["AAPL"]),
        replies=rs.replies(rs.acct(), {"AAPL": _position_without("qty")}),
    )
    run = rs.run_raw(impl_name, scenario)
    assert repr(run.error) == "KeyError('qty')"
    assert rs.targets({"calls": run.run.recorder.calls}).count("broker.submit_sell") == 0
    assert run.calls("persist.create_order_intent") == []
    assert run.calls("broker.submit_buy") == []


# ------------------------------------------------------------------ unrealized_pct call sequence

#: Cost bases tell the three positions apart inside the wrapper (the run copies the records).
_BASIS_NAMES = {1000.0: "HEALTHY", 1001.0: "BREACH", 1002.0: "PEND"}


def _unrealized_pct_scenario() -> object:
    base = SCENARIOS["catastrophe_stop_pending_order"]
    return rs.from_scenario(
        "catastrophe_stop_pending_order",
        signals=rs.blend([], []),
        persist={**base.persist, "get_pending_orders": [_pending("pend-1", "PEND", "SELL")]},
        existing_orders=[{**base.existing_orders[0], "symbol": "PEND", "qty": 10.0}],
        replies=rs.replies(
            rs.acct(),
            {
                "HEALTHY": rs.pos(10.0, 100.0, cost_basis=1000.0, unrealized_pl=-50.0),
                "BREACH": rs.pos(10.0, 60.0, cost_basis=1001.0, unrealized_pl=-400.0),
                "PEND": rs.pos(10.0, 60.0, cost_basis=1002.0, unrealized_pl=-400.0),
            },
        ),
    )


#: One call per held position (the unassessable scan), one per assessable position (inside
#: `find_catastrophe_stops`), then one per breaching symbol with a valid quantity and no pending
#: order (the per-stop percentage; PEND has a pending order and is skipped before it).
_UNREALIZED_PCT_SEQUENCE = [
    "HEALTHY", "BREACH", "PEND",
    "HEALTHY", "BREACH", "PEND",
    "BREACH",
]  # fmt: skip
_UNREALIZED_PCT_VALUES = [-0.05, -0.4, -0.4, -0.05, -0.4, -0.4, -0.4]


def test_unrealized_pct_call_sequence_is_pinned(impl_name):
    """The literal call order of `catastrophe_stop.unrealized_pct` in one run (patched the way
    test_trader.py:3810 does), then a `Mock(side_effect=[...])` of exactly that length passes."""
    real = catastrophe_stop.unrealized_pct
    seen: list[str] = []

    def recording(position: dict) -> float | None:
        seen.append(_BASIS_NAMES[position["cost_basis"]])
        return real(position)

    with patch("ggTrader.paper.trader.catastrophe_stop.unrealized_pct", recording):
        run = rs.run_raw(impl_name, _unrealized_pct_scenario())
    assert run.error is None
    assert run.result["sells"] == ["BREACH"]
    assert seen == _UNREALIZED_PCT_SEQUENCE

    mock = Mock(side_effect=_UNREALIZED_PCT_VALUES)
    with patch("ggTrader.paper.trader.catastrophe_stop.unrealized_pct", mock):
        exact = rs.run_raw(impl_name, _unrealized_pct_scenario())
    assert exact.error is None
    assert mock.call_count == len(_UNREALIZED_PCT_VALUES)


@pytest.mark.parametrize(
    ("length", "error_type", "message", "cause_type"),
    [
        pytest.param(2, StopIteration, "", None, id="short_in_the_unassessable_scan"),
        pytest.param(
            5,
            RuntimeError,
            "generator raised StopIteration",
            StopIteration,
            id="short_inside_find_catastrophe_stops",
        ),
        pytest.param(6, StopIteration, "", None, id="short_in_the_per_stop_call"),
    ],
)
def test_unrealized_pct_one_value_short_keeps_each_sites_exception(
    impl_name, length, error_type, message, cause_type
):
    """Quirk 12: one value short of the sequence, at each of the three call sites.

    Measured on the untouched code. In a plain statement of `_run` the `StopIteration` reaches
    the caller as it is; inside `find_catastrophe_stops` (a `sorted` over a generator
    expression) it reaches the caller as `RuntimeError("generator raised StopIteration")` with
    the `StopIteration` as `__cause__` (PEP 479, in the collaborator's own frame). Each length
    cuts the seven-value sequence before the last call of one site: 2 values end the scan (call
    3), 5 end `find_catastrophe_stops` (call 6), 6 end the per-stop call (call 7).
    """
    mock = Mock(side_effect=_UNREALIZED_PCT_VALUES[:length])
    with patch("ggTrader.paper.trader.catastrophe_stop.unrealized_pct", mock):
        run = rs.run_raw(impl_name, _unrealized_pct_scenario())
    assert type(run.error) is error_type
    assert str(run.error) == message
    if cause_type is None:
        assert run.error.__cause__ is None
    else:
        assert type(run.error.__cause__) is cause_type


def test_a_stop_finder_that_raises_stop_iteration_in_its_own_frame_is_a_runtime_error(impl_name):
    """A fourth case: a patched `find_catastrophe_stops` generator that raises `StopIteration`.

    Its body is a nested generator frame, so the caller gets `RuntimeError` with that cause.
    """

    def finder(positions: dict, threshold: float):
        raise StopIteration
        yield  # pragma: no cover  (makes this a generator function)

    with patch.object(catastrophe_stop, "find_catastrophe_stops", finder):
        run = rs.run_raw(impl_name, SCENARIOS["catastrophe_stop"])
    assert type(run.error) is RuntimeError
    assert str(run.error) == "generator raised StopIteration"
    assert type(run.error.__cause__) is StopIteration


# ------------------------------------------------------------------ late failures (Review Focus 7)


def _two_stops(second_has_qty: bool, *, dry: bool = False) -> object:
    second = rs.pos(10.0, 60.0, **STOPPED_POSITION)
    if not second_has_qty:
        del second["qty"]
    return rs.from_scenario(
        "catastrophe_stop",
        dry_run=dry,
        membership=DRY_MEMBERSHIP if dry else "db",
        signals=rs.blend([], []),
        replies=rs.replies(
            rs.acct(), {"A_STOP": rs.pos(10.0, 60.0, **STOPPED_POSITION), "B_STOP": second}
        ),
    )


def test_a_later_stop_without_qty_raises_after_the_first_stop_order_went_out(impl_name):
    """Two breaching positions, the second (in sorted order) without `qty`.

    Live: the first stop was submitted, its intent saved and attached, its notice sent, then
    `KeyError('qty')` reaches the caller; no second order. A dry run: the first stop's DRY RUN
    notice is sent, then the same `KeyError`.
    """
    run = rs.run_raw(impl_name, _two_stops(False))
    assert repr(run.error) == "KeyError('qty')"
    assert [c["args"][3] for c in run.calls("persist.create_order_intent")] == ["A_STOP"]
    assert [c["args"][0] for c in run.calls("broker.submit_sell")] == ["A_STOP"]
    assert len(run.calls("persist.attach_broker_order")) == 1
    assert run.targets_before_release[-4:] == [
        "lock.verify",
        "broker.submit_sell",
        "persist.attach_broker_order",
        "notifier.send",
    ]
    assert run.messages()[-1].startswith("<b>🛑 Catastrophe stop:</b> A_STOP unrealized -40.0%")

    dry = rs.run_raw(impl_name, _two_stops(False, dry=True))
    assert repr(dry.error) == "KeyError('qty')"
    assert dry.calls("broker.submit_sell") == []
    assert dry.messages() == [
        "<b>🛑 DRY RUN catastrophe stop:</b> A_STOP (qty 10.0, unrealized -40.0% breached "
        "-25.0% floor)"
    ]


def test_exit_validation_without_qty_raises_after_the_stop_orders(impl_name):
    """A stop that submits, then an exit signal on a held position without `qty`.

    The run raises `KeyError('qty')` after the stop order, with no ordinary exit submitted.
    """
    scenario = rs.from_scenario(
        "catastrophe_stop",
        signals=rs.blend([], ["EXIT"]),
        replies=rs.replies(
            rs.acct(),
            {"A_STOP": rs.pos(10.0, 60.0, **STOPPED_POSITION), "EXIT": _position_without("qty")},
        ),
    )
    run = rs.run_raw(impl_name, scenario)
    assert repr(run.error) == "KeyError('qty')"
    assert [c["args"][0] for c in run.calls("broker.submit_sell")] == ["A_STOP"]
    assert run.targets_before_release[-3:] == [
        "broker.submit_sell",
        "persist.attach_broker_order",
        "notifier.send",
    ]
    assert run.messages()[-1].startswith("<b>🛑 Catastrophe stop:</b> A_STOP")


def test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy(impl_name):
    """Two sleeves, the second not in `weights`, both with buy candidates.

    The first sleeve's buy was submitted, then `KeyError` for the second sleeve's name reaches
    the caller (at `sleeve_open_count[universe]`); no second order.
    """
    signals = rs.blend(["B1"], [])
    signals["sleeves"]["midcap"] = rs.sleeve(["B2"], [], "2026-06-19")
    scenario = rs.from_scenario(
        "plain_buys_and_sells", signals=signals, replies=rs.replies(rs.acct(), {})
    )
    assert list(signals["weights"]) == ["sp500"]
    run = rs.run_raw(impl_name, scenario)
    assert repr(run.error) == "KeyError('midcap')"
    assert [c["args"][0] for c in run.calls("broker.submit_buy")] == ["B1"]
    assert run.targets_before_release[-3:] == [
        "lock.verify",
        "broker.submit_buy",
        "persist.attach_broker_order",
    ]


# ------------------------------------------------------------------ the signal try (Review Focus 8)


@pytest.mark.parametrize("dry", [False, True], ids=["live", "read_only_dry_run"])
def test_a_membership_source_failure_sends_the_signal_generation_error_alert(impl_name, dry):
    """A failing membership source is caught by the signal `try`: alert, raise, nothing else.

    The source is an argument of the `generate_core_signals(...)` call, so the failure lands in
    the `except Exception` of the signals block. Runs in the event environment (a `Scenario`
    cannot express a failing source, and a patch made outside `run_scenario` is overwritten).
    """
    scenario = SCENARIOS["dry_run_read_only" if dry else "plain_buys_and_sells"]
    run = rs.run_raw(impl_name, scenario, membership_failure=ValueError("boom"))
    assert type(run.error) is ValueError
    assert str(run.error) == "boom"
    assert run.messages() == ["Paper trading failed: signal generation error\nboom"]
    assert run.calls("signals.generate_core_signals") == []
    after = run.targets[run.targets.index("hook.membership_source") + 1 :]
    assert after == ["notifier.send"] if dry else after == ["notifier.send", "lock.exit"]


def test_a_signal_gate_error_keeps_its_own_notice(impl_name):
    """Scenario `signal_gate_error`: a `SignalGateError` is not the generic generation alert."""
    run = rs.run_raw(impl_name, SCENARIOS["signal_gate_error"])
    assert run.messages() == ["<b>🛑 Signal gate — no orders placed:</b>\n[empty_frame] no bars"]
    assert type(run.error).__name__ == "SignalGateError"


# ------------------------------------------------------------------ SM11: stops before the sweep


def test_stops_are_executed_before_the_sweep_sell_is_sized(impl_name):
    """A stop and a funding sweep sell in one run (`catastrophe_stop` with `sweep_funding_sell`).

    MAX_POSITIONS 3; held NXPI (stopped), F1, F2; three buy signals; cash 1,000 against the
    5,000 reserve. After the stop leaves `strategy_positions` one slot is free, so the day's
    estimated buys are 3,300 and the sweep sells 18.25 SPY (7,300). Sized before the stop it would
    see no free slot and sell only the 4,000 that restores the reserve (10 shares). The stop's
    `submit_sell` precedes the SPY sell.
    """
    scenario = rs.from_scenario(
        "sweep_funding_sell",
        env=rs.SWEEP_STOP_ENV,
        signals=rs.blend(["M1", "M2", "M3"], []),
        fills={"SPY:SELL": [{"status": "filled", "filled_qty": 18.25, "filled_avg_price": 400.0}]},
        replies=rs.replies(
            rs.acct(100_000.0, 1_000.0),
            {
                "NXPI": rs.pos(100.0, 70.0, cost_basis=10_000.0, unrealized_pl=-3000.0),
                "F1": rs.pos(1.0, 100.0),
                "F2": rs.pos(1.0, 100.0),
                "SPY": rs.pos(247.5, 400.0),
            },
        ),
    )
    t = rs.run(impl_name, scenario)
    assert rs.submissions(t, "SELL") == [("SELL", "NXPI", 100.0), ("SELL", "SPY", 18.25)]
    sells = [c for c in t["calls"] if c["target"] == "broker.submit_sell"]
    assert [c["args"][0] for c in sells] == ["NXPI", "SPY"]
    reasons = [c["args"][6] for c in rs.calls(t, "persist.create_order_intent")]
    assert reasons[:2] == ["catastrophe_stop", "cash_sweep"]


# =========================================================================== hook order

#: The call targets the hook-order test pins, in the order they happened.
_HOOK_ORDER_TARGETS = (
    "hook.",
    "membership.",
    "signals.",
    "tape.",
    "lock.verify",
    "broker.submit",
    "broker.get_account",
    "broker.get_positions",
)

#: Measured on the untouched code for `plain_buys_and_sells`: `today_et` once at the freshness
#: gate and once per submitted order (3), `new_uuid` once per submitted order (3).
_LIVE_RUN_HOOK_ORDER = [
    "hook.membership_source",
    "membership.live_membership_source",
    "signals.generate_core_signals",
    "hook.now_pt",
    "tape.refresh_benchmark_tape",
    "hook.today_et",
    "broker.get_account",
    "broker.get_positions",
    "hook.new_uuid",
    "hook.today_et",
    "lock.verify",
    "broker.submit_sell",
    "hook.new_uuid",
    "hook.today_et",
    "lock.verify",
    "broker.submit_buy",
    "hook.new_uuid",
    "hook.today_et",
    "lock.verify",
    "broker.submit_buy",
    "broker.get_account",
    "broker.get_positions",
]


def test_hook_call_order_for_a_live_run(impl_name):
    """The order of the membership source, signal generation, `now_pt`, the tape refresh,
    `today_et` and `new_uuid` against the broker calls, for `plain_buys_and_sells`."""
    run = rs.run_raw(impl_name, SCENARIOS["plain_buys_and_sells"])
    assert run.error is None
    observed = [t for t in run.targets if t.startswith(_HOOK_ORDER_TARGETS)]
    assert observed == _LIVE_RUN_HOOK_ORDER
    assert observed.count("hook.today_et") == 1 + 3
    assert observed.count("hook.new_uuid") == 3
    assert observed.count("hook.now_pt") == 1
    assert observed.count("hook.membership_source") == 1


@pytest.mark.parametrize("name", ["plain_buys_and_sells", "late_run_live"])
def test_late_run_gate_reads_the_clock_after_signal_generation(impl_name, name):
    """`now_pt` is called after `generate_core_signals` returned, once (late run or not)."""
    run = rs.run_raw(impl_name, SCENARIOS[name])
    assert run.error is None
    targets = run.targets
    assert targets.count("hook.now_pt") == 1
    assert targets.index("hook.now_pt") > targets.index("signals.generate_core_signals")
    assert targets.index("hook.membership_source") < targets.index("signals.generate_core_signals")
