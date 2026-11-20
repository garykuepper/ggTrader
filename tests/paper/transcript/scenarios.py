"""The recorded paper-trader scenarios; each docstring says what its golden must show.

Most scenarios are adapted from an existing test in ``tests/paper/test_trader.py`` (cited in
the docstring). Those marked "Newly derived" have no run-level test there: they come from
reading ``trader.py`` (the polling, startup-recovery and some failure paths). Client order ids
are pinned (``00000000-0000-0000-0000-00000000000n``, n counting from 1 in submission order)
and broker order ids are ``ord-n`` in the same order. The goldens record what the trader does
today: a scenario's docstring is the reviewer's expectation and
``test_golden_shows_scenario_behavior`` is what pins it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone

from ggTrader.paper.gates import SignalGateError
from tests.paper.transcript.harness import PT, Scenario, Seq

#: The membership spec the root conftest selects: dry runs may rehearse any source.
FIXTURE_SPEC = "fixture:sp500_v1_2026-06-02"

#: How many scenarios the suite records: the 38 of the first plan, then the follow-up rounds.
EXPECTED_SCENARIO_COUNT = 103

SCENARIOS: dict[str, Scenario] = {}

#: Reads the trader makes that every scenario needs an answer for; a scenario overrides a few.
PERSIST_DEFAULTS: dict[str, object] = {
    "get_accrued_dividend_keys": set(),
    "get_earliest_snapshot": None,
    "get_last_buy_dates": {},
    "get_latest_snapshot": None,
    "get_open_split_states": {},
    "get_peak_value": None,
    "get_pending_orders": [],
    "get_snapshot_history": [],
    "get_total_dividend_accrual": 0.0,
    "get_trade_history_dates": [],
}

_CLOCK_OPEN = {"is_open": True, "next_close": "2026-06-19 16:00:00-04:00"}
_LATE_NOW = datetime(2026, 6, 19, 13, 10, tzinfo=PT)
_LIVE_ENV = {"MAX_POSITIONS": "3"}
_SWEEP_ENV = {"CASH_SWEEP_ENABLED": "true"}


def _acct(
    portfolio_value: float = 100_000.0,
    cash: float = 50_000.0,
    last_equity: float | None = None,
) -> dict:
    """An account read in the shape `AlpacaBroker.get_account` returns."""
    return {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "last_equity": portfolio_value if last_equity is None else last_equity,
        "buying_power": cash * 2,
        "multiplier": 1.0,
    }


def _pos(
    qty: float,
    price: float,
    *,
    cost_basis: float | None = None,
    unrealized_pl: float | None = None,
) -> dict:
    """A position in the shape `AlpacaBroker.get_positions` returns (flat unless told)."""
    market_value = qty * price
    cost = market_value if cost_basis is None else cost_basis
    pl = market_value - cost if unrealized_pl is None else unrealized_pl
    return {
        "qty": qty,
        "market_value": market_value,
        "current_price": price,
        "avg_entry": cost / qty if qty else 0.0,
        "unrealized_pl": pl,
        "unrealized_plpc": pl / cost if cost else 0.0,
        "change_today": 0.0,
        "cost_basis": cost,
    }


def _sleeve(buys: list[str], sells: list[str], as_of: str, last_exit: dict | None = None) -> dict:
    return {
        "buys": buys,
        "sells": sells,
        "as_of": as_of,
        "universe_size": 100,
        "last_exit": last_exit or {},
    }


def _blend(
    buys: list[str],
    sells: list[str],
    as_of: str = "2026-06-19",
    last_exit: dict | None = None,
) -> dict:
    """`generate_core_signals`' real single-sleeve shape (as `_blend` in test_trader.py)."""
    return {
        "sleeves": {"sp500": _sleeve(buys, sells, as_of, last_exit)},
        "weights": {"sp500": 1.0},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }


def _replies(account: object, positions: object, clock: object = _CLOCK_OPEN) -> dict:
    """Broker replies: a scalar entry is used for every call, a list is per call."""
    return {
        "get_account": account if isinstance(account, list) else [account],
        "get_positions": positions if isinstance(positions, list) else [positions],
        "get_clock": clock if isinstance(clock, list) else [clock],
    }


def _pending(order_id: str, symbol: str, side: str) -> dict:
    """A legacy pending-order row (``created_at=None``: never old enough to be flagged stale)."""
    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "created_at": None,
        "flagged_stale": False,
    }


def scenario(build: Callable[[], dict]) -> Callable[[], dict]:
    """Register a scenario: the function name is its name, the function returns the fields."""
    fields = build()
    persist = {**PERSIST_DEFAULTS, **fields.pop("persist", {})}
    name = build.__name__
    if name in SCENARIOS:
        raise ValueError(f"duplicate scenario {name}")
    SCENARIOS[name] = Scenario(name=name, persist=persist, **fields)
    return build


# --------------------------------------------------------------------------- ordinary runs


@scenario
def plain_buys_and_sells() -> dict:
    """A live run that sells one holding and buys two names.

    Adapted from `TestSellExits`, `TestBuyEntries` and `test_loss_halt_reconciles_rejected_exit`
    (the `_intent_store` round trip). Terminal outcome: success, ``sells == ["AAPL"]``,
    ``buys == ["NVDA", "GOOGL"]``, no errors. Discriminating calls in order: SELL AAPL qty 10.0
    (the held quantity), then BUY NVDA and BUY GOOGL, each notional 3300.0 (3.3% of 100,000).
    Intents: each order goes create -> attach -> settle once, so no intent is left unresolved;
    the sell settles before the buys are created, the buys settle after the buy poll.
    Reporting: the sell fills at 150.0; the account is read twice with different equity
    (100,000.0 then 100,053.0) and the final positions are KO, NVDA, GOOGL. The daily summary
    value, the daily P&L (+553.0 against the 99,500.0 snapshot), the total P&L (+53.0) and the
    ``log_snapshot`` payload (value 100053.0, cash 92200.0, the three final positions) all come
    from the final read.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["NVDA", "GOOGL"], ["AAPL"]),
        # The AAPL sale brings 1,500 and the two buys cost 6,600: cash 97,300 -> 92,200. The
        # final account and positions differ from the first read, so the summary, the P&L
        # and the snapshot payload must be built from the final read.
        fills={"AAPL": [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.0}]},
        persist={"get_latest_snapshot": 99_500.0, "get_earliest_snapshot": 100_000.0},
        replies=_replies(
            [
                _acct(100_000.0, 97_300.0),
                _acct(100_053.0, 92_200.0, last_equity=100_000.0),
            ],
            [
                {
                    "AAPL": _pos(10.0, 150.0, cost_basis=1450.0),
                    "KO": _pos(20.0, 60.0),
                },
                {
                    "KO": _pos(20.0, 61.0, cost_basis=1200.0),
                    "NVDA": _pos(33.0, 102.0, cost_basis=3300.0),
                    "GOOGL": _pos(33.0, 99.0, cost_basis=3300.0),
                },
            ],
        ),
    )


@scenario
def total_pnl_two_bases() -> dict:
    """A $20,053 total is +20.15% of prior snapshot and +25.07% of capital.

    Intents: 3 settled, none unresolved.
    """
    result = plain_buys_and_sells()
    result["persist"]["get_earliest_snapshot"] = 80_000.0
    return result


@scenario
def market_closed() -> dict:
    """The broker clock says the market is closed: the run stops before any signal work.

    From `TestMarketGates.test_market_closed_skips_run_entirely`. Terminal outcome: success with
    ``{"buys": [], "sells": [], "errors": []}``; the only notifier text is "Market closed". No
    signal call, no order, no intent row created.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        replies=_replies(_acct(), {}, {"is_open": False}),
    )


@scenario
def late_run_live() -> dict:
    """A live run that starts at 13:10 PT, after the 12:55 cutoff: no orders.

    From `TestFailClosed.test_late_run_places_no_orders`. Terminal outcome: success with
    ``gate == "late_run"`` and empty buys/sells; "Late run" is sent; the benchmark tape refresh
    and the account read never happen.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        now_pt=_LATE_NOW,
        signals=_blend(["MSFT"], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def late_run_dry() -> dict:
    """A dry run at 13:10 PT only warns and still plans its orders.

    From `TestFailClosed.test_late_dry_run_only_warns`. Terminal outcome: success, a "DRY RUN
    late-run notice", then "DRY RUN sell: AAPL (qty 10.0)" and "DRY RUN buy: MSFT ($3300)". No
    broker submit call, no persist write call; ``skipped_writes`` lists the four skipped writes.
    Intents: none created (a dry run records no intent).
    """
    return dict(
        dry_run=True,
        membership=FIXTURE_SPEC,
        env=_LIVE_ENV,
        now_pt=_LATE_NOW,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


# --------------------------------------------------------------------------- signal timestamps


@scenario
def stale_as_of() -> dict:
    """Signals dated 9 days before today: the freshness gate stops the run.

    From `TestMarketGates.test_stale_as_of_skips_trading`. Terminal outcome: success with empty
    buys/sells/errors; "Stale signal data: sp500 as_of=2026-06-10 is 9d old" is sent; no account
    read and no order.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"], as_of="2026-06-10"),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


@scenario
def future_as_of() -> dict:
    """Signals dated 6 days after today: the freshness gate stops the run.

    From `TestMarketGates.test_future_as_of_blocks_buy_and_sell`. Terminal outcome: success with
    empty buys/sells/errors; the text says "6d in the future"; no account read and no order.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"], as_of="2026-06-25"),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


@scenario
def unparseable_as_of() -> dict:
    """An ``as_of`` that is not a date: the freshness gate stops the run.

    From `TestMarketGates.test_unparseable_as_of_blocks_trading`. Terminal outcome: success with
    empty buys/sells/errors; the text says "unparseable"; no account read and no order.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"], as_of="not-a-date"),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


# --------------------------------------------------------------------------- untradable names


@scenario
def untradable_holding() -> dict:
    """A held name the broker marks ``tradable: false`` is never sold and frees its slot.

    From `TestUntradableHoldings.test_untradable_position_is_not_sold_and_frees_its_slot`
    (AVB merged into EQR). With MAX_POSITIONS=1 and only AVB held, a frozen AVB must not fill
    the only slot. Terminal outcome: success, ``sells == []``, ``buys == ["MSFT"]``,
    ``untradable == ["AVB"]``. Discriminating calls: ``get_untradable(["AVB", "MSFT"])``; no
    submit_sell; one BUY MSFT notional 3300.0; "Untradable holdings: AVB" is sent.
    Intents: 1 settled (the MSFT buy), none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "1"},
        untradable={"AVB"},
        signals=_blend(["MSFT"], ["AVB"]),
        replies=_replies(_acct(), {"AVB": _pos(4.05, 184.0, unrealized_pl=0.0)}),
    )


@scenario
def untradable_candidate() -> dict:
    """A buy candidate the broker will not trade is filtered before any order.

    From `TestUntradableAvb.test_untradable_buy_candidate_is_filtered_before_any_order`.
    Terminal outcome: success, ``buys == ["MSFT"]``, ``untradable == ["AVB"]``. Discriminating
    calls: ``get_untradable(["AVB", "MSFT"])``, one BUY MSFT notional 3300.0 and no AVB order;
    "Untradable buy candidates skipped: AVB" is sent.
    Intents: 1 settled (the MSFT buy), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        untradable={"AVB"},
        signals=_blend(["AVB", "MSFT"], []),
        replies=_replies(_acct(), {}),
    )


# --------------------------------------------------------------------------- corporate actions

_MNST_QTY = 20.80410098
_MNST_PL = _MNST_QTY * 47.77 - 1887.11  # the broker's phantom loss, -893.30


@scenario
def split_correction() -> dict:
    """An unapplied 2-for-1 split: reports are corrected, orders use the broker's quantity.

    From `TestSellExits.test_sell_submits_brokers_uncorrected_qty_not_split_adjusted` and
    `TestDailyPnl.test_unapplied_split_corrects_the_daily_summary` (the MNST incident). The
    broker holds 20.80410098 shares; the split-adjusted quantity would be 41.60820196. Terminal
    outcome: success, ``sells == ["MNST"]``. Discriminating calls: ``save_split_correction``
    ("MNST", "2026-06-10", 2.0); SELL MNST qty 20.80410098 (not 41.608...); the daily summary
    carries ``split_corrections == {"MNST": 2.0}`` and the booked equity 100000.0. The sell
    settles; no intent is left unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["MNST"]),
        splits={
            "corp_splits": {"MNST": [(date(2026, 6, 10), 2.0)]},
            "activity_applied": set(),
        },
        replies=_replies(
            _acct(),
            [
                {
                    "MNST": _pos(
                        _MNST_QTY,
                        47.77,
                        cost_basis=1887.11,
                        unrealized_pl=_MNST_QTY * 47.77 - 1887.11,
                    ),
                    "VZ": _pos(77.05, 48.59, cost_basis=3378.63, unrealized_pl=365.22),
                },
                {"VZ": _pos(77.05, 48.59, cost_basis=3378.63, unrealized_pl=365.22)},
            ],
        ),
        persist={"get_latest_snapshot": 99_000.0, "get_earliest_snapshot": 100_000.0},
    )


@scenario
def new_dividend_accrual() -> dict:
    """A cash dividend the broker never credited is accrued and added to reported equity.

    From `TestDividendAccrual.test_held_dividend_is_accrued_and_added_to_reported_equity`. The
    dividend total is read twice (before the write: 0.0, after: 51.8...). Terminal outcome:
    success with no orders. Discriminating calls: ``log_dividend_accrual("VZ", "2026-06-01",
    0.6725, 77.05, 51.8...)``; the daily summary's ``dividend_total`` is the second read and its
    portfolio value is 100000 plus that accrual.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], []),
        dividends={
            "corp_dividends": {"VZ": [(date(2026, 6, 1), 0.6725)]},
            "credited_keys": set(),
        },
        replies=_replies(
            _acct(),
            {"VZ": _pos(77.05, 48.59, cost_basis=3378.63, unrealized_pl=365.22)},
        ),
        persist={
            "get_snapshot_history": [(date(2026, 5, 20), {"VZ": {"qty": 77.05}})],
            "get_total_dividend_accrual": Seq(0.0, 77.05 * 0.6725),
            "get_latest_snapshot": 99_000.0,
        },
    )


# --------------------------------------------------------------------------- risk halts


@scenario
def drawdown_halt() -> dict:
    """Value 16.7% under the persisted peak: buys halt, the exit still goes.

    From `test_loss_limit_allows_protective_exit_but_blocks_new_buys` and
    `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit`. Terminal outcome: success,
    ``sells == ["AAPL"]``, ``buys == []``, errors start with "Max drawdown breached: 16.7%".
    Discriminating calls: no ``submit_buy`` and no ``save_peak_value``; one SELL AAPL qty 10.0;
    the notifier gets "HALTED buys; exits only". The sell settles.
    Intents: 1 settled (the AAPL sell), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        persist={"get_peak_value": 120_000.0},
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


@scenario
def daily_loss_halt() -> dict:
    """Value 4.8% below the broker's previous close: buys halt, the exit still goes.

    From `TestPersistedDrawdown.test_daily_risk_halts_when_broker_previous_close_shows_loss` and
    `test_both_loss_limits_allow_exit_and_preserve_peak`. Terminal outcome: success,
    ``sells == ["AAPL"]``, ``buys == []``, errors start with "Daily loss limit: 4.8%".
    Discriminating calls: ``save_peak_value(100000.0)`` happens (no drawdown halt), no
    ``submit_buy``, one SELL AAPL qty 10.0.
    Intents: 1 settled (the AAPL sell), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct(last_equity=105_000.0), {"AAPL": _pos(10.0, 150.0)}),
    )


@scenario
def daily_loss_halt_with_split() -> dict:
    """A daily-loss halt with a split correction on record says so and still blocks buys.

    From `TestUnappliedSplitAndDailyLoss`. AAPL is booked at 5,000 for a 2-for-1 split the
    broker has not applied; value 95,000 against a 100,000 previous close. Terminal outcome:
    success, ``buys == []``, errors[0] contains "Daily loss limit: 5.0%" and "split correction
    on record for AAPL" and "real loss cannot be ruled out". Discriminating calls: no
    ``submit_buy``; ``save_split_correction("AAPL", "2026-06-15", 2.0)``; the daily summary is
    split-adjusted (AAPL market value doubled).
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        splits={
            "corp_splits": {"AAPL": [(date(2026, 6, 15), 2.0)]},
            "activity_applied": set(),
        },
        persist={"get_peak_value": 100_000.0},
        replies=_replies(
            _acct(portfolio_value=95_000.0, last_equity=100_000.0),
            {"AAPL": _pos(100.0, 50.0, cost_basis=5000.0, unrealized_pl=0.0)},
        ),
    )


@scenario
def peak_unreadable() -> dict:
    """The persisted peak cannot be read: halt before any order.

    From `TestPersistedDrawdown.test_unavailable_risk_history_halts_before_orders`. Terminal
    outcome: success with ``errors == ["Risk history unavailable: could not read persisted
    peak"]``. Discriminating calls: ``get_peak_value`` raises ``RuntimeError: db down``; no
    order, no ``save_peak_value``, no snapshot.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        persist={"get_peak_value": RuntimeError("db down")},
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


@scenario
def peak_invalid() -> dict:
    """The persisted peak is NaN: halt before any order.

    From `TestPersistedDrawdown.test_invalid_persisted_peak_halts_before_orders`. Terminal
    outcome: success with ``errors == ["Invalid stored risk peak: Invalid portfolio value for
    risk peak"]``. Discriminating calls: no order, no ``save_peak_value``, no snapshot.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        persist={"get_peak_value": float("nan")},
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
    )


# --------------------------------------------------------------------------- catastrophe stop

_STOP_ENV = {**_LIVE_ENV, "CATASTROPHE_STOP_ENABLED": "true"}


@scenario
def catastrophe_stop() -> dict:
    """A -40% holding is stopped out once, even with a coincident exit and buy signal.

    From `TestCatastropheStopEnabled.test_stopped_symbol_not_also_sold_by_coincident_strategy_exit`
    and `test_stopped_symbol_not_rebought_by_coincident_buy_signal`. NXPI breaches the -25%
    floor and also appears in both the sell and the buy signals. Terminal outcome: success,
    ``sells == ["NXPI"]``, ``buys == ["MSFT"]``. Discriminating calls: exactly one SELL NXPI qty
    100.0 (intent reason ``catastrophe_stop``) and no BUY NXPI; BUY MSFT notional 3300.0.
    Intents: 2 settled (the NXPI stop sell, the MSFT buy), none unresolved.
    """
    return dict(
        env=_STOP_ENV,
        signals=_blend(["NXPI", "MSFT"], ["NXPI"]),
        replies=_replies(
            _acct(), {"NXPI": _pos(100.0, 60.0, cost_basis=10_000.0, unrealized_pl=-4000.0)}
        ),
    )


@scenario
def catastrophe_stop_pending_order() -> dict:
    """A stop-breaching holding with an order already pending is not sold again.

    From `TestPendingOrderDedup.test_skips_sell_when_symbol_has_pending_order` and
    `test_loss_halt_skips_pending_exit_and_cash_sweep`. NXPI is at -30% and a legacy SELL
    order for it is still working. Terminal outcome: success, ``sells == []``,
    ``buys == ["MSFT"]``. Discriminating calls: ``get_order("pend-1")`` returns ``new``; no
    submit_sell; one BUY MSFT notional 3300.0.
    Intents: 1 settled (the MSFT buy), none unresolved; the legacy pending order is left alone.
    """
    return dict(
        env=_STOP_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": [_pending("pend-1", "NXPI", "SELL")]},
        existing_orders=[
            {
                "id": "pend-1",
                "client_order_id": "legacy-1",
                "symbol": "NXPI",
                "side": "SELL",
                "qty": 20.0,
                "statuses": [{"status": "new"}],
            }
        ],
        replies=_replies(
            _acct(), {"NXPI": _pos(20.0, 70.0, cost_basis=2000.0, unrealized_pl=-600.0)}
        ),
    )


@scenario
def catastrophe_stop_unassessable() -> dict:
    """A holding whose cost basis is NaN cannot be assessed: buys halt, other exits go.

    From `test_unassessable_stop_still_attempts_pending_warning_when_halt_notice_fails` and
    `test_invalid_stop_position_blocks_buys_but_allows_signal_exit`. Terminal outcome: success,
    ``sells == ["AAPL"]``, ``buys == []``, errors[0] == "Cannot assess catastrophe stop for:
    ZZZ". Discriminating calls: no ``submit_buy``; one SELL AAPL qty 10.0. The NaN cost basis
    reaches the daily summary, so the golden holds the tagged ``{"float": "NaN"}``.
    Intents: 1 settled (the AAPL sell), none unresolved.
    """
    zzz = _pos(5.0, 20.0, unrealized_pl=0.0)
    zzz["cost_basis"] = float("nan")
    return dict(
        env=_STOP_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0), "ZZZ": zzz}),
    )


@scenario
def catastrophe_stop_invalid_qty() -> dict:
    """A stop-breaching holding with a negative quantity blocks buys and the sweep buy.

    From `test_invalid_exit_quantity_blocks_buys_but_allows_other_exits[catastrophe]`. AAPL is
    at -40% with qty -10.0; NVDA has a normal exit; the cash sweep is on with 50,000 cash, so a
    sweep buy would otherwise fire. Terminal outcome: success, ``sells == ["NVDA"]``,
    ``buys == []``, errors[0] == "Cannot submit risk-reducing SELL AAPL: invalid long quantity;
    new strategy buys blocked". Discriminating calls: one SELL NVDA qty 5.0, no BUY (neither MSFT
    nor the SPY sweep buy), and only two ``get_account`` reads (no mid-run sweep read).
    Intents: 1 settled (the NVDA sell), none unresolved.
    """
    return dict(
        env={**_STOP_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], ["NVDA"]),
        replies=_replies(
            _acct(),
            {
                "AAPL": _pos(-10.0, 100.0, cost_basis=1000.0, unrealized_pl=-400.0),
                "NVDA": _pos(5.0, 100.0),
            },
        ),
    )


@scenario
def strategy_exit_invalid_qty() -> dict:
    """A strategy exit on a negative-quantity holding blocks buys and the sweep buy.

    From `test_invalid_exit_quantity_blocks_buys_but_allows_other_exits[signal]` and
    `test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy`. AAPL (qty -10.0) and NVDA
    (qty 5.0) both have exit signals; the sweep is on with 50,000 cash. Terminal outcome:
    success, ``sells == ["NVDA"]``, ``buys == []``, errors[0] == "Cannot submit risk-reducing
    SELL AAPL: invalid long quantity; new strategy buys blocked" (reported once). Discriminating
    calls: one SELL NVDA qty 5.0, no BUY at all (no MSFT, no SPY sweep buy), two ``get_account``
    reads only.
    Intents: 1 settled (the NVDA sell), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], ["AAPL", "NVDA"]),
        replies=_replies(
            _acct(),
            {
                "AAPL": _pos(-10.0, 100.0, cost_basis=1000.0, unrealized_pl=0.0),
                "NVDA": _pos(5.0, 100.0),
            },
        ),
    )


# --------------------------------------------------------------------------- cash sweep


@scenario
def sweep_funding_sell() -> dict:
    """Cash cannot cover the day's buy plus the reserve: sell SPY first, then buy.

    From `TestCashSweepEnabled.test_sells_sweep_position_before_submitting_strategy_buy` and
    `test_waits_for_sweep_sell_fill_before_strategy_buy`. Cash 1,000 against a 3,300 buy and a
    5,000 reserve: shortfall 7,300. Terminal outcome: success, ``buys == ["MSFT"]``, ``sells ==
    []`` (the sweep sell is not a strategy sell). Discriminating calls in order: SELL SPY qty
    18.25 (reason ``cash_sweep``), its ``get_order`` poll, then BUY MSFT notional 3300.0; no
    sweep buy afterwards (cash 5,000 is under the 8% trigger). The SPY sell fills at 400.0
    (not the harness default 100.0), the mid-run read shows the cash that implies (5,000.0).
    Intents: 2 settled (the SPY sell at 400.0, the MSFT buy), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], []),
        # SPY is sold at 400: 18.25 shares raise 7,300, so cash goes 1,000 -> 8,300 -> 5,000
        # after the 3,300 MSFT buy (the mid-run read), under the 8% sweep trigger.
        fills={"SPY:SELL": [{"status": "filled", "filled_qty": 18.25, "filled_avg_price": 400.0}]},
        replies=_replies(
            [
                _acct(100_000.0, 1_000.0),
                _acct(100_000.0, 5_000.0),
                _acct(100_033.0, 5_000.0),
            ],
            [
                {"SPY": _pos(247.5, 400.0)},
                {"SPY": _pos(229.25, 400.0), "MSFT": _pos(33.0, 101.0, cost_basis=3300.0)},
            ],
        ),
    )


@scenario
def sweep_buy() -> dict:
    """Idle cash above the 8% trigger is swept into SPY after the strategy's orders.

    From `TestCashSweepEnabled.test_sweep_buy_deploys_leftover_cash_after_strategy_orders`.
    Cash 50,000 on 100,000: sweep target 50,000 - 5,000 reserve. Terminal outcome: success with
    no strategy orders. Discriminating calls: ``get_account`` is read three times (start, sweep
    sizing, post-trade); BUY SPY notional 45000.0 with reason ``cash_sweep``, polled, settled
    through the intent round trip (create, attach, settle); the trade alert shows 450.0 shares.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend([], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def sweep_sell_and_strategy_sell() -> dict:
    """The sweep funding sell goes first and does not count as a freed position slot.

    From `TestCashSweepEnabled.test_sells_sweep_position_before_submitting_strategy_buy` and
    `TestCatastropheStopEnabled.test_slot_math_counts_stop_exactly_once_at_position_cap`.
    MAX_POSITIONS is 2 with AAPL and KO held; AAPL has an exit signal, SPY (the sweep) is held
    and cash is 1,000. Terminal outcome: success, ``sells == ["AAPL"]``, ``buys == ["MSFT"]``.
    Discriminating calls in order: SELL SPY qty 18.25 (``cash_sweep``) before SELL AAPL qty 10.0;
    then exactly one BUY (MSFT 3300.0). NVDA, the second candidate, is refused because only the
    regular sell frees a slot: counting the sweep sell too would let NVDA through.
    The SPY sell fills at 400.0 and the AAPL sell at 150.0, so the mid-run cash read is 6,500.0
    (1,000 + 7,300 + 1,500 - 3,300), under the 8% sweep trigger.
    Intents: 3 settled (SPY sell, AAPL sell, MSFT buy), none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "2", **_SWEEP_ENV},
        signals=_blend(["MSFT", "NVDA"], ["AAPL"]),
        fills={
            "SPY:SELL": [{"status": "filled", "filled_qty": 18.25, "filled_avg_price": 400.0}],
            "AAPL": [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.0}],
        },
        replies=_replies(
            [
                _acct(100_000.0, 1_000.0),
                _acct(100_000.0, 6_500.0),
                _acct(100_033.0, 6_500.0),
            ],
            [
                {
                    "SPY": _pos(240.75, 400.0),
                    "AAPL": _pos(10.0, 150.0),
                    "KO": _pos(20.0, 60.0),
                },
                {
                    "SPY": _pos(222.5, 400.0),
                    "KO": _pos(20.0, 60.0),
                    "MSFT": _pos(33.0, 101.0, cost_basis=3300.0),
                },
            ],
        ),
    )


# --------------------------------------------------------------------------- fills and pending

_AAPL_ONE = {"AAPL": _pos(10.0, 150.0)}
_PARTIAL = {"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 150.0}


@scenario
def sell_partial_fill_working() -> dict:
    """A sell still ``partially_filled`` at the wait deadline blocks the run.

    From `test_exit_must_fill_to_free_a_position_slot` and
    `test_recovery_blocks_ambiguous_or_working_orders`. AAPL (10.0) sells; the broker keeps it
    partially filled at 4.0 shares. Terminal outcome: the run raises ``RuntimeError: Unresolved
    working orders block submissions`` after "Order recovery/pending orders failed" is sent.
    Discriminating calls: SELL AAPL qty 10.0 polled until the 15 second deadline; no
    ``settle_order_intent`` ever; no BUY MSFT. Intent row: 1 unresolved (submitted, ``ord-1``),
    none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        fills={
            "AAPL": [{"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 150.0}]
        },
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def sell_partial_fill_canceled() -> dict:
    """A sell canceled after a partial fill settles and blocks the replacement buy.

    From `test_exit_must_fill_to_free_a_position_slot[canceled-4-False]` and
    `test_recovery_settles_partial_cancellation`. Terminal outcome: success, ``sells == ["AAPL"]``,
    ``buys == []``, errors == ["SELL AAPL not fully filled (canceled); new strategy buys
    blocked"]. Discriminating calls: ``settle_order_intent(.., 4.0, 150.0, "canceled")`` (the
    partial fill is booked), no BUY MSFT. Intent rows: 1 settled (canceled), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        fills={
            "AAPL": [
                {"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 150.0},
                {"status": "canceled", "filled_qty": 4.0, "filled_avg_price": 150.0},
            ]
        },
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def pending_symbol_buy() -> dict:
    """A symbol with a working buy from an earlier run is not bought again.

    From `TestPendingOrderDedup.test_skips_buy_when_symbol_has_pending_order`. Terminal
    outcome: success, ``buys == ["NVDA"]``. Discriminating calls: ``get_order("pend-b1")``
    returns ``new``, no BUY MSFT, one BUY NVDA notional 3300.0.
    Intents: 1 settled (the NVDA buy), none unresolved; the legacy pending order is left alone.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], []),
        persist={"get_pending_orders": [_pending("pend-b1", "MSFT", "BUY")]},
        existing_orders=[
            {
                "id": "pend-b1",
                "client_order_id": "legacy-b1",
                "symbol": "MSFT",
                "side": "BUY",
                "notional": 3300.0,
                "statuses": [{"status": "new"}],
            }
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def pending_symbol_sell() -> dict:
    """A holding with a working sell from an earlier run is not sold again.

    From `TestPendingOrderDedup.test_skips_sell_when_symbol_has_pending_order`. Terminal
    outcome: success, ``sells == []``, no order submitted at all. Discriminating calls:
    ``get_order("pend-s1")`` returns ``new``; no submit_sell.
    Intents: none created; the legacy pending order is left alone.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        persist={"get_pending_orders": [_pending("pend-s1", "AAPL", "SELL")]},
        existing_orders=[
            {
                "id": "pend-s1",
                "client_order_id": "legacy-s1",
                "symbol": "AAPL",
                "side": "SELL",
                "qty": 10.0,
                "statuses": [{"status": "new"}],
            }
        ],
        replies=_replies(_acct(), _AAPL_ONE),
    )


# --------------------------------------------------------------------------- position limits


@scenario
def global_slot_cap() -> dict:
    """The global position cap refuses a candidate that would otherwise trade.

    From `TestBuyEntries` and `test_slot_math_counts_stop_exactly_once_at_position_cap`.
    MAX_POSITIONS is 2 with one holding, so one slot is free. Terminal outcome: success,
    ``buys == ["MSFT"]``. Discriminating calls: exactly one BUY (MSFT 3300.0); NVDA and GOOGL
    are refused by the cap alone (the sleeve cap of 2 and concentration would let NVDA in).
    Intents: 1 settled (the MSFT buy), none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "2"},
        signals=_blend(["MSFT", "NVDA", "GOOGL"], []),
        replies=_replies(_acct(), _AAPL_ONE),
    )


def _two_sleeves(
    sp500: list[str], nasdaq: list[str], weights: dict[str, float], scale: float
) -> dict:
    return {
        "sleeves": {
            "sp500": _sleeve(sp500, [], "2026-06-19"),
            "nasdaq100": _sleeve(nasdaq, [], "2026-06-19"),
        },
        "weights": weights,
        "scale": scale,
        "rebalanced_today": False,
        "fallback_used": False,
    }


@scenario
def sleeve_slot_cap() -> dict:
    """A sleeve's own slot cap refuses a candidate while the global cap would allow it.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from the sleeve-cap math in `risk.sleeve_slot_caps` (no `run`-level test exists).
    MAX_POSITIONS is 4 with two 50% sleeves, so each sleeve gets 2 slots; the sp500 sleeve
    offers three candidates. Terminal outcome: success, ``buys == ["MSFT", "NVDA", "AMZN"]``.
    Discriminating calls: BUY MSFT and BUY NVDA at 1650.0 each, no BUY GOOGL (it would trade if
    the sleeve cap alone were lifted: 3 of 4 global slots), BUY AMZN 1650.0 from the other sleeve.
    Intents: 3 settled, none unresolved.
    Reporting: the first account read is 100,000.0 with no positions; the final read is
    100,033.0 (cash 95,050.0) with the three bought positions, so the summary value, the P&L
    (+233.0 against the 99,800.0 snapshot) and the snapshot payload follow the final read.
    """
    return dict(
        env={"MAX_POSITIONS": "4"},
        signals=_two_sleeves(
            ["MSFT", "NVDA", "GOOGL"], ["AMZN"], {"sp500": 0.5, "nasdaq100": 0.5}, 1.0
        ),
        persist={"get_latest_snapshot": 99_800.0},
        replies=_replies(
            [
                _acct(100_000.0, 100_000.0),
                _acct(100_033.0, 95_050.0, last_equity=100_000.0),
            ],
            [
                {},
                {
                    "MSFT": _pos(16.5, 101.0, cost_basis=1650.0),
                    "NVDA": _pos(16.5, 102.0, cost_basis=1650.0),
                    "AMZN": _pos(16.5, 99.0, cost_basis=1650.0),
                },
            ],
        ),
    )


@scenario
def concentration_limit() -> dict:
    """The 5% concentration limit refuses one candidate that would otherwise trade.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Synthetic (no `run`-level test): with one sleeve at scale 1.0 a new position is 3.3% and
    can never reach 5%, so the scenario uses a sp500 sleeve at weight 1.0 and scale 1.6 (a
    5,280 position, 5.28%) next to a nasdaq100 sleeve at weight 0.1 (528). Real `RiskConfig`
    defaults, MAX_POSITIONS 10. Terminal outcome: success, ``buys == ["AMZN"]``. Discriminating
    calls: no BUY MSFT (it would trade if the limit alone were lifted: slots and sleeve cap
    allow it), one BUY AMZN notional 528.0.
    Intents: 1 settled (the AMZN buy), none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "10"},
        signals=_two_sleeves(["MSFT"], ["AMZN"], {"sp500": 1.0, "nasdaq100": 0.1}, 1.6),
        replies=_replies(_acct(), {}),
    )


@scenario
def missed_exit_catch_up() -> dict:
    """An exit that fired after entry but was missed is caught up; an older one is not.

    From `TestMissedExitCatchUp.test_sells_when_exit_fired_after_entry` and
    `test_keeps_when_exit_predates_entry`. AAPL's last exit bar (2026-06-10) is after its entry
    (2026-06-05); KO's (2026-06-03) is before. Terminal outcome: success, ``sells == ["AAPL"]``.
    Discriminating calls: one SELL AAPL qty 10.0 and no SELL KO.
    Intents: 1 settled (the AAPL sell), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], [], last_exit={"AAPL": "2026-06-10", "KO": "2026-06-03"}),
        persist={"get_last_buy_dates": {"AAPL": date(2026, 6, 5), "KO": date(2026, 6, 5)}},
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0), "KO": _pos(20.0, 60.0)}),
    )


# --------------------------------------------------------------------------- dry runs


@scenario
def dry_run_read_only() -> dict:
    """The default dry run writes nothing and records what it skipped.

    From `TestReadOnlyDryRun.test_dry_run_is_read_only_by_default` and
    `test_pending_orders_are_neither_polled_nor_booked`. Terminal outcome: success with planned
    ``sells == ["AAPL"]``, ``buys == ["MSFT"]`` (NVDA is skipped: a pending order exists) and
    ``skipped_writes == ["init_paper_schema", "refresh_benchmark_tape", "save_peak_value",
    "log_snapshot"]``. Discriminating calls: no persist write call at all (only reads), no
    broker submit, ``generate_core_signals(persist_tape=False)``, no ``get_order`` poll.
    Intents: none created.
    """
    return dict(
        dry_run=True,
        membership=FIXTURE_SPEC,
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], ["AAPL"]),
        persist={"get_pending_orders": [_pending("pend-n1", "NVDA", "BUY")]},
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def dry_run_with_writes() -> dict:
    """A dry run with writes enabled books state but submits nothing.

    From `TestReadOnlyDryRun.test_legacy_dry_run_with_writes_still_persists` and
    `TestReconciliation.test_filled_pending_order_booked_and_cleared`. A pending KO buy has
    filled; AAPL has a NaN broker P&L. Terminal outcome: success, ``buys == ["MSFT"]``,
    ``skipped_writes == []``. Discriminating calls: ``settle_legacy_pending_order("pend-9", 10.0,
    60.0, "filled")``, a "filled (reconciled)" trade alert, ``init_paper_schema``,
    ``save_peak_value``, ``log_snapshot`` and the tape refresh all happen; no submit call. The
    NaN P&L reaches the summary as a tagged value.
    Intents: none created; the filled legacy pending order settles through
    ``settle_legacy_pending_order``.
    """
    return dict(
        dry_run=True,
        read_only=False,
        membership=FIXTURE_SPEC,
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": [_pending("pend-9", "KO", "BUY")]},
        existing_orders=[
            {
                "id": "pend-9",
                "client_order_id": "legacy-9",
                "symbol": "KO",
                "side": "BUY",
                "notional": 600.0,
                "statuses": [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 60.0}],
            }
        ],
        replies=_replies(
            _acct(),
            {"AAPL": _pos(10.0, 150.0, unrealized_pl=float("nan"))},
        ),
    )


# --------------------------------------------------------------------------- failures


@scenario
def submit_error() -> dict:
    """An exception from the broker on submit ends the run after the intent is saved.

    From `TestUntradableAvb.test_submit_error_aborts_run` and
    `test_submit_failure_notifies_client_id_and_stage[submit]`. The first BUY raises. Terminal
    outcome: the run raises ``RuntimeError: insufficient buying power``; the notifier gets
    "Order 00000000-0000-0000-0000-000000000001 failed at submit; investigation required".
    Discriminating calls in order: ``create_order_intent`` (BUY MSFT 3300.0), ``lock.verify``,
    ``submit_buy`` (raised), then nothing: NVDA is never attempted. Intent rows: 1 unresolved
    (never attached), none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], []),
        replies={
            **_replies(_acct(), {}),
            "submit_buy": [RuntimeError("insufficient buying power")],
        },
    )


@scenario
def post_trade_read_failure() -> dict:
    """The account read after trading fails: the run says orders may have been submitted.

    From `test_post_trade_read_failure_reports_orders_may_have_been_submitted`. AAPL sells and
    settles, then the second ``get_account`` raises. Terminal outcome: the run raises
    ``RuntimeError: api down`` after "Account or positions data unavailable after trading:
    orders may have been submitted; snapshot not saved." is sent. Discriminating calls: SELL
    AAPL qty 10.0 first; no ``log_snapshot``; no daily summary.
    Intents: 1 settled (the AAPL sell), none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        replies=_replies([_acct(), RuntimeError("api down")], _AAPL_ONE),
    )


@scenario
def snapshot_write_failure() -> dict:
    """The daily snapshot write fails: the run still ends normally and reports it.

    From `TestErrorHandling.test_db_snapshot_failure_does_not_crash`. Terminal outcome: success
    with ``errors == ["Daily snapshot could not be saved"]``. Discriminating calls:
    ``log_snapshot`` raises ``RuntimeError: write failed``; "Daily snapshot could not be saved."
    is sent after the daily summary.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], []),
        persist={"log_snapshot": RuntimeError("write failed")},
        replies=_replies(_acct(), {}),
    )


@scenario
def sell_rejected_by_broker() -> dict:
    """The broker accepts a sell, then reports it ``rejected``: it settles, other exits go on.

    From `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit` (this pins the path
    next to the code a later task deletes). A drawdown halt is on (peak 120,000) so only exits
    run; AAPL is rejected, MSFT fills. Terminal outcome: success, ``sells == ["AAPL", "MSFT"]``,
    ``buys == []``, errors include "SELL AAPL not fully filled (rejected); new strategy buys
    blocked". Discriminating calls: SELL AAPL qty 10.0 and SELL MSFT qty 5.0 both submitted
    before any poll; ``settle_order_intent`` for AAPL with (0.0, None, "rejected") and for MSFT
    with "filled"; no BUY NVDA. Intent rows: 2 settled, none unresolved.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["NVDA"], ["AAPL", "MSFT"]),
        persist={"get_peak_value": 120_000.0},
        fills={"AAPL": [{"status": "rejected", "filled_qty": 0.0, "filled_avg_price": None}]},
        replies=_replies(
            _acct(),
            {"AAPL": _pos(10.0, 100.0), "MSFT": _pos(5.0, 200.0)},
        ),
    )


# =========================================================================== follow-up scenarios
# Added after review: the branches the later refactor tasks edit, and the cases where the first
# 38 transcripts would have stayed green after a behavior change.

_STALE_NOW = datetime(2026, 6, 10, 15, 0, tzinfo=timezone.utc)
_FRESH = datetime(2026, 6, 18, 15, 0, tzinfo=timezone.utc)


def _held_order(
    order_id: str,
    client_id: str,
    symbol: str,
    side: str,
    amount: float,
    statuses: list[dict],
) -> dict:
    """An order the broker already holds before the run (a BUY by notional, a SELL by qty)."""
    kind = "notional" if side == "BUY" else "qty"
    return {
        "id": order_id,
        "client_order_id": client_id,
        "symbol": symbol,
        "side": side,
        kind: amount,
        "statuses": statuses,
    }


def _prior_intent(client_id: str, symbol: str = "MSFT", amount: float = 3300.0) -> dict:
    """An unresolved BUY intent from an earlier run (the startup recovery input)."""
    return {
        "client_order_id": client_id,
        "run_date": "2026-06-18",
        "side": "BUY",
        "symbol": symbol,
        "amount": amount,
        "amount_kind": "notional",
        "reason": None,
        "broker_order_id": None,
    }


def _recovery(
    statuses: list[dict], replies_extra: dict | None = None, notifier: dict | None = None
) -> dict:
    """A live run that starts with one unresolved prior-run BUY MSFT intent in `statuses`."""
    return dict(
        notifier=notifier or {},
        env=_LIVE_ENV,
        signals=_blend([], []),
        intents=[_prior_intent("prev-1")],
        existing_orders=[_held_order("ord-prev-1", "prev-1", "MSFT", "BUY", 3300.0, statuses)],
        replies={**_replies(_acct(), {}), **(replies_extra or {})},
    )


# --------------------------------------------------------------------------- finding 1


@scenario
def catastrophe_stop_at_position_cap() -> dict:
    """The freed slot of a stopped holding is counted once, with the book at the cap.

    From `TestCatastropheStopEnabled.test_slot_math_counts_stop_exactly_once_at_position_cap`.
    MAX_POSITIONS is 3 and the book holds FILL1, FILL2 and NXPI (-30%, stopped out); MSFT, GOOGL
    and NVDA all want in. Terminal outcome: success, ``sells == ["NXPI"]``,
    ``buys == ["MSFT"]``. Discriminating calls in order: SELL NXPI qty 100.0 (reason
    ``catastrophe_stop``) and then exactly one BUY (MSFT 3300.0). The stop already removed NXPI
    from the strategy positions, so only one slot is free; counting the stop again (as a regular
    sell) would free a second slot and submit BUY GOOGL as well.
    Intents: 2 settled (NXPI stop sell, MSFT buy), none unresolved.
    """
    return dict(
        env=_STOP_ENV,
        signals=_blend(["MSFT", "GOOGL", "NVDA"], []),
        replies=_replies(
            _acct(),
            {
                "FILL1": _pos(1.0, 100.0),
                "FILL2": _pos(1.0, 100.0),
                "NXPI": _pos(100.0, 70.0, cost_basis=10_000.0, unrealized_pl=-3000.0),
            },
        ),
    )


# --------------------------------------------------------------------------- finding 3


@scenario
def sweep_buy_after_strategy_orders() -> dict:
    """The sweep buy is sized from the cash left after the strategy buy, and goes after it.

    From `TestCashSweepEnabled.test_sweep_buy_deploys_leftover_cash_after_strategy_orders`.
    SPY (the sweep) is held at 50,000 with 50,000 cash; MSFT is a strategy buy. The MSFT buy
    leaves 46,700 cash, so the sweep target is 46,700 - 5,000 reserve = 41,700. Terminal
    outcome: success, ``buys == ["MSFT"]``. Discriminating calls in order: BUY MSFT 3300.0, its
    ``get_order`` poll and settlement, a second ``get_account`` read (the sweep sizing), then
    BUY SPY notional 41700.0 (reason ``cash_sweep``, filled at 400.0). A sweep buy placed before
    the strategy orders would be sized from 50,000 cash instead (45,000).
    Intents: 2 settled (MSFT buy, SPY sweep buy), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], []),
        fills={"SPY:BUY": [{"status": "filled", "filled_qty": 104.25, "filled_avg_price": 400.0}]},
        replies=_replies(
            [
                _acct(100_000.0, 50_000.0),
                _acct(100_000.0, 46_700.0),
                _acct(100_033.0, 5_000.0),
            ],
            [
                {"SPY": _pos(125.0, 400.0)},
                {
                    "SPY": _pos(229.25, 400.0),
                    "MSFT": _pos(33.0, 101.0, cost_basis=3300.0),
                },
            ],
        ),
    )


# --------------------------------------------------------------------------- finding 5


@scenario
def intent_create_failure() -> dict:
    """The ledger cannot save the intent: the run stops before the lock check and the broker.

    From `test_submit_failure_notifies_client_id_and_stage[intent]`. Terminal outcome: the run
    raises ``RuntimeError: ledger down`` and sends "Order 00000000-0000-0000-0000-000000000001
    failed at intent; investigation required". Discriminating calls: ``create_order_intent``
    (BUY MSFT 3300.0, raised) and nothing after it: no ``lock.verify``, no ``submit_buy``.
    Intents: none created (the one attempt failed).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], []),
        persist={"create_order_intent": RuntimeError("ledger down")},
        replies=_replies(_acct(), {}),
    )


@scenario
def lock_verify_failure() -> dict:
    """The run lock is lost when the order is about to be submitted: nothing reaches the broker.

    From `test_submit_failure_notifies_client_id_and_stage[lock]`. Terminal outcome: the run
    raises ``RuntimeError: lock lost`` after "failed at lock; investigation required" is sent.
    Discriminating calls in order: ``create_order_intent`` (BUY MSFT 3300.0), ``lock.verify``
    (raised), no ``submit_buy``. Intents: 1 unresolved (created, never attached), none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], []),
        persist={"lock.verify": RuntimeError("lock lost")},
        replies=_replies(_acct(), {}),
    )


@scenario
def attach_failure() -> dict:
    """The broker took the order but the ledger cannot attach its id: the run stops.

    From `test_submit_failure_notifies_client_id_and_stage[attach]`. Terminal outcome: the run
    raises ``RuntimeError: attach failed`` after "failed at attach; investigation required" is
    sent. Discriminating calls in order: ``create_order_intent``, ``lock.verify``,
    ``submit_buy`` (BUY MSFT 3300.0, accepted as ``ord-1``), ``attach_broker_order`` (raised);
    NVDA is never attempted. Intents: 1 unresolved, none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT", "NVDA"], []),
        persist={"attach_broker_order": RuntimeError("attach failed")},
        replies=_replies(_acct(), {}),
    )


@scenario
def failure_notice_fails() -> dict:
    """A failing submit whose failure notice also fails still ends with the original error.

    From `test_submit_failure_notifies_client_id_and_stage` and
    `test_post_trade_account_error_survives_notification_failure`. The broker rejects the BUY
    and the "failed at submit" notice raises ``RuntimeError: telegram down``. Terminal outcome:
    the run raises the broker's ``RuntimeError: rejected``, not the notifier's. Discriminating
    calls: ``submit_buy`` (raised), then the notice ``send`` (raised); nothing else is sent.
    Intents: 1 unresolved, none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        notifier={"failed at submit": RuntimeError("telegram down")},
        replies={**_replies(_acct(), {}), "submit_buy": [RuntimeError("rejected")]},
    )


# --------------------------------------------------------------------------- finding 6: recovery


@scenario
def recovery_intent_filled() -> dict:
    """A prior run's intent whose order has filled is settled at startup.

    From `test_recovery_settles_partial_cancellation` and `test_submit_order_persists_intent...`.
    Terminal outcome: success with no new orders. Discriminating calls: ``get_order_by_client_id``
    for ``prev-1``, ``attach_broker_order("prev-1", "ord-prev-1")``, then
    ``settle_order_intent("prev-1", "ord-prev-1", 33.0, 100.0, "filled")`` before the
    market-clock call. Intents: the seeded row settles, none unresolved.
    """
    return _recovery([{"status": "filled", "filled_qty": 33.0, "filled_avg_price": 100.0}])


@scenario
def recovery_intent_canceled() -> dict:
    """A seeded intent whose order was canceled after a partial fill settles as canceled.

    From `test_recovery_settles_partial_cancellation`. Terminal outcome: success.
    Discriminating call: ``settle_order_intent("prev-1", "ord-prev-1", 2.0, 100.0, "canceled")``.
    Removing "canceled" from the terminal statuses would instead stop the run with
    "Unknown order status". Intents: the seeded row settles, none unresolved.
    """
    return _recovery([{"status": "canceled", "filled_qty": 2.0, "filled_avg_price": 100.0}])


@scenario
def recovery_intent_rejected() -> dict:
    """A seeded intent whose order the broker rejected settles with no fill.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: success. Discriminating call: ``settle_order_intent("prev-1",
    "ord-prev-1", 0.0, None, "rejected")``. Intents: the seeded row settles, none unresolved.
    """
    return _recovery([{"status": "rejected", "filled_qty": 0.0, "filled_avg_price": None}])


@scenario
def recovery_intent_expired() -> dict:
    """A seeded intent whose order expired settles with no fill.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: success. Discriminating call: ``settle_order_intent("prev-1",
    "ord-prev-1", 0.0, None, "expired")``. Intents: the seeded row settles, none unresolved.
    """
    return _recovery([{"status": "expired", "filled_qty": 0.0, "filled_avg_price": None}])


@scenario
def recovery_intent_working() -> dict:
    """A seeded intent whose order is still working stops the run once recovery has read it.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    The lock, the schema call and the recovery reads happen first. Terminal outcome: the run
    raises ``RuntimeError: Unresolved working orders block submissions`` after "Order
    recovery/pending orders failed" is sent. Discriminating calls: ``attach_broker_order`` but
    no ``settle_order_intent``; no market-clock call, no signals.
    Intents: the seeded row stays unresolved.
    """
    return _recovery([{"status": "new"}])


@scenario
def recovery_identity_mismatch() -> dict:
    """A lookup reply with a blank order id is an identity mismatch: the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    The reply is scripted (a malformed broker answer). Terminal outcome: the run raises
    ``ValueError: Order identity mismatch or duplicate legacy order`` after the recovery
    notice is sent. Discriminating calls: no ``attach_broker_order``, no
    ``settle_order_intent``. Intents: the seeded row stays unresolved.
    """
    bad = {
        "id": "",
        "client_order_id": "prev-1",
        "symbol": "MSFT",
        "side": "buy",
        "qty": None,
        "notional": 3300.0,
        "filled_qty": 0.0,
        "filled_avg_price": None,
        "status": "new",
    }
    return _recovery([{"status": "new"}], replies_extra={"get_order_by_client_id": [bad]})


@scenario
def recovery_symbol_mismatch() -> dict:
    """The broker's order for a client id is for another symbol: the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: the run raises ``ValueError: Order identity mismatch or duplicate legacy
    order``. Discriminating calls: ``get_order_by_client_id("prev-1")``, no settle, no attach.
    Intents: the seeded row stays unresolved.
    """
    fields = _recovery([{"status": "filled", "filled_qty": 33.0, "filled_avg_price": 100.0}])
    fields["existing_orders"][0]["symbol"] = "NVDA"
    return fields


@scenario
def recovery_amount_mismatch() -> dict:
    """The broker's order amount differs from the intent by more than a cent: the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    The intent says 3,300.0 notional; the broker holds 3,500.0. Terminal outcome: the run raises
    ``ValueError: Order amount mismatch``. Discriminating calls: no attach, no settle.
    Intents: the seeded row stays unresolved.
    """
    fields = _recovery([{"status": "filled", "filled_qty": 35.0, "filled_avg_price": 100.0}])
    fields["existing_orders"][0]["notional"] = 3500.0
    return fields


@scenario
def recovery_missing_broker_order() -> dict:
    """The broker has no order for a seeded intent (the lookup fails): the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    The lookup reply is a scripted ``RuntimeError: order not found``. Terminal outcome: the run
    raises it after "Order recovery/pending orders failed: order not found" is sent.
    Discriminating calls: ``get_order_by_client_id`` (raised), no attach, no settle.
    Intents: the seeded row stays unresolved.
    """
    return _recovery(
        [{"status": "new"}],
        replies_extra={"get_order_by_client_id": [RuntimeError("order not found")]},
    )


@scenario
def recovery_invalid_fill() -> dict:
    """A canceled order with filled shares but no fill price is an invalid fill: the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: the run raises ``ValueError: Invalid order fill``. Discriminating calls:
    no attach, no settle. Intents: the seeded row stays unresolved.
    """
    return _recovery([{"status": "canceled", "filled_qty": 2.0, "filled_avg_price": None}])


@scenario
def recovery_unknown_status() -> dict:
    """An order status the trader does not know stops the run.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: the run raises ``ValueError: Unknown order status``. Discriminating
    calls: no attach, no settle. Intents: the seeded row stays unresolved.
    """
    return _recovery([{"status": "mystery"}])


# --------------------------------------------------------------------------- finding 6: legacy


@scenario
def legacy_pending_stale() -> dict:
    """Only a legacy pending order open 5+ days and not yet flagged gets a stale alert.

    From `TestStalePendingOrders`. Three working legacy BUY orders: KO (created 9.2 days before
    the pinned clock, not flagged), XOM (same age, already flagged) and CVX (1.2 days old).
    Terminal outcome: success. Discriminating calls: one "Stale pending order" alert naming KO
    ("open 9.2 days") and one ``mark_pending_order_stale("stale-ko")``; nothing for XOM or CVX.
    Intents: none created.
    """
    old = {"created_at": _STALE_NOW}
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], []),
        persist={
            "get_pending_orders": [
                {**_pending("stale-ko", "KO", "BUY"), **old},
                {**_pending("stale-xom", "XOM", "BUY"), **old, "flagged_stale": True},
                {**_pending("fresh-cvx", "CVX", "BUY"), "created_at": _FRESH},
            ]
        },
        existing_orders=[
            _held_order("stale-ko", "legacy-ko", "KO", "BUY", 600.0, [{"status": "new"}]),
            _held_order("stale-xom", "legacy-xom", "XOM", "BUY", 600.0, [{"status": "new"}]),
            _held_order("fresh-cvx", "legacy-cvx", "CVX", "BUY", 600.0, [{"status": "new"}]),
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def legacy_pending_stale_mark_fails() -> dict:
    """The stale alert is sent even when marking the order stale fails.

    From `TestStalePendingOrders.test_stale_pending_order_flagged_once`. Terminal outcome:
    success. Discriminating calls: the "Stale pending order" alert, then
    ``mark_pending_order_stale`` (raised ``RuntimeError: db busy``); the run goes on.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], []),
        persist={
            "get_pending_orders": [{**_pending("stale-ko", "KO", "BUY"), "created_at": _STALE_NOW}],
            "mark_pending_order_stale": RuntimeError("db busy"),
        },
        existing_orders=[
            _held_order("stale-ko", "legacy-ko", "KO", "BUY", 600.0, [{"status": "new"}])
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def legacy_pending_lookup_failure() -> dict:
    """The broker lookup of a legacy pending order fails: the run stops.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Terminal outcome: the run raises ``RuntimeError: lookup failed`` (no notice is sent).
    Discriminating calls: ``get_order("pend-1")`` raised; no signals call, no order.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": [_pending("pend-1", "KO", "BUY")]},
        replies={**_replies(_acct(), {}), "get_order": [RuntimeError("lookup failed")]},
    )


@scenario
def legacy_pending_settlement_failure() -> dict:
    """A filled legacy pending order that cannot be settled stops the run with a notice.

    From `TestReconciliation.test_filled_pending_order_booked_and_cleared`. Terminal outcome:
    the run raises ``RuntimeError: ledger locked`` after "Could not settle pending order pend-1
    — no orders placed: ledger locked" is sent. Discriminating calls:
    ``settle_legacy_pending_order`` (raised), no trade alert, no signals call.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={
            "get_pending_orders": [_pending("pend-1", "KO", "BUY")],
            "settle_legacy_pending_order": RuntimeError("ledger locked"),
        },
        existing_orders=[
            _held_order(
                "pend-1",
                "legacy-1",
                "KO",
                "BUY",
                600.0,
                [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 60.0}],
            )
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def legacy_pending_canceled() -> dict:
    """A canceled legacy pending order is dropped without a trade alert and frees its symbol.

    From `TestReconciliation.test_canceled_pending_order_dropped_without_booking` and
    `TestPendingOrderDedup.test_terminal_pending_order_does_not_block_new_order`. Terminal
    outcome: success, ``buys == ["KO"]``. Discriminating calls:
    ``settle_legacy_pending_order("pend-1", 0.0, None, "canceled")`` with no trade alert for it,
    then BUY KO 3300.0 (the symbol is no longer pending). Removing "canceled" from the legacy
    terminal statuses would leave KO blocked. Intents: 1 settled (the KO buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["KO"], []),
        persist={"get_pending_orders": [_pending("pend-1", "KO", "BUY")]},
        existing_orders=[
            _held_order(
                "pend-1",
                "legacy-1",
                "KO",
                "BUY",
                600.0,
                [{"status": "canceled", "filled_qty": 0.0, "filled_avg_price": None}],
            )
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def pending_orders_read_failure_live() -> dict:
    """The legacy pending-order list cannot be read in a live run: no orders.

    From `test_pending_order_read_failure_stops_live_run`. The first read (the intent recovery)
    works; the second fails. Terminal outcome: the run raises ``RuntimeError: ledger offline``
    after "Could not load pending orders — no orders placed" is sent. Discriminating calls: two
    ``get_pending_orders`` reads, no signals call, no order. Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": Seq([], RuntimeError("ledger offline"))},
        replies=_replies(_acct(), {}),
    )


@scenario
def pending_orders_read_failure_dry() -> dict:
    """A dry run whose pending-order list cannot be read still rehearses its orders.

    From `test_pending_order_read_failure_dry_run_still_rehearses`. Terminal outcome: success,
    ``buys == ["MSFT"]`` (planned only). Discriminating calls: ``get_pending_orders`` raised,
    "DRY RUN buy: MSFT" sent, no submit call.
    """
    return dict(
        dry_run=True,
        membership=FIXTURE_SPEC,
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": RuntimeError("ledger offline")},
        replies=_replies(_acct(), {}),
    )


# --------------------------------------------------------------------------- finding 7


@scenario
def loss_halt_notice_fails_with_pending() -> dict:
    """A failed halt notice does not suppress the separate pending-order warning.

    From `test_loss_halt_notice_failure_does_not_block_exit` and
    `test_loss_halt_skips_pending_exit_and_cash_sweep`. A drawdown halt (peak 120,000) with a
    working legacy BUY for MSFT; the "HALTED buys" notice raises ``RuntimeError: telegram down``.
    Terminal outcome: success, ``sells == ["AAPL"]``, ``buys == []``. Discriminating calls in
    order: the "HALTED buys; exits only" send (raised), then the "Pending orders may still fill"
    send (delivered), then SELL AAPL qty 10.0. Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["NVDA"], ["AAPL"]),
        persist={
            "get_peak_value": 120_000.0,
            "get_pending_orders": [_pending("pend-1", "MSFT", "BUY")],
        },
        existing_orders=[
            _held_order("pend-1", "legacy-1", "MSFT", "BUY", 3300.0, [{"status": "new"}])
        ],
        notifier={"HALTED buys": RuntimeError("telegram down")},
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def catastrophe_stop_unassessable_notice_fails_with_pending() -> dict:
    """A failed unassessable-stop notice does not suppress the pending-order warning.

    From `test_unassessable_stop_still_attempts_pending_warning_when_halt_notice_fails`. ZZZ has
    a NaN cost basis and a working legacy BUY exists; the "HALTED buys" notice raises. Terminal
    outcome: success, ``buys == []``, errors[0] == "Cannot assess catastrophe stop for: ZZZ".
    Discriminating calls: the "HALTED buys; exits only" send (raised) followed by the "Pending
    orders may still fill" send (delivered); no ``submit_buy``. Intents: none created.
    """
    zzz = _pos(5.0, 20.0, unrealized_pl=0.0)
    zzz["cost_basis"] = float("nan")
    return dict(
        env=_STOP_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_pending_orders": [_pending("pend-1", "NVDA", "BUY")]},
        existing_orders=[
            _held_order("pend-1", "legacy-1", "NVDA", "BUY", 3300.0, [{"status": "new"}])
        ],
        notifier={"HALTED buys": RuntimeError("telegram down")},
        replies=_replies(_acct(), {"ZZZ": zzz}),
    )


# --------------------------------------------------------------------------- finding 8


@scenario
def poll_clock_failure() -> dict:
    """A failing clock read while polling is ignored and the poll still completes.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from `_poll_orders` (no `run`-level test). The sell of AAPL is ``new`` twice, then
    ``filled``; the broker clock raises on the poll's read (so the market is assumed open and
    the wait budget is 15 seconds). Terminal outcome: success, ``sells == ["AAPL"]``.
    Discriminating calls: ``get_clock`` raised after the order was submitted, then three
    ``get_order`` polls with two one-second sleeps (the fake clock moves 1.0 each), then settle.
    Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        fills={
            "AAPL": [
                {"status": "new"},
                {"status": "new"},
                {"status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.0},
            ]
        },
        replies=_replies(
            _acct(), _AAPL_ONE, [_CLOCK_OPEN, RuntimeError("clock down"), _CLOCK_OPEN]
        ),
    )


@scenario
def poll_transient_lookup_error() -> dict:
    """A failed lookup in the middle of polling is logged and the next poll succeeds.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from `_poll_orders`. The first ``get_order`` raises ``RuntimeError: blip``; the second
    returns the order, filled. Terminal outcome: success, ``sells == ["AAPL"]``. Discriminating
    calls: two ``get_order("ord-1")`` calls (the first raised) with one one-second sleep between
    them. Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        replies={**_replies(_acct(), _AAPL_ONE), "get_order": [RuntimeError("blip"), None]},
    )


@scenario
def poll_final_lookup_fails() -> dict:
    """When every poll and the final lookup fail, the order is reported as pending.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from `_poll_orders` (the fallback record). Every ``get_order`` raises; the order
    itself is ``filled`` at the broker, so the client-id lookup during recovery settles it.
    Terminal outcome: success, ``sells == ["AAPL"]``, ``buys == []``, errors ==
    ["SELL AAPL not fully filled (pending); new strategy buys blocked"]. Discriminating calls:
    16 failed ``get_order`` polls (15 in the loop, 1 final), 15 sleeps, a trade alert with
    status ``pending`` and amount 1500.0 (the fallback record has no fill), no BUY MSFT.
    Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        fills={"AAPL": [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.0}]},
        replies={**_replies(_acct(), _AAPL_ONE), "get_order": [RuntimeError("poll down")]},
    )


@scenario
def poll_market_closed_short_wait() -> dict:
    """With the market closed during polling the wait budget is 2 seconds, not 15.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from `_poll_orders`. The first clock read (the run gate) says open, the poll's read
    says closed. The AAPL sell stays ``partially_filled``. Terminal outcome: the run raises
    ``RuntimeError: Unresolved working orders block submissions``. Discriminating calls: 3
    ``get_order`` calls (2 in the loop under the 2.0 second budget, 1 final lookup), 2 sleeps.
    Intents: 1 unresolved (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        fills={"AAPL": [_PARTIAL]},
        replies=_replies(
            _acct(),
            _AAPL_ONE,
            [_CLOCK_OPEN, {"is_open": False, "next_close": "2026-06-22 09:30:00-04:00"}],
        ),
    )


# --------------------------------------------------------------------------- finding 9


@scenario
def split_correction_held() -> dict:
    """A held, unapplied-split name stays in the final report at its corrected value.

    From `TestDailyPnl.test_unapplied_split_corrects_the_daily_summary`. MNST (broker qty
    20.80410098, never doubled) is held and not sold; NVDA is bought. Terminal outcome: success,
    ``buys == ["NVDA"]``. Discriminating calls: ``save_split_correction("MNST", "2026-06-10",
    2.0)``; BUY NVDA 3300.0; the daily summary's value is the booked 100,500.0 plus MNST's
    market value again (993.81...), MNST's reported market value is doubled to 1,987.6..., and
    ``booked_equity`` is 100500.0; the ``log_snapshot`` payload carries the corrected value
    and the corrected MNST position. Intents: 1 settled (the NVDA buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["NVDA"], []),
        splits={
            "corp_splits": {"MNST": [(date(2026, 6, 10), 2.0)]},
            "activity_applied": set(),
        },
        persist={"get_latest_snapshot": 99_000.0},
        replies=_replies(
            [_acct(100_000.0, 97_000.0), _acct(100_500.0, 93_700.0, last_equity=100_000.0)],
            [
                {"MNST": _pos(_MNST_QTY, 47.77, cost_basis=1887.11, unrealized_pl=_MNST_PL)},
                {
                    "MNST": _pos(_MNST_QTY, 47.77, cost_basis=1887.11, unrealized_pl=_MNST_PL),
                    "NVDA": _pos(33.0, 101.0, cost_basis=3300.0),
                },
            ],
        ),
    )


# --------------------------------------------------------------------------- finding 10


@scenario
def post_trade_positions_read_failure() -> dict:
    """The positions read after trading fails: the run says orders may have been submitted.

    From `test_post_trade_read_failure_reports_orders_may_have_been_submitted`. Terminal
    outcome: the run raises ``RuntimeError: positions down`` after the "Account or positions
    data unavailable after trading: orders may have been submitted" notice. Discriminating
    calls: SELL AAPL qty 10.0, then ``get_account`` (works), ``get_positions`` (raised);
    no ``log_snapshot``, no daily summary. Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend([], ["AAPL"]),
        replies=_replies(_acct(), [_AAPL_ONE, RuntimeError("positions down")]),
    )


@scenario
def dry_run_catastrophe_stop() -> dict:
    """A dry run reports a catastrophe stop and submits nothing.

    From `test_loss_halt_dry_run_reports_exit_without_submitting_orders`. NXPI is at -40% with
    the stop enabled. Terminal outcome: success, ``sells == ["NXPI"]``, ``buys == ["MSFT"]``
    (planned). Discriminating calls: "DRY RUN catastrophe stop: NXPI (qty 100.0, unrealized
    -40.0% breached -25.0% floor)", "DRY RUN buy: MSFT ($3300 ...)", no submit call and no intent.
    """
    return dict(
        dry_run=True,
        membership=FIXTURE_SPEC,
        env=_STOP_ENV,
        signals=_blend(["MSFT"], []),
        replies=_replies(
            _acct(), {"NXPI": _pos(100.0, 60.0, cost_basis=10_000.0, unrealized_pl=-4000.0)}
        ),
    )


@scenario
def dry_run_sweep_sell_and_buy() -> dict:
    """A dry run announces the sweep funding sell and the sweep buy without ordering.

    From `TestCashSweepEnabled`. The account replies are scripted: cash 1,000 at the start (so a
    funding sell is needed for the MSFT buy) and 50,000 at the sizing read (so a sweep buy fires).
    Terminal outcome: success, ``buys == ["MSFT"]``. Discriminating calls in order: "DRY RUN
    sweep sell: SPY ($7300, funding strategy buys)", "DRY RUN buy: MSFT", "DRY RUN sweep buy: SPY
    ($45000)"; no submit call and no intent.
    """
    return dict(
        dry_run=True,
        membership=FIXTURE_SPEC,
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], []),
        replies=_replies(
            [_acct(100_000.0, 1_000.0), _acct(100_000.0, 50_000.0)],
            {"SPY": _pos(247.5, 400.0)},
        ),
    )


@scenario
def invalid_equity_halt() -> dict:
    """A NaN portfolio value halts before any order.

    From `test_invalid_equity_halts_before_orders`. Terminal outcome: success with
    ``errors == ["Invalid portfolio value; drawdown cannot be assessed"]``. Discriminating
    calls: "HALTED:" notice, no ``save_peak_value``, no order. Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct(float("nan"), 50_000.0, last_equity=100_000.0), _AAPL_ONE),
    )


@scenario
def peak_save_failure() -> dict:
    """The new drawdown peak cannot be saved: halt before any order.

    From `TestPersistedDrawdown.test_failed_peak_save_halts_before_orders`. Terminal outcome:
    success with ``errors == ["Risk history unavailable: could not persist drawdown peak"]``.
    Discriminating calls: ``save_peak_value(100000.0)`` raised, "HALTED:" notice, no order.
    Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        persist={"save_peak_value": RuntimeError("write failed")},
        replies=_replies(_acct(), _AAPL_ONE),
    )


def _acct_without_previous_close(last_equity: float | None) -> dict:
    account = _acct()
    account["last_equity"] = last_equity
    return account


@scenario
def last_equity_missing_halt() -> dict:
    """A missing previous-close equity halts before any order.

    From `test_missing_broker_equity_halts_with_notice_before_orders`. Terminal outcome:
    success with ``errors == ["Invalid broker previous-close equity; daily loss cannot be
    assessed"]``. Discriminating calls: ``save_peak_value`` happened, "HALTED:" notice, no order.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct_without_previous_close(None), _AAPL_ONE),
    )


@scenario
def last_equity_invalid_halt() -> dict:
    """A zero previous-close equity halts before any order.

    From `TestPersistedDrawdown.test_invalid_broker_previous_close_halts`. Terminal outcome:
    success with ``errors == ["Invalid portfolio value or daily baseline; loss cannot be
    assessed"]``. Discriminating calls: "HALTED:" notice, no order.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct_without_previous_close(0.0), _AAPL_ONE),
    )


@scenario
def held_buy_candidate_skipped() -> dict:
    """A buy candidate that is already held is skipped.

    From `TestBuyEntries.test_skips_buy_if_already_holding`. Terminal outcome: success,
    ``buys == ["NVDA"]``. Discriminating calls: one BUY NVDA 3300.0 and no BUY AAPL.
    Intents: 1 settled (the NVDA buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["AAPL", "NVDA"], []),
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def sweep_sizing_failure() -> dict:
    """If the sweep sizing account read fails, no sweep buy is placed and the run goes on.

    Newly derived from reading `trader.py`; no existing run-level test covers it.

    Adapted from the sweep-buy block of `_run`. Cash is 50,000, so a sweep buy would fire; the
    sizing read raises ``RuntimeError: sizing read``. Terminal outcome: success with no orders.
    Discriminating calls: three ``get_account`` calls (the second raised), no ``submit_buy``.
    Intents: none created.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend([], []),
        replies=_replies([_acct(), RuntimeError("sizing read"), _acct()], {}),
    )


@scenario
def invalid_catastrophe_threshold() -> dict:
    """An invalid stop threshold stops the run before the lock, schema or signals.

    From `test_invalid_catastrophe_threshold_stops_before_orders`. ``CATASTROPHE_STOP_PCT`` is
    ``5`` (a gain, not a loss). Terminal outcome: the run raises ``ValueError: Invalid
    CATASTROPHE_STOP_PCT; expected a loss in [-1, 0)`` after a "HALTED:" notice with that text.
    Discriminating calls: only that one ``send``; no ``lock.enter``, no schema init.
    """
    return dict(
        env={**_STOP_ENV, "CATASTROPHE_STOP_PCT": "5"},
        signals=_blend(["MSFT"], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def market_clock_failure_fail_open() -> dict:
    """If the market clock cannot be read the run assumes the market is open and trades.

    From `TestMarketGates.test_market_open_get_clock_failure_fails_open`. Every ``get_clock``
    raises (the poll's read is swallowed as well). Terminal outcome: success,
    ``buys == ["MSFT"]``. Discriminating calls: ``get_clock`` raised at the gate and again in the
    poll, BUY MSFT 3300.0 still submitted. Intents: 1 settled (the MSFT buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        replies=_replies(_acct(), {}, [RuntimeError("clock down")]),
    )


@scenario
def schema_init_failure() -> dict:
    """A failing schema initialisation is logged and trading goes on.

    From `TestErrorHandling.test_db_schema_failure_does_not_block_trading`. Terminal outcome:
    success, ``buys == ["MSFT"]``. Discriminating calls: ``init_paper_schema`` raised
    (``RuntimeError: ddl failed``) and BUY MSFT 3300.0 afterwards. Intents: 1 settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"init_paper_schema": RuntimeError("ddl failed")},
        replies=_replies(_acct(), {}),
    )


@scenario
def signal_gate_error() -> dict:
    """A signal gate failure sends the gate notice and raises: no orders.

    From `TestFailClosed.test_every_gate_failure_places_no_order_and_alerts`. Terminal outcome:
    the run raises ``SignalGateError: [empty_frame] no bars``; "Signal gate — no orders placed:"
    carries that text. Discriminating calls: ``generate_core_signals`` (raised), one ``send``,
    no account read, no tape refresh.
    """
    return dict(
        env=_LIVE_ENV,
        signals=SignalGateError("empty_frame", "no bars"),
        replies=_replies(_acct(), {}),
    )


@scenario
def signal_generation_error() -> dict:
    """Any other signal error sends the failure notice and raises: no orders.

    From `TestFailClosed.test_unexpected_signal_exception_also_fails_closed`. Terminal outcome:
    the run raises ``RuntimeError: yfinance down``; "Paper trading failed: signal generation
    error" carries that text. Discriminating calls: ``generate_core_signals`` (raised), one
    ``send``, no account read.
    """
    return dict(
        env=_LIVE_ENV,
        signals=RuntimeError("yfinance down"),
        replies=_replies(_acct(), {}),
    )


@scenario
def overlay_fallback_and_membership_warning() -> dict:
    """An overlay fallback and a membership-list warning are announced and trading goes on.

    From `TestLiveMembershipGate.test_a_warning_reaches_the_notifier_and_trading_goes_on`. The
    signals say the overlay reused last month's weights and the sleeve carries a membership
    ``warn``. Terminal outcome: success, ``buys == ["MSFT"]``. Discriminating calls in order:
    the "Membership list" warning, the "Overlay fallback" warning, then BUY MSFT 3300.0.
    Intents: 1 settled (the MSFT buy).
    """
    blend = _blend(["MSFT"], [])
    blend["fallback_used"] = True
    blend["sleeves"]["sp500"]["membership_gate"] = {
        "level": "warn",
        "message": "the S&P 500 list was verified 40 days ago",
    }
    return dict(env=_LIVE_ENV, signals=blend, replies=_replies(_acct(), {}))


@scenario
def account_read_failure() -> dict:
    """The first account read fails: halt with a notice, no orders.

    From `test_account_read_failure_halts_with_notice_before_orders`. Terminal outcome: the run
    raises ``RuntimeError: account down`` after "HALTED: Account data unavailable; no orders
    submitted". Discriminating calls: ``get_account`` (raised), no ``get_positions``, no order.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(RuntimeError("account down"), _AAPL_ONE),
    )


@scenario
def positions_read_failure() -> dict:
    """The first positions read fails: halt with a notice, no orders.

    From `test_position_read_failure_halts_with_notice_before_orders`. Terminal outcome: the run
    raises ``RuntimeError: positions down`` after "HALTED: Position data unavailable; no orders
    submitted". Discriminating calls: ``get_positions`` (raised), no order.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        replies=_replies(_acct(), RuntimeError("positions down")),
    )


# =========================================================================== second follow-up
# Terminal-status sites, notifier return values, ledger failures and recovery edge cases.


def _expired_or_rejected_legacy(status: str) -> dict:
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["KO"], []),
        persist={"get_pending_orders": [_pending("pend-1", "KO", "BUY")]},
        existing_orders=[
            _held_order(
                "pend-1",
                "legacy-1",
                "KO",
                "BUY",
                600.0,
                [{"status": status, "filled_qty": 0.0, "filled_avg_price": None}],
            )
        ],
        replies=_replies(_acct(), {}),
    )


@scenario
def legacy_pending_expired() -> dict:
    """An expired legacy pending order is dropped without a trade alert and frees its symbol.

    Newly derived from reading `trader.py` (the legacy terminal-status tuple); the sibling of
    `legacy_pending_canceled`. Terminal outcome: success, ``buys == ["KO"]``. Discriminating
    calls: ``settle_legacy_pending_order("pend-1", 0.0, None, "expired")`` with no trade alert
    for it, then BUY KO 3300.0. Removing "expired" from the legacy terminal statuses would leave
    KO blocked and nothing settled. Intents: 1 settled (the KO buy).
    """
    return _expired_or_rejected_legacy("expired")


@scenario
def legacy_pending_rejected() -> dict:
    """A rejected legacy pending order is dropped without a trade alert and frees its symbol.

    Newly derived from reading `trader.py`; the sibling of `legacy_pending_canceled`. Terminal
    outcome: success, ``buys == ["KO"]``. Discriminating calls:
    ``settle_legacy_pending_order("pend-1", 0.0, None, "rejected")`` with no trade alert for it,
    then BUY KO 3300.0. Removing "rejected" from the legacy terminal statuses would leave KO
    blocked and nothing settled. Intents: 1 settled (the KO buy).
    """
    return _expired_or_rejected_legacy("rejected")


@scenario
def poll_order_expired() -> dict:
    """A sell that goes ``expired`` while polled ends the poll at once and blocks the buys.

    Newly derived from reading `_poll_orders`. The AAPL sell is ``new``, then ``expired`` with no
    fill. Terminal outcome: success, ``sells == ["AAPL"]``, ``buys == []``, errors ==
    ["SELL AAPL not fully filled (expired); new strategy buys blocked"]. Discriminating calls: 2
    ``get_order`` polls and 1 sleep (not the 16 polls of a never-terminal order),
    ``settle_order_intent(.., 0.0, None, "expired")``, a trade alert with status ``expired``, no
    BUY MSFT. Intents: 1 settled (the AAPL sell).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        fills={
            "AAPL": [
                {"status": "new"},
                {"status": "expired", "filled_qty": 0.0, "filled_avg_price": None},
            ]
        },
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def trade_alert_returns_false() -> dict:
    """A trade alert that Telegram does not deliver (returns ``False``) changes nothing.

    Newly derived from reading `trader.py` (it ignores the return value of `trade_alert`). The
    notifier is scripted to return ``False`` for the alert of the MSFT buy. Terminal outcome:
    success, ``buys == ["MSFT"]``. Discriminating calls: BUY MSFT 3300.0, the ``trade_alert``
    call with its rendered text still recorded, and the run carrying on to the daily summary
    and the snapshot. Intents: 1 settled (the MSFT buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        notifier={"trade_alert": False},
        replies=_replies(_acct(), {}),
    )


@scenario
def daily_summary_returns_false() -> dict:
    """A daily summary that Telegram does not deliver (returns ``False``) changes nothing.

    Newly derived from reading `trader.py` (it ignores the return value of `daily_summary`).
    The notifier is scripted to return ``False`` for the summary. Terminal outcome: success,
    ``buys == ["MSFT"]``, no errors. Discriminating calls: the ``daily_summary`` call with its
    rendered text recorded, then the "Risk" line and ``log_snapshot`` as usual.
    Intents: 1 settled (the MSFT buy).
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        notifier={"daily_summary": False},
        replies=_replies(_acct(), {}),
    )


@scenario
def intent_settle_failure() -> dict:
    """The ledger cannot settle a filled order: the run stops with the recovery notice.

    Newly derived from reading `_reconcile_order_intents`. The AAPL sell fills, then
    ``settle_order_intent`` raises ``RuntimeError: ledger write failed``. Terminal outcome: the
    run raises that error after "Order recovery/pending orders failed: ledger write failed" is
    sent. Discriminating calls: SELL AAPL qty 10.0, ``settle_order_intent`` (raised), no BUY.
    Intents: 1 unresolved (the AAPL sell), none settled.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], ["AAPL"]),
        persist={"settle_order_intent": RuntimeError("ledger write failed")},
        replies=_replies(_acct(), _AAPL_ONE),
    )


@scenario
def unresolved_intents_read_failure() -> dict:
    """The ledger cannot be read for unresolved intents at startup: the run stops.

    Newly derived from reading `_reconcile_order_intents`. Terminal outcome: the run raises
    ``RuntimeError: ledger offline`` after "Order recovery/pending orders failed: ledger
    offline" is sent. Discriminating calls: ``get_unresolved_order_intents`` (raised), no
    market-clock call, no signals, no order. Intents: none created.
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["MSFT"], []),
        persist={"get_unresolved_order_intents": RuntimeError("ledger offline")},
        replies=_replies(_acct(), {}),
    )


@scenario
def recovery_notice_fails() -> dict:
    """A failing recovery notice does not hide the recovery error.

    Newly derived from reading `_reconcile_order_intents` (the nested ``except`` that logs a
    failed notice). A seeded intent is still working; the "Order recovery" notice raises
    ``RuntimeError: telegram down``. Terminal outcome: the run raises the recovery error
    ``RuntimeError: Unresolved working orders block submissions``, not the notifier's.
    Discriminating calls: the notice ``send`` (raised) and nothing else sent. Intents: the
    seeded row stays unresolved.
    """
    return _recovery(
        [{"status": "new"}], notifier={"Order recovery": RuntimeError("telegram down")}
    )


@scenario
def recovery_filled_zero_quantity() -> dict:
    """A seeded order the broker calls ``filled`` with no shares is refused.

    Newly derived from reading `_reconcile_order_intents`. Terminal outcome: the run raises
    ``ValueError: Filled order has no fill``. Discriminating calls: no attach, no settle, no
    market-clock call. Intents: the seeded row stays unresolved.
    """
    return _recovery([{"status": "filled", "filled_qty": 0.0, "filled_avg_price": None}])


# --------------------------------------------------------------------------- Phase 4 pins
# Task 32 of the Phase 4 plan: environment-setting behavior and the pending-order tuple fields
# that no earlier scenario shows. Each one was checked against the 96 above; see the commit message.


@scenario
def sweep_min_clip_invalid_fail_soft() -> dict:
    """An unreadable ``SWEEP_MIN_CLIP_USD`` makes sweep-buy sizing fail soft: no sweep buy.

    Based on `sweep_sizing_failure` (same account and empty signals; cash 50,000 would trigger a
    sweep buy). ``SWEEP_MIN_CLIP_USD`` is ``abc``: ``min_clip_usd()`` raises ``ValueError`` inside
    the sweep-buy ``try`` (`trader.py:1351-1362`), which logs and goes on. Terminal outcome:
    success with no orders and no errors. Discriminating calls: three ``get_account`` calls, none
    raised, no ``submit_buy``. Intents: none created.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV, "SWEEP_MIN_CLIP_USD": "abc"},
        signals=_blend([], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def sweep_buy_trigger_invalid_fail_soft() -> dict:
    """An unreadable ``SWEEP_BUY_TRIGGER_PCT`` makes sweep-buy sizing fail soft: no sweep buy.

    Same as `sweep_min_clip_invalid_fail_soft` with ``SWEEP_BUY_TRIGGER_PCT=abc``
    (``buy_trigger_pct()`` raises inside the ``try``, `trader.py:1351-1362`). Terminal outcome:
    success with no orders and no errors. Discriminating calls: three ``get_account`` calls, no
    ``submit_buy``. Intents: none created.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV, "SWEEP_BUY_TRIGGER_PCT": "abc"},
        signals=_blend([], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def sweep_reserve_invalid_ends_run() -> dict:
    """An unreadable ``SWEEP_CASH_RESERVE_PCT`` ends the run at the funding-sell computation.

    Based on `sweep_sizing_failure` with ``SWEEP_CASH_RESERVE_PCT=abc``. ``reserve_pct()`` is read
    in the funding-sell computation (`trader.py:1189-1195`), which has no ``try``. Terminal
    outcome: the run raises ``ValueError: could not convert string to float: 'abc'`` after the
    account and position reads and before any order. Discriminating calls: one ``get_account``,
    no ``submit_buy`` or ``submit_sell``, no snapshot. Intents: none created.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV, "SWEEP_CASH_RESERVE_PCT": "abc"},
        signals=_blend([], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def sweep_reserve_invalid_but_sweep_off() -> dict:
    """With the sweep off nothing reads ``SWEEP_CASH_RESERVE_PCT``, so ``abc`` is harmless.

    Based on `plain_buys_and_sells` with a single MSFT buy. Terminal outcome: success,
    ``buys == ["MSFT"]``, no errors. Discriminating calls: BUY MSFT notional 3300.0 and no error.
    Intents: 1 settled (the MSFT buy), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, "SWEEP_CASH_RESERVE_PCT": "abc"},
        signals=_blend(["MSFT"], []),
        replies=_replies(_acct(), {}),
    )


@scenario
def untradable_symbols_from_env() -> dict:
    """``UNTRADABLE_SYMBOLS=" aapl, ,Msft "`` freezes both held names, upper-cased, blanks dropped.

    Based on `untradable_holding` (`trader.py:881-893`, `untradable.py:38`). AAPL and MSFT are held
    and AAPL carries an exit signal; NVDA is the buy candidate. Terminal outcome: success,
    ``sells == []`` (a frozen name is never sold), ``untradable == ["AAPL", "MSFT"]``.
    Discriminating calls: ``get_untradable(["NVDA"])`` (only the unconfigured name is asked);
    "Untradable holdings: AAPL, MSFT" is sent. Intents: 1 settled (the NVDA buy), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, "UNTRADABLE_SYMBOLS": " aapl, ,Msft "},
        signals=_blend(["NVDA"], ["AAPL"]),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0), "MSFT": _pos(5.0, 400.0)}),
    )


@scenario
def catastrophe_enabled_pct_blank() -> dict:
    """The stop is on and ``CATASTROPHE_STOP_PCT`` is blank: the default -25% applies.

    Based on `catastrophe_stop` with NXPI at -30% (between a blank setting's -25% default and
    nothing). Terminal outcome: success, ``sells == ["NXPI"]``, ``buys == ["MSFT"]``.
    Discriminating calls: one SELL NXPI qty 100.0; BUY MSFT notional 3300.0. Intents: 2 settled
    (the NXPI stop sell, the MSFT buy), none unresolved.
    """
    return dict(
        env={**_STOP_ENV, "CATASTROPHE_STOP_PCT": ""},
        signals=_blend(["MSFT"], []),
        replies=_replies(
            _acct(), {"NXPI": _pos(100.0, 70.0, cost_basis=10_000.0, unrealized_pl=-3000.0)}
        ),
    )


@scenario
def pending_order_all_five_fields() -> dict:
    """A SELL and a BUY whose poll lookups all fail are reported as pending, with their amounts.

    Based on `sweep_sell_and_strategy_sell` and `catastrophe_stop` combined: NXPI at -40% (a
    catastrophe stop), AAPL with an exit signal, SPY held (the sweep position) and cash 1,000 so
    that the funding sell fires, then a BUY MSFT. Submission order: SELL NXPI
    (``catastrophe_stop``), SELL SPY (``cash_sweep``), SELL AAPL, BUY MSFT. ``get_order`` answers
    the first poll of NXPI and AAPL from the broker; every call for SPY and for MSFT fails, so both
    go to the fallback record of `_poll_orders` (`trader.py:460-469`) and are alerted as
    ``pending``. The fallback's own fields (``symbol``, ``side``, ``notional``) are never read after
    the poll, so this golden does not distinguish them; `test_phase4_pins.py` pins them on the poll
    itself. Terminal outcome: success, ``sells == ["NXPI", "AAPL"]`` (the stop first), ``buys ==
    ["MSFT"]``, no errors (a pending sweep sell blocks nothing). Discriminating calls: trade alerts
    for SPY (SELL, status ``pending``, amount 7300.0) and MSFT (BUY, ``pending``, amount 3300.0).
    Intents: 4 settled (recovery finds each order at the broker), none unresolved.
    """
    return dict(
        env={**_STOP_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], ["AAPL"]),
        fills={
            "SPY:SELL": [{"status": "filled", "filled_qty": 18.25, "filled_avg_price": 400.0}],
            "AAPL": [{"status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.0}],
        },
        replies={
            **_replies(
                [
                    _acct(100_000.0, 1_000.0),
                    _acct(100_000.0, 6_500.0),
                    _acct(100_033.0, 6_500.0),
                ],
                [
                    {
                        "NXPI": _pos(100.0, 60.0, cost_basis=10_000.0, unrealized_pl=-4000.0),
                        "SPY": _pos(240.75, 400.0),
                        "AAPL": _pos(10.0, 150.0),
                    },
                    {
                        "SPY": _pos(222.5, 400.0),
                        "MSFT": _pos(33.0, 101.0, cost_basis=3300.0),
                    },
                ],
            ),
            "get_order": [None, RuntimeError("poll down"), None, RuntimeError("poll down")],
        },
    )
