"""The 38 recorded paper-trader scenarios; each docstring says what its golden must show.

Every scenario is built from inputs of an existing test in ``tests/paper/test_trader.py`` (cited
in its docstring). The goldens record what the trader does today: a scenario's docstring is the
reviewer's expectation, ``test_golden_shows_scenario_behavior`` is what pins it. Client order ids
are pinned (``00000000-0000-0000-0000-00000000000n``, n counting from 1 in submission order)
and broker order ids are ``ord-n`` in the same order.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

from tests.paper.transcript.harness import PT, Scenario, Seq

#: The membership spec the root conftest selects: dry runs may rehearse any source.
FIXTURE_SPEC = "fixture:sp500_v1_2026-06-02"

#: How many scenarios the suite records (the plan's 34 base ones plus four splits and additions).
EXPECTED_SCENARIO_COUNT = 38

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
    """A legacy pending-order row (``created_at=None``: the trader's `now` is not pinned)."""
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
    """
    return dict(
        env=_LIVE_ENV,
        signals=_blend(["NVDA", "GOOGL"], ["AAPL"]),
        replies=_replies(
            _acct(),
            {
                "AAPL": _pos(10.0, 150.0, cost_basis=1450.0),
                "KO": _pos(20.0, 60.0),
            },
        ),
    )


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
    sweep buy afterwards (cash 5,000 is under the 8% trigger).
    Intents: 2 settled (the SPY sell, the MSFT buy), none unresolved.
    """
    return dict(
        env={**_LIVE_ENV, **_SWEEP_ENV},
        signals=_blend(["MSFT"], []),
        replies=_replies(
            [_acct(cash=1_000.0), _acct(cash=5_000.0)],
            {"SPY": _pos(50.0, 400.0)},
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
    Intents: 3 settled (SPY sell, AAPL sell, MSFT buy), none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "2", **_SWEEP_ENV},
        signals=_blend(["MSFT", "NVDA"], ["AAPL"]),
        replies=_replies(
            [_acct(cash=1_000.0), _acct(cash=5_000.0)],
            {
                "SPY": _pos(50.0, 400.0),
                "AAPL": _pos(10.0, 150.0),
                "KO": _pos(20.0, 60.0),
            },
        ),
    )


# --------------------------------------------------------------------------- fills and pending

_AAPL_ONE = {"AAPL": _pos(10.0, 150.0)}


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

    Adapted from the sleeve-cap math in `risk.sleeve_slot_caps` (no `run`-level test exists).
    MAX_POSITIONS is 4 with two 50% sleeves, so each sleeve gets 2 slots; the sp500 sleeve
    offers three candidates. Terminal outcome: success, ``buys == ["MSFT", "NVDA", "AMZN"]``.
    Discriminating calls: BUY MSFT and BUY NVDA at 1650.0 each, no BUY GOOGL (it would trade if
    the sleeve cap alone were lifted: 3 of 4 global slots), BUY AMZN 1650.0 from the other sleeve.
    Intents: 3 settled, none unresolved.
    """
    return dict(
        env={"MAX_POSITIONS": "4"},
        signals=_two_sleeves(
            ["MSFT", "NVDA", "GOOGL"], ["AMZN"], {"sp500": 0.5, "nasdaq100": 0.5}, 1.0
        ),
        replies=_replies(_acct(), {}),
    )


@scenario
def concentration_limit() -> dict:
    """The 5% concentration limit refuses one candidate that would otherwise trade.

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
