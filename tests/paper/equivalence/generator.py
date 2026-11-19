"""Build reproducible, bounded inputs for paper-trader equivalence tests."""

from dataclasses import dataclass
from datetime import date, datetime
from random import Random

from tests.paper.transcript.harness import PT, Scenario
from tests.paper.transcript.scenarios import FIXTURE_SPEC, PERSIST_DEFAULTS, _acct, _pos

_SYMBOLS = ("AAPL", "AMZN", "GOOGL", "KO", "MSFT", "NVDA")


@dataclass(frozen=True)
class Case:
    """One scenario and failures injected by the recording runner."""

    scenario: Scenario
    lock_fails_on: int | None
    membership_error: str | None


def build_case(seed: int) -> Case:
    """Generate a deterministic scenario using only a local seeded random source."""
    rng = Random(seed)
    mode = rng.randrange(100)
    special = rng.random()
    if mode < 60 and special < 0.2:
        kind = ("late_stop", "late_exit", "late_sleeve", "duplicate_buy")[int(special / 0.05)]
        first = _pos(10.0, 100.0, cost_basis=2000.0)
        second = _pos(10.0, 100.0, cost_basis=2000.0 if kind == "late_stop" else 1000.0)
        second.pop("qty")
        positions = (
            {"AAPL": first, "KO": second}
            if kind.startswith("late_") and kind != "late_sleeve"
            else {}
        )
        sleeves = {
            "sleeve0": {
                "buys": ["MSFT"] if kind in ("late_sleeve", "duplicate_buy") else [],
                "sells": ["KO"] if kind == "late_exit" else [],
                "as_of": "2026-06-19",
                "universe_size": 100,
                "last_exit": {},
            }
        }
        if kind in ("late_sleeve", "duplicate_buy"):
            sleeves["sleeve1"] = {
                **sleeves["sleeve0"],
                "buys": ["NVDA" if kind == "late_sleeve" else "MSFT"],
            }
        scenario = Scenario(
            name=f"{kind}_{seed}",
            env={
                "MAX_POSITIONS": "5",
                "CATASTROPHE_STOP_ENABLED": "true",
                "CATASTROPHE_STOP_PCT": "-0.25",
            },
            signals={
                "sleeves": sleeves,
                "weights": {
                    "sleeve0": 1.0,
                    **({"sleeve1": 1.0} if kind == "duplicate_buy" else {}),
                },
                "scale": 1.0,
                "rebalanced_today": False,
                "fallback_used": False,
            },
            persist={**PERSIST_DEFAULTS},
            replies={
                "get_account": [_acct()],
                "get_positions": [positions],
                "get_clock": [{"is_open": True, "next_close": "2026-06-19 16:00:00-04:00"}],
            },
        )
        return Case(scenario, None, None)
    dry_run = mode >= 60
    read_only = not (85 <= mode < 95)
    env = {
        "MAX_POSITIONS": str(rng.choice((2, 3, 5))),
        "CASH_SWEEP_ENABLED": str(rng.choice((True, False))).lower(),
        "CATASTROPHE_STOP_ENABLED": str(rng.choice((True, False))).lower(),
        "CATASTROPHE_STOP_PCT": str(round(rng.uniform(-0.4, -0.1), 3)),
    }
    held = rng.sample(_SYMBOLS, rng.randrange(7))
    positions = {}
    for symbol in held:
        qty = rng.choice((10.0, 20.0, 0.0, -1.0, float("nan"), None))
        position = _pos(10.0, 100.0, cost_basis=rng.choice((1000.0, 2000.0)))
        position["qty"] = qty
        positions[symbol] = position
    if rng.random() < 0.3:
        positions["SPY"] = _pos(100.0, 400.0)
    count = rng.choice((1, 2))
    sleeves = {}
    weights = {}
    available = list(_SYMBOLS)
    for index in range(count):
        name = f"sleeve{index}"
        buys = rng.sample(available, rng.randrange(min(3, len(available)) + 1))
        available = [symbol for symbol in available if symbol not in buys]
        sleeves[name] = {
            "buys": buys,
            "sells": rng.sample(held, rng.randrange(len(held) + 1)),
            "as_of": "2026-06-10" if rng.random() < 0.05 else "2026-06-19",
            "universe_size": 100,
            "last_exit": {symbol: "2026-06-10" for symbol in held if rng.random() < 0.2},
        }
        weights[name] = rng.uniform(0.05, 1.0)
    signals = {
        "sleeves": sleeves,
        "weights": weights,
        "scale": rng.uniform(0.1, 1.0),
        "rebalanced_today": False,
        "fallback_used": rng.random() < 0.1,
    }
    persist = {
        **PERSIST_DEFAULTS,
        "get_last_buy_dates": {
            symbol: rng.choice((date(2026, 6, 5), date(2026, 6, 15))) for symbol in held
        },
        "get_peak_value": rng.choice((None, 100_000.0, 120_000.0)),
    }
    account = _acct(
        cash=rng.choice((1_000.0, 50_000.0)), last_equity=rng.choice((100_000.0, 110_000.0))
    )
    replies = {
        "get_account": [account],
        "get_positions": [positions],
        "get_clock": [{"is_open": True, "next_close": "2026-06-19 16:00:00-04:00"}],
    }
    fills = {}
    for symbol in _SYMBOLS:
        status = rng.choice(("filled", "partial", "new", "rejected", "canceled", "expired"))
        if status == "partial":
            fills[symbol] = [
                {"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 100.0},
                {"status": "filled", "filled_qty": 10.0, "filled_avg_price": 100.0},
            ]
        else:
            fills[symbol] = [
                {
                    "status": status,
                    "filled_qty": 10.0 if status == "filled" else 0.0,
                    "filled_avg_price": 100.0 if status == "filled" else None,
                }
            ]
    existing_orders = []
    if rng.random() < 0.1:
        symbol = rng.choice(_SYMBOLS)
        persist["get_pending_orders"] = [
            {
                "order_id": "pending",
                "symbol": symbol,
                "side": "BUY",
                "created_at": None,
                "flagged_stale": False,
            }
        ]
        existing_orders = [
            {
                "id": "pending",
                "client_order_id": "legacy",
                "symbol": symbol,
                "side": "BUY",
                "notional": 1000.0,
                "statuses": [{"status": "new"}],
            }
        ]
    if rng.random() < 0.1:
        env["UNTRADABLE_SYMBOLS"] = rng.choice(_SYMBOLS)
    if rng.random() < 0.1:
        replies["submit_buy"] = [None] * rng.randrange(3) + [RuntimeError("submit failed")]
    if rng.random() < 0.05:
        replies["get_clock"] = [RuntimeError("clock unavailable")]
    splits = {"corp_splits": {}, "activity_applied": set()}
    dividends = {"corp_dividends": {}, "credited_keys": set()}
    if held and rng.random() < 0.1:
        splits["corp_splits"][held[0]] = [(date(2026, 6, 10), 2.0)]
    if held and rng.random() < 0.1:
        dividends["corp_dividends"][held[0]] = [(date(2026, 6, 1), 0.5)]
        persist["get_snapshot_history"] = [(date(2026, 5, 20), {held[0]: {"qty": 10.0}})]
    notifier = (
        {"trade_alert": rng.choice((False, RuntimeError("notifier unavailable")))}
        if rng.random() < 0.1
        else {}
    )
    scenario = Scenario(
        name=f"seed_{seed}",
        dry_run=dry_run,
        read_only=read_only,
        membership=FIXTURE_SPEC if dry_run else "db",
        env=env,
        now_pt=datetime(2026, 6, 19, 13, 10, tzinfo=PT)
        if mode >= 95
        else datetime(2026, 6, 19, 12, 46, tzinfo=PT),
        signals=signals,
        fills=fills,
        existing_orders=existing_orders,
        splits=splits,
        dividends=dividends,
        notifier=notifier,
        persist=persist,
        replies=replies,
        untradable=set(rng.sample(_SYMBOLS, rng.randrange(3))),
    )
    return Case(
        scenario,
        rng.choice((None, None, None, 1, 2, 3)),
        "membership unavailable" if rng.random() < 0.1 else None,
    )
