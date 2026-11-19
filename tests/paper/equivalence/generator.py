"""Build reproducible, bounded inputs for paper-trader equivalence tests."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime
from random import Random

from tests.paper.transcript.harness import PT, Scenario
from tests.paper.transcript.scenarios import (
    FIXTURE_SPEC,
    PERSIST_DEFAULTS,
    SCENARIOS,
    _acct,
    _blend,
    _pos,
    _replies,
    _two_sleeves,
)

_SYMBOLS = ("AAPL", "AMZN", "GOOGL", "KO", "MSFT", "NVDA")


@dataclass(frozen=True)
class Case:
    """One scenario and failures injected by the recording runner."""

    scenario: Scenario
    lock_fails_on: int | None
    membership_error: str | None


def _vary_symbols(value: object, suffix: str) -> object:
    """Vary synthetic symbols consistently in the structured cases."""
    symbols = {
        "AAPL",
        "KO",
        "MSFT",
        "NVDA",
        "NXPI",
        "MNST",
        "VZ",
        "FLAT",
        "EX",
        "B1",
        "B2",
        "B3",
        "M1",
        "M2",
        "M3",
        "F1",
        "F2",
        "GOOGL",
        "AMZN",
    }
    if isinstance(value, str):
        symbol, sep, rest = value.partition(":")
        return symbol + suffix + (sep + rest if sep else "") if symbol in symbols else value
    if isinstance(value, dict):
        return {_vary_symbols(k, suffix): _vary_symbols(v, suffix) for k, v in value.items()}
    if isinstance(value, list):
        return [_vary_symbols(item, suffix) for item in value]
    if isinstance(value, tuple):
        return tuple(_vary_symbols(item, suffix) for item in value)
    if isinstance(value, set):
        return {_vary_symbols(item, suffix) for item in value}
    return value


def _varied_scenario(scenario: Scenario, seed: int) -> Scenario:
    """Vary symbols, position sizes, and sleeve names while preserving scenario roles."""
    variant = seed // 80
    if not variant:
        return scenario
    suffix = chr(ord("A") + variant - 1)
    changes = {
        name: _vary_symbols(getattr(scenario, name), suffix)
        for name in (
            "signals",
            "replies",
            "splits",
            "dividends",
            "fills",
            "persist",
            "existing_orders",
            "untradable",
        )
    }
    scenario = replace(scenario, **changes)
    if scenario.name.startswith("two_caps_"):
        signals = scenario.signals
        sleeves = signals["sleeves"]
        weights = signals["weights"]
        signals["sleeves"] = {f"{name}_{suffix}": sleeve for name, sleeve in sleeves.items()}
        signals["weights"] = {f"{name}_{suffix}": weight for name, weight in weights.items()}
        if variant % 2:
            for sleeve in signals["sleeves"].values():
                sleeve["buys"].reverse()
    if scenario.name.startswith("slot_stop_") and variant % 2:
        scenario.signals["sleeves"]["sp500"]["buys"].append(f"B4{suffix}")
    for position_set in scenario.replies.get("get_positions", []):
        for symbol, position in position_set.items():
            if symbol == "SPY":
                continue
            for key in ("qty", "market_value", "cost_basis", "unrealized_pl"):
                if key in position and isinstance(position[key], (int, float)):
                    position[key] *= 1.0 + variant / 20
    return scenario


def build_case(seed: int) -> Case:
    """Generate a deterministic scenario using only a local seeded random source."""
    rng = Random(seed)
    mode = rng.randrange(100)
    special = seed % 80
    if special in (8, 9, 10, 11, 12, 13, 14, 16):
        if special == 8:
            scenario = Scenario(
                name=f"slot_stop_{seed}",
                env={"MAX_POSITIONS": "3", "CATASTROPHE_STOP_ENABLED": "true"},
                signals=_blend(["B1", "B2", "B3"], ["EX"]),
                persist={**PERSIST_DEFAULTS},
                replies=_replies(
                    _acct(),
                    {
                        "FLAT": _pos(1.0, 100.0),
                        "EX": _pos(1.0, 100.0),
                        "NXPI": _pos(10.0, 70.0, cost_basis=1000.0),
                    },
                ),
            )
        elif special == 9:
            scenario = replace(deepcopy(SCENARIOS["split_correction"]), name=f"split_sold_{seed}")
        elif special == 10:
            scenario = replace(
                deepcopy(SCENARIOS["sell_rejected_by_broker"]), name=f"rejected_exit_{seed}"
            )
        elif special == 11:
            scenario = Scenario(
                name=f"two_caps_{seed}",
                env={"MAX_POSITIONS": "5"},
                signals=_two_sleeves(
                    ["AAPL", "MSFT"], ["KO", "NVDA"], {"sp500": 0.2, "nasdaq100": 0.2}, 1.0
                ),
                persist={**PERSIST_DEFAULTS},
                replies=_replies(_acct(), {}),
            )
        elif special == 14:
            scenario = replace(deepcopy(SCENARIOS["catastrophe_stop"]), name=f"stopped_buy_{seed}")
        elif special == 16:
            scenario = replace(
                deepcopy(SCENARIOS["sell_partial_fill_canceled"]), name=f"partial_exit_{seed}"
            )
        elif special == 13:
            scenario = replace(
                deepcopy(SCENARIOS["catastrophe_stop_pending_order"]),
                name=f"pending_stop_{seed}",
            )
        else:
            scenario = Scenario(
                name=f"stop_sweep_{seed}",
                env={
                    "MAX_POSITIONS": "3",
                    "CATASTROPHE_STOP_ENABLED": "true",
                    "CASH_SWEEP_ENABLED": "true",
                },
                signals=_blend(["M1", "M2", "M3"], []),
                persist={**PERSIST_DEFAULTS},
                replies=_replies(
                    _acct(cash=1000.0),
                    {
                        "NXPI": _pos(100.0, 70.0, cost_basis=10000.0),
                        "F1": _pos(1.0, 100.0),
                        "F2": _pos(1.0, 100.0),
                        "SPY": _pos(247.5, 400.0),
                    },
                ),
            )
        for account in scenario.replies["get_account"]:
            account["portfolio_value"] += (seed // 80) * 100.0
        return Case(_varied_scenario(scenario, seed), None, None)
    if special < 4:
        kind = ("late_stop", "late_exit", "late_sleeve", "duplicate_buy")[special]
        price = rng.choice((80.0, 100.0, 120.0))
        qty = 10.0 + seed // 80
        first = _pos(qty, price, cost_basis=qty * 2.0 * price)
        second = _pos(qty, price, cost_basis=qty * (2.0 if kind == "late_stop" else 1.0) * price)
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
                "scale": 0.6 + (seed // 80) * 0.04,
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
        return Case(_varied_scenario(scenario, seed), None, None)
    if special in (6, 7, 15):
        price = rng.choice((80.0, 100.0, 120.0))
        positions = {symbol: _pos(10.0 + seed // 80, price) for symbol in ("AAPL", "KO")}
        if special in (7, 15):
            positions["KO"]["cost_basis"] = float("nan")
        scenario = Scenario(
            name=f"{'sell_failure' if special == 6 else 'unassessable_stop'}_{seed}",
            env={"MAX_POSITIONS": "5", "CATASTROPHE_STOP_ENABLED": "true"},
            signals={
                "sleeves": {
                    "sleeve0": {
                        "buys": ["MSFT"],
                        "sells": ["AAPL", "KO"] if special == 6 else ["AAPL"],
                        "as_of": "2026-06-19",
                        "universe_size": 100,
                        "last_exit": {},
                    }
                },
                "weights": {"sleeve0": 1.0},
                "scale": 1.0,
                "rebalanced_today": False,
                "fallback_used": False,
            },
            persist={**PERSIST_DEFAULTS},
            notifier={"HALTED buys": RuntimeError("notice failed")} if special == 15 else {},
            replies={
                "get_account": [_acct()],
                "get_positions": [positions],
                "get_clock": [{"is_open": True, "next_close": "2026-06-19 16:00:00-04:00"}],
                **({"submit_sell": [None, RuntimeError("submit failed")]} if special == 6 else {}),
            },
        )
        return Case(_varied_scenario(scenario, seed), None, None)
    dry_run = mode >= 60
    read_only = 60 <= mode < 85 or mode >= 95
    env = {
        "MAX_POSITIONS": str(rng.choice((2, 3, 5))),
        "CASH_SWEEP_ENABLED": str(rng.choice((True, False))).lower(),
        "CATASTROPHE_STOP_ENABLED": str(rng.choice((True, False))).lower(),
        "CATASTROPHE_STOP_PCT": str(round(rng.uniform(-0.4, -0.1), 3)),
    }
    held = rng.sample(_SYMBOLS, rng.choice((0, 0, 1, 1, 2, 3, 4, 5, 6)))
    positions = {}
    for symbol in held:
        qty = rng.choice((10.0, 20.0, 0.0, -1.0, float("nan"), None))
        position = _pos(
            qty if qty in (10.0, 20.0) else 10.0, 100.0, cost_basis=rng.choice((1000.0, 2000.0))
        )
        position["qty"] = qty
        if rng.random() < 0.12:
            position["cost_basis"] = float("nan")
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
        status = rng.choice(
            (
                "default",
                "default",
                "partial",
                "partial_canceled",
                "new",
                "rejected",
                "canceled",
                "expired",
            )
        )
        if status == "default":
            continue
        if status in ("partial", "partial_canceled"):
            qty = positions.get(symbol, {}).get("qty")
            if qty not in (10.0, 20.0):
                continue
            fills[f"{symbol}:SELL"] = [
                {"status": "partially_filled", "filled_qty": 4.0, "filled_avg_price": 100.0},
                {
                    "status": "filled" if status == "partial" else "canceled",
                    "filled_qty": qty if status == "partial" else 4.0,
                    "filled_avg_price": 100.0,
                },
            ]
        else:
            fills[symbol] = [{"status": status, "filled_qty": 0.0, "filled_avg_price": None}]
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
    if rng.random() < 0.1:
        replies["submit_sell"] = [None] * rng.randrange(1, 3) + [RuntimeError("submit failed")]
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
    if special == 4:
        dry_run, read_only = False, False
    if special == 5:
        dry_run, read_only = False, False
        env["CASH_SWEEP_ENABLED"] = "true"
        positions = {"SPY": _pos(240.0, 400.0)}
        sleeves = {"sleeve0": sleeves["sleeve0"]}
        signals["sleeves"] = sleeves
        sleeves["sleeve0"]["as_of"] = "2026-06-19"
        sleeves["sleeve0"]["buys"] = ["MSFT"]
        env.pop("UNTRADABLE_SYMBOLS", None)
        sleeves["sleeve0"]["sells"] = []
        replies["get_account"] = [_acct(cash=1_000.0 + 100.0 * (seed // 80)), _acct(cash=50_000.0)]
        replies["get_positions"] = [positions]
        replies["get_clock"] = [{"is_open": True, "next_close": "2026-06-19 16:00:00-04:00"}]
        replies.pop("submit_buy", None)
        replies.pop("submit_sell", None)
        fills = {}
        persist = {**PERSIST_DEFAULTS}
    scenario = Scenario(
        name=f"seed_{seed}",
        dry_run=dry_run,
        read_only=read_only,
        membership=FIXTURE_SPEC if dry_run else "db",
        env=env,
        now_pt=datetime(2026, 6, 19, 13, 10 + seed // 80, tzinfo=PT)
        if mode >= 95 or special == 4
        else datetime(2026, 6, 19, 12, 46, tzinfo=PT),
        signals=signals,
        fills=fills,
        existing_orders=existing_orders,
        splits=splits,
        dividends=dividends,
        notifier=notifier,
        persist=persist,
        replies=replies,
        untradable=set() if special == 5 else set(rng.sample(_SYMBOLS, rng.randrange(3))),
    )
    return Case(
        scenario,
        None if special in (4, 5) else rng.choice((None, None, None, None, None, 1, 2, 3)),
        None
        if special in (4, 5)
        else (f"membership unavailable source {seed % 11}" if rng.random() < 0.1 else None),
    )
