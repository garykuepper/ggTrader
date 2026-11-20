"""Identify exercised late-stage paper-trader behaviors from observed calls."""

import math
from collections.abc import Callable

from ggTrader.paper.catastrophe_stop import unrealized_pct
from tests.paper.equivalence.generator import Case


def _calls(evidence: dict, target: str) -> list[dict]:
    return [c for c in evidence["transcript"]["calls"] if c["target"] == target]


def _stop(evidence: dict) -> bool:
    return any(
        c["args"][6] == "catastrophe_stop" for c in _calls(evidence, "persist.create_order_intent")
    )


def _missing_qty(evidence: dict, exit_signal: bool) -> bool:
    case = evidence["case"]
    signals = case.scenario.signals
    positions = case.scenario.replies["get_positions"][0]
    exits = {s for sleeve in signals["sleeves"].values() for s in sleeve["sells"]}
    missing = {symbol for symbol, position in positions.items() if "qty" not in position}
    stopped = [c["args"][3] for c in _intents(evidence, "catastrophe_stop")]
    threshold = float(case.scenario.env.get("CATASTROPHE_STOP_PCT", "-0.25"))
    for symbol in missing:
        try:
            pct = unrealized_pct(positions[symbol])
        except ValueError:
            continue
        if exit_signal:
            eligible = symbol in exits and (pct is None or pct > threshold)
        else:
            eligible = (
                pct is not None and pct <= threshold and any(first < symbol for first in stopped)
            )
        if evidence["transcript"]["raised"] == "KeyError: 'qty'" and stopped and eligible:
            return True
    return False


def _late_sleeve(evidence: dict) -> bool:
    signals = evidence["case"].scenario.signals
    missing = set(signals["sleeves"]) - set(signals["weights"])
    raised = evidence["transcript"]["raised"]
    for sleeve in missing:
        if raised != f"KeyError: '{sleeve}'":
            continue
        prior_candidates = {
            symbol
            for name, signal in signals["sleeves"].items()
            if name != sleeve
            for symbol in signal["buys"]
        }
        if signals["sleeves"][sleeve]["buys"] and any(
            c["args"][0] in prior_candidates and not c["raised"]
            for c in _calls(evidence, "broker.submit_buy")
        ):
            return True
    return False


def _membership(evidence: dict) -> bool:
    return any(
        (c["raised"] or "").startswith("ValueError:")
        for c in _calls(evidence, "membership.live_membership_source")
    ) and any("signal generation error" in c["args"][0] for c in _calls(evidence, "notifier.send"))


def _duplicate(evidence: dict) -> bool:
    buys = [c["args"][0] for c in _calls(evidence, "broker.submit_buy") if not c["raised"]]
    signals = evidence["case"].scenario.signals
    return any(
        buys.count(symbol) >= 2
        and sum(symbol in sleeve["buys"] for sleeve in signals["sleeves"].values()) >= 2
        for symbol in set(buys)
    )


def _intents(evidence: dict, reason: str | None = None) -> list[dict]:
    """Return intents with exactly this reason; None selects ordinary strategy intents."""
    return [c for c in _calls(evidence, "persist.create_order_intent") if c["args"][6] == reason]


def _messages(evidence: dict) -> str:
    calls = _calls(evidence, "notifier.send")
    logs = evidence["logs"]
    return "\n".join([c["args"][0] for c in calls] + [r[3] for r in logs])


def _buys(evidence: dict) -> set[str]:
    return {
        symbol
        for sleeve in evidence["case"].scenario.signals["sleeves"].values()
        for symbol in sleeve["buys"]
    }


def _failed_after_order(evidence: dict) -> bool:
    submissions = [
        c
        for c in evidence["transcript"]["calls"]
        if c["target"] in ("broker.submit_buy", "broker.submit_sell")
    ]
    return any(
        c["raised"] and any(not prev["raised"] for prev in submissions[:i])
        for i, c in enumerate(submissions)
    )


def _rejected_then_exit(evidence: dict) -> bool:
    rejected = {
        c["args"][0]
        for c in _calls(evidence, "persist.settle_order_intent")
        if c["args"][-1] == "rejected"
    }
    sells = _calls(evidence, "broker.submit_sell")
    return any(c["args"][2] in rejected and i + 1 < len(sells) for i, c in enumerate(sells))


def _invalid_exit(evidence: dict) -> bool:
    scenario = evidence["case"].scenario
    positions = scenario.replies["get_positions"][0]
    exits = {s for sleeve in scenario.signals["sleeves"].values() for s in sleeve["sells"]}
    errors = (evidence["transcript"]["result"] or {}).get("errors", [])
    for symbol in exits & positions.keys():
        qty = positions[symbol].get("qty")
        if (
            qty is None or not isinstance(qty, (int, float)) or not math.isfinite(qty) or qty <= 0
        ) and any(f"SELL {symbol}: invalid long quantity" in error for error in errors):
            return True
    return False


def _loss_halt(evidence: dict) -> bool:
    notices = [c["args"][0] for c in _calls(evidence, "notifier.send")]
    return any(
        "HALTED buys" in msg and "Cannot assess catastrophe stop" not in msg for msg in notices
    )


def _split_sold(evidence: dict) -> bool:
    sold = {c["args"][0] for c in _calls(evidence, "broker.submit_sell")}
    for c in _calls(evidence, "notifier.daily_summary"):
        if sold & set(c["kwargs"].get("split_corrections", {})):
            return True
    return False


def _sweep_side(evidence: dict, side: str) -> bool:
    return any(c["args"][2] == side for c in _intents(evidence, "cash_sweep"))


def _funding_sweep(evidence: dict) -> bool:
    scenario = evidence["case"].scenario
    account = scenario.replies["get_account"][0]
    positions = scenario.replies["get_positions"][0]
    symbol = scenario.env.get("SWEEP_SYMBOL", "SPY")
    if symbol not in positions:
        return False
    reserve_pct = float(scenario.env.get("SWEEP_CASH_RESERVE_PCT", "0.05"))
    reserve_deficit = max(0.0, account["portfolio_value"] * reserve_pct - account["cash"])
    return any(
        c["args"][2] == "SELL"
        and c["args"][4] * positions[symbol]["current_price"] > reserve_deficit
        for c in _intents(evidence, "cash_sweep")
    )


def _sleeves_at_caps(evidence: dict) -> bool:
    messages = _messages(evidence)
    return (
        sum(
            f"Sleeve {name} slot cap reached" in messages
            for name in evidence["case"].scenario.signals["sleeves"]
        )
        >= 2
    )


WITNESSES: dict[str, Callable[[dict], bool]] = {
    "W1": lambda e: any(
        f"Skipping BUY {symbol}: catastrophe-stopped this run" in _messages(e)
        for symbol in {c["args"][3] for c in _intents(e, "catastrophe_stop")} & _buys(e)
    ),
    "W2": lambda e: (
        _stop(e)
        and any(c["args"][2] == "SELL" for c in _intents(e))
        and bool(_buys(e))
        and "Max positions reached" in _messages(e)
    ),
    "W3": lambda e: bool(_buys(e)) and "not fully filled" in _messages(e),
    "W4": lambda e: (
        bool(_buys(e))
        and e["case"].scenario.env.get("CASH_SWEEP_ENABLED") == "true"
        and _invalid_exit(e)
    ),
    "W5": _split_sold,
    "W6": lambda e: (
        bool(e["transcript"]["skipped_writes"])
        and (
            e["case"].scenario.read_only
            if e["case"].scenario.read_only is not None
            else e["case"].scenario.dry_run
        )
    ),
    "W7": _failed_after_order,
    "W8": _rejected_then_exit,
    "W9": lambda e: "Cannot assess catastrophe stop" in _messages(e),
    "W10": lambda e: _loss_halt(e) and any(c["args"][2] == "SELL" for c in _intents(e)),
    "W11": lambda e: _sweep_side(e, "SELL") and _sweep_side(e, "BUY"),
    "W12": lambda e: (
        not e["case"].scenario.dry_run
        and (e["transcript"]["result"] or {}).get("gate") == "late_run"
    ),
    "W13": lambda e: "order already pending" in _messages(e),
    "W14": _sleeves_at_caps,
    "W15": lambda e: _stop(e) and _funding_sweep(e),
    "W16": lambda evidence: _missing_qty(evidence, False),
    "W17": lambda evidence: _missing_qty(evidence, True),
    "W18": _late_sleeve,
    "W19": _membership,
    "W20": _duplicate,
}


def reached_witnesses(case: Case, transcript: dict, logs: tuple) -> frozenset[str]:
    """Return the names supported by recorded events and their scenario."""
    evidence = {"case": case, "transcript": transcript, "logs": logs}
    return frozenset(name for name, predicate in WITNESSES.items() if predicate(evidence))


def witness_projection(name: str, case: Case, transcript: dict, logs: tuple) -> tuple:
    """Extract only events and inputs relevant to one reached witness."""
    calls = transcript["calls"]
    intents = [c for c in calls if c["target"] == "persist.create_order_intent"]
    submits = [c for c in calls if c["target"] in ("broker.submit_buy", "broker.submit_sell")]
    notices = [c["args"][0] for c in calls if c["target"] == "notifier.send"]
    messages = notices + [r[3] for r in logs]
    sides = tuple((c["target"], *c["args"][:2], c["raised"]) for c in submits)
    reasons = tuple((c["args"][2], c["args"][3], c["args"][4], c["args"][6]) for c in intents)
    if name == "W1":
        return tuple(r for r in reasons if r[3] == "catastrophe_stop"), tuple(
            m for m in messages if "catastrophe-stopped this run" in m
        )
    if name == "W2":
        return reasons, sides, tuple(m for m in messages if "Max positions reached" in m)
    if name == "W3":
        return sides, tuple(m for m in messages if "not fully filled" in m)
    if name == "W4":
        return tuple((transcript["result"] or {}).get("errors", []))
    if name == "W5":
        return sides, tuple(
            str(c["kwargs"].get("split_corrections"))
            for c in calls
            if c["target"] == "notifier.daily_summary"
        )
    if name == "W6":
        return tuple(transcript["skipped_writes"])
    if name == "W7":
        return sides
    if name == "W8":
        rejected = tuple(
            (c["args"][2], c["args"][4])
            for c in calls
            if c["target"] == "persist.settle_order_intent" and c["args"][-1] == "rejected"
        )
        return sides, rejected
    if name == "W9":
        return tuple(m for m in messages if "Cannot assess catastrophe stop" in m)
    if name == "W10":
        return sides, tuple(m for m in notices if "HALTED buys" in m)
    if name == "W11":
        return tuple(r for r in reasons if r[3] == "cash_sweep")
    if name == "W12":
        return tuple(m for m in notices if "Late run" in m)
    if name == "W13":
        return tuple(m for m in messages if "order already pending" in m)
    if name == "W14":
        return tuple(m for m in messages if "slot cap reached" in m)
    if name == "W15":
        return tuple(r for r in reasons if r[3] in ("cash_sweep", "catastrophe_stop"))
    if name in ("W16", "W17"):
        positions = case.scenario.replies["get_positions"][0]
        missing = tuple(
            sorted(symbol for symbol, position in positions.items() if "qty" not in position)
        )
        return reasons, missing, transcript["raised"]
    if name == "W18":
        return sides, transcript["raised"]
    if name == "W19":
        return tuple(
            c["raised"] for c in calls if c["target"] == "membership.live_membership_source"
        ), tuple(m for m in notices if "signal generation error" in m)
    if name == "W20":
        return sides
    raise ValueError(f"unknown witness {name}")
