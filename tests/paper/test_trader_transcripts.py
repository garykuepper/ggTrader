"""Golden transcripts of `PaperTrader.run` over 38 scenarios: the refactor's safety net.

Every later change to the live paper trader must reproduce `golden/trader_transcripts/*.json`
byte for byte. To record a deliberate change, run the tests once with
``GGT_REGEN_TRANSCRIPTS=1`` (they write the files and fail), read the diff, and rerun.
"""

from __future__ import annotations

import difflib
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ggTrader.paper.alpaca_broker import AlpacaBroker
from tests.paper.transcript import harness
from tests.paper.transcript.harness import (
    ENV_VARS,
    FakeBroker,
    HarnessError,
    MembershipToken,
    Recorder,
    Scenario,
    normalize,
    run_scenario,
    to_json,
)
from tests.paper.transcript.scenarios import (
    EXPECTED_SCENARIO_COUNT,
    FIXTURE_SPEC,
    SCENARIOS,
    _acct,
    _blend,
    _pos,
    _replies,
)

REPO = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "golden" / "trader_transcripts"
REGEN_ENV = "GGT_REGEN_TRANSCRIPTS"

#: Run in a child process: no pytest fixture or monkeypatch reaches it, so it sets up its own.
CHILD_CODE = """
import json, sys
from unittest.mock import patch
from tests.netguard import blocked_connections
from tests.paper.transcript.harness import run_scenario, to_json
from tests.paper.transcript.scenarios import SCENARIOS

with patch("ggTrader.utils.config.load_dotenv", lambda *a, **k: False), blocked_connections():
    out = {name: to_json(run_scenario(s)) for name, s in SCENARIOS.items()}
sys.stdout.write("TRANSCRIPTS " + json.dumps(out) + "\\n")
"""

_PERSIST_WRITES = (
    "persist.init_paper_schema",
    "persist.log_snapshot",
    "persist.save_peak_value",
    "persist.save_split_correction",
    "persist.clear_split_correction",
    "persist.log_dividend_accrual",
    "persist.log_trade",
    "persist.log_pending_order",
    "persist.clear_pending_order",
    "persist.mark_pending_order_stale",
    "persist.create_order_intent",
    "persist.attach_broker_order",
    "persist.settle_order_intent",
    "persist.settle_legacy_pending_order",
    "tape.refresh_benchmark_tape",
)


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _unified_diff(expected: str, actual: str, name: str) -> str:
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"golden/{name}.json",
            tofile=f"{name} (this run)",
            lineterm="",
        )
    )
    shown = lines[:120]
    more = f"\n... {len(lines) - 120} more diff lines" if len(lines) > 120 else ""
    return "\n".join(shown) + more


# --------------------------------------------------------------------------- the goldens


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_transcript_matches_golden(name: str) -> None:
    actual = to_json(run_scenario(SCENARIOS[name]))
    path = _golden_path(name)
    if os.environ.get(REGEN_ENV):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        pytest.fail(f"regenerated {path.name}: review the diff and rerun without {REGEN_ENV}")
    assert path.exists(), f"no golden for {name}: run once with {REGEN_ENV}=1 and review it"
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, "transcript differs from its golden:\n" + _unified_diff(
        expected, actual, name
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_transcript_is_deterministic(name: str) -> None:
    first = to_json(run_scenario(SCENARIOS[name]))
    second = to_json(run_scenario(SCENARIOS[name]))
    assert first == second


def test_transcripts_stable_across_hash_seeds() -> None:
    """Two child processes with different hash seeds each reproduce every golden."""
    outputs: dict[str, dict[str, str]] = {}
    for seed in ("1", "2"):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONHASHSEED": seed,
            "GGT_MEMBERSHIP": FIXTURE_SPEC,  # REGEN_ENV is deliberately absent
        }
        done = subprocess.run(
            [sys.executable, "-c", CHILD_CODE],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert done.returncode == 0, (
            f"child with PYTHONHASHSEED={seed} failed:\n{done.stderr[-3000:]}"
        )
        marked = [ln for ln in done.stdout.splitlines() if ln.startswith("TRANSCRIPTS ")]
        assert len(marked) == 1, f"child output has no transcript line:\n{done.stdout[-2000:]}"
        outputs[seed] = json.loads(marked[0].removeprefix("TRANSCRIPTS "))
    for seed, produced in outputs.items():
        assert sorted(produced) == sorted(SCENARIOS)
        for name, text in produced.items():
            expected = _golden_path(name).read_text(encoding="utf-8")
            assert text == expected, f"PYTHONHASHSEED={seed}, {name}:\n" + _unified_diff(
                expected, text, name
            )


def test_every_scenario_has_a_golden() -> None:
    assert len(SCENARIOS) == EXPECTED_SCENARIO_COUNT == 38
    missing = sorted(name for name in SCENARIOS if not _golden_path(name).exists())
    assert not missing, f"scenarios without a golden: {missing}"


def test_no_orphan_goldens() -> None:
    files = {path.stem for path in GOLDEN_DIR.glob("*.json")}
    assert sorted(files - set(SCENARIOS)) == []


# --------------------------------------------------------------------------- what each shows


def _load(name: str) -> dict:
    return json.loads(_golden_path(name).read_text(encoding="utf-8"))


def _targets(t: dict) -> list[str]:
    return [c["target"] for c in t["calls"]]


def _calls(t: dict, target: str) -> list[dict]:
    return [c for c in t["calls"] if c["target"] == target]


def _sends(t: dict) -> list[str]:
    return [c["args"][0] for c in _calls(t, "notifier.send")]


def _index(t: dict, target: str, *, nth: int = 0) -> int:
    return [i for i, c in enumerate(t["calls"]) if c["target"] == target][nth]


def _submits(t: dict) -> list[tuple[str, str, float]]:
    """(side, symbol, amount) of every broker submit, in order."""
    out = []
    for c in t["calls"]:
        if c["target"] in ("broker.submit_buy", "broker.submit_sell"):
            out.append(
                (c["target"].removeprefix("broker.submit_").upper(), c["args"][0], c["args"][1])
            )
    return out


def _intent_rows(t: dict) -> tuple[list[str], list[str]]:
    """(settled, unresolved) client ids, derived from the recorded intent-ledger calls."""
    created = [c["args"][0] for c in _calls(t, "persist.create_order_intent")]
    settled = [c["args"][0] for c in _calls(t, "persist.settle_order_intent")]
    return settled, [cid for cid in created if cid not in settled]


def _no_orders(t: dict) -> None:
    assert _submits(t) == []
    assert _calls(t, "persist.create_order_intent") == []


def _no_persist_writes(t: dict) -> None:
    assert [x for x in _targets(t) if x in _PERSIST_WRITES] == []


def _check_plain_buys_and_sells(t: dict) -> None:
    assert t["raised"] is None and t["result"]["sells"] == ["AAPL"]
    assert t["result"]["buys"] == ["NVDA", "GOOGL"] and t["result"]["errors"] == []
    assert _submits(t) == [
        ("SELL", "AAPL", 10.0),
        ("BUY", "NVDA", 3300.0),
        ("BUY", "GOOGL", 3300.0),
    ]
    settled, unresolved = _intent_rows(t)
    assert len(settled) == 3 and unresolved == []
    for cid in settled:  # the intent round trip: create, attach, then settle
        order = [c["target"] for c in t["calls"] if c["args"][:1] == [cid]]
        assert order.index("persist.create_order_intent") < order.index(
            "persist.attach_broker_order"
        )
        assert order.index("persist.attach_broker_order") < order.index(
            "persist.settle_order_intent"
        )


def _check_market_closed(t: dict) -> None:
    assert t["result"] == {"buys": [], "sells": [], "errors": []}
    assert _calls(t, "signals.generate_core_signals") == []
    assert any("Market closed" in s for s in _sends(t))
    _no_orders(t)


def _check_late_run_live(t: dict) -> None:
    assert t["result"]["gate"] == "late_run" and t["result"]["buys"] == []
    assert any("Late run" in s for s in _sends(t))
    assert _calls(t, "tape.refresh_benchmark_tape") == [] and _calls(t, "broker.get_account") == []
    _no_orders(t)


def _check_late_run_dry(t: dict) -> None:
    sends = _sends(t)
    assert any("DRY RUN late-run notice" in s for s in sends)
    assert any("DRY RUN sell:</b> AAPL" in s for s in sends)
    assert any("DRY RUN buy:</b> MSFT" in s for s in sends)
    assert t["result"]["sells"] == ["AAPL"] and t["result"]["buys"] == ["MSFT"]
    _no_orders(t)
    _no_persist_writes(t)
    assert len(t["skipped_writes"]) == 4


def _stale_check(fragment: str) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        assert t["result"] == {"buys": [], "sells": [], "errors": []}
        assert any("Stale signal data" in s and fragment in s for s in _sends(t))
        assert _calls(t, "broker.get_account") == []
        _no_orders(t)

    return check


def _check_untradable_holding(t: dict) -> None:
    assert _calls(t, "broker.get_untradable")[0]["args"] == [["AVB", "MSFT"]]
    assert t["result"]["sells"] == [] and t["result"]["buys"] == ["MSFT"]
    assert t["result"]["untradable"] == ["AVB"]
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]
    assert any("Untradable holdings:" in s and "AVB" in s for s in _sends(t))


def _check_untradable_candidate(t: dict) -> None:
    assert t["result"]["buys"] == ["MSFT"] and t["result"]["untradable"] == ["AVB"]
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]
    assert any("Untradable buy candidates skipped:" in s and "AVB" in s for s in _sends(t))


def _check_split_correction(t: dict) -> None:
    held = 20.80410098
    assert _submits(t) == [("SELL", "MNST", held)]
    assert held * 2.0 != _submits(t)[0][2]  # never the split-adjusted quantity
    assert _calls(t, "persist.save_split_correction")[0]["args"] == ["MNST", "2026-06-10", 2.0]
    summary = _calls(t, "notifier.daily_summary")[0]
    assert summary["kwargs"]["split_corrections"] == {"MNST": 2.0}
    assert summary["kwargs"]["booked_equity"] == 100000.0


def _check_new_dividend_accrual(t: dict) -> None:
    accrual = _calls(t, "persist.log_dividend_accrual")[0]["args"]
    assert accrual[:4] == ["VZ", "2026-06-01", 0.6725, 77.05]
    assert accrual[4] == pytest.approx(51.82, abs=0.01)
    assert len(_calls(t, "persist.get_total_dividend_accrual")) == 2
    summary = _calls(t, "notifier.daily_summary")[0]
    assert summary["kwargs"]["dividend_total"] == pytest.approx(51.82, abs=0.01)
    assert summary["args"][0] == pytest.approx(100_000.0 + 51.82, abs=0.01)
    _no_orders(t)


def _check_drawdown_halt(t: dict) -> None:
    assert "persist.save_peak_value" not in _targets(t)
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    assert t["result"]["buys"] == [] and t["result"]["errors"][0].startswith(
        "Max drawdown breached: 16.7%"
    )
    assert any("HALTED buys; exits only" in s for s in _sends(t))


def _check_daily_loss_halt(t: dict) -> None:
    assert _calls(t, "persist.save_peak_value")[0]["args"] == [100000.0]
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    assert t["result"]["buys"] == [] and t["result"]["errors"][0].startswith(
        "Daily loss limit: 4.8%"
    )


def _check_daily_loss_halt_with_split(t: dict) -> None:
    error = t["result"]["errors"][0]
    assert "Daily loss limit: 5.0%" in error and "split correction on record for AAPL" in error
    assert "real loss cannot be ruled out" in error
    assert [s for s in _submits(t) if s[0] == "BUY"] == []
    assert _calls(t, "persist.save_split_correction")[0]["args"] == ["AAPL", "2026-06-15", 2.0]
    positions = _calls(t, "notifier.daily_summary")[0]["args"][2]
    assert positions["AAPL"]["market_value"] == 10_000.0  # split-adjusted, booked 5,000


def _check_peak_unreadable(t: dict) -> None:
    assert t["result"]["errors"] == ["Risk history unavailable: could not read persisted peak"]
    assert _calls(t, "persist.get_peak_value")[0]["raised"] == "RuntimeError: db down"
    assert "persist.save_peak_value" not in _targets(t) and "persist.log_snapshot" not in _targets(
        t
    )
    _no_orders(t)


def _check_peak_invalid(t: dict) -> None:
    assert t["result"]["errors"] == [
        "Invalid stored risk peak: Invalid portfolio value for risk peak"
    ]
    assert "persist.save_peak_value" not in _targets(t) and "persist.log_snapshot" not in _targets(
        t
    )
    _no_orders(t)


def _check_catastrophe_stop(t: dict) -> None:
    sells = [s for s in _submits(t) if s[0] == "SELL"]
    assert sells == [("SELL", "NXPI", 100.0)]
    assert [s for s in _submits(t) if s[0] == "BUY" and s[1] == "NXPI"] == []
    stop = _calls(t, "persist.create_order_intent")[0]["args"]
    assert stop[2:4] == ["SELL", "NXPI"] and stop[6] == "catastrophe_stop"
    assert t["result"]["sells"] == ["NXPI"] and t["result"]["buys"] == ["MSFT"]


def _check_catastrophe_stop_pending_order(t: dict) -> None:
    assert _calls(t, "broker.get_order")[0]["args"] == ["pend-1"]
    assert [s for s in _submits(t) if s[0] == "SELL"] == []
    assert t["result"]["sells"] == [] and t["result"]["buys"] == ["MSFT"]


def _check_catastrophe_stop_unassessable(t: dict) -> None:
    assert t["result"]["errors"][0] == "Cannot assess catastrophe stop for: ZZZ"
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    positions = _calls(t, "notifier.daily_summary")[0]["args"][2]
    assert positions["ZZZ"]["cost_basis"] == {"float": "NaN"}


def _invalid_qty_check(t: dict) -> None:
    assert t["result"]["errors"][0] == (
        "Cannot submit risk-reducing SELL AAPL: invalid long quantity; new strategy buys blocked"
    )
    assert _submits(t) == [("SELL", "NVDA", 5.0)]  # no strategy buy, and no SPY sweep buy
    assert t["result"]["sells"] == ["NVDA"] and t["result"]["buys"] == []
    assert len(_calls(t, "broker.get_account")) == 2  # the mid-run sweep sizing read never ran


def _check_sweep_funding_sell(t: dict) -> None:
    assert _submits(t) == [("SELL", "SPY", 18.25), ("BUY", "MSFT", 3300.0)]
    intent = _calls(t, "persist.create_order_intent")[0]["args"]
    assert intent[2:4] == ["SELL", "SPY"] and intent[6] == "cash_sweep"
    assert (
        _index(t, "broker.submit_sell")
        < _index(t, "broker.get_order")
        < _index(t, "broker.submit_buy")
    )
    assert t["result"]["sells"] == [] and t["result"]["buys"] == ["MSFT"]


def _check_sweep_buy(t: dict) -> None:
    assert _submits(t) == [("BUY", "SPY", 45000.0)]
    assert len(_calls(t, "broker.get_account")) == 3
    intent = _calls(t, "persist.create_order_intent")[0]["args"]
    assert intent[2:4] == ["BUY", "SPY"] and intent[6] == "cash_sweep"
    settled, unresolved = _intent_rows(t)
    assert len(settled) == 1 and unresolved == []
    alert = _calls(t, "notifier.trade_alert")[0]
    assert alert["kwargs"]["qty"] == 450.0


def _check_sweep_sell_and_strategy_sell(t: dict) -> None:
    assert _submits(t) == [
        ("SELL", "SPY", 18.25),
        ("SELL", "AAPL", 10.0),
        ("BUY", "MSFT", 3300.0),
    ]
    assert [s for s in _submits(t) if s[1] == "NVDA"] == []  # the sweep sell frees no slot
    assert t["result"]["sells"] == ["AAPL"] and t["result"]["buys"] == ["MSFT"]


def _check_sell_partial_fill_working(t: dict) -> None:
    assert t["raised"] == "RuntimeError: Unresolved working orders block submissions"
    assert t["result"] is None
    assert _submits(t) == [("SELL", "AAPL", 10.0)]  # no buy follows
    assert _calls(t, "persist.settle_order_intent") == []
    settled, unresolved = _intent_rows(t)
    assert settled == [] and len(unresolved) == 1
    assert any("Order recovery/pending orders failed" in s for s in _sends(t))
    assert len(_calls(t, "broker.get_order")) > 2  # polled up to the deadline, not just once


def _check_sell_partial_fill_canceled(t: dict) -> None:
    settle = _calls(t, "persist.settle_order_intent")[0]["args"]
    assert settle[2:] == [4.0, 150.0, "canceled"]
    assert [s for s in _submits(t) if s[0] == "BUY"] == []
    assert t["result"]["errors"] == [
        "SELL AAPL not fully filled (canceled); new strategy buys blocked"
    ]
    settled, unresolved = _intent_rows(t)
    assert len(settled) == 1 and unresolved == []


def _check_pending_symbol_buy(t: dict) -> None:
    assert _calls(t, "broker.get_order")[0]["args"] == ["pend-b1"]
    assert _submits(t) == [("BUY", "NVDA", 3300.0)]
    assert t["result"]["buys"] == ["NVDA"]


def _check_pending_symbol_sell(t: dict) -> None:
    assert _calls(t, "broker.get_order")[0]["args"] == ["pend-s1"]
    assert _submits(t) == [] and t["result"]["sells"] == []


def _check_global_slot_cap(t: dict) -> None:
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]
    assert t["result"]["buys"] == ["MSFT"]


def _check_sleeve_slot_cap(t: dict) -> None:
    assert _submits(t) == [
        ("BUY", "MSFT", 1650.0),
        ("BUY", "NVDA", 1650.0),
        ("BUY", "AMZN", 1650.0),
    ]
    assert t["result"]["buys"] == ["MSFT", "NVDA", "AMZN"]


def _check_concentration_limit(t: dict) -> None:
    assert _submits(t) == [("BUY", "AMZN", 528.0)]
    assert t["result"]["buys"] == ["AMZN"]


def _check_missed_exit_catch_up(t: dict) -> None:
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    assert t["result"]["sells"] == ["AAPL"]


def _check_dry_run_read_only(t: dict) -> None:
    _no_persist_writes(t)
    assert t["skipped_writes"] == [
        "init_paper_schema",
        "refresh_benchmark_tape",
        "save_peak_value",
        "log_snapshot",
    ]
    assert _calls(t, "signals.generate_core_signals")[0]["kwargs"]["persist_tape"] is False
    assert _calls(t, "broker.get_order") == [] and _submits(t) == []
    assert t["result"]["sells"] == ["AAPL"] and t["result"]["buys"] == ["MSFT"]
    assert t["result"]["skipped_writes"] == t["skipped_writes"]


def _check_dry_run_with_writes(t: dict) -> None:
    assert t["skipped_writes"] == []
    assert _calls(t, "persist.settle_legacy_pending_order")[0]["args"] == [
        "pend-9",
        10.0,
        60.0,
        "filled",
    ]
    alerts = _calls(t, "notifier.trade_alert")
    assert alerts[0]["kwargs"]["status"] == "filled (reconciled)"
    for target in ("persist.init_paper_schema", "persist.save_peak_value", "persist.log_snapshot"):
        assert target in _targets(t)
    assert "tape.refresh_benchmark_tape" in _targets(t) and _submits(t) == []
    summary = _calls(t, "notifier.daily_summary")[0]
    assert summary["args"][2]["AAPL"]["unrealized_pl"] == {"float": "NaN"}
    assert t["result"]["buys"] == ["MSFT"]


def _check_submit_error(t: dict) -> None:
    assert t["raised"] == "RuntimeError: insufficient buying power" and t["result"] is None
    assert _targets(t).index("persist.create_order_intent") < _targets(t).index("broker.submit_buy")
    assert _calls(t, "broker.submit_buy")[0]["raised"] == "RuntimeError: insufficient buying power"
    assert len(_calls(t, "broker.submit_buy")) == 1  # NVDA is never attempted
    assert _calls(t, "persist.attach_broker_order") == []
    assert any(
        "Order 00000000-0000-0000-0000-000000000001 failed at submit" in s for s in _sends(t)
    )
    settled, unresolved = _intent_rows(t)
    assert settled == [] and len(unresolved) == 1


def _check_post_trade_read_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: api down" and t["result"] is None
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    assert any("orders may have been submitted; snapshot not saved" in s for s in _sends(t))
    assert "persist.log_snapshot" not in _targets(t)
    assert _calls(t, "notifier.daily_summary") == []


def _check_snapshot_write_failure(t: dict) -> None:
    assert t["result"]["errors"] == ["Daily snapshot could not be saved"]
    assert _calls(t, "persist.log_snapshot")[0]["raised"] == "RuntimeError: write failed"
    assert _sends(t)[-1] == "<b>⚠️ Daily snapshot could not be saved.</b>"


def _check_sell_rejected_by_broker(t: dict) -> None:
    assert _submits(t) == [("SELL", "AAPL", 10.0), ("SELL", "MSFT", 5.0)]
    settles = {c["args"][0]: c["args"][2:] for c in _calls(t, "persist.settle_order_intent")}
    first, second = (c["args"][0] for c in _calls(t, "persist.create_order_intent"))
    assert settles[first] == [0.0, None, "rejected"] and settles[second][-1] == "filled"
    assert [s for s in _submits(t) if s[0] == "BUY"] == []
    assert any("SELL AAPL not fully filled (rejected)" in e for e in t["result"]["errors"])
    settled, unresolved = _intent_rows(t)
    assert len(settled) == 2 and unresolved == []


BEHAVIOR_CHECKS: dict[str, Callable[[dict], None]] = {
    "plain_buys_and_sells": _check_plain_buys_and_sells,
    "market_closed": _check_market_closed,
    "late_run_live": _check_late_run_live,
    "late_run_dry": _check_late_run_dry,
    "stale_as_of": _stale_check("9d old"),
    "future_as_of": _stale_check("6d in the future"),
    "unparseable_as_of": _stale_check("unparseable"),
    "untradable_holding": _check_untradable_holding,
    "untradable_candidate": _check_untradable_candidate,
    "split_correction": _check_split_correction,
    "new_dividend_accrual": _check_new_dividend_accrual,
    "drawdown_halt": _check_drawdown_halt,
    "daily_loss_halt": _check_daily_loss_halt,
    "daily_loss_halt_with_split": _check_daily_loss_halt_with_split,
    "peak_unreadable": _check_peak_unreadable,
    "peak_invalid": _check_peak_invalid,
    "catastrophe_stop": _check_catastrophe_stop,
    "catastrophe_stop_pending_order": _check_catastrophe_stop_pending_order,
    "catastrophe_stop_unassessable": _check_catastrophe_stop_unassessable,
    "catastrophe_stop_invalid_qty": _invalid_qty_check,
    "strategy_exit_invalid_qty": _invalid_qty_check,
    "sweep_funding_sell": _check_sweep_funding_sell,
    "sweep_buy": _check_sweep_buy,
    "sweep_sell_and_strategy_sell": _check_sweep_sell_and_strategy_sell,
    "sell_partial_fill_working": _check_sell_partial_fill_working,
    "sell_partial_fill_canceled": _check_sell_partial_fill_canceled,
    "pending_symbol_buy": _check_pending_symbol_buy,
    "pending_symbol_sell": _check_pending_symbol_sell,
    "global_slot_cap": _check_global_slot_cap,
    "sleeve_slot_cap": _check_sleeve_slot_cap,
    "concentration_limit": _check_concentration_limit,
    "missed_exit_catch_up": _check_missed_exit_catch_up,
    "dry_run_read_only": _check_dry_run_read_only,
    "dry_run_with_writes": _check_dry_run_with_writes,
    "submit_error": _check_submit_error,
    "post_trade_read_failure": _check_post_trade_read_failure,
    "snapshot_write_failure": _check_snapshot_write_failure,
    "sell_rejected_by_broker": _check_sell_rejected_by_broker,
}


#: (settled, unresolved) intent rows each golden ends with: the `Intents:` line of its docstring.
EXPECTED_INTENT_ROWS: dict[str, tuple[int, int]] = {
    "plain_buys_and_sells": (3, 0),
    "untradable_holding": (1, 0),
    "untradable_candidate": (1, 0),
    "split_correction": (1, 0),
    "drawdown_halt": (1, 0),
    "daily_loss_halt": (1, 0),
    "catastrophe_stop": (2, 0),
    "catastrophe_stop_pending_order": (1, 0),
    "catastrophe_stop_unassessable": (1, 0),
    "catastrophe_stop_invalid_qty": (1, 0),
    "strategy_exit_invalid_qty": (1, 0),
    "sweep_funding_sell": (2, 0),
    "sweep_buy": (1, 0),
    "sweep_sell_and_strategy_sell": (3, 0),
    "sell_partial_fill_working": (0, 1),
    "sell_partial_fill_canceled": (1, 0),
    "pending_symbol_buy": (1, 0),
    "global_slot_cap": (1, 0),
    "sleeve_slot_cap": (3, 0),
    "concentration_limit": (1, 0),
    "missed_exit_catch_up": (1, 0),
    "submit_error": (0, 1),
    "post_trade_read_failure": (1, 0),
    "sell_rejected_by_broker": (2, 0),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_final_intent_rows(name: str) -> None:
    settled, unresolved = _intent_rows(_load(name))
    assert (len(settled), len(unresolved)) == EXPECTED_INTENT_ROWS.get(name, (0, 0))


def test_every_scenario_has_a_behavior_check() -> None:
    assert sorted(BEHAVIOR_CHECKS) == sorted(SCENARIOS)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_shows_scenario_behavior(name: str) -> None:
    """The recorded golden shows what the scenario's name and docstring claim."""
    BEHAVIOR_CHECKS[name](_load(name))


# --------------------------------------------------------------------------- the harness itself


def test_to_json_tags_special_values_and_sorts_sets() -> None:
    text = to_json(
        {
            "b": [float("nan"), float("inf"), float("-inf"), 1.5, 0.1 + 0.2],
            "a": {3, 1, 2},
            "d": date(2026, 6, 19),
            "n": Decimal("1.10"),
            "tok": MembershipToken("db"),
            "t": (1, "x"),
        }
    )
    assert text.endswith("\n") and text.index('"a"') < text.index('"b"')  # sorted keys
    data = json.loads(text)
    assert data["b"][:3] == [{"float": "NaN"}, {"float": "Infinity"}, {"float": "-Infinity"}]
    assert data["b"][3:] == [1.5, 0.30000000000000004]
    assert data["a"] == [1, 2, 3] and data["d"] == "2026-06-19" and data["n"] == "1.10"
    assert data["tok"] == {"membership": "db"} and data["t"] == [1, "x"]


def test_to_json_refuses_unknown_types() -> None:
    class Opaque:
        pass

    with pytest.raises(TypeError):
        to_json({"x": Opaque()})
    with pytest.raises(TypeError):
        normalize({1: "non-string key"})


def test_recorder_snapshots_arguments_at_call_time() -> None:
    recorder = Recorder()
    payload = {"positions": {"AAPL": {"qty": 1.0}}}
    recorder.begin("x", [payload], {})
    payload["positions"]["AAPL"]["qty"] = 99.0  # the trader mutates replies after the call
    assert recorder.calls[0]["args"] == [{"positions": {"AAPL": {"qty": 1.0}}}]
    with pytest.raises(HarnessError):
        recorder.begin("x", [object()], {})


def test_fake_broker_implements_every_public_alpaca_broker_method() -> None:
    public = {
        name
        for name, member in inspect.getmembers(AlpacaBroker, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public <= {name for name in dir(FakeBroker) if not name.startswith("_")}


def test_env_vars_cover_every_variable_the_paper_package_reads() -> None:
    found: set[str] = set()
    pattern = re.compile(r'"([A-Z][A-Z0-9_]{3,})"')
    for path in (REPO / "src" / "ggTrader" / "paper").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"environ|getenv|ENV_VAR\s*=|ENV_UNTRADABLE\s*=|MEMBERSHIP_ENV", line):
                found.update(pattern.findall(line))
    assert found, "the grep found no environment variable names: the pattern is stale"
    assert found <= set(ENV_VARS), (
        f"unlisted environment variables: {sorted(found - set(ENV_VARS))}"
    )


def test_harness_never_patches_the_global_clock() -> None:
    source = inspect.getsource(harness)
    assert not re.search(r"patch(\.object)?\(\s*(time\b|[\"']time\.)", source)
    assert "monkeypatch" not in source


def _base_scenario(**fields: object) -> Scenario:
    persist = {
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
    persist.update(fields.pop("persist", {}))  # type: ignore[arg-type]
    defaults: dict[str, object] = {
        "name": "probe",
        "env": {"MAX_POSITIONS": "3"},
        "signals": _blend([], []),
        "replies": _replies(_acct(), {}),
        "persist": persist,
    }
    return Scenario(**{**defaults, **fields})  # type: ignore[arg-type]


def test_missing_persist_reply_is_a_harness_error_not_a_trading_outcome() -> None:
    probe = _base_scenario()
    del probe.persist["get_peak_value"]
    with pytest.raises(HarnessError, match="persist.get_peak_value"):
        run_scenario(probe)


def test_missing_broker_reply_is_a_harness_error() -> None:
    probe = _base_scenario()
    del probe.replies["get_positions"]
    with pytest.raises(HarnessError, match="broker.get_positions"):
        run_scenario(probe)


def test_unpinned_pending_order_timestamp_is_a_harness_error() -> None:
    row = {"order_id": "p", "symbol": "KO", "side": "BUY", "created_at": date(2026, 6, 1)}
    with pytest.raises(HarnessError, match="created_at"):
        run_scenario(_base_scenario(persist={"get_pending_orders": [row]}))


def test_runaway_poll_loop_is_a_harness_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "MAX_POLLS", 3)
    probe = _base_scenario(
        signals=_blend([], ["AAPL"]),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
        fills={
            "AAPL": [{"status": "partially_filled", "filled_qty": 1.0, "filled_avg_price": 1.0}]
        },
    )
    with pytest.raises(HarnessError, match="runaway"):
        run_scenario(probe)
