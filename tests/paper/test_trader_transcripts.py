"""Golden transcripts of `PaperTrader.run` over every scenario: the refactor's safety net.

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
    _MNST_QTY,
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


_MNST_MV = _MNST_QTY * 47.77  # the broker's (uncorrected) market value of the MNST holding


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
    assert len(SCENARIOS) == EXPECTED_SCENARIO_COUNT == 103
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


def _intent_rows(t: dict, seeded: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """(settled, unresolved) client ids at the end of the run.

    Derived from the recorded ledger calls: the rows the run created plus the `seeded` ones the
    scenario started with, minus those a ``settle_order_intent`` call settled.
    """
    created = [c["args"][0] for c in _calls(t, "persist.create_order_intent") if not c["raised"]]
    settled = [c["args"][0] for c in _calls(t, "persist.settle_order_intent") if not c["raised"]]
    return settled, [cid for cid in [*seeded, *created] if cid not in settled]


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
    # The broker quantity is sold; split correction changes value, not quantity.
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


def _check_full_poll_deadline(t: dict) -> None:
    """15 one-second sleeps and 16 polls (15 in the loop, 1 final): the open-market budget."""
    sleeps = _calls(t, "clock.sleep")
    assert [c["args"] for c in sleeps] == [[1.0]] * 15
    assert len(_calls(t, "broker.get_order")) == 16
    reads = [c["kwargs"]["returned"] for c in _calls(t, "clock.time")]
    assert reads[0] == harness.CLOCK_START and reads[-1] == harness.CLOCK_START + 15.0


def _check_sell_partial_fill_working(t: dict) -> None:
    assert t["raised"] == "RuntimeError: Unresolved working orders block submissions"
    assert t["result"] is None
    assert _submits(t) == [("SELL", "AAPL", 10.0)]  # no buy follows
    assert _calls(t, "persist.settle_order_intent") == []
    settled, unresolved = _intent_rows(t)
    assert settled == [] and len(unresolved) == 1
    assert any("Order recovery/pending orders failed" in s for s in _sends(t))
    assert len(_calls(t, "broker.get_order")) > 2  # polled up to the deadline, not just once
    _check_full_poll_deadline(t)


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


# ----- follow-up checks: reporting, ordering, failures, recovery, polling


def _summary(t: dict) -> dict:
    return _calls(t, "notifier.daily_summary")[0]


def _snapshot(t: dict) -> list:
    return _calls(t, "persist.log_snapshot")[0]["args"]


def _both(*checks: Callable[[dict], None]) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        for one in checks:
            one(t)

    return check


def _check_plain_reporting(t: dict) -> None:
    summary = _summary(t)
    assert summary["args"][0] == 100_053.0 and summary["args"][1] == 553.0  # final equity, P&L
    assert summary["kwargs"]["total_pnl"] == 53.0
    assert sorted(summary["args"][2]) == ["GOOGL", "KO", "NVDA"]  # the final positions
    snapshot = _snapshot(t)
    assert snapshot[1:3] == [100_053.0, 92_200.0]
    assert sorted(snapshot[3]) == ["GOOGL", "KO", "NVDA"]
    first = _calls(t, "persist.settle_order_intent")[0]["args"]
    assert first[2:] == [10.0, 150.0, "filled"]


def _check_sleeve_reporting(t: dict) -> None:
    summary = _summary(t)
    assert summary["args"][0] == 100_033.0 and summary["args"][1] == 233.0
    assert sorted(summary["args"][2]) == ["AMZN", "MSFT", "NVDA"]
    assert _snapshot(t)[1:3] == [100_033.0, 95_050.0]


def _check_sweep_funding_fills(t: dict) -> None:
    settles = [c["args"] for c in _calls(t, "persist.settle_order_intent")]
    assert settles[0][2:] == [18.25, 400.0, "filled"]  # the SPY sell fills at 400
    assert len(_calls(t, "broker.get_account")) == 3
    assert _summary(t)["args"][0] == 100_033.0 and _snapshot(t)[2] == 5_000.0


def _check_sweep_sell_strategy_fills(t: dict) -> None:
    settles = [c["args"] for c in _calls(t, "persist.settle_order_intent")]
    assert settles[0][2:] == [18.25, 400.0, "filled"]
    assert settles[1][2:] == [10.0, 150.0, "filled"]
    assert _summary(t)["args"][0] == 100_033.0


def _check_catastrophe_stop_at_position_cap(t: dict) -> None:
    assert _submits(t) == [("SELL", "NXPI", 100.0), ("BUY", "MSFT", 3300.0)]
    assert t["result"]["sells"] == ["NXPI"] and t["result"]["buys"] == ["MSFT"]
    assert [s for s in _submits(t) if s[1] in ("GOOGL", "NVDA")] == []


def _check_sweep_buy_after_strategy_orders(t: dict) -> None:
    assert _submits(t) == [("BUY", "MSFT", 3300.0), ("BUY", "SPY", 41700.0)]
    first, second = (i for i, c in enumerate(t["calls"]) if c["target"] == "broker.submit_buy")
    sizing = [i for i, c in enumerate(t["calls"]) if c["target"] == "broker.get_account"][1]
    assert first < sizing < second  # sized from the account read after the MSFT buy
    sweep = _calls(t, "persist.create_order_intent")[1]["args"]
    assert sweep[3:7] == ["SPY", 41700.0, "notional", "cash_sweep"]
    settled, unresolved = _intent_rows(t)
    assert len(settled) == 2 and unresolved == []


def _failed_at(stage: str) -> Callable[[str], bool]:
    return lambda s: f"failed at {stage}; investigation required" in s


def _check_intent_create_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: ledger down" and t["result"] is None
    assert _calls(t, "persist.create_order_intent")[0]["raised"] == "RuntimeError: ledger down"
    assert "lock.verify" not in _targets(t) and _submits(t) == []
    assert any(_failed_at("intent")(s) for s in _sends(t))


def _check_lock_verify_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: lock lost"
    assert _calls(t, "lock.verify")[0]["raised"] == "RuntimeError: lock lost"
    assert _submits(t) == [] and any(_failed_at("lock")(s) for s in _sends(t))
    settled, unresolved = _intent_rows(t)
    assert settled == [] and len(unresolved) == 1


def _check_attach_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: attach failed"
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]  # the broker took it, NVDA never tried
    assert _calls(t, "persist.attach_broker_order")[0]["raised"] == "RuntimeError: attach failed"
    assert any(_failed_at("attach")(s) for s in _sends(t))


def _check_failure_notice_fails(t: dict) -> None:
    assert t["raised"] == "RuntimeError: rejected"  # not the notifier's error
    failing = [c for c in _calls(t, "notifier.send") if "failed at submit" in c["args"][0]]
    assert len(failing) == 1 and failing[0]["raised"] == "RuntimeError: telegram down"


def _recovery_settles(status: str, qty: float, price: float | None) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        assert t["raised"] is None and _submits(t) == []
        assert _calls(t, "broker.get_order_by_client_id")[0]["args"] == ["prev-1"]
        settle = _calls(t, "persist.settle_order_intent")
        assert [c["args"] for c in settle] == [["prev-1", "ord-prev-1", qty, price, status]]
        assert _index(t, "persist.settle_order_intent") < _index(t, "broker.get_clock")

    return check


def _recovery_stops(error: str) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        assert t["raised"] == error and t["result"] is None
        assert _calls(t, "persist.settle_order_intent") == []
        assert _calls(t, "broker.get_clock") == [] and _submits(t) == []
        assert any("Order recovery/pending orders failed" in s for s in _sends(t))
        settled, unresolved = _intent_rows(t, ("prev-1",))
        assert settled == [] and unresolved == ["prev-1"]  # the seeded row is still open

    return check


def _check_recovery_working(t: dict) -> None:
    _recovery_stops("RuntimeError: Unresolved working orders block submissions")(t)
    assert _calls(t, "persist.attach_broker_order")[0]["args"] == ["prev-1", "ord-prev-1"]


def _check_recovery_missing_order(t: dict) -> None:
    _recovery_stops("RuntimeError: order not found")(t)
    assert (
        _calls(t, "broker.get_order_by_client_id")[0]["raised"] == "RuntimeError: order not found"
    )


def _check_legacy_pending_stale(t: dict) -> None:
    alerts = [s for s in _sends(t) if "Stale pending order" in s]
    assert len(alerts) == 1 and "KO" in alerts[0] and "stale-ko" in alerts[0]
    assert "9.2 days" in alerts[0]
    assert [c["args"] for c in _calls(t, "persist.mark_pending_order_stale")] == [["stale-ko"]]
    assert [c["args"][0] for c in _calls(t, "broker.get_order")] == [
        "stale-ko",
        "stale-xom",
        "fresh-cvx",
    ]


def _check_legacy_pending_stale_mark_fails(t: dict) -> None:
    assert t["raised"] is None
    assert len([s for s in _sends(t) if "Stale pending order" in s]) == 1
    assert _calls(t, "persist.mark_pending_order_stale")[0]["raised"] == "RuntimeError: db busy"


def _check_legacy_pending_lookup_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: lookup failed" and t["result"] is None
    assert _calls(t, "broker.get_order")[0]["raised"] == "RuntimeError: lookup failed"
    assert _calls(t, "signals.generate_core_signals") == [] and _sends(t) == []


def _check_legacy_pending_settlement_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: ledger locked"
    assert _calls(t, "persist.settle_legacy_pending_order")[0]["raised"] == (
        "RuntimeError: ledger locked"
    )
    assert any("Could not settle pending order pend-1" in s for s in _sends(t))
    assert _calls(t, "notifier.trade_alert") == []
    assert _calls(t, "signals.generate_core_signals") == []


def _check_legacy_pending_canceled(t: dict) -> None:
    settle = _calls(t, "persist.settle_legacy_pending_order")
    assert [c["args"] for c in settle] == [["pend-1", 0.0, None, "canceled"]]
    assert [c["args"][3] for c in _calls(t, "notifier.trade_alert")] == ["ord-1"]  # only the buy
    assert _submits(t) == [("BUY", "KO", 3300.0)] and t["result"]["buys"] == ["KO"]


def _check_pending_orders_read_failure_live(t: dict) -> None:
    assert t["raised"] == "RuntimeError: ledger offline"
    reads = _calls(t, "persist.get_pending_orders")
    assert len(reads) == 2 and reads[1]["raised"] == "RuntimeError: ledger offline"
    assert any("Could not load pending orders" in s for s in _sends(t))
    assert _calls(t, "signals.generate_core_signals") == [] and _submits(t) == []


def _check_pending_orders_read_failure_dry(t: dict) -> None:
    assert t["raised"] is None and _submits(t) == []
    assert _calls(t, "persist.get_pending_orders")[0]["raised"] == "RuntimeError: ledger offline"
    assert any("DRY RUN buy:</b> MSFT" in s for s in _sends(t))
    assert t["result"]["buys"] == ["MSFT"]


def _pending_warning_survives(t: dict) -> None:
    sends = _calls(t, "notifier.send")
    halted = next(i for i, c in enumerate(sends) if "HALTED buys; exits only" in c["args"][0])
    assert sends[halted]["raised"] == "RuntimeError: telegram down"
    warning = sends[halted + 1]
    assert "Pending orders may still fill" in warning["args"][0] and warning["raised"] is None


def _check_loss_halt_notice_fails_with_pending(t: dict) -> None:
    _pending_warning_survives(t)
    assert _submits(t) == [("SELL", "AAPL", 10.0)]  # exits still go
    assert (
        _index(t, "broker.submit_sell")
        > [
            i
            for i, c in enumerate(t["calls"])
            if c["target"] == "notifier.send" and "Pending orders may still fill" in c["args"][0]
        ][0]
    )


def _check_unassessable_notice_fails_with_pending(t: dict) -> None:
    _pending_warning_survives(t)
    assert t["result"]["errors"] == ["Cannot assess catastrophe stop for: ZZZ"]
    assert [s for s in _submits(t) if s[0] == "BUY"] == []


def _check_poll_clock_failure(t: dict) -> None:
    clocks = _calls(t, "broker.get_clock")
    assert len(clocks) == 2 and clocks[1]["raised"] == "RuntimeError: clock down"
    assert len(_calls(t, "broker.get_order")) == 3
    assert [c["args"] for c in _calls(t, "clock.sleep")] == [[1.0], [1.0]]
    assert t["raised"] is None and t["result"]["sells"] == ["AAPL"]


def _check_total_pnl_two_bases(t: dict) -> None:
    assert any(
        "+20.15% of prior snapshot, +25.07% of starting capital" in message for message in _sends(t)
    )


def _check_poll_transient_lookup_error(t: dict) -> None:
    polls = _calls(t, "broker.get_order")
    assert [c["raised"] for c in polls] == ["RuntimeError: blip", None]
    assert [c["args"] for c in _calls(t, "clock.sleep")] == [[1.0]]
    assert t["result"]["sells"] == ["AAPL"]


def _check_poll_final_lookup_fails(t: dict) -> None:
    polls = _calls(t, "broker.get_order")
    assert len(polls) == 16 and all(c["raised"] == "RuntimeError: poll down" for c in polls)
    assert len(_calls(t, "clock.sleep")) == 15
    alert = _calls(t, "notifier.trade_alert")[0]
    assert alert["args"][:3] == ["SELL", "AAPL", 1500.0] and alert["kwargs"]["status"] == "pending"
    assert t["result"]["errors"] == [
        "SELL AAPL not fully filled (pending); new strategy buys blocked"
    ]
    assert [s for s in _submits(t) if s[0] == "BUY"] == []


def _check_poll_market_closed(t: dict) -> None:
    assert t["raised"] == "RuntimeError: Unresolved working orders block submissions"
    assert len(_calls(t, "broker.get_order")) == 3  # 2 in the 2-second loop, 1 final
    assert len(_calls(t, "clock.sleep")) == 2


def _check_split_correction_held(t: dict) -> None:
    mnst = _MNST_MV
    summary = _summary(t)
    assert summary["args"][0] == pytest.approx(100_500.0 + mnst, abs=1e-6)
    assert summary["args"][2]["MNST"]["market_value"] == pytest.approx(2 * mnst, abs=1e-6)
    assert summary["args"][2]["MNST"]["qty"] == _MNST_QTY  # quantity is never corrected
    assert summary["kwargs"]["split_corrections"] == {"MNST": 2.0}
    assert summary["kwargs"]["booked_equity"] == 100_500.0
    snapshot = _snapshot(t)
    assert snapshot[1] == summary["args"][0]
    assert snapshot[3]["MNST"]["market_value"] == pytest.approx(2 * mnst, abs=1e-6)
    assert _submits(t) == [("BUY", "NVDA", 3300.0)] and t["result"]["sells"] == []


def _check_post_trade_positions_read_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: positions down"
    assert [c["raised"] for c in _calls(t, "broker.get_positions")][-1] == (
        "RuntimeError: positions down"
    )
    accounts = _calls(t, "broker.get_account")
    assert len(accounts) == 2 and accounts[1]["raised"] is None
    assert any("orders may have been submitted" in s for s in _sends(t))
    assert "persist.log_snapshot" not in _targets(t) and _calls(t, "notifier.daily_summary") == []


def _check_dry_run_catastrophe_stop(t: dict) -> None:
    assert any(
        "DRY RUN catastrophe stop:</b> NXPI (qty 100.0, unrealized -40.0% breached -25.0% floor)"
        in s
        for s in _sends(t)
    )
    assert _submits(t) == [] and _calls(t, "persist.create_order_intent") == []
    assert t["result"]["sells"] == ["NXPI"] and t["result"]["buys"] == ["MSFT"]


def _check_dry_run_sweep(t: dict) -> None:
    sends = [s for s in _sends(t) if "DRY RUN" in s]
    assert [s.split("</b>")[0].split("DRY RUN ")[1] for s in sends] == [
        "sweep sell:",
        "buy:",
        "sweep buy:",
    ]
    assert "SPY ($7300" in sends[0] and "SPY ($45000)" in sends[2]
    assert _submits(t) == [] and _calls(t, "persist.create_order_intent") == []


def _halt_check(error: str, *, peak_saved: bool) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        assert t["result"]["errors"] == [error] and t["raised"] is None
        assert any(s == f"<b>🛑 HALTED:</b> {error}" for s in _sends(t))
        assert ("persist.save_peak_value" in _targets(t)) == peak_saved
        _no_orders(t)

    return check


def _check_peak_save_failure(t: dict) -> None:
    _halt_check("Risk history unavailable: could not persist drawdown peak", peak_saved=True)(t)
    assert _calls(t, "persist.save_peak_value")[0]["raised"] == "RuntimeError: write failed"


def _check_held_buy_candidate_skipped(t: dict) -> None:
    assert _submits(t) == [("BUY", "NVDA", 3300.0)] and t["result"]["buys"] == ["NVDA"]


def _check_sweep_sizing_failure(t: dict) -> None:
    accounts = _calls(t, "broker.get_account")
    assert [c["raised"] for c in accounts] == [None, "RuntimeError: sizing read", None]
    assert _submits(t) == [] and t["raised"] is None


def _check_invalid_catastrophe_threshold(t: dict) -> None:
    assert t["raised"] == "ValueError: Invalid CATASTROPHE_STOP_PCT; expected a loss in [-1, 0)"
    assert _targets(t) == ["notifier.send"]  # before the lock, the schema and the signals
    assert _sends(t)[0].startswith("<b>🛑 HALTED:</b> Invalid CATASTROPHE_STOP_PCT")


def _check_market_clock_failure_fail_open(t: dict) -> None:
    clocks = _calls(t, "broker.get_clock")
    assert len(clocks) >= 2 and all(c["raised"] == "RuntimeError: clock down" for c in clocks)
    assert _submits(t) == [("BUY", "MSFT", 3300.0)] and t["result"]["buys"] == ["MSFT"]


def _check_schema_init_failure(t: dict) -> None:
    assert _calls(t, "persist.init_paper_schema")[0]["raised"] == "RuntimeError: ddl failed"
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]


def _check_signal_gate_error(t: dict) -> None:
    assert t["raised"] == "SignalGateError: [empty_frame] no bars" and t["result"] is None
    assert _calls(t, "signals.generate_core_signals")[0]["raised"] == t["raised"]
    assert _sends(t) == ["<b>🛑 Signal gate — no orders placed:</b>\n[empty_frame] no bars"]
    assert _calls(t, "broker.get_account") == [] and _calls(t, "tape.refresh_benchmark_tape") == []


def _check_signal_generation_error(t: dict) -> None:
    assert t["raised"] == "RuntimeError: yfinance down"
    assert _sends(t) == ["Paper trading failed: signal generation error\nyfinance down"]
    assert _calls(t, "broker.get_account") == []


def _check_overlay_fallback_and_membership_warning(t: dict) -> None:
    sends = _sends(t)
    assert "Membership list" in sends[0] and "Overlay fallback" in sends[1]
    assert _submits(t) == [("BUY", "MSFT", 3300.0)]


def _read_failure_check(call: str, notice: str) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        assert t["raised"] == "RuntimeError: " + (
            "account down" if "account" in call else "positions down"
        )
        assert _calls(t, f"broker.{call}")[0]["raised"] == t["raised"]
        assert f"<b>🛑 HALTED:</b> {notice}" in _sends(t)
        _no_orders(t)

    return check


def _legacy_dropped(status: str) -> Callable[[dict], None]:
    def check(t: dict) -> None:
        settle = _calls(t, "persist.settle_legacy_pending_order")
        assert [c["args"] for c in settle] == [["pend-1", 0.0, None, status]]
        assert [c["args"][3] for c in _calls(t, "notifier.trade_alert")] == ["ord-1"]
        assert _submits(t) == [("BUY", "KO", 3300.0)] and t["result"]["buys"] == ["KO"]

    return check


def _check_poll_order_expired(t: dict) -> None:
    assert len(_calls(t, "broker.get_order")) == 2 and len(_calls(t, "clock.sleep")) == 1
    settle = _calls(t, "persist.settle_order_intent")[0]["args"]
    assert settle[2:] == [0.0, None, "expired"]
    assert _calls(t, "notifier.trade_alert")[0]["kwargs"]["status"] == "expired"
    assert [s for s in _submits(t) if s[0] == "BUY"] == []
    assert t["result"]["errors"] == [
        "SELL AAPL not fully filled (expired); new strategy buys blocked"
    ]


def _undelivered_check(method: str, text: str) -> Callable[[dict], None]:
    """The scripted ``False`` is invisible to the trader: it ignores the return value."""

    def check(t: dict) -> None:
        name = [n for n, s in SCENARIOS.items() if s.notifier == {method: False}]
        assert name, "no scenario scripts this notifier outcome"
        assert (
            t["raised"] is None and t["result"]["buys"] == ["MSFT"] and t["result"]["errors"] == []
        )
        assert len(_calls(t, f"notifier.{method}")) == 1
        assert any(text in s for s in _sends(t))  # the rendered text is recorded anyway
        assert "persist.log_snapshot" in _targets(t)  # the run goes on to the snapshot

    return check


def _check_intent_settle_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: ledger write failed"
    assert _calls(t, "persist.settle_order_intent")[0]["raised"] == t["raised"]
    assert any("Order recovery/pending orders failed" in s for s in _sends(t))
    assert _submits(t) == [("SELL", "AAPL", 10.0)]
    settled, unresolved = _intent_rows(t)
    assert settled == [] and len(unresolved) == 1


def _check_unresolved_intents_read_failure(t: dict) -> None:
    assert t["raised"] == "RuntimeError: ledger offline"
    assert _calls(t, "persist.get_unresolved_order_intents")[0]["raised"] == t["raised"]
    assert any("Order recovery/pending orders failed" in s for s in _sends(t))
    assert _calls(t, "broker.get_clock") == [] and _submits(t) == []


def _check_recovery_notice_fails(t: dict) -> None:
    assert t["raised"] == "RuntimeError: Unresolved working orders block submissions"
    notices = _calls(t, "notifier.send")
    assert len(notices) == 1 and notices[0]["raised"] == "RuntimeError: telegram down"
    settled, unresolved = _intent_rows(t, ("prev-1",))
    assert settled == [] and unresolved == ["prev-1"]


def _sweep_float_fail_soft(t: dict) -> None:
    """A bad sweep-buy float is logged and skipped: three clean account reads, no order."""
    accounts = _calls(t, "broker.get_account")
    assert [c["raised"] for c in accounts] == [None, None, None]
    assert _submits(t) == [] and t["raised"] is None and t["result"]["errors"] == []


def _check_sweep_reserve_invalid_ends_run(t: dict) -> None:
    assert t["raised"] == "ValueError: could not convert string to float: 'abc'"
    assert t["result"] is None and len(_calls(t, "broker.get_account")) == 1
    assert _submits(t) == [] and _calls(t, "persist.log_snapshot") == []


def _check_sweep_reserve_invalid_but_sweep_off(t: dict) -> None:
    assert t["raised"] is None and t["result"]["buys"] == ["MSFT"]
    assert t["result"]["errors"] == [] and _submits(t) == [("BUY", "MSFT", 3300.0)]


def _check_untradable_symbols_from_env(t: dict) -> None:
    assert _calls(t, "broker.get_untradable")[0]["args"] == [["NVDA"]]
    assert t["result"]["untradable"] == ["AAPL", "MSFT"] and t["result"]["sells"] == []
    assert _submits(t) == [("BUY", "NVDA", 3300.0)]
    assert any("Untradable holdings:</b> AAPL, MSFT" in s for s in _sends(t))


def _check_catastrophe_enabled_pct_blank(t: dict) -> None:
    assert _submits(t) == [("SELL", "NXPI", 100.0), ("BUY", "MSFT", 3300.0)]
    assert any("-30.0% breached -25.0% floor" in s for s in _sends(t))
    assert t["result"]["sells"] == ["NXPI"] and t["result"]["buys"] == ["MSFT"]


def _check_pending_order_all_five_fields(t: dict) -> None:
    assert _submits(t) == [
        ("SELL", "NXPI", 100.0),
        ("SELL", "SPY", 18.25),
        ("SELL", "AAPL", 10.0),
        ("BUY", "MSFT", 3300.0),
    ]
    reasons = [c["args"][6] for c in _calls(t, "persist.create_order_intent")]
    assert reasons == ["catastrophe_stop", "cash_sweep", None, None]
    pending = {
        c["args"][1]: c["args"][:3] + [c["kwargs"]["status"]]
        for c in _calls(t, "notifier.trade_alert")
        if c["kwargs"]["status"] == "pending"
    }
    assert pending == {
        "SPY": ["SELL", "SPY", 7300.0, "pending"],
        "MSFT": ["BUY", "MSFT", 3300.0, "pending"],
    }
    assert t["result"]["sells"] == ["NXPI", "AAPL"] and t["result"]["buys"] == ["MSFT"]
    assert t["result"]["errors"] == []


BEHAVIOR_CHECKS: dict[str, Callable[[dict], None]] = {
    "plain_buys_and_sells": _both(_check_plain_buys_and_sells, _check_plain_reporting),
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
    "sweep_funding_sell": _both(_check_sweep_funding_sell, _check_sweep_funding_fills),
    "sweep_buy": _check_sweep_buy,
    "sweep_sell_and_strategy_sell": _both(
        _check_sweep_sell_and_strategy_sell, _check_sweep_sell_strategy_fills
    ),
    "sell_partial_fill_working": _check_sell_partial_fill_working,
    "sell_partial_fill_canceled": _check_sell_partial_fill_canceled,
    "pending_symbol_buy": _check_pending_symbol_buy,
    "pending_symbol_sell": _check_pending_symbol_sell,
    "global_slot_cap": _check_global_slot_cap,
    "sleeve_slot_cap": _both(_check_sleeve_slot_cap, _check_sleeve_reporting),
    "concentration_limit": _check_concentration_limit,
    "missed_exit_catch_up": _check_missed_exit_catch_up,
    "dry_run_read_only": _check_dry_run_read_only,
    "dry_run_with_writes": _check_dry_run_with_writes,
    "submit_error": _check_submit_error,
    "post_trade_read_failure": _check_post_trade_read_failure,
    "snapshot_write_failure": _check_snapshot_write_failure,
    "sell_rejected_by_broker": _check_sell_rejected_by_broker,
    "catastrophe_stop_at_position_cap": _check_catastrophe_stop_at_position_cap,
    "sweep_buy_after_strategy_orders": _check_sweep_buy_after_strategy_orders,
    "intent_create_failure": _check_intent_create_failure,
    "lock_verify_failure": _check_lock_verify_failure,
    "attach_failure": _check_attach_failure,
    "failure_notice_fails": _check_failure_notice_fails,
    "recovery_intent_filled": _recovery_settles("filled", 33.0, 100.0),
    "recovery_intent_canceled": _recovery_settles("canceled", 2.0, 100.0),
    "recovery_intent_rejected": _recovery_settles("rejected", 0.0, None),
    "recovery_intent_expired": _recovery_settles("expired", 0.0, None),
    "recovery_intent_working": _check_recovery_working,
    "recovery_identity_mismatch": _recovery_stops(
        "ValueError: Order identity mismatch or duplicate legacy order"
    ),
    "recovery_symbol_mismatch": _recovery_stops(
        "ValueError: Order identity mismatch or duplicate legacy order"
    ),
    "recovery_amount_mismatch": _recovery_stops("ValueError: Order amount mismatch"),
    "recovery_missing_broker_order": _check_recovery_missing_order,
    "recovery_invalid_fill": _recovery_stops("ValueError: Invalid order fill"),
    "recovery_unknown_status": _recovery_stops("ValueError: Unknown order status"),
    "legacy_pending_stale": _check_legacy_pending_stale,
    "legacy_pending_stale_mark_fails": _check_legacy_pending_stale_mark_fails,
    "legacy_pending_lookup_failure": _check_legacy_pending_lookup_failure,
    "legacy_pending_settlement_failure": _check_legacy_pending_settlement_failure,
    "legacy_pending_canceled": _check_legacy_pending_canceled,
    "pending_orders_read_failure_live": _check_pending_orders_read_failure_live,
    "pending_orders_read_failure_dry": _check_pending_orders_read_failure_dry,
    "loss_halt_notice_fails_with_pending": _check_loss_halt_notice_fails_with_pending,
    "catastrophe_stop_unassessable_notice_fails_with_pending": (
        _check_unassessable_notice_fails_with_pending
    ),
    "poll_clock_failure": _check_poll_clock_failure,
    "total_pnl_two_bases": _check_total_pnl_two_bases,
    "poll_transient_lookup_error": _check_poll_transient_lookup_error,
    "poll_final_lookup_fails": _check_poll_final_lookup_fails,
    "poll_market_closed_short_wait": _check_poll_market_closed,
    "split_correction_held": _check_split_correction_held,
    "post_trade_positions_read_failure": _check_post_trade_positions_read_failure,
    "dry_run_catastrophe_stop": _check_dry_run_catastrophe_stop,
    "dry_run_sweep_sell_and_buy": _check_dry_run_sweep,
    "invalid_equity_halt": _halt_check(
        "Invalid portfolio value; drawdown cannot be assessed", peak_saved=False
    ),
    "peak_save_failure": _check_peak_save_failure,
    "last_equity_missing_halt": _halt_check(
        "Invalid broker previous-close equity; daily loss cannot be assessed", peak_saved=True
    ),
    "last_equity_invalid_halt": _halt_check(
        "Invalid portfolio value or daily baseline; loss cannot be assessed", peak_saved=True
    ),
    "held_buy_candidate_skipped": _check_held_buy_candidate_skipped,
    "sweep_sizing_failure": _check_sweep_sizing_failure,
    "invalid_catastrophe_threshold": _check_invalid_catastrophe_threshold,
    "market_clock_failure_fail_open": _check_market_clock_failure_fail_open,
    "schema_init_failure": _check_schema_init_failure,
    "signal_gate_error": _check_signal_gate_error,
    "signal_generation_error": _check_signal_generation_error,
    "overlay_fallback_and_membership_warning": _check_overlay_fallback_and_membership_warning,
    "account_read_failure": _read_failure_check(
        "get_account", "Account data unavailable; no orders submitted"
    ),
    "positions_read_failure": _read_failure_check(
        "get_positions", "Position data unavailable; no orders submitted"
    ),
    "legacy_pending_expired": _legacy_dropped("expired"),
    "legacy_pending_rejected": _legacy_dropped("rejected"),
    "poll_order_expired": _check_poll_order_expired,
    "trade_alert_returns_false": _undelivered_check("trade_alert", "Paper Trade"),
    "daily_summary_returns_false": _undelivered_check("daily_summary", "Paper Portfolio Summary"),
    "intent_settle_failure": _check_intent_settle_failure,
    "unresolved_intents_read_failure": _check_unresolved_intents_read_failure,
    "recovery_notice_fails": _check_recovery_notice_fails,
    "recovery_filled_zero_quantity": _recovery_stops("ValueError: Filled order has no fill"),
    # Phase 4 pins (Task 32)
    "sweep_min_clip_invalid_fail_soft": _sweep_float_fail_soft,
    "sweep_buy_trigger_invalid_fail_soft": _sweep_float_fail_soft,
    "sweep_reserve_invalid_ends_run": _check_sweep_reserve_invalid_ends_run,
    "sweep_reserve_invalid_but_sweep_off": _check_sweep_reserve_invalid_but_sweep_off,
    "untradable_symbols_from_env": _check_untradable_symbols_from_env,
    "catastrophe_enabled_pct_blank": _check_catastrophe_enabled_pct_blank,
    "pending_order_all_five_fields": _check_pending_order_all_five_fields,
}


#: (settled, unresolved) intent rows each golden ends with: the `Intents:` line of its docstring.
EXPECTED_INTENT_ROWS: dict[str, tuple[int, int]] = {
    "plain_buys_and_sells": (3, 0),
    "total_pnl_two_bases": (3, 0),
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
    "attach_failure": (0, 1),
    "catastrophe_stop_at_position_cap": (2, 0),
    "failure_notice_fails": (0, 1),
    "held_buy_candidate_skipped": (1, 0),
    "legacy_pending_canceled": (1, 0),
    "lock_verify_failure": (0, 1),
    "loss_halt_notice_fails_with_pending": (1, 0),
    "market_clock_failure_fail_open": (1, 0),
    "overlay_fallback_and_membership_warning": (1, 0),
    "poll_clock_failure": (1, 0),
    "poll_final_lookup_fails": (1, 0),
    "poll_market_closed_short_wait": (0, 1),
    "poll_transient_lookup_error": (1, 0),
    "post_trade_positions_read_failure": (1, 0),
    "recovery_intent_canceled": (1, 0),  # the seeded row settles
    "recovery_intent_expired": (1, 0),
    "recovery_intent_filled": (1, 0),
    "recovery_intent_rejected": (1, 0),
    "schema_init_failure": (1, 0),
    "split_correction_held": (1, 0),
    "sweep_buy_after_strategy_orders": (2, 0),
    # seeded rows that startup recovery leaves unresolved
    "recovery_intent_working": (0, 1),
    "recovery_identity_mismatch": (0, 1),
    "recovery_symbol_mismatch": (0, 1),
    "recovery_amount_mismatch": (0, 1),
    "recovery_missing_broker_order": (0, 1),
    "recovery_invalid_fill": (0, 1),
    "recovery_unknown_status": (0, 1),
    "recovery_notice_fails": (0, 1),
    "recovery_filled_zero_quantity": (0, 1),
    # second follow-up
    "legacy_pending_expired": (1, 0),
    "legacy_pending_rejected": (1, 0),
    "poll_order_expired": (1, 0),
    "trade_alert_returns_false": (1, 0),
    "daily_summary_returns_false": (1, 0),
    "intent_settle_failure": (0, 1),
    # Phase 4 pins (Task 32)
    "sweep_reserve_invalid_but_sweep_off": (1, 0),
    "untradable_symbols_from_env": (1, 0),
    "catastrophe_enabled_pct_blank": (2, 0),
    "pending_order_all_five_fields": (4, 0),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_final_intent_rows(name: str) -> None:
    seeded = tuple(row["client_order_id"] for row in SCENARIOS[name].intents)
    settled, unresolved = _intent_rows(_load(name), seeded)
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


def test_removed_sleep_is_a_harness_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poll loop that never sleeps never advances the fake clock: it must not hang."""
    monkeypatch.setattr(harness.FakeTime, "sleep", lambda self, seconds: None)
    probe = _base_scenario(
        signals=_blend([], ["AAPL"]),
        replies=_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
        fills={
            "AAPL": [{"status": "partially_filled", "filled_qty": 1.0, "filled_avg_price": 1.0}]
        },
    )
    with pytest.raises(HarnessError, match="runaway"):
        run_scenario(probe)


def test_scripted_intent_failure_is_recorded_and_stops_submission() -> None:
    probe = _base_scenario(
        signals=_blend(["AAPL"], []),
        persist={"create_order_intent": RuntimeError("ledger down")},
    )
    transcript = run_scenario(probe)
    assert transcript["raised"] == "RuntimeError: ledger down"
    assert (
        _calls(transcript, "persist.create_order_intent")[0]["raised"]
        == "RuntimeError: ledger down"
    )
    assert not _submits(transcript)
    assert any("failed at intent" in message for message in _sends(transcript))


def test_fake_clock_records_every_read_and_sleep_and_enforces_its_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder()
    clock = harness.FakeTime(recorder)
    assert clock.time() == harness.CLOCK_START
    clock.sleep(2.5)
    assert clock.monotonic() == harness.CLOCK_START + 2.5
    assert [(c["target"], c["args"], c["kwargs"]) for c in recorder.calls] == [
        ("clock.time", [], {"returned": harness.CLOCK_START}),
        ("clock.sleep", [2.5], {}),
        ("clock.monotonic", [], {"returned": harness.CLOCK_START + 2.5}),
    ]
    monkeypatch.setattr(harness, "MAX_POLLS", 2)
    clock.time()  # the second event at this instant (the monotonic read was the first)
    with pytest.raises(HarnessError, match="runaway"):
        clock.time()  # the third at one instant: no sleep moved the clock
    clock.sleep(1.0)  # a sleep advances the clock and resets the count
    clock.time()


def test_long_legitimate_poll_of_many_orders_is_not_a_runaway() -> None:
    """14 working orders polled for 15 iterations make over 200 lookups: that is allowed."""
    symbols = [f"S{i:02d}" for i in range(14)]
    probe = _base_scenario(
        signals=_blend([], symbols),
        replies=_replies(_acct(), {s: _pos(10.0, 50.0) for s in symbols}),
        fills={
            s: [{"status": "partially_filled", "filled_qty": 1.0, "filled_avg_price": 50.0}]
            for s in symbols
        },
    )
    transcript = run_scenario(probe)
    assert len(_calls(transcript, "broker.get_order")) > harness.MAX_POLLS
    assert transcript["raised"] == "RuntimeError: Unresolved working orders block submissions"


def test_scripted_false_is_returned_by_every_notifier_method() -> None:
    recorder = Recorder()
    notifier = harness.RecordingNotifier(
        recorder, {"trade_alert": False, "daily_summary": False, "quiet": False}
    )
    assert notifier.trade_alert("BUY", "MSFT", 10.0, "ord-1", qty=1.0, price=10.0) is False
    assert notifier.daily_summary(1000.0, 5.0, {}) is False
    assert notifier.send("be quiet") is False
    assert notifier.send("hello") is True
    # The rendered text is still recorded for the formatted methods.
    texts = [c["args"][0] for c in recorder.calls if c["target"] == "notifier.send"]
    assert any("Paper Trade" in s for s in texts)
    assert any("Paper Portfolio Summary" in s for s in texts)


def test_signal_gate_error_copy_registration_is_scoped() -> None:
    import copyreg

    from ggTrader.paper.gates import SignalGateError

    assert SignalGateError not in copyreg.dispatch_table
    run_scenario(_base_scenario(signals=SignalGateError("empty_frame", "no bars")))
    assert SignalGateError not in copyreg.dispatch_table  # restored after the run


def test_scripted_notifier_outcomes_are_recorded() -> None:
    recorder = Recorder()
    notifier = harness.RecordingNotifier(
        recorder, {"quiet": False, "boom": RuntimeError("telegram down"), "daily_summary": False}
    )
    assert notifier.send("hello") is True
    assert notifier.send("be quiet") is False
    with pytest.raises(RuntimeError, match="telegram down"):
        notifier.send("boom!")
    assert [c["raised"] for c in recorder.calls] == [None, None, "RuntimeError: telegram down"]
    assert notifier.trade_alert("BUY", "MSFT", 10.0, "ord-1") is True  # the real formatting runs
    assert [c["target"] for c in recorder.calls][-2:] == ["notifier.trade_alert", "notifier.send"]


def test_scripted_lookup_replies_fall_through_on_none() -> None:
    probe = _base_scenario(
        signals=_blend([], ["AAPL"]),
        replies={
            **_replies(_acct(), {"AAPL": _pos(10.0, 150.0)}),
            "get_order": [RuntimeError("blip"), None],
        },
    )
    transcript = run_scenario(probe)
    polls = _calls(transcript, "broker.get_order")
    assert [c["raised"] for c in polls] == ["RuntimeError: blip", None]
