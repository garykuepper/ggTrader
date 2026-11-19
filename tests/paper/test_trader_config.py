"""`TraderConfig` holds the trader's run-time settings, one snapshot per run (Phase 4, Task 43).

The record calls the existing accessors (`cash_sweep`, `catastrophe_stop`, `untradable`), which
stay the only place the variable names are written. A malformed float is stored with its original
`ValueError` and re-raised where the old code parsed it, so the fail time of each setting is kept
even though the read moved to the top of `run()` (Deviation 4).
"""

from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path

import pytest

from ggTrader.paper import config as config_module
from ggTrader.paper import trader as trader_module
from ggTrader.paper.config import TraderConfig
from tests.paper.test_trader_ledger import _build_trader, _trader_environment
from tests.paper.transcript.harness import ENV_VARS, run_scenario
from tests.paper.transcript.scenarios import SCENARIOS

CONFIG_SOURCE = Path(config_module.__file__)
TRADER_SOURCE = Path(trader_module.__file__)

SWEEP_ON = {"CASH_SWEEP_ENABLED": "true"}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _from_env(**overrides: bool) -> TraderConfig:
    return TraderConfig.from_env(**{"dry_run": True, "read_only": True, **overrides})


# --------------------------------------------------------------------------- the record


def test_from_env_defaults() -> None:
    cfg = _from_env(dry_run=False, read_only=False)
    assert cfg.dry_run is False
    assert cfg.read_only is False
    assert cfg.stop_enabled is False
    assert cfg.stop_threshold is None
    assert cfg.stop_threshold_error is None
    assert cfg.sweep_enabled is False
    assert cfg.sweep_symbol is None
    assert cfg.untradable == frozenset()
    assert cfg.sweep_reserve_pct is None
    assert cfg.sweep_min_clip_usd is None
    assert cfg.sweep_buy_trigger_pct is None
    assert dict(cfg.sweep_parse_errors) == {}


def test_from_env_carries_the_two_flags() -> None:
    cfg = _from_env(dry_run=True, read_only=False)
    assert (cfg.dry_run, cfg.read_only) == (True, False)


def test_from_env_reads_each_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATASTROPHE_STOP_ENABLED", "true")
    monkeypatch.setenv("CATASTROPHE_STOP_PCT", "-0.1")
    monkeypatch.setenv("CASH_SWEEP_ENABLED", "yes")
    monkeypatch.setenv("SWEEP_SYMBOL", " voo ")
    monkeypatch.setenv("UNTRADABLE_SYMBOLS", " aapl, ,Msft ")
    monkeypatch.setenv("SWEEP_CASH_RESERVE_PCT", "0.1")
    monkeypatch.setenv("SWEEP_MIN_CLIP_USD", "250")
    monkeypatch.setenv("SWEEP_BUY_TRIGGER_PCT", "0.2")
    cfg = _from_env()
    assert cfg.stop_enabled is True
    assert cfg.stop_threshold == -0.1
    assert cfg.stop_threshold_error is None
    assert cfg.sweep_enabled is True
    assert cfg.sweep_symbol == "VOO"
    assert cfg.untradable == frozenset({"AAPL", "MSFT"})
    assert (cfg.reserve_pct(), cfg.min_clip_usd(), cfg.buy_trigger_pct()) == (0.1, 250.0, 0.2)
    assert dict(cfg.sweep_parse_errors) == {}


def test_blank_floats_take_the_accessor_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATASTROPHE_STOP_ENABLED", "true")
    monkeypatch.setenv("CATASTROPHE_STOP_PCT", "")
    monkeypatch.setenv("CASH_SWEEP_ENABLED", "true")
    cfg = _from_env()
    assert cfg.stop_threshold == -0.25
    assert cfg.sweep_symbol == "SPY"
    assert (cfg.reserve_pct(), cfg.min_clip_usd(), cfg.buy_trigger_pct()) == (0.05, 500.0, 0.08)


def test_invalid_stop_pct_is_stored_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATASTROPHE_STOP_ENABLED", "true")
    monkeypatch.setenv("CATASTROPHE_STOP_PCT", "5")
    cfg = _from_env()
    assert cfg.stop_enabled is True
    assert cfg.stop_threshold is None
    assert isinstance(cfg.stop_threshold_error, ValueError)
    assert (
        str(cfg.stop_threshold_error) == "Invalid CATASTROPHE_STOP_PCT; expected a loss in [-1, 0)"
    )


def test_stop_pct_is_not_read_while_the_stop_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATASTROPHE_STOP_PCT", "5")
    cfg = _from_env()
    assert (cfg.stop_enabled, cfg.stop_threshold, cfg.stop_threshold_error) == (False, None, None)


@pytest.mark.parametrize(
    ("variable", "accessor"),
    [
        ("SWEEP_CASH_RESERVE_PCT", "reserve_pct"),
        ("SWEEP_MIN_CLIP_USD", "min_clip_usd"),
        ("SWEEP_BUY_TRIGGER_PCT", "buy_trigger_pct"),
    ],
)
def test_invalid_sweep_floats_are_stored_not_raised(
    monkeypatch: pytest.MonkeyPatch, variable: str, accessor: str
) -> None:
    monkeypatch.setenv("CASH_SWEEP_ENABLED", "true")
    monkeypatch.setenv(variable, "abc")
    cfg = _from_env()  # does not raise
    assert list(cfg.sweep_parse_errors) == [accessor]
    assert isinstance(cfg.sweep_parse_errors[accessor], ValueError)
    assert cfg.sweep_enabled is True


@pytest.mark.parametrize("accessor", ["reserve_pct", "min_clip_usd", "buy_trigger_pct"])
def test_accessors_re_raise_the_stored_error_with_the_original_message(
    monkeypatch: pytest.MonkeyPatch, accessor: str
) -> None:
    variable = {
        "reserve_pct": "SWEEP_CASH_RESERVE_PCT",
        "min_clip_usd": "SWEEP_MIN_CLIP_USD",
        "buy_trigger_pct": "SWEEP_BUY_TRIGGER_PCT",
    }[accessor]
    monkeypatch.setenv("CASH_SWEEP_ENABLED", "true")
    monkeypatch.setenv(variable, "abc")
    cfg = _from_env()
    with pytest.raises(ValueError, match=r"^could not convert string to float: 'abc'$") as info:
        getattr(cfg, accessor)()
    assert info.value is cfg.sweep_parse_errors[accessor]  # the same object, not a copy
    for other in {"reserve_pct", "min_clip_usd", "buy_trigger_pct"} - {accessor}:
        assert isinstance(getattr(cfg, other)(), float)  # the others still answer


def test_sweep_off_reads_no_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWEEP_MIN_CLIP_USD", "abc")
    monkeypatch.setenv("SWEEP_CASH_RESERVE_PCT", "abc")
    monkeypatch.setenv("SWEEP_BUY_TRIGGER_PCT", "abc")
    monkeypatch.setenv("SWEEP_SYMBOL", "voo")
    cfg = _from_env()
    assert dict(cfg.sweep_parse_errors) == {}
    assert cfg.sweep_symbol is None
    assert cfg.sweep_min_clip_usd is None


def test_config_is_frozen() -> None:
    cfg = _from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sweep_enabled = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        cfg.sweep_parse_errors["reserve_pct"] = ValueError("x")  # type: ignore[index]


# --------------------------------------------------------------------------- no new env reads


def test_from_env_adds_no_new_environment_variable() -> None:
    text = CONFIG_SOURCE.read_text(encoding="utf-8")
    assert "os.environ" not in text
    assert "getenv" not in text
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "os" not in imported


def test_trader_py_reads_no_environment() -> None:
    tree = ast.parse(TRADER_SOURCE.read_text(encoding="utf-8"))
    assert not [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert not [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(a.name == "os" for a in node.names)
    ]
    assert "os.environ" not in TRADER_SOURCE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the trader's runs


def _targets(run) -> list[str]:  # type: ignore[no-untyped-def]
    return [c["target"] for c in run.recorder.calls]


def test_env_is_read_at_run_time_not_construction() -> None:
    with _trader_environment("plain_buys_and_sells", wrap_persist=True) as run:
        trader = _build_trader(run)
        os.environ["CATASTROPHE_STOP_ENABLED"] = "true"
        os.environ["CATASTROPHE_STOP_PCT"] = "5"
        with pytest.raises(ValueError, match=r"^Invalid CATASTROPHE_STOP_PCT"):
            trader.run()
    sends = [c for c in run.recorder.calls if c["target"] == "notifier.send"]
    assert [c["args"] for c in sends] == [
        ["<b>🛑 HALTED:</b> Invalid CATASTROPHE_STOP_PCT; expected a loss in [-1, 0)"]
    ]


def test_catastrophe_stop_error_is_still_raised_before_the_lock() -> None:
    scenario = SCENARIOS["invalid_catastrophe_threshold"]
    assert scenario.dry_run is False  # a live trader: the lock would be entered
    with _trader_environment("invalid_catastrophe_threshold", wrap_persist=True) as run:
        trader = _build_trader(run)
        with pytest.raises(ValueError, match=r"^Invalid CATASTROPHE_STOP_PCT"):
            trader.run()
    targets = _targets(run)
    assert "lock.enter" not in targets
    assert "persist.init_paper_schema" not in targets
    assert targets == ["notifier.send"]


@pytest.mark.parametrize("name", ["market_closed", "stale_as_of"])
def test_early_exits_with_a_malformed_sweep_float_return_their_normal_result(name: str) -> None:
    base = SCENARIOS[name]
    clean = dataclasses.replace(base, env={**base.env, **SWEEP_ON})
    bad = dataclasses.replace(base, env={**base.env, **SWEEP_ON, "SWEEP_CASH_RESERVE_PCT": "abc"})
    expected = run_scenario(clean)
    actual = run_scenario(bad)
    assert actual == expected  # same result, same Telegram text, same calls
    assert actual["raised"] is None


def test_lock_failure_with_a_malformed_reserve_ends_where_the_reserve_was_always_read() -> None:
    """Not an early exit: the funding-sell step reads the reserve before the lock is checked.

    The brief expected this run to match the one with the variable unset. It never did (the
    unmodified trader also raises the parse error here); the test pins that the snapshot did not
    move the failure: same error, same point, no order and no lock check reached.
    """
    base = SCENARIOS["lock_verify_failure"]
    clean = run_scenario(dataclasses.replace(base, env={**base.env, **SWEEP_ON}))
    bad = run_scenario(
        dataclasses.replace(base, env={**base.env, **SWEEP_ON, "SWEEP_CASH_RESERVE_PCT": "abc"})
    )
    assert clean["raised"] == "RuntimeError: lock lost"
    assert bad["raised"] == "ValueError: could not convert string to float: 'abc'"
    targets = [c["target"] for c in bad["calls"]]
    assert "lock.verify" not in targets
    assert "broker.submit_buy" not in targets
    assert "persist.create_order_intent" not in targets
    # everything the bad run did before it ended (it only leaves the lock after), the clean run
    # did first: same prefix, same Telegram text
    assert targets[-1] == "lock.exit"
    assert clean["calls"][: len(bad["calls"]) - 1] == bad["calls"][:-1]


def test_two_runs_on_one_trader_each_take_their_own_snapshot() -> None:
    base = SCENARIOS["sweep_min_clip_invalid_fail_soft"]
    assert base.env["SWEEP_MIN_CLIP_USD"] == "abc"
    with _trader_environment("sweep_min_clip_invalid_fail_soft", wrap_persist=True) as run:
        trader = _build_trader(run)
        trader.run()
        first_buys = [c for c in run.recorder.calls if c["target"] == "broker.submit_buy"]
        assert first_buys == []  # the bad setting failed soft: no sweep buy
        del os.environ["SWEEP_MIN_CLIP_USD"]  # the owner fixes it between runs
        trader.run()
    buys = [c for c in run.recorder.calls if c["target"] == "broker.submit_buy"]
    assert [c["args"][0] for c in buys] == ["SPY"]  # the second run took the new value


def test_one_run_never_mixes_two_values() -> None:
    """The environment changes mid-run; the sweep buy still uses the value read at the top."""
    base = SCENARIOS["sweep_min_clip_invalid_fail_soft"]
    env = {k: v for k, v in base.env.items() if k != "SWEEP_MIN_CLIP_USD"}
    plain = dataclasses.replace(base, env=env)
    sweep_buys = [c for c in run_scenario(plain)["calls"] if c["target"] == "broker.submit_buy"]
    assert [c["args"][0] for c in sweep_buys] == ["SPY"]  # the control: valid settings buy SPY

    with _trader_environment("sweep_min_clip_invalid_fail_soft", wrap_persist=True) as run:
        del os.environ["SWEEP_MIN_CLIP_USD"]
        real_get_account = run.broker.get_account

        def corrupting_get_account() -> dict:
            os.environ["SWEEP_MIN_CLIP_USD"] = "abc"  # changes after the run has started
            return real_get_account()

        run.broker.get_account = corrupting_get_account  # type: ignore[method-assign]
        _build_trader(run).run()
    buys = [c for c in run.recorder.calls if c["target"] == "broker.submit_buy"]
    assert [c["args"][0] for c in buys] == ["SPY"]  # the snapshot, not the later environment
