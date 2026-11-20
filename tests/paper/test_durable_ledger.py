"""A live run refuses a ledger that cannot survive the process (decision D4).

The refusal is the very first statement of `PaperTrader.run()`: before any clock or environment
read and before any broker, ledger or notifier call. The paper tests' autouse fixture lifts it
(`conftest.py`); every test here turns it back on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper import ledger as ledger_module
from ggTrader.paper import trader as trader_module
from ggTrader.paper.ledger import InMemoryLedger, NonDurableLedgerError, ReadOnlyLedger
from ggTrader.paper.persist import PgPaperLedger
from ggTrader.paper.trader import PaperTrader
from tests.paper.sql_recorder import PERSIST_ENGINE, RecordingEngine
from tests.paper.support import SpyLedger

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Reached(Exception):
    """Raised by a probe to prove the run got past the refusal."""


@pytest.fixture(autouse=True)
def refusal_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the refusal back on (the package fixture lifts it for the other paper tests)."""
    monkeypatch.setattr(ledger_module, "ALLOW_NON_DURABLE_LEDGER", False)


def _trader(ledger: object, *, dry_run: bool, **extra: object) -> PaperTrader:
    return PaperTrader(
        MagicMock(name="broker"),
        MagicMock(name="notifier"),
        dry_run=dry_run,
        membership="db",
        ledger=ledger,  # type: ignore[arg-type]
        **extra,  # type: ignore[arg-type]
    )


def _arm_untouchable_hooks(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Make every clock and environment read of `run()` raise, and return the probes."""
    probes = {
        "from_env": MagicMock(side_effect=_Reached("env read")),
        "today_et": MagicMock(side_effect=_Reached("clock read")),
        "now_pt": MagicMock(side_effect=_Reached("clock read")),
    }
    monkeypatch.setattr(trader_module.TraderConfig, "from_env", probes["from_env"])
    monkeypatch.setattr(trader_module, "_today_et", probes["today_et"])
    monkeypatch.setattr(trader_module, "_now_pt", probes["now_pt"])
    return probes


@pytest.mark.parametrize("make", [InMemoryLedger, SpyLedger], ids=["in-memory", "spy"])
def test_live_run_refuses_non_durable_ledger(
    make: type[InMemoryLedger], monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = _arm_untouchable_hooks(monkeypatch)
    ledger = make()
    trader = _trader(ledger, dry_run=False)
    with pytest.raises(NonDurableLedgerError, match=make.__name__):
        trader.run()
    assert not any(probe.called for probe in probes.values())
    assert not trader._broker.mock_calls  # type: ignore[attr-defined]
    assert not trader._notifier.mock_calls  # type: ignore[attr-defined]
    assert not ledger.schema_initialized
    assert ledger.lock_entries == 0


def test_live_run_refuses_in_memory_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    assert issubclass(NonDurableLedgerError, RuntimeError)
    trader = _trader(InMemoryLedger(), dry_run=False)
    with pytest.raises(NonDurableLedgerError, match="InMemoryLedger"):
        trader.run()


def test_live_run_refuses_spy_ledger() -> None:
    with pytest.raises(NonDurableLedgerError, match="SpyLedger"):
        _trader(SpyLedger(), dry_run=False).run()


def test_live_run_refuses_a_ledger_that_does_not_say_it_is_durable() -> None:
    with pytest.raises(NonDurableLedgerError, match="MagicMock"):
        _trader(MagicMock(spec=["paper_run_lock"]), dry_run=False).run()


def test_allow_flag_lifts_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "ALLOW_NON_DURABLE_LEDGER", True)
    _arm_untouchable_hooks(monkeypatch)
    with pytest.raises(_Reached, match="env read"):  # got past the refusal to the env read
        _trader(InMemoryLedger(), dry_run=False).run()


def test_dry_run_allows_in_memory_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_untouchable_hooks(monkeypatch)
    with pytest.raises(_Reached, match="env read"):
        _trader(InMemoryLedger(), dry_run=True).run()


def test_read_only_run_allows_in_memory_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_untouchable_hooks(monkeypatch)
    trader = _trader(InMemoryLedger(), dry_run=True, read_only=True)
    assert isinstance(trader._ledger, ReadOnlyLedger)
    with pytest.raises(_Reached, match="env read"):
        trader.run()


def test_dry_run_with_writes_allows_in_memory_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_untouchable_hooks(monkeypatch)
    with pytest.raises(_Reached, match="env read"):
        _trader(InMemoryLedger(), dry_run=True, read_only=False).run()


def test_live_run_accepts_pg_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RecordingEngine([])
    ledger = PgPaperLedger()
    assert ledger.durable is True
    monkeypatch.setattr(
        PgPaperLedger, "paper_run_lock", MagicMock(side_effect=_Reached("lock taken"))
    )
    trader = _trader(ledger, dry_run=False)
    monkeypatch.setattr(
        trader_module.TraderConfig,
        "from_env",
        MagicMock(return_value=MagicMock(stop_enabled=False, stop_threshold=None)),
    )
    with patch(PERSIST_ENGINE, return_value=engine):
        with pytest.raises(_Reached, match="lock taken"):  # past the refusal, at the lock
            trader.run()


def test_the_default_ledger_is_durable() -> None:
    with patch.object(trader_module, "PgPaperLedger", PgPaperLedger):
        trader = PaperTrader(MagicMock(), MagicMock(), dry_run=False, membership="db")
    assert trader._ledger.durable is True


def test_read_only_wrapper_follows_inner_durability() -> None:
    skipped: list[str] = []
    log = MagicMock()
    assert ReadOnlyLedger(PgPaperLedger(), skipped, log).durable is True
    assert ReadOnlyLedger(InMemoryLedger(), skipped, log).durable is False
    assert ReadOnlyLedger(SpyLedger(), skipped, log).durable is False


def test_every_ledger_class_declares_durable() -> None:
    assert PgPaperLedger.durable is True
    assert InMemoryLedger.durable is False
    assert SpyLedger.durable is False


def test_the_flag_defaults_to_false_in_a_fresh_interpreter() -> None:
    code = (
        "import ggTrader.paper.ledger as m; "
        "print(m.ALLOW_NON_DURABLE_LEDGER is False, m.NonDurableLedgerError.__mro__[1].__name__)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.stdout.split() == ["True", "RuntimeError"]
