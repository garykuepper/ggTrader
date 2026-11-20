"""Prevent external connections throughout paper tests."""

import pytest

from tests.netguard import blocked_connections, guard_applies


@pytest.fixture(autouse=True)
def no_real_connections(request: pytest.FixtureRequest):
    """Guard each paper test unless it is marked integration."""
    markers = {marker.name for marker in request.node.iter_markers()}
    if guard_applies(request.node.nodeid, markers):
        with blocked_connections():
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def allow_non_durable_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lift the live-run refusal of a non-durable ledger for every paper test, then restore it.

    Tests drive live runs with in-memory ledgers; the refusal's own tests turn it back on. The
    package import (`tests/paper/__init__.py`) also lifts it, for session fixtures and children.
    """
    from ggTrader.paper import ledger as ledger_module

    monkeypatch.setattr(ledger_module, "ALLOW_NON_DURABLE_LEDGER", True, raising=False)
