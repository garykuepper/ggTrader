"""Engine test fixtures: run on both engines and restore the process-wide setting."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ggTrader.engine import config

ENGINES = ("rust", "numba")


@pytest.fixture(autouse=True)
def _restore_engine() -> Iterator[None]:
    """Tests pin the engine process-wide; put it back for the rest of the suite."""
    before = config.active_engine()
    yield
    config.set_engine(before)


@pytest.fixture(params=ENGINES)
def engine(request: pytest.FixtureRequest) -> str:
    """Run the test once per engine (L2/L3: the whole contract holds on both)."""
    config.set_engine(request.param)
    return request.param


@pytest.fixture
def use_engine() -> Iterator[None]:
    """Marker fixture for tests that call configure() themselves."""
    yield
