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
