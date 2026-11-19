"""Two tests that hang on purpose; tests/test_timeout_guard.py runs them in a child pytest.

The file name does not match ``test_*.py``, so a normal run never collects it. A spin loop
allocates no memory, so this cannot exhaust RAM even if the timeout guard were broken.
"""

import pytest


def test_hangs() -> None:
    while True:
        pass


@pytest.mark.timeout_seconds(1)
def test_marked_hangs() -> None:
    while True:
        pass
