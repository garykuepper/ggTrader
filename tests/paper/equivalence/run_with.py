"""Run a scenario under a chosen implementation of `PaperTrader._run` (immutable set; Task 47).

This file defines the one way an implementation is selected. `impl.IMPLS` (the only thing Task 74
edits) decides which names are valid, read at call time, so once `"legacy"` leaves `IMPLS` the
patch below is unreachable by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal
from unittest.mock import patch

from ggTrader.paper.trader import PaperTrader
from tests.paper.equivalence import impl
from tests.paper.transcript.harness import Scenario, run_scenario

Impl = Literal["new", "legacy"]


@contextmanager
def implementation(impl_name: str) -> Iterator[None]:
    """Select `impl_name` for the duration of the `with`; raise `ValueError` if not in `IMPLS`."""
    if impl_name not in impl.IMPLS:
        raise ValueError(f"unknown implementation {impl_name!r}; expected one of {impl.IMPLS}")
    if impl_name == "new":
        yield
    elif impl_name == "legacy":
        with patch.object(PaperTrader, "_run", PaperTrader._run_legacy):
            yield
    else:
        raise ValueError(f"no selection rule for implementation {impl_name!r}")


def run_with(impl_name: str, scenario: Scenario) -> dict:
    """Run `scenario` under `impl_name` and return its normalized transcript."""
    with implementation(impl_name):
        return run_scenario(scenario)
