"""Wall-clock limit for one test, so a hang becomes a failure instead of an out-of-memory kill.

On 2026-11-18 a paper test looped forever and kept allocating until the memory cap (or, with two
runs at once, the machine) gave out. A spin loop that allocates nothing would never reach a cap at
all. ``time_limit`` interrupts the test in the main thread after a fixed time with
``TestTimedOut``, which pytest reports as an ordinary failure with a message that says what
happened. It cannot interrupt a single long call into C code (the Rust simulator, numba) until
that call returns; the memory cap and the runner's time limit cover that case.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

#: Tests that assert on the process's own SIGALRM timer, which the per-test limit (a running
#: timer around every test) would make fail for a reason that has nothing to do with them. Each
#: one is frozen study code that cannot be edited; the root conftest runs it without the limit.
UNTIMED_TESTS: frozenset[str] = frozenset(
    {
        "research/intraday/_lib/tests/test_intraday_probe.py"
        "::test_alarm_bounds_total_body_time_and_restores_handler",
    }
)


class TestTimedOut(Exception):
    """A test ran longer than its wall-clock limit."""

    __test__ = False  # the name starts with "Test": keep pytest from trying to collect it


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise ``TestTimedOut`` in this thread if the block runs longer than ``seconds``.

    Does nothing for ``seconds <= 0``, off the main thread, or where ``SIGALRM`` does not exist.
    The previous handler and any running timer are restored on exit (the remaining time of an
    outer timer is reduced by the time spent inside).
    """
    usable = (
        seconds > 0
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def expired(signum: int, frame: object) -> None:
        raise TestTimedOut(f"test exceeded {seconds:g}s (a hang?)")

    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_remaining, previous_interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            left = max(previous_remaining - (time.monotonic() - started), 0.001)
            signal.setitimer(signal.ITIMER_REAL, left, previous_interval)
