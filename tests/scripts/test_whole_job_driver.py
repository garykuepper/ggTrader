"""Driver behaviour with real short-lived child processes and a local socket pair only."""

import socket
import sys

import pytest

from scripts import whole_job_driver as driver
from scripts.whole_job_prototype import PhaseResult

SLEEP = (sys.executable, "-c", "import time; time.sleep(30)")


def test_lost_broker_connection_kills_the_phase() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left:
        right.close()
        assert driver.run_child(SLEEP, 20, left) == PhaseResult(75, terminal=True)


def test_deadline_kills_the_phase() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        assert driver.run_child(SLEEP, 0.3, left) == PhaseResult(124, terminal=True)


@pytest.mark.parametrize(
    "code,expected",
    [("0", PhaseResult(0)), ("1", PhaseResult(1)), ("137", PhaseResult(137, terminal=True))],
)
def test_exit_status_mapping(code: str, expected: PhaseResult) -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        command = (sys.executable, "-c", f"raise SystemExit({code})")
        assert driver.run_child(command, 20, left) == expected


def test_signalled_child_is_terminal() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        command = (sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
        assert driver.run_child(command, 20, left) == PhaseResult(137, terminal=True)


class FakeLink:
    def __init__(self, answers: dict[str, str]) -> None:
        self.sock = None
        self.answers = answers
        self.asked: list[tuple[str, ...]] = []

    def ask(self, *words: str) -> str:
        self.asked.append(words)
        return self.answers.get(words[0], "")

    run_phase = driver.BrokerLink.run_phase


def test_refused_phase_stops_before_any_command(monkeypatch) -> None:
    link = FakeLink({"phase": "refused held"})
    monkeypatch.setattr(driver, "run_child", lambda *a: pytest.fail("phase ran"))
    assert driver.drive(link) == 75
    assert link.asked == [("phase", "core"), ("done", "75")]


def test_terminal_first_phase_skips_second(monkeypatch) -> None:
    link = FakeLink({"phase": "ok core"})
    ran: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        driver, "run_child", lambda cmd, s, sock: ran.append(cmd) or PhaseResult(137, True)
    )
    assert driver.drive(link) == 137
    assert ran == [driver.PHASE_COMMANDS["core"]]


def test_no_registration_means_no_work() -> None:
    assert driver.drive(None) == 75
