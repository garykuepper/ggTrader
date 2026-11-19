"""Main process of the fixed whole-job test unit: runs phases only when the broker admits them.

It is started by the broker as ``ggtrader-whole-job.service`` (see deploy/whole_job/). It takes
no paths or PIDs: the broker socket and the phase commands are constants, and the commands run
relative to the unit's fixed ``WorkingDirectory``. Each phase is an ordinary child process in
the unit's own cgroup; it never starts a scope or service (``GGT_TEST_NO_CGROUP=1``), and an
unprivileged system user without a user manager cannot start one. The broker answers ``go``
once the job is bound, and ``ok`` before each phase. A refusal, a lost broker connection or a
deadline ends the job: the driver kills its phase's process group and exits non-zero. This is
cooperative cleanup only; the broker's pinned ``cgroup.kill`` is the authoritative stop.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.whole_job_broker import (  # noqa: E402
    JOB_SOCKET,
    MAX_MESSAGE,
    PHASES,
    REGISTER_SECONDS,
)
from scripts.whole_job_prototype import PhaseResult, run_job  # noqa: E402

PHASE_COMMANDS = {
    "core": ("scripts/run_tests.sh", "--tier", "full", "tests"),
    "studies": ("scripts/run_tests.sh", "--studies"),
}
PHASE_RUN_SECONDS = 1800
REFUSED_STATUS = 75
TIMEOUT_STATUS = 124
POLL_SECONDS = 0.1


class BrokerLink:
    """The driver's side of the broker protocol over one connected SEQPACKET socket."""

    def __init__(self, sock: socket.socket, nonce: str) -> None:
        self.sock = sock
        self.nonce = nonce

    def ask(self, *words: str) -> str:
        self.sock.send(" ".join((words[0], self.nonce, *words[1:])).encode("ascii"))
        return self.sock.recv(MAX_MESSAGE).decode("ascii")

    def run_phase(self, ticket: object, phase: str, seconds: int) -> PhaseResult:
        if self.ask("phase", phase) != f"ok {phase}":
            return PhaseResult(REFUSED_STATUS, terminal=True)
        return run_child(PHASE_COMMANDS[phase], seconds, self.sock)


def run_child(command: Sequence[str], seconds: float, sock: socket.socket) -> PhaseResult:
    """Run one phase; end it if the deadline passes or the broker connection drops."""
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "GGT_TEST_NO_CGROUP": "1",
        "GGT_TEST_MAX_SECONDS": str(int(seconds)),
    }
    child = subprocess.Popen(list(command), env=env, start_new_session=True, close_fds=True)
    poller = select.poll()
    poller.register(sock.fileno(), select.POLLIN | select.POLLHUP)
    end = time.monotonic() + seconds
    while True:
        code = child.poll()
        if code is not None:
            if code < 0:
                return PhaseResult(128 - code, terminal=True)
            return PhaseResult(code, terminal=code == TIMEOUT_STATUS or code >= 128)
        left = end - time.monotonic()
        if left <= 0:
            _kill_group(child)
            return PhaseResult(TIMEOUT_STATUS, terminal=True)
        if poller.poll(min(left, POLL_SECONDS) * 1000):
            _kill_group(child)
            return PhaseResult(REFUSED_STATUS, terminal=True)


def _kill_group(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def connect() -> BrokerLink | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
    sock.settimeout(REGISTER_SECONDS)
    sock.connect(JOB_SOCKET)
    hello = sock.recv(MAX_MESSAGE).decode("ascii").split(" ")
    if len(hello) != 2 or hello[0] != "hello":
        sock.close()
        return None
    link = BrokerLink(sock, hello[1])
    if link.ask("register") != "go":
        sock.close()
        return None
    sock.settimeout(None)
    return link


def drive(link: BrokerLink | None) -> int:
    """Run the fixed phases through the broker and report the result before exiting."""
    if link is None:
        return REFUSED_STATUS
    status = run_job(link, None, PHASES, PHASE_RUN_SECONDS)
    try:
        link.ask("done", str(status))
    except OSError:
        pass
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Whole-job driver: run only as the fixed test unit's main process."
    )
    parser.parse_args()
    return drive(connect())


if __name__ == "__main__":
    sys.exit(main())
