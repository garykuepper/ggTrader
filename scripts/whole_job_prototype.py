"""Offline protocol sketch; this module is not wired to the test launchers.

An eventual privileged broker must authenticate and pin each job independently of its
caller, validate the caller's host cgroup namespace and membership, and atomically reject
stale identities and persistent holds before starting a phase. It must enforce that phase's
deadline and, for interruption or timeout, durably hold the job before stopping its entire
verified cgroup. A terminal result is returned only once that operation has succeeded.
Nested cooperating commands must remain in that cgroup; launching a new scope is forbidden.
Registration must bind an opaque ticket to the boot and a pinned cgroup identity. An external
stop must persist the hold before killing that exact cgroup; after an unclean broker exit or
reboot, admission must remain closed until an explicit authorized release. The broker must
detect termination it did not initiate without assuming that every SIGKILL is an OOM.
No broker, cgroup verification, registration or stop operation is implemented here.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PhaseResult:
    """A broker-reported phase status and whether the entire job was stopped."""

    status: int
    terminal: bool = False


class Broker(Protocol):
    """Authority boundary, not an implementation of registration or cgroup control."""

    def run_phase(self, ticket: object, phase: str, seconds: int) -> PhaseResult:
        """Validate and run a phase, persisting a hold before any terminal stop."""
        ...


def run_job(
    broker: Broker,
    ticket: object,
    phases: tuple[str, ...],
    seconds: int,
    *,
    studies: bool = False,
    empty_studies: bool = False,
) -> int:
    """Run one or two phases through an injected authority; never launch a process here."""
    if seconds <= 0 or not 1 <= len(phases) <= 2:
        raise ValueError("one or two phases and a positive deadline are required")
    status = 0
    for index, phase in enumerate(phases):
        result = broker.run_phase(ticket, phase, seconds)
        if result.terminal:
            if not 1 <= result.status <= 255:
                raise RuntimeError("terminal result must report failure")
            return result.status
        if result.status < 0 or result.status == 124 or result.status >= 128:
            raise RuntimeError("broker returned an unheld terminal status")
        accepts_empty = (studies and index == 0) or (empty_studies and index == 1)
        phase_status = 0 if accepts_empty and result.status == 5 else result.status
        if status == 0:
            status = phase_status
    return status
