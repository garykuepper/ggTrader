"""Whole-job broker for one fixed systemd test service: a guest-test candidate, not installed.

Trust model (see deploy/whole_job/README.md). The broker is a fixed root system service. It
admits work from exactly one fixed, nondelegated, unprivileged system service
(``TEST_UNIT``) and accepts start, stop, status and acknowledge requests only from the fixed
root oneshot operator template (``ggtrader-job-operator@<action>.service``). Every request is
authenticated per message: the kernel attaches the sender's pidfd (``SCM_PIDFD``), PID 1 maps
it to a unit and invocation (``GetUnitByPIDFD``), and the broker requires the sender to be that
unit's current ``MainPID``, in that unit's cgroup, before and after the lookup. No request
carries a path, PID, cgroup or unit name: they are constants here and in the unit files.

A job is bound in the root-owned admission record (``whole_job_admission``) to the boot, a
random attempt, the service invocation, the cgroup id and the device/inode of pinned
``cgroup.kill`` and ``cgroup.events`` handles. The handles go into the broker's systemd file
descriptor store; a successful notification or barrier proves nothing, so the broker reads the
store back (``DumpUnitFileDescriptorStore``) and admits the job only when the store holds
exactly the expected pair. The driver runs no phase until the broker answers ``go``, and asks
again before each phase; each answer is given under the admission lock.

Stop persists a hold, writes the pinned ``cgroup.kill`` and waits (bounded) for the pinned
``cgroup.events`` to report the cgroup empty or removed; it then removes the stored pair and
reads the store back, and only when it is empty is the record cleared and the pinned handles
closed. Any failure leaves the job held with its record and handles. A broker restart starts
held: the job may finish its current phase but no further phase is admitted, and the store must
again match exactly or the broker stays held as unverifiable. A restart with no recorded job is
idle only when nothing of an unrecorded invocation can remain: a stored exact pair is killed
and drained through its own pinned handles, any other stored entry stays as evidence, and with
nothing stored the unit must be inactive and its cgroup absent or empty; otherwise the broker
stays unverifiable. A record from an earlier boot stays held until an operator acknowledges
that exact attempt. A termination the broker did not initiate is reported as such, never as an
out-of-memory kill.

The broker never forks while the admission lock is held: PID 1 queries run ``busctl`` only
outside it. Kernel and systemd behaviour this relies on was observed in a disposable systemd
259.9 guest, not on Encom; see the README for which sequences remain untested.
"""

from __future__ import annotations

import argparse
import array
import errno
import fcntl
import grp
import hmac
import json
import os
import re
import secrets
import select
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.whole_job_admission import AdmissionState  # noqa: E402

TEST_UNIT = "ggtrader-whole-job.service"
TEST_CGROUP = "/system.slice/ggtrader-whole-job.service"
CGROUP_ROOT = "/sys/fs/cgroup"
BROKER_UNIT = "ggtrader-job-broker.service"
OPERATOR_UNIT = re.compile(
    r"ggtrader-job-operator@(status|start|stop|acknowledge)(?::([0-9a-f]{32}))?\.service"
)
JOB_GROUP = "ggtrader-job"
RUNTIME_DIR = "/run/ggtrader-job-broker"
STATE_DIR = "/var/lib/ggtrader-job-broker"
JOB_SOCKET = "/run/ggtrader-job-broker/job.sock"
OPERATOR_SOCKET = "/run/ggtrader-job-broker/operator.sock"
KILL_NAME = "job-kill"
EVENTS_NAME = "job-events"
PHASES = ("core", "studies")
REGISTER_SECONDS = 60.0
PHASE_SECONDS = 1860.0
STOP_SECONDS = 20.0
STOP_RETRY_SECONDS = 15.0
IO_SECONDS = 5.0
MAX_MESSAGE = 512
SO_PASSPIDFD = 76
SCM_PIDFD = 4
ACCMODE = os.O_RDONLY | os.O_WRONLY | os.O_RDWR
BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RECORD = re.compile(
    r"v1;([0-9a-f]{32});([0-9a-f]{32});([0-9]{1,20});([0-9]{1,20}):([0-9]{1,20});"
    r"([0-9]{1,20}):([0-9]{1,20})"
)


class Refused(Exception):
    """A request was not authorized or not valid in the current state; nothing was admitted."""


FileId = tuple[int, int]


@dataclass(frozen=True)
class UnitState:
    active_state: str
    main_pid: int
    invocation: str
    control_group: str
    control_group_id: int
    delegate: bool


@dataclass(frozen=True)
class Peer:
    unit: str
    invocation: str
    pid: int
    control_group: str
    control_group_id: int


@dataclass(frozen=True)
class StoreEntry:
    """One ``DumpUnitFileDescriptorStore`` row, reduced to what is compared."""

    name: str
    mode: int
    dev: int
    inode: int
    flags: int


@dataclass(frozen=True)
class JobRecord:
    attempt: str
    invocation: str
    cgroup_id: int
    kill: FileId
    events: FileId

    def encode(self) -> str:
        return (
            f"v1;{self.attempt};{self.invocation};{self.cgroup_id};"
            f"{self.kill[0]}:{self.kill[1]};{self.events[0]}:{self.events[1]}"
        )

    @classmethod
    def parse(cls, text: str) -> JobRecord:
        match = RECORD.fullmatch(text)
        if match is None:
            raise ValueError("invalid job record")
        a, inv, cg, kd, ki, ed, ei = match.groups()
        return cls(a, inv, int(cg), (int(kd), int(ki)), (int(ed), int(ei)))


class Handles:
    """Pinned ``cgroup.kill`` (write-only) and ``cgroup.events`` (read-only) descriptors."""

    def __init__(self, kill: int, events: int) -> None:
        self.kill = kill
        self.events = events

    def ids(self) -> tuple[FileId, FileId]:
        return file_id(self.kill, os.O_WRONLY), file_id(self.events, os.O_RDONLY)

    def close(self) -> None:
        for fd in (self.kill, self.events):
            try:
                os.close(fd)
            except OSError:
                pass


def file_id(fd: int, accmode: int) -> FileId:
    """Device and inode of a regular file opened with exactly ``accmode``."""
    info = os.fstat(fd)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if not stat.S_ISREG(info.st_mode) or flags & ACCMODE != accmode or flags & os.O_PATH:
        raise Refused("pinned handle has the wrong type or access mode")
    return info.st_dev, info.st_ino


class System(Protocol):
    """Host operations. ``LinuxSystem`` is the real one; tests inject a fixture."""

    def now(self) -> float: ...
    def boot_id(self) -> str: ...
    def pidfd_pid(self, pidfd: int) -> int: ...
    def unit_by_pidfd(self, pidfd: int) -> tuple[str, str]: ...
    def unit_state(self, unit: str) -> UnitState: ...
    def proc_cgroup(self, pid: int) -> str: ...
    def start_unit(self, unit: str) -> None: ...
    def open_job_cgroup(self, cgroup_id: int) -> Handles: ...
    def populated(self, handles: Handles) -> bool | None: ...
    def kill(self, handles: Handles) -> None: ...
    def wait_empty(self, handles: Handles, seconds: float) -> bool: ...
    def job_cgroup_populated(self) -> bool | None: ...
    def fdstore_add(self, name: str, fd: int) -> None: ...
    def fdstore_remove(self, name: str) -> None: ...
    def fdstore_barrier(self) -> None: ...
    def fdstore_dump(self) -> list[StoreEntry]: ...


def authenticate(system: System, pidfd: int, unit: str, cgroup: str | None) -> Peer:
    """Map a kernel-attached sender pidfd to a fixed unit's current main process, or refuse."""
    first = system.pidfd_pid(pidfd)
    if first <= 0:
        raise Refused("sender has exited or is outside this PID namespace")
    found, invocation = system.unit_by_pidfd(pidfd)
    if found != unit:
        raise Refused("sender is not in the expected unit")
    state = system.unit_state(unit)
    if (
        state.delegate
        or state.main_pid != first
        or state.invocation != invocation
        or not re.fullmatch(r"[0-9a-f]{32}", invocation)
        or (cgroup is not None and state.control_group != cgroup)
        or system.proc_cgroup(first) != state.control_group
        or system.unit_by_pidfd(pidfd) != (unit, invocation)
        or system.pidfd_pid(pidfd) != first
    ):
        raise Refused("sender is not the unit's current main process")
    return Peer(unit, invocation, first, state.control_group, state.control_group_id)


class Broker:
    """Admission, registration, phase and stop sequencing for the one fixed test unit."""

    def __init__(
        self, system: System, state: AdmissionState, log: Callable[[str], None] = print
    ) -> None:
        self.system = system
        self.state = state
        self.log = log
        self.boot = system.boot_id()
        if BOOT_ID.fullmatch(self.boot) is None:
            raise ValueError("invalid boot id")
        self.mode = "starting"
        self.record: JobRecord | None = None
        self.record_boot = ""
        self.handles: Handles | None = None
        self.driver: Peer | None = None
        self.pending: tuple[str, float] | None = None
        self.phases: list[str] = []
        self.deadline: float | None = None
        self.reported: int | None = None
        self.outcome = ""
        self.retry_at: float | None = None

    # Startup ------------------------------------------------------------------------------

    def startup(self, inherited: dict[str, list[int]]) -> None:
        """Verify the persisted record against the inherited store; admission stays held."""
        try:
            snap = self.state.snapshot()
            entries = self.system.fdstore_dump()
        except BaseException:
            self._close_inherited(inherited)
            self.mode = "unverifiable"
            raise
        if not snap["job"]:
            self._startup_unrecorded(inherited, entries)
            return
        try:
            record = JobRecord.parse(snap["job"])
        except ValueError:
            self._close_inherited(inherited)
            self.mode = "unverifiable"
            self.log("persisted job record is malformed; admission stays held")
            return
        self.record, self.record_boot = record, snap["boot"]
        if snap["boot"] != self.boot:
            self._close_inherited(inherited)
            if entries:
                self.mode = "unverifiable"
                self.log("prior-boot record but the descriptor store is not empty")
            else:
                self.mode = "prior-boot"
                self.log(f"job {record.attempt} belongs to an earlier boot; acknowledge to clear")
            return
        handles = self._restore(record, inherited, entries)
        if handles is None:
            self.mode = "unverifiable"
            self.log(f"job {record.attempt}: stored handles do not match; admission stays held")
            return
        self.handles = handles
        self.mode = "held"
        self.outcome = "broker restarted; job held, no further phase is admitted"
        self.log(f"job {record.attempt}: pinned handles restored; job held")

    def _restore(
        self, record: JobRecord, inherited: dict[str, list[int]], entries: list[StoreEntry]
    ) -> Handles | None:
        kill = inherited.get(KILL_NAME, [])
        events = inherited.get(EVENTS_NAME, [])
        ok = set(inherited) <= {KILL_NAME, EVENTS_NAME} and len(kill) == 1 and len(events) == 1
        if ok:
            handles = Handles(kill[0], events[0])
            try:
                ok = handles.ids() == (record.kill, record.events)
            except (OSError, Refused):
                ok = False
            ok = ok and store_matches(entries, record)
            if ok:
                return handles
        self._close_inherited(inherited)
        return None

    def _startup_unrecorded(
        self, inherited: dict[str, list[int]], entries: list[StoreEntry]
    ) -> None:
        """No recorded job: go idle only once no unrecorded invocation can remain.

        Stored descriptors without a record mean a registration was cut short after the pair was
        stored (a crash, or a refused registration whose identity could not be recorded). That
        invocation never got ``go``. An exact pair is killed and drained through its own pinned
        handles; anything else stays in the store as evidence and the broker stays unverifiable.
        With nothing stored, the fixed unit must be inactive and its cgroup absent or empty.
        """
        if inherited or entries:
            handles = self._orphan_pair(inherited, entries)
            if handles is None:
                self.mode = "unverifiable"
                self.outcome = "descriptor store holds entries with no recorded job; kept"
                self.log(f"{self.outcome}; admission stays held")
                return
            try:
                self.system.kill(handles)
                if not self.system.wait_empty(handles, STOP_SECONDS):
                    raise TimeoutError("unrecorded cgroup did not empty")
                self._drain_verified()
            except BaseException as exc:  # noqa: BLE001 - fail closed, keep what is stored
                handles.close()
                self.mode = "unverifiable"
                self.outcome = f"unrecorded stored invocation not shown stopped ({exc!r})"
                self.log(f"{self.outcome}; admission stays held")
                return
            handles.close()
            self.log("unrecorded stored invocation killed; cgroup empty and store drained")
        try:
            unit = self.system.unit_state(TEST_UNIT)
            live = self.system.job_cgroup_populated()
        except Exception as exc:  # noqa: BLE001 - fail closed but keep serving status
            self.mode = "unverifiable"
            self.outcome = f"no recorded job and the test unit could not be checked ({exc!r})"
            self.log(f"{self.outcome}; restart the broker to recheck")
            return
        # A stored pair just emptied may leave the unit briefly deactivating; its pinned cgroup
        # is the evidence there. Without one, the unit's own state must also show it stopped.
        stopped = bool(inherited or entries) or unit.active_state in ("inactive", "failed")
        if live is True or not stopped:
            self.mode = "unverifiable"
            self.outcome = (
                f"no recorded job but the test unit is {unit.active_state} "
                f"(cgroup populated: {live}); restart the broker once it has stopped"
            )
            self.log(f"{self.outcome}; admission stays held")
            return
        self.mode = "idle"

    def _orphan_pair(
        self, inherited: dict[str, list[int]], entries: list[StoreEntry]
    ) -> Handles | None:
        """The inherited pair if it is exactly one pinned ``kill``/``events`` pair, else None."""
        kill = inherited.get(KILL_NAME, [])
        events = inherited.get(EVENTS_NAME, [])
        if set(inherited) == {KILL_NAME, EVENTS_NAME} and len(kill) == 1 and len(events) == 1:
            handles = Handles(kill[0], events[0])
            try:
                if store_holds(entries, *handles.ids()):
                    return handles
            except (OSError, Refused):
                pass
        self._close_inherited(inherited)
        return None

    @staticmethod
    def _close_inherited(inherited: dict[str, list[int]]) -> None:
        for fds in inherited.values():
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _drain_verified(self) -> None:
        """Remove the pinned pair and anything else stored, then require the store read empty.

        Never called under the admission lock: the read-back is a PID 1 query (a fork).
        """
        names = {entry.name for entry in self.system.fdstore_dump()} | {KILL_NAME, EVENTS_NAME}
        for name in sorted(names):
            self.system.fdstore_remove(name)
        self.system.fdstore_barrier()
        if self.system.fdstore_dump():
            raise RuntimeError("descriptor store not empty after removal")

    # Operator requests ---------------------------------------------------------------------

    def operator(self, action: str, attempt: str | None) -> dict[str, Any]:
        if action == "status" and attempt is None:
            return self.status()
        if action == "start" and attempt is None:
            return {"started": self.start(), **self.status()}
        if action == "stop" and attempt is not None:
            self.stop(attempt)
            return self.status()
        if action == "acknowledge" and attempt is not None:
            self.acknowledge(attempt)
            return self.status()
        raise Refused("unsupported operator request")

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "attempt": self.record.attempt if self.record else None,
            "invocation": self.record.invocation if self.record else None,
            "pending": self.pending[0] if self.pending else None,
            "phases": list(self.phases),
            "reported": self.reported,
            "outcome": self.outcome,
        }

    def start(self) -> str:
        """Start the fixed unit; nothing runs until its main process registers and gets ``go``."""
        if self.mode != "idle":
            raise Refused(f"cannot start while {self.mode}")
        snap = self.state.snapshot()
        if not snap["held"] or snap["job"]:
            raise Refused("admission record is not held and clear")
        if self.system.fdstore_dump():
            raise Refused("descriptor store is not empty")
        if self.system.unit_state(TEST_UNIT).active_state not in ("inactive", "failed"):
            raise Refused("test unit is already active")
        attempt = secrets.token_hex(16)
        self.pending = (attempt, self.system.now() + REGISTER_SECONDS)
        self.mode = "pending"
        self.record = None
        self.phases, self.reported, self.deadline, self.outcome = [], None, None, ""
        self.retry_at = None
        try:
            self.system.start_unit(TEST_UNIT)
        except BaseException:
            self.pending, self.mode = None, "idle"
            raise
        self.log(f"attempt {attempt}: start requested")
        return attempt

    def stop(self, attempt: str) -> None:
        """Stop only the named attempt; a stale request never holds a replacement."""
        if self.record is None or not hmac.compare_digest(attempt, self.record.attempt):
            raise Refused("attempt is not the current job")
        if self.mode not in ("running", "held"):
            raise Refused(f"cannot stop while {self.mode}")
        self._stop("operator stop")

    def acknowledge(self, attempt: str) -> None:
        """Clear a record from an earlier boot: its cgroup cannot exist on this boot."""
        record = self.record
        if self.mode != "prior-boot" or record is None:
            raise Refused("no prior-boot record to acknowledge")
        if not hmac.compare_digest(attempt, record.attempt) or self.record_boot == self.boot:
            raise Refused("attempt or boot does not match")
        self.state.clear(record.encode(), self.record_boot, self.state.epoch)
        self.mode, self.outcome = "idle", f"prior-boot job {record.attempt} acknowledged"
        self.log(self.outcome)

    # Driver requests -----------------------------------------------------------------------

    def register(self, pidfd: int, send_go: Callable[[], None]) -> None:
        """Bind the fixed unit's main process and pinned cgroup, then answer ``go``."""
        if self.mode != "pending" or self.pending is None:
            raise Refused("no start is pending")
        attempt, deadline = self.pending
        if self.system.now() > deadline:
            raise Refused("registration window expired")
        peer = authenticate(self.system, pidfd, TEST_UNIT, TEST_CGROUP)
        handles = self.system.open_job_cgroup(peer.control_group_id)
        record: JobRecord | None = None
        try:
            kill_id, events_id = handles.ids()
            record = JobRecord(attempt, peer.invocation, peer.control_group_id, kill_id, events_id)
            if self.system.populated(handles) is not True:
                raise Refused("pinned cgroup is not populated")
            if authenticate(self.system, pidfd, TEST_UNIT, TEST_CGROUP) != peer:
                raise Refused("sender identity changed during registration")
            self.system.fdstore_add(KILL_NAME, handles.kill)
            self.system.fdstore_add(EVENTS_NAME, handles.events)
            self.system.fdstore_barrier()
            if not store_matches(self.system.fdstore_dump(), record):
                raise Refused("descriptor store does not hold exactly the pinned pair")
            self.state.release(record.encode(), self.boot)
        except BaseException:
            self._abort_unadmitted(handles, record)
            raise
        self.record, self.record_boot, self.handles, self.driver = record, self.boot, handles, peer
        self.pending, self.mode = None, "running"
        try:
            with self.state.admission(record.encode(), self.boot):
                self._check_live(pidfd)
                send_go()
        except BaseException:
            self.mode = "held"
            self._stop_quietly("admission failed after registration")
            raise
        self.log(f"attempt {attempt}: invocation {peer.invocation} admitted")

    def phase(self, pidfd: int, name: str, send_ok: Callable[[], None]) -> None:
        """Admit the next phase in fixed order under the admission lock."""
        if self.mode != "running" or self.record is None or self.driver is None:
            raise Refused(f"no phase is admitted while {self.mode}")
        if name not in PHASES or PHASES[len(self.phases) : len(self.phases) + 1] != (name,):
            raise Refused("phase is out of order or repeated")
        if authenticate(self.system, pidfd, TEST_UNIT, TEST_CGROUP) != self.driver:
            raise Refused("sender is not the registered driver")
        try:
            with self.state.admission(self.record.encode(), self.boot):
                self._check_live(pidfd)
                self.phases.append(name)
                self.deadline = self.system.now() + PHASE_SECONDS
                send_ok()
        except PermissionError:
            self.mode = "held"  # Refused before launch: the current phase may finish, no more.
            raise
        except BaseException:
            self.mode = "held"
            self._stop_quietly("phase admission failed")
            raise
        self.log(f"attempt {self.record.attempt}: phase {name} admitted")

    def done(self, pidfd: int, status: int) -> None:
        """Record the driver's own exit report; it is a label, not evidence of cleanup."""
        if self.driver is None or self.mode not in ("running", "held"):
            raise Refused("no registered driver")
        if not 0 <= status <= 255:
            raise Refused("invalid status")
        if authenticate(self.system, pidfd, TEST_UNIT, TEST_CGROUP) != self.driver:
            raise Refused("sender is not the registered driver")
        self.reported = status

    def driver_closed(self) -> None:
        """The driver's connection ended: stop whatever remains of the job."""
        if self.mode == "running":
            self._stop_quietly(self._ended_reason())

    def _check_live(self, pidfd: int) -> None:
        """In-process checks only (no fork while the admission lock is held)."""
        if self.driver is None or self.system.pidfd_pid(pidfd) != self.driver.pid:
            raise Refused("driver exited")
        if self.handles is None or self.record is None:
            raise Refused("no pinned handles")
        if self.handles.ids() != (self.record.kill, self.record.events):
            raise Refused("pinned handles changed")

    # Periodic checks and stop --------------------------------------------------------------

    def tick(self) -> None:
        now = self.system.now()
        if self.mode == "pending" and self.pending and now > self.pending[1]:
            self.log(f"attempt {self.pending[0]}: no registration; start abandoned")
            self.pending, self.mode = None, "idle"
            self.outcome = "start abandoned: driver did not register"
            return
        if self.mode not in ("running", "held") or self.handles is None:
            return
        if self.retry_at is not None and now < self.retry_at:
            return
        if self.mode == "running" and self.deadline is not None and now > self.deadline:
            self._stop_quietly("phase deadline exceeded")
            return
        try:
            populated = self.system.populated(self.handles)
        except OSError as exc:
            self.log(f"cannot read pinned cgroup events: {exc}")
            return
        if not populated:
            self._stop_quietly(self._ended_reason())

    def _ended_reason(self) -> str:
        if self.reported is not None:
            return f"job ended; driver reported status {self.reported}"
        return "job ended without a driver report; cause not determined"

    def _stop_quietly(self, reason: str) -> None:
        try:
            self._stop(reason)
        except BaseException as exc:  # noqa: BLE001 - logged, admission stays held
            self.retry_at = self.system.now() + STOP_RETRY_SECONDS
            self.log(f"stop failed ({reason}): {exc!r}; admission stays held")

    def _stop(self, reason: str) -> None:
        """Hold, kill the pinned cgroup, verify it empty, drain and verify the store, then clear.

        The record and the pinned handles are kept until every step is verified, so a failure
        at any point leaves the job held and retryable through the same handles.
        """
        record, handles = self.record, self.handles
        if record is None or handles is None:
            raise Refused("no pinned job")
        job = record.encode()
        self.deadline = None
        try:
            with self.state.held_job(job, self.boot, self.state.epoch):
                self.mode = "stopping"
                if handles.ids() != (record.kill, record.events):
                    raise Refused("pinned handles changed")
                self.system.kill(handles)
                if not self.system.wait_empty(handles, STOP_SECONDS):
                    raise TimeoutError("pinned cgroup did not empty")
            self._drain_verified()  # outside the admission lock: it queries PID 1
            self.state.clear(job, self.boot, self.state.epoch)
        except BaseException:
            self.mode = "held"
            self.outcome = f"stop incomplete ({reason}); admission held"
            raise
        handles.close()
        self.handles, self.retry_at = None, None
        self.mode, self.driver, self.outcome = "idle", None, reason
        self.log(f"attempt {record.attempt}: {reason}; cleared")

    def _abort_unadmitted(self, handles: Handles, record: JobRecord | None) -> None:
        """A registration failed before admission: kill that exact cgroup, then forget it.

        The handles are closed and the identity dropped only after the cgroup is verified empty
        and the descriptor store verified empty. Otherwise the job stays held: its identity is
        persisted (held) and the pinned handles kept, so ``@stop:<attempt>``, the cgroup
        emptying, or a broker restart (which restores the stored pair) can still clear it.
        """
        self.pending = None
        try:
            self.system.kill(handles)
            if not self.system.wait_empty(handles, STOP_SECONDS):
                raise TimeoutError("unadmitted cgroup did not empty")
            self._drain_verified()
        except BaseException as exc:  # noqa: BLE001 - fail closed: keep the job held
            self._hold_unadmitted(handles, record, exc)
            return
        handles.close()
        self.mode = "idle"
        self.outcome = "registration refused; unadmitted invocation stopped"
        self.log(self.outcome)

    def _hold_unadmitted(
        self, handles: Handles, record: JobRecord | None, exc: BaseException
    ) -> None:
        """Keep an unadmitted job that was not shown stopped as held, never as idle."""
        self.handles, self.driver = handles, None
        if record is not None:
            try:
                self.state.retain(record.encode(), self.boot)
                self.record, self.record_boot = record, self.boot
                self.mode = "held"
                self.outcome = (
                    f"registration refused; unadmitted invocation stop not verified ({exc!r}); "
                    "held until a verified stop"
                )
                self.log(f"attempt {record.attempt}: {self.outcome}")
                return
            except BaseException as hold_exc:  # noqa: BLE001 - falls through to unverifiable
                exc = hold_exc
        self.record = record
        self.mode = "unverifiable"
        self.outcome = (
            f"registration refused; unadmitted invocation stop not verified and not recorded "
            f"({exc!r}); manual investigation needed"
        )
        self.log(self.outcome)


def store_matches(entries: list[StoreEntry], record: JobRecord) -> bool:
    """True only for exactly the pinned pair: names, regular files, device, inode, access."""
    return store_holds(entries, record.kill, record.events)


def store_holds(entries: list[StoreEntry], kill: FileId, events: FileId) -> bool:
    """True only when the store is exactly this ``kill``/``events`` pair."""
    if len(entries) != 2:
        return False
    found = {
        (entry.name, (entry.dev, entry.inode), entry.flags & ACCMODE)
        for entry in entries
        if stat.S_ISREG(entry.mode) and not entry.flags & os.O_PATH
    }
    return found == {(KILL_NAME, kill, os.O_WRONLY), (EVENTS_NAME, events, os.O_RDONLY)}


# Linux implementation ------------------------------------------------------------------------


def pidfd_pid(pidfd: int) -> int:
    """The PID a pidfd refers to: -1 once it has exited, 0 outside this namespace."""
    with open(f"/proc/self/fdinfo/{pidfd}", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("Pid:"):
                return int(line.split()[1])
    raise Refused("descriptor is not a pidfd")


def receive(sock: socket.socket) -> tuple[bytes, int | None]:
    """One message and the kernel-attached sender pidfd; any other ancillary data is refused."""
    data, ancillary, flags, _ = sock.recvmsg(
        MAX_MESSAGE, socket.CMSG_SPACE(4) * 2, socket.MSG_CMSG_CLOEXEC
    )
    pidfd = None
    extra: list[int] = []
    for level, kind, payload in ancillary:
        fds = array.array("i")
        fds.frombytes(payload[: len(payload) - len(payload) % fds.itemsize])
        if level == socket.SOL_SOCKET and kind == SCM_PIDFD and len(fds) == 1 and pidfd is None:
            pidfd = fds[0]
        elif level == socket.SOL_SOCKET and kind in (SCM_PIDFD, socket.SCM_RIGHTS):
            extra.extend(fds)
    for fd in extra:
        os.close(fd)
    if extra or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
        if pidfd is not None:
            os.close(pidfd)
        raise Refused("unexpected ancillary data or truncated message")
    return data, pidfd


def inherited_fds(environ: dict[str, str]) -> dict[str, list[int]]:
    """Descriptors systemd passed from the store (``LISTEN_FDS``), keyed by name."""
    if environ.get("LISTEN_PID") != str(os.getpid()):
        return {}
    count = int(environ.get("LISTEN_FDS", "0"))
    names = environ.get("LISTEN_FDNAMES", "").split(":") if count else []
    if count < 0 or len(names) != count:
        raise ValueError("inconsistent LISTEN_FDS")
    result: dict[str, list[int]] = {}
    for offset, name in enumerate(names):
        fd = 3 + offset
        os.set_inheritable(fd, False)
        result.setdefault(name, []).append(fd)
    return result


class LinuxSystem:
    """Real host operations: ``busctl`` for PID 1, pinned cgroupfs handles, sd_notify."""

    def __init__(self, notify_socket: str) -> None:
        if not notify_socket:
            raise ValueError("NOTIFY_SOCKET is required (Type=notify)")
        self.notify_address = (
            "\0" + notify_socket[1:] if notify_socket.startswith("@") else notify_socket
        )

    def now(self) -> float:
        return time.monotonic()

    def boot_id(self) -> str:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as handle:
            return handle.read().strip()

    def pidfd_pid(self, pidfd: int) -> int:
        return pidfd_pid(pidfd)

    def _busctl(self, *args: str, pass_fds: tuple[int, ...] = ()) -> list[Any]:
        result = subprocess.run(
            ["/usr/bin/busctl", "--system", "--json=short", *args],
            capture_output=True,
            text=True,
            timeout=IO_SECONDS,
            pass_fds=pass_fds,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            check=False,
        )
        if result.returncode != 0:
            raise Refused(f"PID 1 query failed: {result.stderr.strip()[:200]}")
        return [json.loads(line)["data"] for line in result.stdout.splitlines() if line.strip()]

    def _manager(self, method: str, signature: str, *args: str, **kw: Any) -> Any:
        return self._busctl(
            "call", "org.freedesktop.systemd1", "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager", method, signature, *args, **kw,
        )[0]  # fmt: skip

    def unit_by_pidfd(self, pidfd: int) -> tuple[str, str]:
        _, unit, invocation = self._manager("GetUnitByPIDFD", "h", str(pidfd), pass_fds=(pidfd,))
        return unit, bytes(invocation).hex()

    def unit_state(self, unit: str) -> UnitState:
        (path,) = self._manager("LoadUnit", "s", unit)
        dest = ("get-property", "org.freedesktop.systemd1", path)
        active, invocation = self._busctl(
            *dest, "org.freedesktop.systemd1.Unit", "ActiveState", "InvocationID"
        )
        main_pid, group, group_id, delegate = self._busctl(
            *dest, "org.freedesktop.systemd1.Service",
            "MainPID", "ControlGroup", "ControlGroupId", "Delegate",
        )  # fmt: skip
        return UnitState(
            active, int(main_pid), bytes(invocation).hex(), group, int(group_id), bool(delegate)
        )

    def proc_cgroup(self, pid: int) -> str:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        if len(lines) != 1 or not lines[0].startswith("0::/"):
            raise Refused("process is not in a unified cgroup hierarchy")
        return lines[0][3:]

    def start_unit(self, unit: str) -> None:
        self._manager("StartUnit", "ss", unit, "fail")

    def open_job_cgroup(self, cgroup_id: int) -> Handles:
        flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory = os.open(CGROUP_ROOT, flags)
        try:
            for part in TEST_CGROUP.strip("/").split("/"):
                child = os.open(part, flags, dir_fd=directory)
                os.close(directory)
                directory = child
            if os.fstat(directory).st_ino != cgroup_id:
                raise Refused("cgroup directory is not the unit's cgroup")
            kill = os.open(
                "cgroup.kill", os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory
            )
            try:
                events = os.open(
                    "cgroup.events", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory
                )
            except BaseException:
                os.close(kill)
                raise
            handles = Handles(kill, events)
            device = os.fstat(directory).st_dev
            if os.fstat(directory).st_ino != cgroup_id or any(
                fid[0] != device for fid in handles.ids()
            ):
                handles.close()
                raise Refused("cgroup changed while opening its handles")
            return handles
        finally:
            os.close(directory)

    def job_cgroup_populated(self) -> bool | None:
        """Populated state of whatever cgroup is at the fixed unit's path; None if absent.

        Not pinned: used only at startup with no recorded job, to refuse idle while any cgroup
        at that path still holds processes. It can only make the broker stricter. Only the
        unit's own directory may be missing; anything else missing or unreadable raises.
        """
        flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        *parents, leaf = TEST_CGROUP.strip("/").split("/")
        directory = os.open(CGROUP_ROOT, flags)
        try:
            for part in parents:
                child = os.open(part, flags, dir_fd=directory)
                os.close(directory)
                directory = child
            try:
                child = os.open(leaf, flags, dir_fd=directory)
            except FileNotFoundError:
                return None  # systemd removes a stopped unit's cgroup once it is empty
            os.close(directory)
            directory = child
            events = os.open(
                "cgroup.events", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory
            )
            handles = Handles(-1, events)  # populated() reads only the events descriptor
            try:
                return self.populated(handles)
            finally:
                os.close(events)
        finally:
            os.close(directory)

    def populated(self, handles: Handles) -> bool | None:
        try:
            text = os.pread(handles.events, 4096, 0).decode("ascii")
        except OSError as exc:
            if exc.errno == errno.ENODEV:
                return None  # The cgroup was removed, which requires it to be empty.
            raise
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key == "populated" and value in ("0", "1"):
                return value == "1"
        raise OSError(errno.EIO, "cgroup.events has no populated line")

    def kill(self, handles: Handles) -> None:
        try:
            os.write(handles.kill, b"1")
        except OSError as exc:
            if exc.errno != errno.ENODEV:
                raise

    def wait_empty(self, handles: Handles, seconds: float) -> bool:
        poller = select.poll()
        poller.register(handles.events, select.POLLPRI | select.POLLERR)
        end = time.monotonic() + seconds
        while True:
            if not self.populated(handles):
                return True
            left = end - time.monotonic()
            if left <= 0:
                return False
            poller.poll(min(left, 0.2) * 1000)

    def _notify(self, message: str, fds: tuple[int, ...] = ()) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
            sock.connect(self.notify_address)
            ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
            sock.sendmsg([message.encode("ascii")], ancillary if fds else [])

    def notify_ready(self) -> None:
        self._notify("READY=1")

    def fdstore_add(self, name: str, fd: int) -> None:
        self._notify(f"FDSTORE=1\nFDNAME={name}\nFDPOLL=0", (fd,))

    def fdstore_remove(self, name: str) -> None:
        self._notify(f"FDSTOREREMOVE=1\nFDNAME={name}")

    def fdstore_barrier(self) -> None:
        read, write = os.pipe2(os.O_CLOEXEC)
        try:
            self._notify("BARRIER=1", (write,))
            os.close(write)
            write = -1
            poller = select.poll()
            poller.register(read, select.POLLHUP)
            if not poller.poll(IO_SECONDS * 1000):
                raise TimeoutError("notification barrier timed out")
        finally:
            os.close(read)
            if write >= 0:
                os.close(write)

    def fdstore_dump(self) -> list[StoreEntry]:
        rows = self._manager("DumpUnitFileDescriptorStore", "s", BROKER_UNIT)[0]
        return [
            StoreEntry(name, int(mode), os.makedev(int(major), int(minor)), int(inode), int(flags))
            for name, mode, major, minor, inode, _rmajor, _rminor, _path, flags in rows
        ]


# Socket server -------------------------------------------------------------------------------


class Server:
    """Single-threaded challenge/response loop over two fixed SEQPACKET listeners."""

    def __init__(
        self,
        broker: Broker,
        job: socket.socket,
        operator: socket.socket,
        operator_unit: Callable[[int], tuple[str, str | None]] | None = None,
    ) -> None:
        self.broker = broker
        self.listeners = {job.fileno(): ("job", job), operator.fileno(): ("operator", operator)}
        self.connections: dict[int, tuple[str, socket.socket, str]] = {}
        self.operator_unit = operator_unit or self._operator_unit

    def _operator_unit(self, pidfd: int) -> tuple[str, str | None]:
        unit, _ = self.broker.system.unit_by_pidfd(pidfd)
        match = OPERATOR_UNIT.fullmatch(unit)
        if match is None:
            raise Refused("sender is not a fixed operator unit")
        authenticate(self.broker.system, pidfd, unit, None)
        return match.group(1), match.group(2)

    def serve_once(self, timeout: float = 1.0) -> None:
        poller = select.poll()
        for fd in [*self.listeners, *self.connections]:
            poller.register(fd, select.POLLIN)
        handles = self.broker.handles
        if handles is not None and (
            self.broker.retry_at is None or self.broker.system.now() >= self.broker.retry_at
        ):
            poller.register(handles.events, select.POLLPRI | select.POLLERR)
        for fd, _ in poller.poll(timeout * 1000):
            if fd in self.listeners:
                self._accept(*self.listeners[fd])
            elif fd in self.connections:
                self._handle(fd)
        self.broker.tick()

    def _accept(self, kind: str, listener: socket.socket) -> None:
        conn, _ = listener.accept()
        conn.settimeout(IO_SECONDS)
        conn.setsockopt(socket.SOL_SOCKET, SO_PASSPIDFD, 1)
        nonce = secrets.token_hex(16)
        try:
            conn.send(f"hello {nonce}".encode("ascii"))
        except OSError:
            conn.close()
            return
        self.connections[conn.fileno()] = (kind, conn, nonce)

    def _close(self, fd: int) -> None:
        kind, conn, _ = self.connections.pop(fd)
        conn.close()
        if kind == "job-driver":
            self.broker.driver_closed()

    def _handle(self, fd: int) -> None:
        kind, conn, nonce = self.connections[fd]
        try:
            data, pidfd = receive(conn)
        except (OSError, Refused):
            self._close(fd)
            return
        if not data:
            if pidfd is not None:
                os.close(pidfd)
            self._close(fd)
            return
        try:
            words = data.decode("ascii").split(" ")
            if pidfd is None or len(words) < 2 or not hmac.compare_digest(words[1], nonce):
                raise Refused("unauthenticated message")
            if kind == "operator":
                self._operator(conn, pidfd, words)
            else:
                self._job(fd, conn, pidfd, words)
        except Exception as exc:  # noqa: BLE001 - refuse the request, keep serving
            reason = str(exc) if isinstance(exc, Refused) else type(exc).__name__
            self.broker.log(f"{kind} request refused: {reason}")
            self._reply(conn, {"refused": reason} if kind == "operator" else f"refused {reason}")
            if kind == "operator":
                self._close(fd)
        finally:
            if pidfd is not None:
                os.close(pidfd)

    @staticmethod
    def _reply(conn: socket.socket, message: str | dict[str, Any]) -> None:
        text = message if isinstance(message, str) else json.dumps(message, sort_keys=True)
        try:
            conn.send(text.encode("ascii")[: MAX_MESSAGE * 8])
        except OSError:
            pass

    def _operator(self, conn: socket.socket, pidfd: int, words: list[str]) -> None:
        if words[0] != "request" or len(words) != 2:
            raise Refused("malformed operator request")
        uid = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)[4:8]
        if int.from_bytes(uid, sys.byteorder) != 0:
            raise Refused("operator must be root")
        action, attempt = self.operator_unit(pidfd)
        result = self.broker.operator(action, attempt)
        self._reply(conn, result)
        self._close(conn.fileno())

    def _job(self, fd: int, conn: socket.socket, pidfd: int, words: list[str]) -> None:
        kind = self.connections[fd][0]
        if words[0] == "register" and len(words) == 2 and kind == "job":
            self.broker.register(pidfd, lambda: conn.send(b"go"))
            self.connections[fd] = ("job-driver", conn, self.connections[fd][2])
        elif words[0] == "phase" and len(words) == 3 and kind == "job-driver":
            self.broker.phase(pidfd, words[2], lambda: conn.send(f"ok {words[2]}".encode()))
        elif words[0] == "done" and len(words) == 3 and kind == "job-driver":
            if not words[2].isdigit():
                raise Refused("invalid status")
            self.broker.done(pidfd, int(words[2]))
            conn.send(b"ok done")
        else:
            raise Refused("malformed job request")


def _listen(path: str, mode: int, gid: int) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    old = os.umask(0o177)
    try:
        sock.bind(path)
    finally:
        os.umask(old)
    os.chown(path, 0, gid, follow_symlinks=False)
    os.chmod(path, mode, follow_symlinks=False)
    sock.listen(8)
    return sock


def serve() -> int:
    """Run as the fixed root broker unit (Type=notify, NotifyAccess=main)."""
    if os.geteuid() != 0:
        print("the broker must run as root from its fixed unit", file=sys.stderr)
        return 2
    notify = os.environ.pop("NOTIFY_SOCKET", "")
    inherited = inherited_fds(dict(os.environ))
    for key in ("LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"):
        os.environ.pop(key, None)
    system = LinuxSystem(notify)
    directory = os.open(STATE_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    state = AdmissionState(directory, expected_owner=0)
    broker = Broker(system, state, log=lambda text: print(text, file=sys.stderr, flush=True))
    broker.startup(inherited)
    gid = grp.getgrnam(JOB_GROUP).gr_gid
    job = _listen(JOB_SOCKET, 0o660, gid)
    operator = _listen(OPERATOR_SOCKET, 0o600, 0)
    system.notify_ready()
    server = Server(broker, job, operator)
    while True:
        server.serve_once()


def request() -> int:
    """Run as a fixed root operator unit; the broker derives the action from the unit name."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC) as sock:
        sock.settimeout(STOP_SECONDS + 4 * IO_SECONDS)
        sock.connect(OPERATOR_SOCKET)
        hello = sock.recv(MAX_MESSAGE).decode("ascii").split(" ")
        if len(hello) != 2 or hello[0] != "hello":
            print("unexpected broker greeting", file=sys.stderr)
            return 1
        sock.send(f"request {hello[1]}".encode("ascii"))
        answer = sock.recv(MAX_MESSAGE * 8).decode("ascii")
    print(answer)
    return 1 if '"refused"' in answer else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Whole-job broker candidate (fixed units and paths; not installed)."
    )
    parser.add_argument(
        "command",
        choices=("serve", "request"),
        help="serve: run as the fixed broker unit; request: run as a fixed operator unit",
    )
    args = parser.parse_args()
    return serve() if args.command == "serve" else request()


if __name__ == "__main__":
    sys.exit(main())
