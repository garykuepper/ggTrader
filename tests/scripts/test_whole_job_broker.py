"""Broker sequencing against an injected host fixture, plus the kernel pieces it can run safely.

The fixture stands in for PID 1, cgroupfs and the descriptor store: these tests prove the
broker's ordering and refusals, not systemd or kernel behaviour. The only real kernel paths
exercised are SCM_PIDFD over a socket pair, pidfd fdinfo and descriptor passing to a local
datagram socket; no unit, cgroup or host setting is touched.
"""

import array
import fcntl
import json
import os
import socket
import stat
from pathlib import Path

import pytest

from scripts.whole_job_admission import AdmissionState
from scripts.whole_job_broker import (
    EVENTS_NAME,
    KILL_NAME,
    PHASE_SECONDS,
    REGISTER_SECONDS,
    STOP_ATTEMPTS,
    STOP_EXHAUSTED_SECONDS,
    STOP_RETRY_SECONDS,
    TEST_CGROUP,
    TEST_UNIT,
    Broker,
    Handles,
    JobRecord,
    LinuxSystem,
    Refused,
    Server,
    StoreEntry,
    UnitState,
    inherited_fds,
    pidfd_pid,
    receive,
    store_matches,
)

BOOT = "0f0e0d0c-0b0a-4908-8706-050403020100"
OTHER_BOOT = "1f0e0d0c-0b0a-4908-8706-050403020100"
INVOCATION = "ab" * 16
DRIVER_PID = 4242
DRIVER_PIDFD = 900
CGROUP_ID = 777


class FakeSystem:
    """PID 1, cgroupfs and descriptor-store fixture; asserts no PID 1 call holds the lock."""

    def __init__(self, tmp: Path, directory: Path) -> None:
        self.tmp = tmp
        self.directory = directory
        self.clock = 1000.0
        self.boot = BOOT
        self.pidfds = {DRIVER_PIDFD: DRIVER_PID}
        self.units = {
            TEST_UNIT: UnitState("inactive", 0, "", TEST_CGROUP, CGROUP_ID, False),
        }
        self.pid_units = {DRIVER_PID: TEST_UNIT}
        self.cgroups = {DRIVER_PID: TEST_CGROUP}
        self.started: list[str] = []
        self.store: list[tuple[str, int]] = []
        self.dump_override: list[StoreEntry] | None = None
        self.populated_value: bool | None = True
        self.path_populated: bool | None = None  # the unit's cgroup looked up by path
        self.empties_on_kill = True
        self.kills: list[dict] = []
        self.opened = 0

    # Guard: forking (busctl) is forbidden while the admission lock is held.
    def _unlocked(self) -> None:
        fd = os.open(self.directory / "admission.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AssertionError("PID 1 query while the admission lock is held") from None
        finally:
            os.close(fd)

    def now(self) -> float:
        return self.clock

    def boot_id(self) -> str:
        return self.boot

    def pidfd_pid(self, pidfd: int) -> int:
        return self.pidfds.get(pidfd, -1)

    def unit_by_pidfd(self, pidfd: int) -> tuple[str, str]:
        self._unlocked()
        pid = self.pidfd_pid(pidfd)
        if pid <= 0:
            raise Refused("ESRCH")
        unit = self.pid_units[pid]
        return unit, self.units[unit].invocation

    def unit_state(self, unit: str) -> UnitState:
        self._unlocked()
        return self.units[unit]

    def proc_cgroup(self, pid: int) -> str:
        return self.cgroups[pid]

    def start_unit(self, unit: str) -> None:
        self._unlocked()
        self.started.append(unit)
        self.units[unit] = UnitState(
            "active", DRIVER_PID, INVOCATION, TEST_CGROUP, CGROUP_ID, False
        )
        self.populated_value = True
        self.path_populated = True

    def open_job_cgroup(self, cgroup_id: int) -> Handles:
        assert cgroup_id == CGROUP_ID
        self.opened += 1
        kill = self.tmp / f"kill-{self.opened}"
        events = self.tmp / f"events-{self.opened}"
        kill.write_text("")
        events.write_text("populated 1\n")
        return Handles(os.open(kill, os.O_WRONLY), os.open(events, os.O_RDONLY))

    def populated(self, handles: Handles) -> bool | None:
        return self.populated_value

    def kill(self, handles: Handles) -> None:
        self.kills.append(json.loads((self.directory / "admission.json").read_text()))
        if self.empties_on_kill:
            self.populated_value = None
            self.path_populated = None
            self.units[TEST_UNIT] = UnitState("failed", 0, "", TEST_CGROUP, CGROUP_ID, False)

    def wait_empty(self, handles: Handles, seconds: float) -> bool:
        return not self.populated_value

    def job_cgroup_populated(self) -> bool | None:
        return self.path_populated

    def fdstore_add(self, name: str, fd: int) -> None:
        self.store.append((name, os.dup(fd)))

    def fdstore_remove(self, name: str) -> None:
        for entry in [e for e in self.store if e[0] == name]:
            os.close(entry[1])
            self.store.remove(entry)

    def fdstore_barrier(self) -> None:
        self._unlocked()

    def fdstore_dump(self) -> list[StoreEntry]:
        self._unlocked()
        if self.dump_override is not None:
            return self.dump_override
        return [entry(name, fd) for name, fd in self.store]

    def inherited(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for name, fd in self.store:
            result.setdefault(name, []).append(os.dup(fd))
        return result


def entry(name: str, fd: int) -> StoreEntry:
    info = os.fstat(fd)
    return StoreEntry(name, info.st_mode, info.st_dev, info.st_ino, fcntl.fcntl(fd, fcntl.F_GETFL))


@pytest.fixture
def env(tmp_path: Path):
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    system = FakeSystem(tmp_path, directory)
    logs: list[str] = []
    try:
        yield system, fd, directory, logs
    finally:
        os.close(fd)


def make(env) -> Broker:
    system, fd, _, logs = env
    broker = Broker(system, AdmissionState(fd, expected_owner=os.geteuid()), log=logs.append)
    return broker


def admitted(env) -> tuple[Broker, list[str]]:
    broker = make(env)
    broker.startup({})
    broker.start()
    sent: list[str] = []
    broker.register(DRIVER_PIDFD, lambda: sent.append("go"))
    return broker, sent


def persisted(directory: Path) -> dict:
    return json.loads((directory / "admission.json").read_text())


def test_start_register_binds_attempt_invocation_handles_and_exact_store(env) -> None:
    system, _, directory, _ = env
    broker, sent = admitted(env)
    assert sent == ["go"] and broker.mode == "running"
    record = JobRecord.parse(persisted(directory)["job"])
    assert record.invocation == INVOCATION and record.cgroup_id == CGROUP_ID
    assert persisted(directory)["boot"] == BOOT and not persisted(directory)["held"]
    assert (record.kill, record.events) == broker.handles.ids()
    assert sorted(name for name, _ in system.store) == [EVENTS_NAME, KILL_NAME]
    assert system.started == [TEST_UNIT]


def test_record_round_trips_and_rejects_garbage() -> None:
    record = JobRecord("a" * 32, "b" * 32, 5, (1, 2), (3, 4))
    assert JobRecord.parse(record.encode()) == record
    for bad in ["", record.encode() + ";x", record.encode().replace("v1", "v2"), "v1;" + "x" * 99]:
        with pytest.raises(ValueError):
            JobRecord.parse(bad)


def test_nothing_runs_without_an_operator_start(env) -> None:
    system, _, directory, _ = env
    broker = make(env)
    broker.startup({})
    system.start_unit(TEST_UNIT)  # e.g. a root `systemctl start` that bypassed the broker
    with pytest.raises(Refused):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert persisted(directory)["held"] and not persisted(directory)["job"]


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong-unit",
        "not-main-pid",
        "invocation-changed",
        "wrong-cgroup",
        "proc-cgroup",
        "delegated",
        "exited",
    ],
)
def test_registration_refuses_unverified_sender(env, tamper: str) -> None:
    system, _, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    state = system.units[TEST_UNIT]
    if tamper == "wrong-unit":
        system.pid_units[DRIVER_PID] = "other.service"
        system.units["other.service"] = state
    elif tamper == "not-main-pid":
        system.units[TEST_UNIT] = UnitState("active", 1, INVOCATION, TEST_CGROUP, CGROUP_ID, False)
    elif tamper == "invocation-changed":
        system.units[TEST_UNIT] = UnitState("active", DRIVER_PID, "cd" * 16, TEST_CGROUP, 1, False)
        system.unit_by_pidfd = lambda pidfd: (TEST_UNIT, INVOCATION)
    elif tamper == "wrong-cgroup":
        system.units[TEST_UNIT] = UnitState(
            "active", DRIVER_PID, INVOCATION, "/x", CGROUP_ID, False
        )
    elif tamper == "proc-cgroup":
        system.cgroups[DRIVER_PID] = TEST_CGROUP + "/escaped"
    elif tamper == "delegated":
        system.units[TEST_UNIT] = UnitState(
            "active", DRIVER_PID, INVOCATION, TEST_CGROUP, CGROUP_ID, True
        )
    else:
        system.pidfds[DRIVER_PIDFD] = -1
    with pytest.raises(Refused):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert persisted(directory)["held"] and not persisted(directory)["job"]
    assert system.opened == 0 and system.store == []


@pytest.mark.parametrize("tamper", ["missing", "extra", "duplicate", "inode", "flags", "empty"])
def test_store_must_hold_exactly_the_pinned_pair_before_go(env, tamper: str) -> None:
    system, _, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    real_dump = system.fdstore_dump

    def dump() -> list[StoreEntry]:
        rows = real_dump()
        if not rows or tamper == "empty":
            return [] if tamper == "empty" else rows
        kill = next(r for r in rows if r.name == KILL_NAME)
        if tamper == "missing":
            return [r for r in rows if r is not kill]
        if tamper == "extra":
            return [*rows, StoreEntry("other", kill.mode, kill.dev, kill.inode + 1, kill.flags)]
        if tamper == "duplicate":
            return [*rows, kill]
        if tamper == "inode":
            return [r if r is not kill else StoreEntry(KILL_NAME, r.mode, r.dev, 1, r.flags)
                    for r in rows]  # fmt: skip
        return [r if r is not kill else StoreEntry(KILL_NAME, r.mode, r.dev, r.inode, os.O_RDWR)
                for r in rows]  # fmt: skip

    system.fdstore_dump = dump
    with pytest.raises(Refused, match="exactly"):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert persisted(directory)["held"] and not persisted(directory)["job"]
    assert len(system.kills) == 1 and system.store == []  # unadmitted invocation stopped
    assert broker.mode == "idle"


def test_store_comparison_needs_regular_files_and_access_modes(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_text("")
    kill, events = os.open(path, os.O_WRONLY), os.open(path, os.O_RDONLY)
    try:
        info = os.fstat(kill)
        record = JobRecord(
            "a" * 32, "b" * 32, 1, (info.st_dev, info.st_ino), (info.st_dev, info.st_ino)
        )
        rows = [entry(KILL_NAME, kill), entry(EVENTS_NAME, events)]
        assert store_matches(rows, record)
        assert not store_matches(rows[::-1][:1], record)
        assert not store_matches([entry(KILL_NAME, events), entry(EVENTS_NAME, kill)], record)
        fifo = [StoreEntry(r.name, stat.S_IFIFO | 0o600, r.dev, r.inode, r.flags) for r in rows]
        assert not store_matches(fifo, record)
    finally:
        os.close(kill)
        os.close(events)


def test_phases_are_admitted_once_in_order_and_only_from_the_driver(env) -> None:
    system, _, _, _ = env
    broker, _ = admitted(env)
    sent: list[str] = []
    with pytest.raises(Refused):
        broker.phase(DRIVER_PIDFD, "studies", lambda: sent.append("studies"))
    broker.phase(DRIVER_PIDFD, "core", lambda: sent.append("core"))
    with pytest.raises(Refused):
        broker.phase(DRIVER_PIDFD, "core", lambda: sent.append("again"))
    system.pidfds[901] = 5151  # a descendant sending on an inherited socket
    system.pid_units[5151] = TEST_UNIT
    system.cgroups[5151] = TEST_CGROUP
    with pytest.raises(Refused):
        broker.phase(901, "studies", lambda: sent.append("child"))
    assert sent == ["core"] and broker.phases == ["core"]


def test_stale_stop_never_holds_a_replacement(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    with pytest.raises(Refused):
        broker.stop("0" * 32)
    assert not persisted(directory)["held"] and system.kills == []


def test_stop_persists_hold_before_kill_and_clears_only_after_empty(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    attempt = broker.record.attempt
    job = persisted(directory)["job"]
    broker.stop(attempt)
    assert system.kills == [{**system.kills[0], "held": True, "job": job}]
    assert persisted(directory)["held"] and persisted(directory)["job"] == ""
    assert system.store == [] and broker.mode == "idle" and broker.handles is None
    broker.start()  # an explicit operator start is the only release


def test_stop_that_does_not_empty_stays_held_and_blocks_start(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    system.empties_on_kill = False
    with pytest.raises(TimeoutError):
        broker.stop(broker.record.attempt)
    assert persisted(directory)["held"] and persisted(directory)["job"]
    assert broker.mode == "held"
    with pytest.raises(Refused):
        broker.start()
    with pytest.raises(Refused):
        broker.phase(DRIVER_PIDFD, "core", lambda: pytest.fail("phase admitted"))
    system.empties_on_kill = True
    broker.stop(broker.record.attempt)
    assert broker.mode == "idle" and not persisted(directory)["job"]


def test_broker_restart_restores_exact_handles_and_holds(env) -> None:
    system, fd, directory, logs = env
    broker, _ = admitted(env)
    broker.phase(DRIVER_PIDFD, "core", lambda: None)
    attempt = broker.record.attempt
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and persisted(directory)["held"]
    with pytest.raises(Refused):
        restarted.phase(DRIVER_PIDFD, "studies", lambda: pytest.fail("phase admitted"))
    with pytest.raises(PermissionError):
        broker.phase(DRIVER_PIDFD, "studies", lambda: pytest.fail("stale instance admitted"))
    restarted.stop(attempt)
    assert restarted.mode == "idle" and not persisted(directory)["job"]


@pytest.mark.parametrize("tamper", ["missing", "extra-name", "duplicate", "other-file"])
def test_restart_with_inexact_store_stays_unverifiable(env, tamper: str) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    inherited = system.inherited()
    if tamper == "missing":
        os.close(inherited.pop(KILL_NAME)[0])
    elif tamper == "extra-name":
        inherited["stray"] = [os.dup(inherited[KILL_NAME][0])]
    elif tamper == "duplicate":
        inherited[KILL_NAME].append(os.dup(inherited[KILL_NAME][0]))
    else:
        other = system.tmp / "other"
        other.write_text("")
        os.close(inherited[KILL_NAME][0])
        inherited[KILL_NAME] = [os.open(other, os.O_WRONLY)]
    restarted = make(env)
    restarted.startup(inherited)
    assert restarted.mode == "unverifiable" and restarted.handles is None
    for call in (restarted.start, lambda: restarted.stop(broker.record.attempt)):
        with pytest.raises(Refused):
            call()
    assert persisted(directory)["held"] and persisted(directory)["job"]


def test_unreported_termination_is_cleared_without_claiming_oom(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    system.populated_value = False
    broker.tick()
    assert broker.mode == "idle" and not persisted(directory)["job"]
    assert "without a driver report" in broker.outcome and "oom" not in broker.outcome.lower()


def test_reported_completion_then_close_clears(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.done(DRIVER_PIDFD, 0)
    broker.driver_closed()
    assert broker.outcome == "job ended; driver reported status 0"
    assert persisted(directory)["held"] and not persisted(directory)["job"]


def test_phase_deadline_stops_the_job(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.phase(DRIVER_PIDFD, "core", lambda: None)
    system.clock += PHASE_SECONDS + 1
    broker.tick()
    assert len(system.kills) == 1 and broker.outcome == "phase deadline exceeded"
    assert persisted(directory)["held"] and not persisted(directory)["job"]


def test_unregistered_start_is_abandoned(env) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    system.clock += REGISTER_SECONDS + 1
    broker.tick()
    assert broker.mode == "idle"
    with pytest.raises(Refused):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))


def test_prior_boot_record_needs_acknowledgement_of_that_attempt(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    attempt = broker.record.attempt
    for _, fd in list(system.store):
        os.close(fd)
    system.store = []
    system.boot = OTHER_BOOT
    system.units[TEST_UNIT] = UnitState("inactive", 0, "", TEST_CGROUP, CGROUP_ID, False)
    rebooted = make(env)
    rebooted.startup({})
    assert rebooted.mode == "prior-boot"
    with pytest.raises(Refused):
        rebooted.start()
    with pytest.raises(Refused):
        rebooted.acknowledge("0" * 32)
    rebooted.acknowledge(attempt)
    assert rebooted.mode == "idle" and not persisted(directory)["job"]
    rebooted.start()


def test_prior_boot_record_with_stored_handles_is_unverifiable(env) -> None:
    system, _, _, _ = env
    admitted(env)
    system.boot = OTHER_BOOT
    rebooted = make(env)
    rebooted.startup(system.inherited())
    assert rebooted.mode == "unverifiable"


def test_stray_store_entry_without_a_record_stays_unresolved(env) -> None:
    system, _, _, logs = env
    path = system.tmp / "orphan"
    path.write_text("")
    system.store.append((KILL_NAME, os.open(path, os.O_WRONLY)))
    broker = make(env)
    broker.startup(system.inherited())
    assert broker.mode == "unverifiable" and len(system.store) == 1  # kept, not drained
    assert system.kills == []
    with pytest.raises(Refused):
        broker.start()


def test_start_refuses_active_unit_or_nonempty_store(env) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    system.units[TEST_UNIT] = UnitState("active", 1, INVOCATION, TEST_CGROUP, CGROUP_ID, False)
    with pytest.raises(Refused):
        broker.start()
    system.units[TEST_UNIT] = UnitState("failed", 0, "", TEST_CGROUP, CGROUP_ID, False)
    path = system.tmp / "stray"
    path.write_text("")
    system.store.append(("stray", os.open(path, os.O_RDONLY)))
    with pytest.raises(Refused):
        broker.start()


def test_operator_actions_are_fixed(env) -> None:
    broker = make(env)
    broker.startup({})
    assert broker.operator("status", None)["mode"] == "idle"
    for action, attempt in [("stop", None), ("start", "a" * 32), ("kill", None)]:
        with pytest.raises(Refused):
            broker.operator(action, attempt)


# Real kernel pieces (no unit, cgroup or host setting is touched) ---------------------------


def test_scm_pidfd_identifies_the_sender_per_message() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        right.setsockopt(socket.SOL_SOCKET, 76, 1)
        left.send(b"register x")
        data, pidfd = receive(right)
        try:
            assert data == b"register x" and pidfd_pid(pidfd) == os.getpid()
        finally:
            os.close(pidfd)


def test_receive_refuses_passed_descriptors() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        right.setsockopt(socket.SOL_SOCKET, 76, 1)
        read, write = os.pipe()
        try:
            left.sendmsg([b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [read]))])
        finally:
            os.close(read)
            os.close(write)
        with pytest.raises(Refused):
            receive(right)


def test_inherited_descriptors_require_our_pid(monkeypatch) -> None:
    assert inherited_fds({"LISTEN_PID": "1", "LISTEN_FDS": "2", "LISTEN_FDNAMES": "a:b"}) == {}
    with pytest.raises(ValueError):
        inherited_fds({"LISTEN_PID": str(os.getpid()), "LISTEN_FDS": "2", "LISTEN_FDNAMES": "a"})


def test_notify_passes_descriptor_with_fdpoll_off(tmp_path: Path) -> None:
    path = str(tmp_path / "notify")
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        server.bind(path)
        target = tmp_path / "f"
        target.write_text("")
        fd = os.open(target, os.O_RDONLY)
        try:
            LinuxSystem(path).fdstore_add(EVENTS_NAME, fd)
        finally:
            os.close(fd)
        data, ancillary, _, _ = server.recvmsg(256, socket.CMSG_SPACE(4))
        assert data == b"FDSTORE=1\nFDNAME=job-events\nFDPOLL=0"
        ((level, kind, payload),) = ancillary
        received = array.array("i", payload)[0]
        assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS)
        assert os.fstat(received).st_ino == target.stat().st_ino
        os.close(received)


def test_server_protocol_challenge_register_and_phase(env, tmp_path: Path) -> None:
    system, _, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    operator = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.bind(str(tmp_path / "job.sock"))
    operator.bind(str(tmp_path / "operator.sock"))
    listener.listen(1)
    operator.listen(1)
    # The fixture maps the real kernel-attached pidfd of this process to the driver.
    system.pidfd_pid = lambda pidfd: DRIVER_PID if pidfd_pid(pidfd) == os.getpid() else -1
    server = Server(broker, listener, operator)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with listener, operator:
        client.connect(str(tmp_path / "job.sock"))
        server.serve_once(1)
        nonce = client.recv(512).decode().split(" ")[1]
        client.send(b"register wrong-nonce")
        server.serve_once(1)
        assert client.recv(512).startswith(b"refused")
        client.send(f"register {nonce}".encode())
        server.serve_once(1)
        assert client.recv(512) == b"go"
        client.send(f"phase {nonce} core".encode())
        server.serve_once(1)
        assert client.recv(512) == b"ok core"
        client.send(f"done {nonce} 0".encode())
        server.serve_once(1)
        assert client.recv(512) == b"ok done"
        client.close()
        server.serve_once(1)  # connection closed: the job is stopped and cleared
    assert broker.mode == "idle" and not persisted(directory)["job"]


def test_failed_record_write_stops_the_unadmitted_invocation(env, monkeypatch) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()

    def fail(job: str, boot: str) -> None:
        raise OSError("disk")

    monkeypatch.setattr(broker.state, "release", fail)
    with pytest.raises(OSError):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert len(system.kills) == 1 and system.store == [] and broker.mode == "idle"


def test_go_send_failure_holds_and_stops(env) -> None:
    system, _, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()

    def broken() -> None:
        raise BrokenPipeError

    with pytest.raises(BrokenPipeError):
        broker.register(DRIVER_PIDFD, broken)
    assert len(system.kills) == 1 and system.kills[0]["held"]
    assert persisted(directory)["held"] and not persisted(directory)["job"]


# A registration that fails before admission must not forget a job it could not stop.


def _refused_registration(env, monkeypatch) -> Broker:
    """Start, then fail the record write after the pinned pair is stored exactly."""
    broker = make(env)
    broker.startup({})
    broker.start()

    def fail(job: str, boot: str) -> None:
        raise OSError("disk")

    monkeypatch.setattr(broker.state, "release", fail)
    with pytest.raises(OSError):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    return broker


def _assert_held_with_handles(broker: Broker, system: FakeSystem, directory: Path) -> str:
    assert broker.mode == "held"
    assert broker.handles is not None and broker.record is not None
    assert broker.handles.ids() == (broker.record.kill, broker.record.events)  # still open
    state = persisted(directory)
    assert state["held"] and JobRecord.parse(state["job"]) == broker.record
    assert "invocation stopped" not in broker.outcome and "held until" in broker.outcome
    assert broker.status()["attempt"] == broker.record.attempt
    with pytest.raises(Refused):
        broker.start()
    return broker.record.attempt


@pytest.mark.parametrize("failure", ["kill-raises", "never-empties"])
def test_unadmitted_invocation_that_does_not_stop_stays_held(env, monkeypatch, failure) -> None:
    system, _, directory, _ = env
    if failure == "kill-raises":

        def broken_kill(handles: Handles) -> None:
            raise OSError("EBUSY")

        monkeypatch.setattr(system, "kill", broken_kill)
    else:
        system.empties_on_kill = False
    broker = _refused_registration(env, monkeypatch)
    attempt = _assert_held_with_handles(broker, system, directory)
    assert len(system.store) == 2  # the pinned pair is kept for a broker restart
    with pytest.raises(Refused):
        broker.phase(DRIVER_PIDFD, "core", lambda: pytest.fail("phase admitted"))
    monkeypatch.undo()
    system.empties_on_kill = True
    broker.stop(attempt)  # the operator stop retries through the same pinned handles
    assert broker.mode == "idle" and broker.handles is None
    assert persisted(directory)["held"] and not persisted(directory)["job"]
    assert system.store == []


def test_held_unadmitted_invocation_clears_once_its_cgroup_empties(env, monkeypatch) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    broker = _refused_registration(env, monkeypatch)
    _assert_held_with_handles(broker, system, directory)
    system.empties_on_kill = True
    system.populated_value = False
    broker.tick()  # the recorded stop intent waits out its retry pause
    assert broker.mode == "held"
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()
    assert broker.mode == "idle" and not persisted(directory)["job"] and system.store == []


def test_held_unadmitted_invocation_survives_a_broker_restart(env, monkeypatch) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    broker = _refused_registration(env, monkeypatch)
    attempt = _assert_held_with_handles(broker, system, directory)
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.record.attempt == attempt
    assert len(system.store) == 2  # not drained as orphans
    system.empties_on_kill = True
    restarted.stop(attempt)
    assert restarted.mode == "idle" and system.store == []


def test_unverified_store_drain_keeps_the_identity_held(env, monkeypatch) -> None:
    system, _, directory, _ = env
    real_remove = system.fdstore_remove
    monkeypatch.setattr(system, "fdstore_remove", lambda name: None)  # removal silently ignored
    broker = _refused_registration(env, monkeypatch)
    assert len(system.kills) == 1  # the cgroup was killed and emptied
    _assert_held_with_handles(broker, system, directory)
    assert len(system.store) == 2
    monkeypatch.setattr(system, "fdstore_remove", real_remove)
    broker.stop(broker.record.attempt)
    assert broker.mode == "idle" and system.store == [] and not persisted(directory)["job"]


def test_unadmitted_failure_without_a_recordable_identity_is_unverifiable(env, monkeypatch) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    broker = make(env)
    broker.startup({})
    broker.start()

    def no_ids(self: Handles) -> tuple:
        raise Refused("pinned handle has the wrong type or access mode")

    monkeypatch.setattr(Handles, "ids", no_ids)
    with pytest.raises(Refused):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert broker.mode == "unverifiable" and broker.handles is not None
    for call in (broker.start, lambda: broker.stop("0" * 32)):
        with pytest.raises(Refused):
            call()


# A stop forgets the job only after the cgroup is empty AND the store removal is verified.


def test_stop_keeps_record_and_handles_until_store_removal_is_verified(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    attempt = broker.record.attempt
    real_remove = system.fdstore_remove
    monkeypatch.setattr(system, "fdstore_remove", lambda name: None)  # removal silently ignored
    with pytest.raises(RuntimeError):
        broker.stop(attempt)
    assert broker.mode == "held" and broker.record is not None and broker.handles is not None
    assert broker.handles.ids() == (broker.record.kill, broker.record.events)  # still open
    assert persisted(directory)["held"] and persisted(directory)["job"] == broker.record.encode()
    assert len(system.store) == 2 and "stop incomplete" in broker.outcome
    with pytest.raises(Refused):
        broker.start()
    monkeypatch.setattr(system, "fdstore_remove", real_remove)
    broker.stop(attempt)
    assert broker.mode == "idle" and broker.handles is None and system.store == []
    assert persisted(directory)["held"] and not persisted(directory)["job"]


def test_retained_job_with_persistent_store_failure_stays_held_on_tick_and_stop(
    env, monkeypatch
) -> None:
    system, _, directory, logs = env
    real_remove = system.fdstore_remove
    monkeypatch.setattr(system, "fdstore_remove", lambda name: None)
    broker = _refused_registration(env, monkeypatch)
    attempt = _assert_held_with_handles(broker, system, directory)
    assert system.populated_value is None  # the cgroup itself was killed and is gone
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()  # retries the recorded stop after its pause: the store still will not drain
    assert broker.mode == "held" and broker.record.attempt == attempt
    assert broker.handles is not None and broker.handles.ids() == (
        broker.record.kill,
        broker.record.events,
    )
    assert persisted(directory)["job"] == broker.record.encode() and len(system.store) == 2
    assert any("stop failed" in line for line in logs)
    with pytest.raises(RuntimeError):
        broker.stop(attempt)  # an operator stop does not paper over it either
    assert broker.mode == "held" and broker.handles is not None
    assert persisted(directory)["held"] and persisted(directory)["job"]
    with pytest.raises(Refused):
        broker.start()
    monkeypatch.setattr(system, "fdstore_remove", real_remove)
    broker.stop(attempt)
    assert broker.mode == "idle" and system.store == [] and not persisted(directory)["job"]


def test_tick_retries_a_failed_stop_only_after_a_pause(env, monkeypatch) -> None:
    system, _, _, logs = env
    monkeypatch.setattr(system, "fdstore_remove", lambda name: None)
    broker = _refused_registration(env, monkeypatch)
    broker.tick()
    failures = sum("stop failed" in line for line in logs)
    broker.tick()
    assert sum("stop failed" in line for line in logs) == failures  # no retry every second
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()
    assert sum("stop failed" in line for line in logs) == failures + 1


def test_events_read_failure_defers_retry(env, monkeypatch) -> None:
    system, _, _, logs = env
    broker = make(env)
    broker.startup({})
    broker.start()
    broker.register(DRIVER_PIDFD, lambda: None)

    def fail(_handles):
        raise OSError("invalid events")

    monkeypatch.setattr(system, "populated", fail)
    broker.tick()
    assert broker.retry_at == system.clock + STOP_RETRY_SECONDS
    broker.tick()
    assert sum("cannot read pinned cgroup events" in line for line in logs) == 1


def test_phase_deadline_still_stops_during_events_read_backoff(env, monkeypatch) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    broker.register(DRIVER_PIDFD, lambda: None)
    broker.deadline = system.clock + 1

    def fail(_handles):
        raise OSError("invalid events")

    monkeypatch.setattr(system, "populated", fail)
    broker.tick()
    system.clock += 2
    broker.tick()
    assert len(system.kills) == 1


def test_failed_kill_is_retried_when_job_remains_populated(env, monkeypatch) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    broker.register(DRIVER_PIDFD, lambda: None)
    kill = system.kill
    calls = []

    def fail_once(handles):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary kill failure")
        kill(handles)

    monkeypatch.setattr(system, "kill", fail_once)
    broker._stop_quietly("deadline")
    assert broker.mode == "held" and system.populated_value
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()
    assert len(calls) == 2
    assert broker.mode == "idle"


def test_held_job_is_not_killed_after_events_read_error(env, monkeypatch) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    broker.register(DRIVER_PIDFD, lambda: None)
    broker.mode = "held"
    populated = system.populated

    def fail(_handles):
        raise OSError("temporary read error")

    monkeypatch.setattr(system, "populated", fail)
    broker.tick()
    monkeypatch.setattr(system, "populated", populated)
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()
    assert not system.kills
    assert broker.mode == "held"


def test_server_ignores_ready_cgroup_events_during_stop_cooldown(env, monkeypatch) -> None:
    system, _, _, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    broker.register(DRIVER_PIDFD, lambda: None)
    broker.retry_at = system.clock + STOP_RETRY_SECONDS
    registrations = []

    class Poll:
        def register(self, fd, mask):
            registrations.append(fd)

        def poll(self, timeout):
            return []

    monkeypatch.setattr("scripts.whole_job_broker.select.poll", Poll)
    with socket.socket() as job, socket.socket() as operator:
        server = Server(broker, job, operator)
        server.serve_once()
        assert registrations == [job.fileno(), operator.fileno()]
        registrations.clear()
        system.clock = broker.retry_at
        server.serve_once()
        assert registrations == [job.fileno(), operator.fileno(), broker.handles.events]
        registrations.clear()
        broker.mode = "unverifiable"
        server.serve_once()
        assert registrations == [job.fileno(), operator.fileno()]


def test_record_clear_failure_after_drain_keeps_handles_and_restart_is_unverifiable(
    env, monkeypatch
) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)

    def fail(job: str, boot: str, epoch: str) -> None:
        raise OSError("disk")

    monkeypatch.setattr(broker.state, "clear", fail)
    with pytest.raises(OSError):
        broker.stop(broker.record.attempt)
    assert broker.mode == "held" and broker.handles is not None and system.store == []
    assert persisted(directory)["held"] and persisted(directory)["job"]
    restarted = make(env)  # a crash here: the record survives, the store is empty
    restarted.startup(system.inherited())
    assert restarted.mode == "unverifiable" and persisted(directory)["job"]


# Restart without a recorded job: idle only once nothing of an unrecorded invocation remains.


class Crash(BaseException):
    """The broker process dies at this point: no exception handler of its own runs."""


def _crash_after_store_push(env, monkeypatch) -> Broker:
    broker = make(env)
    broker.startup({})
    broker.start()

    def crash(job: str, boot: str) -> None:
        raise Crash

    monkeypatch.setattr(broker.state, "release", crash)
    monkeypatch.setattr(broker, "_abort_unadmitted", lambda handles, record: None)
    with pytest.raises(Crash):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    return broker


def test_crash_between_store_push_and_record_kills_and_drains_on_restart(env, monkeypatch) -> None:
    system, _, directory, _ = env
    _crash_after_store_push(env, monkeypatch)
    assert len(system.store) == 2 and not persisted(directory)["job"]
    assert system.kills == []
    restarted = make(env)
    restarted.startup(system.inherited())
    assert len(system.kills) == 1  # killed through the stored pinned handle
    assert restarted.mode == "idle" and system.store == []
    restarted.start()


def test_crash_between_store_push_and_record_with_live_cgroup_stays_unresolved(
    env, monkeypatch
) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    _crash_after_store_push(env, monkeypatch)
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "unverifiable" and len(system.store) == 2  # store kept
    assert persisted(directory)["held"] and not persisted(directory)["job"]
    with pytest.raises(Refused):
        restarted.start()
    again = make(env)  # the condition survives a further restart
    again.startup(system.inherited())
    assert again.mode == "unverifiable" and len(system.store) == 2
    system.empties_on_kill = True
    resolved = make(env)
    resolved.startup(system.inherited())
    assert resolved.mode == "idle" and system.store == []


def test_failed_retain_then_restart_stays_unresolved_until_the_cgroup_empties(
    env, monkeypatch
) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False

    def no_retain(job: str, boot: str) -> None:
        raise OSError("disk")

    broker = make(env)
    broker.startup({})
    broker.start()
    monkeypatch.setattr(broker.state, "release", no_retain)
    monkeypatch.setattr(broker.state, "abort_release", no_retain)
    with pytest.raises(OSError):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert broker.mode == "unverifiable" and not persisted(directory)["job"]
    assert len(system.store) == 2
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "unverifiable" and len(system.store) == 2
    with pytest.raises(Refused):
        restarted.start()
    system.empties_on_kill = True
    resolved = make(env)
    resolved.startup(system.inherited())
    assert resolved.mode == "idle" and system.store == []


def test_identity_failure_then_restart_with_a_live_unrecorded_job_is_unverifiable(
    env, monkeypatch
) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    broker = make(env)
    broker.startup({})
    broker.start()
    real_ids = Handles.ids

    def no_ids(self: Handles) -> tuple:
        raise Refused("pinned handle has the wrong type or access mode")

    monkeypatch.setattr(Handles, "ids", no_ids)
    with pytest.raises(Refused):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    monkeypatch.setattr(Handles, "ids", real_ids)
    assert broker.mode == "unverifiable" and system.store == []
    assert not persisted(directory)["job"]
    restarted = make(env)  # nothing recorded, nothing stored: only the live cgroup remains
    restarted.startup(system.inherited())
    assert restarted.mode == "unverifiable"
    with pytest.raises(Refused):
        restarted.start()
    # The job ends on its own; a later restart sees no unit cgroup and an inactive unit.
    system.path_populated = None
    system.units[TEST_UNIT] = UnitState("failed", 0, "", TEST_CGROUP, CGROUP_ID, False)
    resolved = make(env)
    resolved.startup(system.inherited())
    assert resolved.mode == "idle"


@pytest.mark.parametrize("active", ["active", "activating", "deactivating"])
def test_restart_with_active_unit_and_no_record_is_unverifiable(env, active: str) -> None:
    system, _, _, _ = env
    system.units[TEST_UNIT] = UnitState(active, DRIVER_PID, INVOCATION, TEST_CGROUP, 1, False)
    broker = make(env)
    broker.startup({})
    assert broker.mode == "unverifiable"


def test_restart_with_populated_unit_cgroup_and_no_record_is_unverifiable(env) -> None:
    system, _, _, _ = env
    system.path_populated = True  # e.g. leftovers after a failed final SIGKILL
    system.units[TEST_UNIT] = UnitState("failed", 0, "", TEST_CGROUP, CGROUP_ID, False)
    broker = make(env)
    broker.startup({})
    assert broker.mode == "unverifiable"


@pytest.mark.parametrize("content", [None, "populated 0\nfrozen 0\n", "populated 1\n", "junk\n"])
def test_job_cgroup_populated_reads_the_fixed_path(tmp_path: Path, monkeypatch, content) -> None:
    import scripts.whole_job_broker as broker_module

    monkeypatch.setattr(broker_module, "CGROUP_ROOT", str(tmp_path))
    group = tmp_path / TEST_CGROUP.strip("/")
    group.parent.mkdir(parents=True)
    if content is not None:
        group.mkdir()
        (group / "cgroup.events").write_text(content)
    system = LinuxSystem(str(tmp_path / "notify"))
    if content == "junk\n":
        with pytest.raises(OSError):
            system.job_cgroup_populated()
        return
    expected = {None: None, "populated 0\nfrozen 0\n": False, "populated 1\n": True}[content]
    assert system.job_cgroup_populated() is expected


def test_job_cgroup_lookup_fails_closed_when_more_than_the_unit_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.whole_job_broker as broker_module

    monkeypatch.setattr(broker_module, "CGROUP_ROOT", str(tmp_path / "absent"))
    with pytest.raises(OSError):
        LinuxSystem(str(tmp_path / "notify")).job_cgroup_populated()
    monkeypatch.setattr(broker_module, "CGROUP_ROOT", str(tmp_path))
    (tmp_path / TEST_CGROUP.strip("/")).mkdir(parents=True)  # unit cgroup without events file
    with pytest.raises(OSError):
        LinuxSystem(str(tmp_path / "notify")).job_cgroup_populated()


def test_unit_check_failure_at_restart_is_unverifiable_not_idle(env, monkeypatch) -> None:
    system, _, _, _ = env

    def broken() -> bool | None:
        raise OSError("EIO")

    monkeypatch.setattr(system, "job_cgroup_populated", broken)
    broker = make(env)
    broker.startup({})
    assert broker.mode == "unverifiable"
    with pytest.raises(Refused):
        broker.start()


# Durable stop intent: a requested stop survives a broker restart, a plain hold does not kill,
# and a retry only ever goes through the exact restored pinned pair, a bounded number of times.


def _kill_targets(system: FakeSystem, monkeypatch) -> list:
    """Record the device/inode pair of every handle the broker writes ``cgroup.kill`` through."""
    targets: list = []
    real = system.kill

    def kill(handles: Handles) -> None:
        targets.append(handles.ids())
        real(handles)

    monkeypatch.setattr(system, "kill", kill)
    return targets


def _failed_stop(env) -> tuple[Broker, str]:
    """An operator stop whose kill does not empty the cgroup (stop intent recorded, held)."""
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.phase(DRIVER_PIDFD, "core", lambda: None)
    system.empties_on_kill = False
    with pytest.raises(TimeoutError):
        broker.stop(broker.record.attempt)
    state = persisted(directory)
    assert state["held"] and state["stop"] == state["job"] and state["stops"] == 1
    return broker, broker.record.attempt


def test_stop_intent_is_durable_before_the_kill(env) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    job = persisted(directory)["job"]
    broker.stop(broker.record.attempt)
    assert system.kills[0]["held"] and (system.kills[0]["stop"], system.kills[0]["stops"]) == (
        job,
        1,
    )
    state = persisted(directory)
    assert (state["job"], state["stop"], state["stops"]) == ("", "", 0)
    assert not broker.status()["stop_requested"] and broker.status()["stop_attempts"] == 0


def test_crash_mid_stop_is_retried_after_restart_through_the_exact_pinned_pair(
    env, monkeypatch
) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    record = broker.record
    real_kill = system.kill

    def crash(handles: Handles) -> None:
        raise Crash()

    monkeypatch.setattr(system, "kill", crash)
    with pytest.raises(Crash):
        broker.stop(record.attempt)
    monkeypatch.setattr(system, "kill", real_kill)
    assert persisted(directory)["stop"] == record.encode() and system.populated_value
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup(system.inherited())
    status = restarted.status()
    assert status["mode"] == "held" and status["stop_requested"] and status["stop_attempts"] == 1
    assert "retried" in status["outcome"] and targets == []  # startup itself kills nothing
    restarted.tick()
    assert targets == [(record.kill, record.events)]  # only the exact recorded pair
    assert restarted.mode == "idle" and not persisted(directory)["job"] and system.store == []
    with pytest.raises(PermissionError):
        broker.stop(record.attempt)  # the crashed instance cannot act on the new epoch


def test_failed_stop_is_retried_after_restart_and_the_count_continues(env, monkeypatch) -> None:
    system, _, directory, _ = env
    _, attempt = _failed_stop(env)
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.stop_requested and restarted.stop_retry
    with pytest.raises(Refused):
        restarted.phase(DRIVER_PIDFD, "studies", lambda: pytest.fail("phase admitted"))
    restarted.tick()  # second attempt: still does not empty
    assert len(system.kills) == 2 and persisted(directory)["stops"] == 2
    assert restarted.mode == "held" and restarted.record.attempt == attempt
    system.empties_on_kill = True
    system.clock += STOP_RETRY_SECONDS + 1
    restarted.tick()
    assert len(system.kills) == 3 and restarted.mode == "idle" and system.store == []
    assert persisted(directory)["held"] and not persisted(directory)["job"]


# Fail-closed restart (Sol): every same-boot record whose exact pinned pair is restored becomes a
# stop target, whatever the record said, and the inferred intent is durable before the kill.


def _assert_restart_stops_exact_pair(env, monkeypatch, record: JobRecord) -> Broker:
    system, _, directory, _ = env
    targets = _kill_targets(system, monkeypatch)
    before = len(system.kills)
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.stop_requested and restarted.stop_retry
    assert targets == []  # startup itself kills nothing
    with pytest.raises(Refused):
        restarted.phase(DRIVER_PIDFD, "studies", lambda: pytest.fail("phase admitted"))
    restarted.tick()
    assert targets == [(record.kill, record.events)]  # only the exact recorded pair
    at_kill = system.kills[before]
    assert at_kill["held"] and at_kill["job"] == record.encode()
    assert (at_kill["stop"], at_kill["stops"]) == (record.encode(), 1)  # durable before the kill
    assert restarted.mode == "idle" and system.store == [] and not persisted(directory)["job"]
    assert restarted.outcome == "stop after broker restart"
    return restarted


def test_released_job_is_stopped_after_restart(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.phase(DRIVER_PIDFD, "core", lambda: None)  # mid-phase: it is not allowed to finish
    state = persisted(directory)
    assert not state["held"] and (state["stop"], state["stops"]) == ("", 0)
    _assert_restart_stops_exact_pair(env, monkeypatch, broker.record)


def test_plain_held_job_is_stopped_after_restart(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.state.hold(broker.record.encode(), BOOT, broker.state.epoch)
    state = persisted(directory)
    assert state["held"] and (state["stop"], state["stops"]) == ("", 0)  # no recorded intent
    _assert_restart_stops_exact_pair(env, monkeypatch, broker.record)


def test_inferred_restart_stop_that_does_not_empty_is_bounded(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    system.empties_on_kill = False
    restarted = make(env)
    restarted.startup(system.inherited())
    for _ in range(STOP_ATTEMPTS * 3):
        restarted.tick()
        system.clock += STOP_RETRY_SECONDS + 1
    assert len(system.kills) == STOP_ATTEMPTS and persisted(directory)["stops"] == STOP_ATTEMPTS
    assert persisted(directory)["stop"] == broker.record.encode()
    assert restarted.mode == "held" and not restarted.stop_retry and "used up" in restarted.outcome
    again = make(env)  # the next restart keeps the exhausted budget: no new kill
    again.startup(system.inherited())
    assert again.stop_requested and not again.stop_retry and again.stop_attempts == STOP_ATTEMPTS
    for _ in range(3):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        again.tick()
    assert len(system.kills) == STOP_ATTEMPTS and again.mode == "held"


def test_refused_phase_hold_is_not_a_stop_intent(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    broker.state.hold(broker.record.encode(), BOOT, broker.state.epoch)  # e.g. a concurrent hold
    with pytest.raises(PermissionError):
        broker.phase(DRIVER_PIDFD, "core", lambda: pytest.fail("phase admitted"))
    assert broker.mode == "held" and not broker.stop_requested
    assert persisted(directory)["stop"] == "" and system.kills == []
    _assert_restart_stops_exact_pair(env, monkeypatch, broker.record)  # a restart fails closed


@pytest.mark.parametrize("record_state", ["released", "held"])
@pytest.mark.parametrize("tamper", ["missing", "other-file", "extra-name"])
def test_restart_without_intent_and_mismatched_pair_never_kills(
    env, monkeypatch, record_state: str, tamper: str
) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    if record_state == "held":
        broker.state.hold(broker.record.encode(), BOOT, broker.state.epoch)
    inherited = system.inherited()
    if tamper == "missing":
        os.close(inherited.pop(KILL_NAME)[0])
    elif tamper == "other-file":
        other = system.tmp / "other-events"
        other.write_text("")
        os.close(inherited[EVENTS_NAME][0])
        inherited[EVENTS_NAME] = [os.open(other, os.O_RDONLY)]
    else:
        inherited["stray"] = [os.dup(inherited[KILL_NAME][0])]
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup(inherited)
    assert restarted.mode == "unverifiable" and restarted.handles is None
    for _ in range(STOP_ATTEMPTS + 1):
        system.clock += STOP_EXHAUSTED_SECONDS
        restarted.tick()
    with pytest.raises(Refused):
        restarted.stop(broker.record.attempt)
    assert targets == []
    state = persisted(directory)
    assert state["job"] == broker.record.encode() and (state["stop"], state["stops"]) == ("", 0)


@pytest.mark.parametrize("record_state", ["released", "held"])
def test_prior_boot_record_without_intent_is_never_killed(
    env, monkeypatch, record_state: str
) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    attempt = broker.record.attempt
    if record_state == "held":
        broker.state.hold(broker.record.encode(), BOOT, broker.state.epoch)
    for _, fd in system.store:  # a reboot empties the store
        os.close(fd)
    system.store.clear()
    system.boot = OTHER_BOOT
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup({})
    assert restarted.mode == "prior-boot" and not restarted.stop_requested
    for _ in range(3):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        restarted.tick()
    assert targets == [] and persisted(directory)["stop"] == ""
    restarted.acknowledge(attempt)
    assert restarted.mode == "idle" and targets == []


@pytest.mark.parametrize("tamper", ["missing", "other-file", "extra-name"])
def test_stop_intent_with_stale_handles_is_unverifiable_and_never_retried(
    env, monkeypatch, tamper: str
) -> None:
    system, _, directory, _ = env
    broker, attempt = _failed_stop(env)
    inherited = system.inherited()
    if tamper == "missing":
        os.close(inherited.pop(EVENTS_NAME)[0])
    elif tamper == "other-file":
        other = system.tmp / "other-kill"
        other.write_text("")
        os.close(inherited[KILL_NAME][0])
        inherited[KILL_NAME] = [os.open(other, os.O_WRONLY)]
    else:
        inherited["stray"] = [os.dup(inherited[KILL_NAME][0])]
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup(inherited)
    status = restarted.status()
    assert status["mode"] == "unverifiable" and status["stop_requested"]
    assert "not retried" in status["outcome"] and restarted.handles is None
    for _ in range(STOP_ATTEMPTS + 1):
        system.clock += STOP_EXHAUSTED_SECONDS
        restarted.tick()
    for call in (restarted.start, lambda: restarted.stop(attempt)):
        with pytest.raises(Refused):
            call()
    assert targets == []  # nothing is killed: not the old pair, not anything else
    assert persisted(directory)["stop"] == persisted(directory)["job"]  # evidence kept


def test_stop_intent_for_another_job_refuses_startup(env, monkeypatch) -> None:
    system, _, directory, _ = env
    _failed_stop(env)
    state = persisted(directory)
    other = JobRecord("0" * 32, "cd" * 16, CGROUP_ID, (1, 2), (3, 4)).encode()
    tampered = {**state, "stop": other}
    (directory / "admission.json").write_text(json.dumps(tampered))
    targets = _kill_targets(system, monkeypatch)
    with pytest.raises(ValueError):
        make(env)  # corrupt state: no broker instance, no kill
    assert targets == [] and persisted(directory) == tampered


def test_prior_boot_stop_intent_is_not_retried_and_clears_on_acknowledge(env, monkeypatch) -> None:
    system, _, directory, _ = env
    _, attempt = _failed_stop(env)
    for name, fd in system.store:  # a reboot empties the store
        os.close(fd)
    system.store.clear()
    system.boot = OTHER_BOOT
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup({})
    assert restarted.mode == "prior-boot" and restarted.status()["stop_requested"]
    restarted.tick()
    assert targets == []
    restarted.acknowledge(attempt)
    assert restarted.mode == "idle" and not restarted.status()["stop_requested"]
    assert (persisted(directory)["stop"], persisted(directory)["stops"]) == ("", 0)


def test_uninterruptible_job_retries_are_bounded_and_survive_no_restart(env, monkeypatch) -> None:
    """A task in uninterruptible sleep ignores SIGKILL until its I/O ends: the cgroup stays
    populated whatever the broker does. Retries stop at the budget, across restarts too, and the
    job still clears on its own once the task finally dies."""
    system, _, directory, logs = env
    broker, attempt = _failed_stop(env)
    for _ in range(STOP_ATTEMPTS * 3):
        system.clock += STOP_RETRY_SECONDS + 1
        broker.tick()
    assert len(system.kills) == STOP_ATTEMPTS and persisted(directory)["stops"] == STOP_ATTEMPTS
    assert broker.mode == "held" and not broker.stop_retry and "used up" in broker.outcome
    assert broker.retry_at is not None and broker.retry_at - system.clock > STOP_RETRY_SECONDS
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.stop_requested and not restarted.stop_retry
    assert "used up" in restarted.outcome
    for _ in range(3):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        restarted.tick()
    assert len(system.kills) == STOP_ATTEMPTS  # a restart does not re-arm the retries
    system.populated_value = None  # the I/O completed and the pending SIGKILL took effect
    restarted.tick()
    assert restarted.mode == "idle" and system.store == [] and not persisted(directory)["job"]
    assert restarted.outcome.startswith("job ended")


def test_operator_stop_after_exhaustion_gets_a_fresh_budget(env) -> None:
    system, _, directory, _ = env
    broker, attempt = _failed_stop(env)
    for _ in range(STOP_ATTEMPTS):
        system.clock += STOP_RETRY_SECONDS + 1
        broker.tick()
    assert not broker.stop_retry and persisted(directory)["stops"] == STOP_ATTEMPTS
    with pytest.raises(TimeoutError):
        broker.stop(attempt)
    assert persisted(directory)["stops"] == 1 and broker.stop_retry
    system.empties_on_kill = True
    system.clock += STOP_RETRY_SECONDS + 1
    broker.tick()
    assert broker.mode == "idle" and not persisted(directory)["job"]


def test_stop_intent_write_failure_is_still_bounded(env, monkeypatch) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise OSError("disk")

    monkeypatch.setattr(broker.state, "held_job", broken)
    with pytest.raises(OSError):
        broker.stop(broker.record.attempt)
    for _ in range(STOP_ATTEMPTS * 3):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        broker.tick()
    assert len(calls) == STOP_ATTEMPTS and system.kills == []  # no kill without the intent
    assert broker.mode == "held" and not broker.stop_retry


def test_unadmitted_stop_intent_is_retried_after_restart(env, monkeypatch) -> None:
    system, _, directory, _ = env
    system.empties_on_kill = False
    broker = _refused_registration(env, monkeypatch)
    record = broker.record
    assert persisted(directory)["stop"] == record.encode() and persisted(directory)["stops"] == 1
    targets = _kill_targets(system, monkeypatch)
    system.empties_on_kill = True
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.stop_retry
    restarted.tick()
    assert targets == [(record.kill, record.events)]
    assert restarted.mode == "idle" and system.store == [] and not persisted(directory)["job"]


# Sol blocker 1: once the stop budget is used up, an already-empty cgroup is cleaned up without
# writing cgroup.kill again and without counting another stop attempt, however long the store
# refuses to drain.


@pytest.mark.parametrize("start", ["operator-stop", "job-ended"])
def test_exhausted_stop_cleans_an_empty_cgroup_without_new_kills(env, start: str) -> None:
    system, _, directory, _ = env
    broker, _ = admitted(env)
    attempt = broker.record.attempt
    # The real _drain_verified fails every time: the store reads back non-empty after removal.
    system.dump_override = [entry(KILL_NAME, broker.handles.kill)]
    if start == "operator-stop":
        with pytest.raises(RuntimeError):
            broker.stop(attempt)
    else:
        system.populated_value = None  # the job ended on its own; the cgroup is already empty
        broker.tick()
    assert len(system.kills) == 1 and broker.mode == "held"
    for _ in range(STOP_ATTEMPTS * 3):  # well over 9 ticks, each past any pause
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        broker.tick()
        assert len(system.kills) <= STOP_ATTEMPTS
        assert persisted(directory)["stops"] <= STOP_ATTEMPTS
    assert len(system.kills) == STOP_ATTEMPTS and persisted(directory)["stops"] == STOP_ATTEMPTS
    assert broker.stop_attempts == STOP_ATTEMPTS and not broker.stop_retry
    assert broker.mode == "held" and broker.handles is not None
    assert persisted(directory)["job"] == broker.record.encode()
    assert "used up" in broker.outcome
    system.dump_override = None  # the store drains at last: a verified-empty cleanup clears it
    system.clock += STOP_EXHAUSTED_SECONDS + 1
    broker.tick()
    assert len(system.kills) == STOP_ATTEMPTS  # cleared without another kill
    assert broker.mode == "idle" and broker.handles is None and system.store == []
    state = persisted(directory)
    assert state["held"] and (state["job"], state["stop"], state["stops"]) == ("", "", 0)


def test_exhausted_cleanup_after_restart_never_kills(env) -> None:
    system, _, directory, _ = env
    broker, attempt = _failed_stop(env)
    for _ in range(STOP_ATTEMPTS):
        system.clock += STOP_RETRY_SECONDS + 1
        broker.tick()
    assert len(system.kills) == STOP_ATTEMPTS
    restarted = make(env)
    restarted.startup(system.inherited())
    system.populated_value = None
    system.dump_override = [entry(KILL_NAME, restarted.handles.kill)]
    for _ in range(10):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        restarted.tick()
    assert len(system.kills) == STOP_ATTEMPTS and persisted(directory)["stops"] == STOP_ATTEMPTS
    assert restarted.mode == "held" and restarted.record.attempt == attempt
    system.dump_override = None
    system.clock += STOP_EXHAUSTED_SECONDS + 1
    restarted.tick()
    assert restarted.mode == "idle" and len(system.kills) == STOP_ATTEMPTS


def test_exhausted_cleanup_waits_while_the_cgroup_is_still_populated(env) -> None:
    system, _, directory, _ = env
    broker, _ = _failed_stop(env)  # never empties: an uninterruptible task
    for _ in range(STOP_ATTEMPTS * 2):
        system.clock += STOP_EXHAUSTED_SECONDS + 1
        broker.tick()
    assert len(system.kills) == STOP_ATTEMPTS and broker.mode == "held"
    assert persisted(directory)["job"] == broker.record.encode()


# Sol blocker 2: a registration whose release record was renamed into place but whose directory
# sync failed must still leave a durable stop intent when the rollback kill fails.


def _torn_release(monkeypatch, fd: int, directory: Path, *, persistent: bool = False) -> list:
    """Fail the directory fsync right after the release record is actually renamed into place."""
    real_replace, real_fsync = os.replace, os.fsync
    armed: list[bool] = []

    def replace(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if dst == "admission.json" and not persisted(directory)["held"]:
            armed.append(True)  # the release is on disk now, not yet synced

    def fsync(target: int) -> None:
        if target == fd and armed and (persistent or len(armed) == 1):
            armed.append(False)
            raise OSError("directory sync failed")
        real_fsync(target)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", fsync)
    return armed


@pytest.mark.parametrize("kill_failure", ["kill-raises", "never-empties"])
def test_torn_release_rollback_records_a_durable_stop_intent(
    env, monkeypatch, kill_failure: str
) -> None:
    system, fd, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    armed = _torn_release(monkeypatch, fd, directory)
    if kill_failure == "kill-raises":

        def broken_kill(handles: Handles) -> None:
            raise OSError("EBUSY")

        monkeypatch.setattr(system, "kill", broken_kill)
    else:
        system.empties_on_kill = False
    with pytest.raises(OSError, match="directory sync failed"):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert armed[:2] == [True, False]  # the failure came after the actual rename
    record = broker.record
    assert broker.mode == "held" and broker.handles is not None
    state = persisted(directory)
    assert state["held"] and state["job"] == record.encode()
    assert (state["stop"], state["stops"]) == (record.encode(), 1)  # durable stop intent
    assert broker.stop_requested and broker.stop_attempts == 1
    monkeypatch.undo()
    system.empties_on_kill = True
    targets = _kill_targets(system, monkeypatch)
    restarted = make(env)
    restarted.startup(system.inherited())
    assert restarted.mode == "held" and restarted.stop_requested and restarted.stop_retry
    restarted.tick()
    assert targets == [(record.kill, record.events)]  # retried through the exact pinned pair
    assert restarted.mode == "idle" and system.store == [] and not persisted(directory)["job"]


def test_torn_release_instance_never_admits_again(env, monkeypatch) -> None:
    system, fd, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    system.empties_on_kill = False
    _torn_release(monkeypatch, fd, directory)
    with pytest.raises(OSError):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    monkeypatch.undo()
    system.empties_on_kill = True
    broker.stop(broker.record.attempt)  # the in-process stop still works through the handles
    assert broker.mode == "idle" and system.store == []
    broker.start()
    with pytest.raises(PermissionError):  # a release on the torn instance is refused
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert persisted(directory)["held"] and not persisted(directory)["job"]
    assert broker.mode == "idle" and system.store == []  # the unadmitted retry was stopped


def test_torn_release_with_failing_rollback_write_is_unverifiable(env, monkeypatch) -> None:
    system, fd, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    system.empties_on_kill = False
    _torn_release(monkeypatch, fd, directory, persistent=True)
    with pytest.raises(OSError):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert broker.mode == "unverifiable" and broker.handles is not None
    for call in (broker.start, lambda: broker.stop(broker.record.attempt)):
        with pytest.raises(Refused):
            call()


def test_torn_release_with_failed_rollback_rename_is_stopped_after_restart(
    env, monkeypatch
) -> None:
    """The release record is renamed into place, the directory sync fails, the rollback kill does
    not empty the cgroup and the rollback's own rename fails: the disk keeps the unsynced
    release. A restart treats that released record as a stop target through the exact pair."""
    system, fd, directory, _ = env
    broker = make(env)
    broker.startup({})
    broker.start()
    system.empties_on_kill = False
    real_replace, real_fsync = os.replace, os.fsync
    events: list[str] = []

    def replace(src, dst, *args, **kwargs):
        if dst == "admission.json" and "renamed" in events:
            events.append("rollback-rename-failed")
            raise OSError("rollback rename failed")
        real_replace(src, dst, *args, **kwargs)
        if dst == "admission.json" and not persisted(directory)["held"]:
            events.append("renamed")

    def fsync(target: int) -> None:
        if target == fd and events == ["renamed"]:
            events.append("sync-failed")
            raise OSError("directory sync failed")
        real_fsync(target)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="directory sync failed"):
        broker.register(DRIVER_PIDFD, lambda: pytest.fail("go sent"))
    assert events == ["renamed", "sync-failed", "rollback-rename-failed"]
    record = broker.record
    assert broker.mode == "unverifiable" and len(system.kills) == 1
    state = persisted(directory)
    assert not state["held"] and state["job"] == record.encode()  # the unsynced release
    assert (state["stop"], state["stops"]) == ("", 0)  # no intent reached the disk
    assert not [p for p in directory.iterdir() if p.name.startswith(".admission-")]
    monkeypatch.undo()
    system.empties_on_kill = True
    _assert_restart_stops_exact_pair(env, monkeypatch, record)
