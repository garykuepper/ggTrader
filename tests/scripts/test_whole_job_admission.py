"""Filesystem fixture tests for the offline admission store."""

import fcntl
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from scripts.whole_job_admission import AdmissionState


@pytest.fixture
def directory(tmp_path: Path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield fd
    finally:
        os.close(fd)


def store(directory: int) -> AdmissionState:
    return AdmissionState(directory, expected_owner=os.geteuid())


def test_restart_starts_held_and_invalidates_prior_instance(directory: int) -> None:
    first = store(directory)
    with pytest.raises(PermissionError):
        with first.admission("job-a", "boot-a"):
            pass
    first.release("job-a", "boot-a")
    with first.admission("job-a", "boot-a"):
        pass
    restarted = store(directory)
    with pytest.raises(PermissionError):
        with restarted.admission("job-a", "boot-a"):
            pass
    with pytest.raises(PermissionError):
        first.release("job-a", "boot-a")


def test_identity_mismatch_and_explicit_hold_refuse_admission(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        state.release("job-b", "boot-a")
    for job, boot in [("job-b", "boot-a"), ("job-a", "boot-b")]:
        with pytest.raises(PermissionError):
            with state.admission(job, boot):
                pass
    state.hold("job-a", "boot-a", state.epoch)
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_failed_launch_persists_a_hold(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    with pytest.raises(RuntimeError, match="launch failed"):
        with state.admission("job-a", "boot-a"):
            raise RuntimeError("launch failed")
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_rejects_writable_directory(tmp_path: Path) -> None:
    tmp_path.chmod(0o777)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(PermissionError):
            store(fd)
    finally:
        os.close(fd)


def test_lock_symlink_is_not_followed(directory: int, tmp_path: Path) -> None:
    (tmp_path / "admission.lock").symlink_to(tmp_path / "target")
    with pytest.raises(OSError):
        store(directory)
    assert not (tmp_path / "target").exists()


def test_corrupt_state_is_not_treated_as_open(directory: int, tmp_path: Path) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    (tmp_path / "admission.json").write_text("broken")
    with pytest.raises((ValueError, PermissionError)):
        with state.admission("job-a", "boot-a"):
            pass


def test_failed_directory_sync_disables_instance(directory: int, monkeypatch) -> None:
    state = store(directory)
    fsync = os.fsync

    def fail_directory(fd: int) -> None:
        if fd == directory:
            raise OSError("sync failed")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="sync failed"):
        state.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_hold_waits_for_in_progress_launch(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    entered = threading.Event()
    completed = threading.Event()

    def hold() -> None:
        entered.set()
        state.hold("job-a", "boot-a", state.epoch)
        completed.set()

    with state.admission("job-a", "boot-a"):
        thread = threading.Thread(target=hold)
        thread.start()
        assert entered.wait(2)
        assert not completed.wait(0.1)
    thread.join(timeout=2)
    assert completed.is_set()
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_corrupt_state_latches_and_startup_preserves_evidence(
    directory: int, tmp_path: Path
) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    path = tmp_path / "admission.json"
    saved = path.read_bytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        state.hold("job-a", "boot-a", state.epoch)
    with pytest.raises(ValueError):
        store(directory)
    assert path.read_bytes() == b"corrupt"
    path.write_bytes(saved)
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_epoch_failure_is_latched(directory: int, tmp_path: Path) -> None:
    first = store(directory)
    first.release("job-a", "boot-a")
    path = tmp_path / "admission.json"
    old = path.read_bytes()
    store(directory)
    with pytest.raises(PermissionError):
        first.hold("job-a", "boot-a", first.epoch)
    path.write_bytes(old)
    with pytest.raises(PermissionError):
        with first.admission("job-a", "boot-a"):
            pass


def test_short_reads_are_completed(directory: int, monkeypatch) -> None:
    state = store(directory)
    original = os.read
    monkeypatch.setattr(os, "read", lambda fd, size: original(fd, min(size, 4)))
    state.release("job-a", "boot-a")
    with state.admission("job-a", "boot-a"):
        pass


def test_inheritable_directory_fd_is_rejected(directory: int) -> None:
    os.set_inheritable(directory, True)
    try:
        with pytest.raises(PermissionError):
            store(directory)
    finally:
        os.set_inheritable(directory, False)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "mode"])
def test_rejects_unsafe_state_file(directory: int, tmp_path: Path, kind: str) -> None:
    state = store(directory)
    path = tmp_path / "admission.json"
    if kind == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "target")
    elif kind == "hardlink":
        os.link(path, tmp_path / "hardlink")
    else:
        path.chmod(0o644)
    with pytest.raises((OSError, PermissionError)):
        state.hold("job-a", "boot-a", state.epoch)


def test_fifo_state_fails_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "admission.json", 0o600)
    code = (
        "import os, sys\n"
        "from scripts.whole_job_admission import AdmissionState\n"
        "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)\n"
        "AdmissionState(fd, expected_owner=os.geteuid())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, timeout=2, check=False
    )
    assert result.returncode != 0


def test_lock_replacement_while_waiting_is_rejected(
    directory: int, tmp_path: Path, monkeypatch
) -> None:
    state = store(directory)
    original = fcntl.flock

    def replace_lock(fd: int, operation: int) -> None:
        original(fd, operation)
        path = tmp_path / "admission.lock"
        path.unlink()
        path.touch(mode=0o600)

    monkeypatch.setattr(fcntl, "flock", replace_lock)
    with pytest.raises(PermissionError):
        state.hold("job-a", "boot-a", state.epoch)


def test_failed_fdopen_closes_fd(directory: int, monkeypatch) -> None:
    state = store(directory)
    opened = []

    def fail(fd: int, *args, **kwargs):
        opened.append(fd)
        raise OSError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", fail)
    with pytest.raises(OSError, match="fdopen failed"):
        state.release("job-a", "boot-a")
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_failed_replace_disables_instance(directory: int, monkeypatch) -> None:
    state = store(directory)

    def fail(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        state.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        state.hold("job-a", "boot-a", state.epoch)


def test_lock_open_failure_is_latched(directory: int, tmp_path: Path) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    lock = tmp_path / "admission.lock"
    lock.unlink()
    lock.symlink_to(tmp_path / "missing")
    with pytest.raises(OSError):
        state.hold("job-a", "boot-a", state.epoch)
    lock.unlink()
    lock.touch(mode=0o600)
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


def test_exception_cannot_overwrite_replacement_epoch(directory: int, tmp_path: Path) -> None:
    first = store(directory)
    first.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        with first.admission("job-a", "boot-a"):
            (tmp_path / "admission.lock").unlink()
            second = store(directory)
            second.clear("job-a", "boot-a", second.epoch)
            second.release("job-b", "boot-a")
            raise RuntimeError("launch failed")
    with second.admission("job-b", "boot-a"):
        pass


def test_exception_cannot_overwrite_changed_epoch_with_same_lock(
    directory: int, tmp_path: Path
) -> None:
    first = store(directory)
    first.release("job-a", "boot-a")
    path = tmp_path / "admission.json"
    with pytest.raises(PermissionError):
        with first.admission("job-a", "boot-a"):
            data = json.loads(path.read_text())
            data["epoch"] = "a" * 64
            path.write_text(json.dumps(data))
            raise RuntimeError("launch failed")
    assert json.loads(path.read_text())["epoch"] == "a" * 64


def test_directory_becoming_public_disables_instance(directory: int, tmp_path: Path) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(PermissionError):
            state.hold("job-a", "boot-a", state.epoch)
    finally:
        tmp_path.chmod(0o700)
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass


@pytest.mark.parametrize(
    "changes",
    [
        {"epoch": ""},
        {"epoch": "x" * 64},
        {"held": False, "job": "", "boot": ""},
        {"held": True, "job": "job-a", "boot": ""},
        {"held": False, "job": "x" * 513, "boot": "boot-a"},
    ],
)
def test_invalid_serialized_identity_is_rejected(
    directory: int, tmp_path: Path, changes: dict[str, object]
) -> None:
    state = store(directory)
    path = tmp_path / "admission.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, PermissionError)):
        state.hold("job-a", "boot-a", state.epoch)


def test_oversized_serialized_identity_cannot_be_released(directory: int) -> None:
    state = store(directory)
    with pytest.raises(ValueError):
        state.release("🙂" * 512, "🙂" * 512)
    with pytest.raises(PermissionError):
        with state.admission("🙂" * 512, "🙂" * 512):
            pass


@pytest.mark.parametrize("previously_held", [False, True])
def test_restart_retains_previous_job_and_requires_explicit_cleanup(
    directory: int, previously_held: bool
) -> None:
    first = store(directory)
    first.release("job-a", "boot-a")
    if previously_held:
        first.hold("job-a", "boot-a", first.epoch)
    restarted = store(directory)
    snapshot = restarted.snapshot()
    assert snapshot == {"epoch": restarted.epoch, "held": True, "job": "job-a", "boot": "boot-a"}
    with pytest.raises(PermissionError):
        restarted.release("job-b", "boot-b")
    restarted.clear("job-a", "boot-a", restarted.epoch)
    restarted.release("job-b", "boot-b")


def test_stale_stop_cannot_hold_replacement(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        state.clear("job-a", "boot-a", state.epoch)
    state.hold("job-a", "boot-a", state.epoch)
    with pytest.raises(PermissionError):
        state.clear("job-a", "boot-a", "0" * 64)
    state.clear("job-a", "boot-a", state.epoch)
    state.release("job-b", "boot-b")
    for job, boot, epoch in [
        ("job-a", "boot-a", state.epoch),
        ("job-b", "boot-a", state.epoch),
        ("job-b", "boot-b", "0" * 64),
    ]:
        with pytest.raises(PermissionError):
            state.hold(job, boot, epoch)
        with state.admission("job-b", "boot-b"):
            pass
    state.hold("job-b", "boot-b", state.epoch)
    with pytest.raises(PermissionError):
        with state.admission("job-b", "boot-b"):
            pass


def test_held_job_blocks_cleanup_during_stop_operation(directory: int, tmp_path: Path) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    started = threading.Event()
    finished = threading.Event()

    def cleanup() -> None:
        started.set()
        state.clear("job-a", "boot-a", state.epoch)
        finished.set()

    with state.held_job("job-a", "boot-a", state.epoch):
        assert json.loads((tmp_path / "admission.json").read_text())["held"]
        thread = threading.Thread(target=cleanup)
        thread.start()
        assert started.wait(2)
        assert not finished.wait(0.1)
    thread.join(timeout=2)
    assert finished.is_set()


def test_held_job_failure_stays_held_and_stale_identity_never_enters(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    entered = False
    with pytest.raises(PermissionError):
        with state.held_job("job-b", "boot-a", state.epoch):
            entered = True
    assert not entered
    with pytest.raises(RuntimeError, match="stop failed"):
        with state.held_job("job-a", "boot-a", state.epoch):
            raise RuntimeError("stop failed")
    assert state.snapshot()["held"]
    restarted = store(directory)
    assert restarted.snapshot()["job"] == "job-a"
    assert restarted.snapshot()["held"]


def test_held_job_write_failure_never_enters(directory: int, monkeypatch) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")

    def fail(*args, **kwargs):
        raise OSError("sync failed")

    monkeypatch.setattr(os, "fsync", fail)
    entered = False
    with pytest.raises(OSError, match="sync failed"):
        with state.held_job("job-a", "boot-a", state.epoch):
            entered = True
    assert not entered


def test_held_job_detects_lock_replacement_even_on_stop_failure(
    directory: int, tmp_path: Path
) -> None:
    first = store(directory)
    first.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        with first.held_job("job-a", "boot-a", first.epoch):
            (tmp_path / "admission.lock").unlink()
            replacement = store(directory)
            replacement.clear("job-a", "boot-a", replacement.epoch)
            replacement.release("job-b", "boot-a")
            raise RuntimeError("stop failed")
    with replacement.admission("job-b", "boot-a"):
        pass


def test_retain_records_identity_without_releasing_admission(directory: int) -> None:
    state = store(directory)
    state.retain("job-a", "boot-a")
    snap = state.snapshot()
    assert snap["held"] and (snap["job"], snap["boot"]) == ("job-a", "boot-a")
    with pytest.raises(PermissionError):
        with state.admission("job-a", "boot-a"):
            pass
    with pytest.raises(PermissionError):
        state.retain("job-b", "boot-a")  # a retained identity is never replaced
    with pytest.raises(PermissionError):
        state.release("job-b", "boot-a")
    restarted = store(directory)
    assert restarted.snapshot()["job"] == "job-a"
    restarted.clear("job-a", "boot-a", restarted.epoch)
    assert not restarted.snapshot()["job"]


def test_retain_refuses_while_admission_is_released(directory: int) -> None:
    state = store(directory)
    state.release("job-a", "boot-a")
    with pytest.raises(PermissionError):
        state.retain("job-a", "boot-a")
    with pytest.raises(ValueError):
        store(directory).retain("", "boot-a")
