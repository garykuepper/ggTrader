"""Filesystem fixture tests for the offline admission store."""

import os
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
    state.hold()
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
        state.hold()
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
