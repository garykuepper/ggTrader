"""Durable fail-closed admission state, not a broker or cgroup authority.

The caller must supply a securely opened, private directory descriptor that remains open,
authenticate all requests, and independently verify boot and pinned cgroup identities.
Production must use the default root owner. This module does not launch or stop processes.
Admission is locked through the caller's launch operation; an exception records a hold.
Each broker startup creates a new epoch and persists a hold before serving requests.
"""

import fcntl
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class AdmissionState:
    """Serialize durable release, admission and hold operations in one trusted directory."""

    def __init__(self, directory: int, *, expected_owner: int = 0) -> None:
        self.directory = directory
        self.owner = expected_owner
        info = os.fstat(directory)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != self.owner or info.st_mode & 0o077:
            raise PermissionError("admission directory is not private and trusted")
        self.epoch = secrets.token_hex(32)
        self.failed = False
        with self._lock():
            self._read(missing_ok=True)
            self._write({"epoch": self.epoch, "held": True, "job": "", "boot": ""})

    def _check_file(self, fd: int) -> None:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != self.owner
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise PermissionError("untrusted admission state file")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        fd = os.open(
            "admission.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=self.directory,
        )
        try:
            self._check_file(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read(self, *, missing_ok: bool = False) -> dict[str, Any] | None:
        try:
            fd = os.open(
                "admission.json",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=self.directory,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise PermissionError("admission state missing") from None
        try:
            self._check_file(fd)
            data = os.read(fd, 4097)
        finally:
            os.close(fd)
        if len(data) > 4096:
            raise ValueError("admission state too large")
        state = json.loads(data)
        if (
            not isinstance(state, dict)
            or set(state) != {"epoch", "held", "job", "boot"}
            or not all(isinstance(state[key], str) for key in ("epoch", "job", "boot"))
            or not isinstance(state["held"], bool)
        ):
            raise ValueError("invalid admission state")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        try:
            self._write_state(state)
        except BaseException:
            self.failed = True
            raise

    def _write_state(self, state: dict[str, Any]) -> None:
        name = f".admission-{secrets.token_hex(16)}"
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=self.directory,
        )
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(state, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, "admission.json", src_dir_fd=self.directory, dst_dir_fd=self.directory)
            os.fsync(self.directory)
        finally:
            try:
                os.unlink(name, dir_fd=self.directory)
            except FileNotFoundError:
                pass

    def _current(self) -> dict[str, Any]:
        if self.failed:
            raise PermissionError("admission instance has a failed state write")
        state = self._read()
        if state is None or state["epoch"] != self.epoch:
            raise PermissionError("stale admission instance")
        return state

    def release(self, job: str, boot: str) -> None:
        """Bind a hold release to an independently verified job and boot."""
        if not all(isinstance(value, str) and 0 < len(value) <= 512 for value in (job, boot)):
            raise ValueError("invalid job or boot identifier")
        with self._lock():
            state = self._current()
            if not state["held"]:
                raise PermissionError("admission is already released")
            state.update(held=False, job=job, boot=boot)
            self._write(state)

    def hold(self) -> None:
        """Persist a hold; the caller must separately stop the verified cgroup."""
        with self._lock():
            state = self._current()
            state["held"] = True
            self._write(state)

    @contextmanager
    def admission(self, job: str, boot: str) -> Iterator[None]:
        """Hold the admission lock while the caller validates and launches a job."""
        with self._lock():
            state = self._current()
            if state["held"] or state["job"] != job or state["boot"] != boot:
                raise PermissionError("job is held or identity does not match")
            try:
                yield
            except BaseException:
                state["held"] = True
                self._write(state)
                raise
