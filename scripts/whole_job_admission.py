"""Durable fail-closed admission state, not a broker or cgroup authority.

The caller must supply a securely opened, private directory descriptor that remains open,
authenticate all requests, and independently verify boot and pinned cgroup identities.
Production must use the default root owner. This module does not launch or stop processes.
Admission is locked through the caller's launch operation; an exception records a hold.
Each broker startup creates a new epoch and persists a hold before serving requests.
The broker must never fork while admission is locked: close-on-exec does not close a
lock inherited by a forked child until that child execs or exits. Corrupt prior state
is preserved and startup refuses admission until an operator investigates it.
The directory must remain under trusted control: metadata checks detect some changes,
but cannot make a check and write atomic against a privileged concurrent replacement.
Job IDs must uniquely identify individual launches. State does not authorize a kill or
prove cleanup: a trusted broker must verify the pinned cgroup before stop or clear.
"""

import fcntl
import json
import os
import re
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
        self.failed = False
        info = os.fstat(directory)
        self.directory_id = (info.st_dev, info.st_ino)
        self._check_directory()
        self.epoch = secrets.token_hex(32)
        with self._lock():
            previous = self._read(missing_ok=True)
            self._write(
                {
                    "epoch": self.epoch,
                    "held": True,
                    "job": previous["job"] if previous else "",
                    "boot": previous["boot"] if previous else "",
                }
            )

    def _check_directory(self) -> None:
        try:
            info = os.fstat(self.directory)
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != self.owner
                or info.st_mode & 0o077
                or info.st_nlink == 0
                or (info.st_dev, info.st_ino) != self.directory_id
                or os.get_inheritable(self.directory)
            ):
                raise PermissionError("admission directory is not private and trusted")
        except BaseException:
            self.failed = True
            raise

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
    def _lock(self) -> Iterator[int]:
        try:
            self._check_directory()
            fd = os.open(
                "admission.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                0o600,
                dir_fd=self.directory,
            )
        except BaseException:
            self.failed = True
            raise
        try:
            try:
                self._check_file(fd)
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._check_lock(fd)
            except BaseException:
                self.failed = True
                raise
            try:
                yield fd
            finally:
                self._check_lock(fd)
        finally:
            os.close(fd)

    def _check_lock(self, fd: int) -> None:
        try:
            self._check_directory()
            self._check_file(fd)
            held = os.fstat(fd)
            current = os.stat("admission.lock", dir_fd=self.directory, follow_symlinks=False)
            if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
                raise PermissionError("admission lock was replaced")
        except BaseException:
            self.failed = True
            raise

    def _read(self, *, missing_ok: bool = False) -> dict[str, Any] | None:
        try:
            fd = os.open(
                "admission.json",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=self.directory,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise PermissionError("admission state missing") from None
        try:
            self._check_file(fd)
            data = b""
            while len(data) < 4097:
                chunk = os.read(fd, 4097 - len(data))
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(fd)
        if len(data) > 4096:
            raise ValueError("admission state too large")
        state = json.loads(data)
        self._validate(state)
        return state

    @staticmethod
    def _validate(state: Any) -> None:
        if (
            not isinstance(state, dict)
            or set(state) != {"epoch", "held", "job", "boot"}
            or not all(isinstance(state[key], str) for key in ("epoch", "job", "boot"))
            or not isinstance(state["held"], bool)
            or re.fullmatch(r"[0-9a-f]{64}", state["epoch"]) is None
            or len(state["job"]) > 512
            or len(state["boot"]) > 512
            or bool(state["job"]) != bool(state["boot"])
            or (not state["held"] and not state["job"])
            or len(json.dumps(state).encode("utf-8")) > 4096
        ):
            raise ValueError("invalid admission state")

    def _write(self, state: dict[str, Any]) -> None:
        try:
            self._check_directory()
            self._validate(state)
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
            try:
                file = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                raise
            with file:
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
            raise PermissionError("admission instance has failed")
        try:
            state = self._read()
            if state is None or state["epoch"] != self.epoch:
                raise PermissionError("stale admission instance")
            return state
        except BaseException:
            self.failed = True
            raise

    def release(self, job: str, boot: str) -> None:
        """Bind a hold release to an independently verified job and boot."""
        if not all(isinstance(value, str) and 0 < len(value) <= 512 for value in (job, boot)):
            raise ValueError("invalid job or boot identifier")
        with self._lock():
            state = self._current()
            if not state["held"] or state["job"]:
                raise PermissionError("prior admission must be held and cleared")
            state.update(held=False, job=job, boot=boot)
            self._write(state)

    def retain(self, job: str, boot: str) -> None:
        """Record an identity that stays held, without ever releasing admission.

        For a job that was never admitted but could not be shown stopped: the identity must
        survive a broker restart so that only a verified stop or explicit cleanup can clear it.
        """
        if not all(isinstance(value, str) and 0 < len(value) <= 512 for value in (job, boot)):
            raise ValueError("invalid job or boot identifier")
        with self._lock():
            state = self._current()
            if not state["held"] or state["job"]:
                raise PermissionError("admission must be held and clear to retain a job")
            state.update(job=job, boot=boot)
            self._write(state)

    def snapshot(self) -> dict[str, Any]:
        """Read this epoch's held state and retained identity under the lock."""
        with self._lock():
            return self._current().copy()

    def hold(self, job: str, boot: str, epoch: str) -> None:
        """Hold only the expected identity; this does not authorize a cgroup stop."""
        with self.held_job(job, boot, epoch):
            pass

    @contextmanager
    def held_job(self, job: str, boot: str, epoch: str) -> Iterator[None]:
        """Persist an identity-matched hold and keep the admission lock until exit.

        This is a lifecycle primitive, not authorization. A future broker must authenticate
        the peer and validate a pinned cgroup before using this around any stop operation.
        """
        with self._lock():
            state = self._match(job, boot, epoch)
            state["held"] = True
            self._write(state)
            yield

    def clear(self, job: str, boot: str, epoch: str) -> None:
        """Forget a held identity after the trusted broker independently verifies cleanup."""
        with self._lock():
            state = self._match(job, boot, epoch)
            if not state["held"]:
                raise PermissionError("active job must be held before cleanup")
            state.update(job="", boot="")
            self._write(state)

    def _match(self, job: str, boot: str, epoch: str) -> dict[str, Any]:
        state = self._current()
        if (
            not job
            or not boot
            or (state["job"], state["boot"], state["epoch"]) != (job, boot, epoch)
        ):
            raise PermissionError("job identity or epoch does not match")
        return state

    @contextmanager
    def admission(self, job: str, boot: str) -> Iterator[None]:
        """Hold the admission lock while the caller validates and launches a job."""
        with self._lock() as lock_fd:
            state = self._current()
            if state["held"] or state["job"] != job or state["boot"] != boot:
                raise PermissionError("job is held or identity does not match")
            try:
                yield
            except BaseException:
                self._check_lock(lock_fd)
                state = self._current()
                self._check_lock(lock_fd)
                state["held"] = True
                self._write(state)
                raise
