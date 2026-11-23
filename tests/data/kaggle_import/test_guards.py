"""Tests for import safety checks."""

import hashlib
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ggTrader.data.kaggle_import.guards import (
    BackupStale,
    HashMismatch,
    HashMissing,
    backup_fresh,
    free_space_ok,
    verify_sha256,
)


def test_hash_lookup_and_mismatch(tmp_path):
    path = tmp_path / "sub" / "data"
    path.parent.mkdir()
    path.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  sub/data\n")
    assert verify_sha256(path, sums) == digest
    path.write_bytes(b"changed")
    with pytest.raises(HashMismatch, match="data"):
        verify_sha256(path, sums)
    sums.write_text("")
    with pytest.raises(HashMissing):
        verify_sha256(path, sums)


def test_space_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=110))
    assert free_space_ok(tmp_path, 10, 100)
    assert not free_space_ok(tmp_path, 11, 100)


def test_backup_age(tmp_path):
    now = datetime.now(timezone.utc)
    with pytest.raises(BackupStale):
        backup_fresh(tmp_path, now=now)
    old = tmp_path / "old.dump"
    old.touch()
    os.utime(old, ((now - timedelta(days=8)).timestamp(),) * 2)
    with pytest.raises(BackupStale):
        backup_fresh(tmp_path, now=now)
    fresh = tmp_path / "fresh.dump"
    fresh.touch()
    os.utime(fresh, ((now - timedelta(days=6)).timestamp(),) * 2)
    assert backup_fresh(tmp_path, now=now) == fresh
