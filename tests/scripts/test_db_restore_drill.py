"""Exercise restore drill safety with isolated fakes."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import db_restore_drill as drill


class Connection:
    """Provide a scratch connection with a configurable extension version."""

    autocommit = False

    def __init__(self, version: str) -> None:
        self.version = version

    def cursor(self) -> "Connection":
        return self

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: str) -> None:
        pass

    def fetchone(self) -> tuple[str]:
        return (self.version,)

    def close(self) -> None:
        pass


def setup_drill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str = "2.1") -> tuple:
    """Build inputs and fake external operations."""
    root = tmp_path / "scratch"
    monkeypatch.setattr(drill, "SCRATCH_ROOT", root)
    monkeypatch.setattr(drill, "snapshot", lambda conn: {"counts": {"a": 1}, "digests": {"a": "x"}})
    dump = tmp_path / "data.dump"
    dump.write_bytes(b"dump")
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'dump').hexdigest()}  data.dump\n")
    counts = tmp_path / "counts.json"
    counts.write_text(
        json.dumps({"timescaledb": "2.1", "after": {"counts": {"a": 1}, "digests": {"a": "x"}}})
    )
    out = tmp_path / "report.json"
    calls = []

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="127.0.0.1:54321\n")

    def connect(**kwargs: object) -> Connection:
        return Connection(version)

    return (
        ["--dump", str(dump), "--counts", str(counts), "--out", str(out)],
        calls,
        runner,
        connect,
        root,
        out,
    )


def test_refuses_name_not_ours() -> None:
    with pytest.raises(ValueError):
        drill.refuse_unsafe("ggtrader_db", drill.SCRATCH_ROOT / "test")


def test_refuses_scratch_dir_outside_scratch_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        drill.refuse_unsafe(drill.container_name("abc"), tmp_path)


def test_compare_flags_missing_table_and_changed_count() -> None:
    result = drill.compare(
        {"counts": {"a": 1, "b": 2}, "digests": {}}, {"counts": {"a": 2, "c": 1}, "digests": {}}
    )
    assert not result["ok"]
    assert all(not entry["ok"] for entry in result["tables"].values())


def test_compare_checks_digests() -> None:
    result = drill.compare(
        {"counts": {}, "digests": {"a": "x"}}, {"counts": {}, "digests": {"a": "y"}}
    )
    assert not result["ok"] and not result["digests"]["a"]["ok"]


def test_extension_version_mismatch_aborts_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch, "wrong")
    assert drill.main(args, runner=runner, connect=connect) != 0
    assert not any("pg_restore" in command for command in calls)
    assert not list(root.iterdir())


def test_cleanup_removes_only_its_own_container_and_folder_even_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    root.mkdir()
    (root / "other").mkdir()

    def fail_restore(command: list[str], **kwargs: object) -> SimpleNamespace:
        result = runner(command, **kwargs)
        if "pg_restore" in command:
            raise subprocess.CalledProcessError(1, command)
        return result

    assert drill.main(args, runner=fail_restore, connect=connect) != 0
    run = next(c for c in calls if c[1] == "run")
    name = run[run.index("--name") + 1]
    assert name.startswith("ggt_restore_drill_")
    assert "127.0.0.1::5432" in run
    assert [c for c in calls if c[1] == "rm"] == [["docker", "rm", "-f", name]]
    assert sorted(p.name for p in root.iterdir()) == ["other"]
    assert all(name in c for c in calls if c[1] in {"exec", "rm", "stop"})


def test_report_has_no_password_or_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    assert drill.main(args, runner=runner, connect=connect) == 0
    run = next(c for c in calls if c[1] == "run")
    password = next(x.split("=", 1)[1] for x in run if x.startswith("POSTGRES_PASSWORD="))
    report = out.read_text()
    assert password not in report and "://" not in report
    assert json.loads(report)["ok"]
    assert not list(root.iterdir())


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        drill.main(["--help"])
    assert exc.value.code == 0
