"""Exercise restore drill safety with isolated fakes."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg2 import sql

from scripts import db_restore_drill as drill


class Connection:
    """Provide a scratch connection with a configurable extension version."""

    autocommit = False

    def __init__(self, version: str, events: list | None = None) -> None:
        self.version = version
        self.events = events if events is not None else []

    def cursor(self) -> "Connection":
        return self

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: object) -> None:
        self.events.append(query)

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
    assert drill.main(args, runner=runner, connect=connect, disk_usage=ample_space) != 0
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

    assert drill.main(args, runner=fail_restore, connect=connect, disk_usage=ample_space) != 0
    run = next(c for c in calls if c[1] == "run")
    name = run[run.index("--name") + 1]
    assert name.startswith("ggt_restore_drill_")
    assert "127.0.0.1::5432" in run
    assert [c for c in calls if c[1] == "rm"] == [["docker", "rm", "-f", name]]
    assert sorted(p.name for p in root.iterdir()) == ["other"]
    assert all(name in c for c in calls if c[1] in {"exec", "rm", "stop"})


def test_report_has_no_password_or_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    monkeypatch.setattr(drill.secrets, "token_urlsafe", lambda _: "private-test-value")
    assert drill.main(args, runner=runner, connect=connect, disk_usage=ample_space) == 0
    password = "private-test-value"
    report = out.read_text()
    assert password not in report and "://" not in report
    assert json.loads(report)["ok"]
    assert not list(root.iterdir())


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        drill.main(["--help"])
    assert exc.value.code == 0


def test_cleanup_empties_folder_through_container_before_rmdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def simulate(command: list[str], **kwargs: object) -> SimpleNamespace:
        result = runner(command, **kwargs)
        if command[1:3] == ["run", "-d"]:
            folder = Path(command[command.index("-v") + 1].split(":")[0])
            (folder / "owned").write_text("data")
        if command[1:3] == ["run", "--rm"]:
            folder = Path(command[command.index("-v") + 1].split(":")[0])
            assert (folder / "owned").exists()
            (folder / "owned").unlink()
        return result

    assert drill.main(args, runner=simulate, connect=connect, disk_usage=ample_space) == 0
    assert any(command[1:3] == ["run", "--rm"] for command in calls)
    assert not list(root.iterdir())


def test_cleanup_failure_is_reported_and_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def fail(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["run", "-d"]:
            folder = Path(command[command.index("-v") + 1].split(":")[0])
            (folder / "owned").write_text("data")
        if command[1:3] == ["run", "--rm"]:
            raise subprocess.CalledProcessError(1, command)
        return runner(command, **kwargs)

    assert drill.main(args, runner=fail, connect=connect, disk_usage=ample_space) == 1
    assert json.loads(out.read_text()) == {
        "ok": False,
        "stage": "cleanup",
        "error": "CalledProcessError",
    }
    assert len(list(root.iterdir())) == 1


def test_failed_docker_run_still_attempts_removal_of_its_own_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def fail(command: list[str], **kwargs: object) -> SimpleNamespace:
        result = runner(command, **kwargs)
        if command[1:3] == ["run", "-d"]:
            raise subprocess.CalledProcessError(1, command)
        return result

    assert drill.main(args, runner=fail, connect=connect, disk_usage=ample_space) == 1
    name = calls[0][calls[0].index("--name") + 1]
    assert ["docker", "rm", "-f", name] in calls
    assert json.loads(out.read_text())["stage"] == "start"


def test_refuses_when_scratch_space_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    assert (
        drill.main(
            args, runner=runner, connect=connect, disk_usage=lambda path: SimpleNamespace(free=1)
        )
        == 2
    )
    assert "space" in capsys.readouterr().err.lower()
    assert not calls and not root.exists()
    assert json.loads(out.read_text())["stage"] == "verify"


def test_failure_writes_report_with_stage_and_no_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch, "wrong")
    assert drill.main(args, runner=runner, connect=connect, disk_usage=ample_space) == 1
    assert json.loads(out.read_text()) == {"ok": False, "stage": "connect", "error": "ValueError"}


def test_restore_stderr_saved_next_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def noisy(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "pg_restore" in command:
            raise subprocess.CalledProcessError(1, command, stderr=b"restore problem\n")
        return runner(command, **kwargs)

    assert drill.main(args, runner=noisy, connect=connect, disk_usage=ample_space) == 1
    assert out.with_suffix(".restore.log").read_text() == "restore problem\n"
    assert json.loads(out.read_text())["stage"] == "restore"


def test_password_not_in_docker_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def inspect(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["run", "-d"]:
            password = kwargs["env"]["POSTGRES_PASSWORD"]
            assert "POSTGRES_PASSWORD" in command
            assert all(password not in argument for argument in command)
        return runner(command, **kwargs)

    assert drill.main(args, runner=inspect, connect=connect, disk_usage=ample_space) == 0


def ample_space(path: Path) -> SimpleNamespace:
    return SimpleNamespace(free=10**15)


def test_missing_container_after_failed_start_is_not_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def fail(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["run", "-d"]:
            raise subprocess.CalledProcessError(1, command)
        if command[1] == "rm":
            raise subprocess.CalledProcessError(1, command, stderr=b"nO SuCh CoNtAiNeR")
        return runner(command, **kwargs)

    assert drill.main(args, runner=fail, connect=connect, disk_usage=ample_space) == 1
    assert json.loads(out.read_text())["stage"] == "start"
    assert not list(root.iterdir())


def test_mismatch_report_keeps_table_and_digest_detail_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    monkeypatch.setattr(drill, "snapshot", lambda conn: {"counts": {"a": 2}, "digests": {"a": "y"}})
    assert drill.main(args, runner=runner, connect=connect, disk_usage=ample_space) == 1
    report = json.loads(out.read_text())
    assert report["stage"] == "compare" and report["error"] == "SnapshotMismatch"
    assert report["tables"]["a"] == {"expected": 1, "actual": 2, "ok": False}
    assert report["digests"]["a"] == {"expected": "x", "actual": "y", "ok": False}


def test_cleanup_failure_does_not_hide_primary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch, "wrong")

    def fail(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1] == "rm":
            raise OSError("private detail")
        return runner(command, **kwargs)

    assert drill.main(args, runner=fail, connect=connect, disk_usage=ample_space) == 1
    assert json.loads(out.read_text()) == {
        "ok": False,
        "stage": "connect",
        "error": "ValueError",
        "cleanup_error": "OSError",
    }


def test_empty_scratch_folder_skips_throwaway_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    assert drill.main(args, runner=runner, connect=connect, disk_usage=ample_space) == 0
    assert not any(command[1:3] == ["run", "--rm"] for command in calls)
    assert not list(root.iterdir())


def test_throwaway_cleanup_runs_as_root_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)

    def simulate(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["run", "-d"]:
            folder = Path(command[command.index("-v") + 1].split(":")[0])
            (folder / "owned").touch()
        if command[1:3] == ["run", "--rm"]:
            assert command[command.index("-u") + 1] == "0"
            folder = Path(command[command.index("-v") + 1].split(":")[0])
            (folder / "owned").unlink()
        return runner(command, **kwargs)

    assert drill.main(args, runner=simulate, connect=connect, disk_usage=ample_space) == 0
    assert any(command[1:3] == ["run", "--rm"] for command in calls)


def test_happy_path_runs_sql_and_restore_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    events: list[str] = []

    def record(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "pg_restore" in command:
            events.append("pg_restore")
        return runner(command, **kwargs)

    def new_connection(**kwargs: object) -> Connection:
        return Connection("2.1", events)

    assert drill.main(args, runner=record, connect=new_connection, disk_usage=ample_space) == 0
    assert events == [
        "CREATE DATABASE ggtrader",
        "CREATE EXTENSION timescaledb",
        "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'",
        sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier("universe_reader")),
        sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier("universe_loader")),
        "SELECT timescaledb_pre_restore()",
        "pg_restore",
        "SELECT timescaledb_post_restore()",
        "ANALYZE",
    ]


def test_keep_flag_skips_all_cleanup_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    assert (
        drill.main(args + ["--keep"], runner=runner, connect=connect, disk_usage=ample_space) == 0
    )
    assert not any(c[1] in {"rm", "stop"} or c[1:3] == ["run", "--rm"] for c in calls)
    assert len(list(root.iterdir())) == 1


def test_restore_connects_as_ggtrader_not_postgres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    connections = []

    def record_connect(**kwargs: object) -> Connection:
        connections.append(kwargs)
        return connect(**kwargs)

    assert drill.main(args, runner=runner, connect=record_connect, disk_usage=ample_space) == 0
    assert all(item["user"] == "ggtrader" for item in connections)
    assert connections[0]["dbname"] == "postgres"
    run = next(c for c in calls if c[1:3] == ["run", "-d"])
    assert "POSTGRES_USER=ggtrader" in run
    restore = next(c for c in calls if "pg_restore" in c)
    assert restore[restore.index("-U") + 1] == "ggtrader"


def test_roles_created_before_restore_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls, runner, connect, root, out = setup_drill(tmp_path, monkeypatch)
    events = []

    def record(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "pg_restore" in command:
            events.append("restore")
        return runner(command, **kwargs)

    def connect_record(**kwargs: object) -> Connection:
        return Connection("2.1", events)

    assert drill.main(args, runner=record, connect=connect_record, disk_usage=ample_space) == 0
    roles = [
        sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(name))
        for name in ("universe_reader", "universe_loader")
    ]
    for role in roles:
        assert (
            events.index("CREATE EXTENSION timescaledb")
            < events.index(role)
            < events.index("SELECT timescaledb_pre_restore()")
            < events.index("restore")
        )
